#!/usr/bin/env python
"""Replay a recorded calibration through today's staged search, with no GPU.

A profile keeps everything the search decided on: the machine (`hardware`), every engine's prepared model
(`prepared_all`), the options and workload, and every trial it ran (`calibration_table`). This rebuilds the
memory planner's configurations from the first two, runs the search exactly as `rigtune calibrate` wires it,
and answers each trial from the table by its configuration key. A configuration never measured is answered as
a failed launch, so a pick reached past one is not evidence. Each profile lands in one of three classes:

  exact         the search asked for every recorded trial and nothing else, and reached the recorded pick
  inconclusive  it reached the recorded pick, but past trials never measured (or under an option the replay
                cannot apply), so the pick it would have reached on hardware is unknown
  different     it reached another pick

    PYTHONPATH=. python benchmarks/replay_search.py benchmarks/strategies/results-*/profiles/**/*.json

(PYTHONPATH makes the script use this checkout's rigtune rather than an installed one.) It exits non-zero only
on a regression against BASELINE, the class each recorded profile had when the replay was written: a profile
that was exact and no longer is, or that reached its pick and now reaches another. Profiles that already
differed then cannot fail it. Run it before shipping a change to the search.

What the replay holds fixed. Engine features rigtune detects from the installed software are taken from the
recording: the vLLM version from the profile's `hardware`, suffix decoding where the table holds a suffix trial,
FlashInfer where it holds an fp8_e5m2 trial on vLLM. The predictor and any memory fit made with
`rigtune memory-report --apply` are left out (they live on the machine that calibrated, not in the profile).
The quality probe is off: it records drift without changing the pick, but a profile recorded under the drift
gate that was later withdrawn (`quality` set to a threshold) is at best inconclusive. Budgeted profiles are not
replayed: their stages were cut by wall-clock time, which a replay does not have.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from unittest import mock

from rigtune.backends import registry
from rigtune.calibrate.workload import Workload
from rigtune.memory import plan as memory_plan
from rigtune.models import Config, ModelSpec, Profile, TrialMetrics, TrialResult
from rigtune.pipeline import PlanResult, SearchOptions, calibrate

STRATEGIES = Path(__file__).resolve().parent / "strategies"
CLASSES = ("exact", "inconclusive", "different")  # best first

# Each recorded profile's class when this script was written (2026-11-12, search as of rigtune 0.2.1), by path
# under benchmarks/strategies. The three that differ were made by older searches: an earlier warm-up, and two
# `extract` runs whose recorded draft-model trial at batch 512 today's search no longer asks for.
BASELINE = {
    "results-14b/profiles/4090/balanced-file-dolly-d03d5896-quality-0.020-quant-auto+awq+gptq+w8a8-ttft-p95.json":
        "inconclusive",
    "results-14b/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95.json": "exact",
    "results-dolly/p95-old-warmup/profiles/balanced-file-dolly-d03d5896-ttft-p95.json": "different",
    "results-dolly/p95/profiles/balanced-file-dolly-d03d5896-ttft-p95.json": "inconclusive",
    "results-dolly/profiles/balanced-file-dolly-d03d5896.json": "inconclusive",
    "results-l4/p95/profiles/balanced-sharegpt-ttft-p95.json": "inconclusive",
    "results-llama/profiles/4090/balanced-file-dolly-d03d5896-levels-all-ttft-p95.json": "inconclusive",
    "results-llama/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95-before-fix.json": "inconclusive",
    "results-llama/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95-explore.json": "inconclusive",
    "results-llama/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95-specfirst.json": "exact",
    "results-llama/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95.json": "inconclusive",
    "results-llama/profiles/a100/balanced-file-dolly-d03d5896-ttft-p95-specfirst.json": "exact",
    "results-llama/profiles/a100/balanced-file-dolly-d03d5896-ttft-p95.json": "inconclusive",
    "results-llama/profiles/a40/balanced-file-dolly-d03d5896-ttft-p95-specfirst.json": "exact",
    "results-llama/profiles/a40/balanced-file-dolly-d03d5896-ttft-p95.json": "inconclusive",
    "results-llama/profiles/a40/qwen2.5-3b-balanced-file-dolly-d03d5896-ttft-p95.json": "inconclusive",
    "results-llama/profiles/h100/balanced-file-dolly-d03d5896-ttft-p95.json": "inconclusive",
    "results-real/profiles/extract-run1.profile.json": "different",
    "results-real/profiles/extract-run2.profile.json": "different",
    "results-real/profiles/extract-run3.profile.json": "inconclusive",
    "results-real/sglang/profiles/balanced-sharegpt.json": "exact",
}


def options_of(profile: Profile) -> Tuple[SearchOptions, List[str]]:
    """The SearchOptions whose key() is the profile's `options`, and the recorded options the replay cannot apply.
    Balanced profiles without a `ttft` entry were picked by the median (see SearchOptions.key)."""
    o = profile.options
    pct = o.get("ttft", "p50" if profile.objective == "balanced" else "p95")
    opts = SearchOptions(
        quants=o["quant"].split("+") if "quant" in o else None,
        kv_quant=o.get("kv") != "off", speculative=o.get("spec") != "off", prefix_cache=o.get("prefix") != "off",
        phase_tuning=o.get("prefill") != "off", combine=o.get("combine") != "off", confirm=o.get("confirm") != "off",
        all_levels=o.get("levels") == "all", explore=o.get("explore") == "on",
        budget_s=float(o["budget"].rstrip("s")) if "budget" in o else None,
        ttft_percentile=int(pct.lstrip("p")),
        quality_probe=False,  # it needs the prompts, which the profile does not keep
    )
    cannot = []
    if o.get("quality") not in (None, "on"):  # the withdrawn drift gate, at a threshold: it could refuse
        cannot.append(f"quality={o['quality']}")
    expected = dict(opts.key(profile.objective), **({"quality": o["quality"]} if "quality" in o else {}))
    cannot += [f"{k}={v}" for k, v in sorted(o.items()) if expected.get(k) != v]
    return opts, cannot


def workload_of(profile: Profile) -> Workload:
    w = profile.workload_spec
    fields = ("name", "n_prompts", "prefill_tokens", "decode_tokens", "ttft_ceiling_ms", "tpot_ceiling_ms",
              "shared_prefix_tokens", "seed", "source", "natural_stop", "requests_per_slot")
    kw = {k: w[k] for k in fields if k in w}
    if "concurrencies" in w:
        kw["concurrencies"] = tuple(w["concurrencies"])
    return Workload(**kw)


def plan_of(profile: Profile, reg, workload: Workload) -> PlanResult:
    """The planner's result rebuilt from the profile: each engine's grid, kept where it fits this machine.
    `prepared_all` keeps the engines in the order calibration listed them."""
    hw = profile.hardware
    spec = profile.prepared.spec if profile.prepared else ModelSpec(hf_id=profile.model_id)
    result = PlanResult(hw=hw, spec=spec, candidates=list(profile.prepared_all))
    for name, prepared in profile.prepared_all.items():
        backend = reg[name]
        grid = backend.candidate_configs(hw, prepared, min_ctx=workload.min_ctx)
        result.prepared[name] = prepared
        result.feasible[name] = memory_plan(hw, prepared, grid, backend.memory_model(hw))
        result.considered[name] = len(grid)
    return result


class RecordedRunner:
    """Answers each trial with the recorded one of the same configuration key. Confirmation runs are answered
    by the recorded confirmation runs of that key in order, then by its search trial. A key never recorded is
    answered as a failed launch and listed in `missing`."""

    def __init__(self, table: Sequence[TrialResult]):
        self.search: Dict[str, List[TrialResult]] = {}
        self.confirm: Dict[str, List[TrialResult]] = {}
        for t in table:
            (self.confirm if t.stage == "confirm" else self.search).setdefault(t.config.key(), []).append(t)
        self.used: Dict[str, int] = {}
        self.missing: List[Tuple[str, str]] = []  # (stage, config key)
        self.ran: List[Tuple[str, str]] = []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        key = cfg.key()
        self.ran.append((stage, key))
        if stage == "confirm" and self.confirm.get(key):
            rec = self.confirm[key].pop(0)
        elif self.search.get(key):
            n = self.used.get(key, 0)
            rec = self.search[key][min(n, len(self.search[key]) - 1)]
            self.used[key] = n + 1
        else:
            self.missing.append((stage, key))
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error="not recorded: this configuration was never measured")
        return rec.model_copy(update={"stage": stage, "config": cfg})


def _version_tuple(version: Optional[str]) -> Tuple[int, int]:
    m = re.match(r"(\d+)\.(\d+)", version or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


@contextlib.contextmanager
def recorded_environment(profile: Profile) -> Iterator[None]:
    """Engine features as the recording machine had them, and none of this machine's fitted state."""
    import rigtune.backends.vllm as vllm_backend
    import rigtune.speculative as speculative

    vllm = profile.hardware.backends.get("vllm")
    version = vllm.version if vllm is not None else None
    table = profile.calibration_table
    present = {
        "arctic_inference": any((t.config.spec_decode or "").startswith("suffix") for t in table),
        "flashinfer": any(t.config.backend == "vllm" and t.config.kv_dtype == "fp8_e5m2" for t in table),
    }
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in present:
            return object() if present[name] else None
        return real_find_spec(name, *args, **kwargs)

    with tempfile.TemporaryDirectory() as home, contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, {"RIGTUNE_HOME": home}))
        stack.enter_context(mock.patch.object(importlib.util, "find_spec", find_spec))
        stack.enter_context(mock.patch.object(vllm_backend, "_vllm_version", lambda: _version_tuple(version)))
        stack.enter_context(mock.patch.object(speculative, "_vllm_version", lambda: version))
        yield


@dataclass
class Replay:
    path: str
    recorded: str
    replayed: Optional[str]
    trials: int
    missing: List[Tuple[str, str]] = field(default_factory=list)
    unused: List[str] = field(default_factory=list)  # recorded configurations the replay never asked for
    cannot: List[str] = field(default_factory=list)  # recorded options the replay cannot apply
    skipped: Optional[str] = None  # why the profile was not replayed

    @property
    def same(self) -> bool:
        return self.replayed == self.recorded

    @property
    def verdict(self) -> Optional[str]:
        if self.skipped:
            return None
        if not self.same:
            return "different"
        return "inconclusive" if self.missing or self.cannot else "exact"


def replay(profile: Profile, path: str = "", options: Optional[SearchOptions] = None) -> Replay:
    """Replay one profile; `options` replaces the recorded ones (to replay another search on the same data)."""
    recorded = profile.config.key()
    opts, cannot = options_of(profile)
    if opts.budget_s is not None:
        return Replay(path, recorded, None, 0, skipped="budgeted: its stages were cut by wall-clock time")
    if not profile.prepared_all:
        return Replay(path, recorded, None, 0, skipped="no prepared_all: the planner cannot be rebuilt")
    if profile.phases != "unified" or profile.layout != "single" or profile.power_mode != "off":
        return Replay(path, recorded, None, 0, skipped="not a single-engine calibration")
    workload = workload_of(profile)
    runner = RecordedRunner(profile.calibration_table)
    with recorded_environment(profile):
        reg = registry()
        plan = plan_of(profile, reg, workload)
        try:
            out = calibrate(profile.hardware, plan.spec, profile.objective, plan, reg, workload=workload,
                            runner=runner, options=options or opts)
            replayed: Optional[str] = out.config.key()
        except RuntimeError:
            replayed = None
    asked = {key for _, key in runner.ran}
    unused = sorted({t.config.key() for t in profile.calibration_table} - asked)
    return Replay(path, recorded, replayed, len(runner.ran), missing=runner.missing, unused=unused, cannot=cannot)


def load(path: Path) -> Optional[Profile]:
    try:
        return Profile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def baseline_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(STRATEGIES).as_posix()
    except ValueError:
        return path.as_posix()


def regression(name: str, r: Replay) -> Optional[str]:
    """Why this replay is worse than BASELINE, or None."""
    was = BASELINE.get(name)
    if was is None or r.verdict is None or CLASSES.index(r.verdict) <= CLASSES.index(was):
        return None
    return f"was {was}, now {r.verdict}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("profiles", nargs="+", type=Path, help="profile JSON files (others are ignored)")
    args = ap.parse_args(argv)
    counts: Dict[str, int] = {}
    regressions: List[str] = []
    for path in sorted(args.profiles, key=lambda p: p.as_posix()):
        profile = load(path)
        if profile is None or not profile.calibration_table:
            continue
        r = replay(profile, str(path))
        print(path.as_posix())
        if r.skipped:
            print(f"  skipped: {r.skipped}")
            continue
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
        print(f"  {r.verdict}: recorded {r.recorded}")
        if not r.same:
            print(f"  {' ' * len(r.verdict)}  replayed {r.replayed}")
        print(f"  {r.trials} trials asked for, {len(r.missing)} never recorded; "
              f"{len(r.unused)} recorded configurations not asked for")
        if r.cannot:
            print(f"  recorded with options the replay cannot apply: {', '.join(r.cannot)}")
        for stage, key in r.missing:
            print(f"    not recorded ({stage}): {key}")
        why = regression(baseline_name(path), r)
        if why:
            regressions.append(f"{path.as_posix()}: {why}")
    print("; ".join(f"{counts.get(c, 0)} {c}" for c in CLASSES))
    for line in regressions:
        print(f"REGRESSION {line}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
