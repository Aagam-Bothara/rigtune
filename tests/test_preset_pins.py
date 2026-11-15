"""The four preset objectives, pinned before goals generalised them: every rank, early stop, close call, cache key
and search-rule switch they had must come out exactly as it did (tests/legacy_objectives.py is the frozen copy).

    python -m tests.test_preset_pins     # rewrite tests/golden/presets.json from the code as it is
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from rigtune import cache as profile_cache
from rigtune.calibrate import objectives as live
from rigtune.calibrate.objectives import Constraints
from rigtune.calibrate.search import StagedSearch
from rigtune.calibrate.workload import WORKLOAD_NAMES, get_workload
from rigtune.models import OBJECTIVES, Config, ModelSpec, Profile, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions, default_constraints, idle_tail_for, level_rule
from tests import legacy_objectives as legacy

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).parent / "golden" / "presets.json"
PRESETS = ("throughput", "latency", "balanced", "efficiency")

CONSTRAINTS = {
    "default": Constraints(),
    "ceiling-300": Constraints(ttft_ceiling_ms=300.0),
    "median": Constraints(ttft_percentile=50),
    "tpot-30": Constraints(tpot_ceiling_ms=30.0),
    "floor-abs": Constraints(tok_s_floor_abs=400.0),
    "floor-frac": Constraints(tok_s_floor_frac=0.8, noise_tolerance=0.0),
    "tight": Constraints(ttft_ceiling_ms=120.0, tpot_ceiling_ms=12.0, tok_s_floor_frac=0.9, power_max_loss=0.1),
}


def test_presets_are_the_four_objectives():
    assert OBJECTIVES == PRESETS


def _signature(ranked) -> List[tuple]:
    """Everything a Ranked carries, the trial and level by identity, so two rankings of the same objects compare
    exactly, floats with == and not approximately."""
    return [(id(x.result), id(x.metrics), x.score, x.feasible, x.violation, x.violation_key, x.rate) for x in ranked]


def _recorded_tables() -> Dict[str, List[TrialResult]]:
    out = {}
    for path in sorted((ROOT / "benchmarks").glob("**/profiles/**/*.json"), key=lambda p: p.as_posix()):
        text = path.read_text(encoding="utf-8")
        if '"calibration_table"' not in text:
            continue
        profile = Profile.model_validate_json(text)
        if profile.calibration_table:
            out[path.relative_to(ROOT).as_posix()] = profile.calibration_table
    return out


@pytest.fixture(scope="module")
def recorded() -> Dict[str, List[TrialResult]]:
    tables = _recorded_tables()
    assert len(tables) >= 25  # every recorded profile under benchmarks/ with a calibration table
    return tables


def _same(results: List[TrialResult], objective: str, cons: Constraints) -> None:
    assert _signature(live.rank(results, objective, cons)) == _signature(legacy.rank(results, objective, cons))
    new_enough, old_enough = live.enough_level(objective, cons), legacy.enough_level(objective, cons)
    new_close, old_close = live.close_call_level(objective, cons), legacy.close_call_level(objective, cons)
    assert (new_enough is None) == (old_enough is None) and (new_close is None) == (old_close is None)
    for r in results:
        for m in [r.metrics, *r.metrics.by_concurrency.values()]:
            if old_enough is not None:
                assert new_enough(m) == old_enough(m)
            if old_close is not None:
                assert new_close(m) == old_close(m)


def test_every_recorded_table_ranks_as_before_under_every_preset(recorded):
    for name, table in recorded.items():
        for objective in PRESETS:
            for label, cons in CONSTRAINTS.items():
                _same(table, objective, cons)


def test_every_recorded_table_picks_as_before(recorded):
    """pick's notes name the objective and the violation: the same words for the same trials."""
    for table in recorded.values():
        for objective in PRESETS:
            for cons in (Constraints(), CONSTRAINTS["tight"]):
                winner, notes = live.pick(table, objective, cons)
                old = legacy.rank(table, objective, cons)
                assert winner is old[0].result
                relaxed = [n for n in notes if "least-violating" in n]
                assert relaxed == ([] if old[0].feasible else [
                    f"no config satisfied the {objective} constraint; picked the least-violating one "
                    f"(violation {old[0].violation:.1f})"])


def _maybe(rng: random.Random, value: float) -> Optional[float]:
    roll = rng.random()
    if roll < 0.1:
        return None
    if roll < 0.13:
        return math.nan
    if roll < 0.15:
        return math.inf
    return value


def _synthetic(seed: int) -> List[TrialResult]:
    """Trials of three to five levels with every optional field sometimes missing, NaN or infinite, ties included
    (two trials share a speed), and a failed trial and level among them."""
    rng = random.Random(seed)
    out = []
    for i in range(rng.randint(3, 9)):
        cfg = Config(backend=rng.choice(["vllm", "sglang"]), quant=rng.choice(["bf16", "fp8"]),
                     ctx=rng.choice([2048, 4096, 8192]), batch=rng.choice([16, 64, 256]),
                     gpu_memory_utilization=rng.choice([0.85, 0.9]),
                     kv_dtype=rng.choice(["auto", "auto", "fp8"]), power_limit_w=rng.choice([None, None, 200]))
        by: Dict[str, TrialMetrics] = {}
        base = rng.choice([400.0, 400.0, 800.0, 1200.0 * rng.random() + 100])
        for c in sorted(rng.sample([1, 4, 8, 16, 32, 64], rng.randint(3, 5))):
            failed = 16 if rng.random() < 0.05 else 0
            ttft = 30 + c * rng.uniform(2, 12)
            by[str(c)] = TrialMetrics(
                tok_s=round(base * c / (c + rng.uniform(2, 20)), rng.choice([0, 1, 3])), concurrency=c,
                ttft_ms=_maybe(rng, ttft), ttft_p95_ms=_maybe(rng, ttft * rng.uniform(1.1, 3)),
                tpot_ms=_maybe(rng, rng.uniform(5, 40)), joules_per_token=_maybe(rng, rng.uniform(0.1, 0.9)),
                power_w=_maybe(rng, rng.uniform(100, 300)), peak_mem_mb=_maybe(rng, rng.uniform(1e4, 2e4)),
                requests=16, failed=failed, output_tokens=16 * rng.choice([1, 64, 128]),
                ttft_samples_ms=sorted(ttft * rng.uniform(0.5, 2.5) for _ in range(rng.choice([0, 8, 32]))))
        best = max(by.values(), key=lambda m: m.tok_s)
        summary = best.model_copy(update={"by_concurrency": by, "requests": 16 * len(by),
                                          "output_tokens": sum(m.output_tokens for m in by.values())})
        if rng.random() < 0.15:  # a single-level trial: scored on its summary
            summary = summary.model_copy(update={"by_concurrency": {}})
        out.append(TrialResult(config=cfg, stage="t", metrics=summary,
                               error="boom" if rng.random() < 0.05 else None))
    return out


def test_synthetic_trials_with_missing_fields_rank_as_before():
    for seed in range(100):
        trials = _synthetic(seed)
        for objective in PRESETS:
            for cons in CONSTRAINTS.values():
                _same(trials, objective, cons)


def test_efficiency_at_a_rate_still_takes_its_own_path():
    trials = _synthetic(7)
    cons = Constraints(arrival_rate=0.5)
    assert _signature(live.rank(trials, "efficiency", cons)) == _signature(legacy.rank(trials, "efficiency", cons))


# --------------------------------------------------------------------------- the search, end to end


LEVELS = (1, 8, 32, 64)


class LevelRunner:
    """A deterministic engine model measured level by level as calibrate/measure.py does: busiest first, stopping
    at the first level `enough` accepts, and a level `close_call` flags measured twice as long."""

    QUANT = {"bf16": 1.0, "fp8": 1.3}
    ENGINE = {"vllm": 1.0, "sglang": 0.8}

    def __init__(self, enough, close_call):
        self.enough, self.close_call = enough, close_call
        self.log: List[Dict[str, Any]] = []
        self.calls: Dict[str, int] = {}

    def level(self, cfg: Config, c: int, n: int) -> TrialMetrics:
        running = min(c, cfg.batch)
        speed = self.QUANT[cfg.quant] * self.ENGINE[cfg.backend] * (1 + 0.5 * math.log2(cfg.batch)) * 100
        speed *= 1.1 if cfg.kv_dtype != "auto" else 1.0
        speed *= 1.0 + 0.01 * (n % 3)  # the third and later runs of one configuration differ a little
        tok_s = round(speed * running / (running + 6), 3)
        ttft = 40 + 4 * running + cfg.ctx / 200 + 0.3 * cfg.batch + (0 if cfg.backend == "vllm" else 30)
        samples = sorted(round(ttft * (0.6 + 1.1 * i / 31), 3) for i in range(32))
        power = 120 + 160 * running / (running + 12) + 0.4 * cfg.batch  # a larger batch draws more at any load
        tpot = round(1000 * running / tok_s / 4 + cfg.batch / 40, 3)
        return TrialMetrics(tok_s=tok_s, ttft_ms=samples[15], ttft_p95_ms=samples[30], tpot_ms=tpot,
                            joules_per_token=None if cfg.backend == "sglang" else round(power / tok_s, 5),
                            power_w=power, peak_mem_mb=1e4 + 10 * running, concurrency=c, requests=32,
                            failed=0, output_tokens=32 * 128, ttft_samples_ms=samples)

    def run(self, cfg: Config, stage: str) -> TrialResult:
        n = self.calls[cfg.key()] = self.calls.get(cfg.key(), 0) + 1
        if cfg.gpu_memory_utilization is not None and cfg.gpu_memory_utilization >= 0.95:
            self.log.append({"stage": stage, "config": cfg.key(), "levels": [], "ok": False})
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error="CUDA OOM")
        order = sorted(LEVELS, reverse=True) if self.enough is not None else list(LEVELS)
        by: Dict[str, TrialMetrics] = {}
        measured = []
        for c in order:
            m = self.level(cfg, c, n)
            if self.close_call is not None and self.close_call(m):
                m = m.model_copy(update={"requests": 64, "output_tokens": 64 * 128, "resampled": True})
            by[str(c)] = m
            measured.append(f"{c}{'r' if m.resampled else ''}")
            if self.enough is not None and self.enough(m):
                break
        best = max(by.values(), key=lambda x: x.tok_s)
        summary = best.model_copy(update={"by_concurrency": by, "requests": sum(x.requests for x in by.values()),
                                          "output_tokens": sum(x.output_tokens for x in by.values())})
        self.log.append({"stage": stage, "config": cfg.key(), "levels": measured, "ok": True})
        return TrialResult(config=cfg, stage=stage, metrics=summary)


def _grid() -> List[Config]:
    return [Config(backend=b, quant=q, ctx=ctx, batch=batch, gpu_memory_utilization=g)
            for b in ("vllm", "sglang") for q in ("bf16", "fp8") for ctx in (4096, 8192)
            for batch in (16, 64, 256) for g in (0.9, 0.95)]


def _kv(c: Config) -> List[Config]:
    return [c.model_copy(update={"kv_dtype": "fp8"})] if c.kv_dtype == "auto" else []


SEARCH_CONSTRAINTS = {
    "default": Constraints(ttft_ceiling_ms=250.0),
    "tpot": Constraints(ttft_ceiling_ms=250.0, tpot_ceiling_ms=12.0),
    "median-floor": Constraints(ttft_ceiling_ms=150.0, ttft_percentile=50, tok_s_floor_abs=500.0),
    "strict": Constraints(ttft_ceiling_ms=60.0, strict=True),
}


def search_golden() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for objective in PRESETS:
        for label, cons in SEARCH_CONSTRAINTS.items():
            opts = SearchOptions(ttft_percentile=cons.ttft_percentile)
            runner = LevelRunner(level_rule(objective, cons, opts), live.close_call_level(objective, cons))
            search = StagedSearch(objective=objective, runner=runner, constraints=cons, confirm_top=3,
                                  variant_stages=[("kv", _kv)])
            winner, notes = search.run(_grid())
            out[f"{objective}/{label}"] = {
                "winner": winner.config.key() if winner is not None else None,
                "notes": notes, "trials": runner.log, "idle_tail_s": idle_tail_for(objective, opts)}
    return out


def test_the_search_runs_every_preset_as_before():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(search_golden()))
    assert sorted(got) == sorted(golden)
    assert sorted(k for k in golden if got[k] != golden[k]) == []
    # The golden exercises what it pins: early stops, re-measured close calls and a refusal.
    trials = [t for case in golden.values() for t in case["trials"]]
    assert any(t["levels"] and len(t["levels"]) < len(LEVELS) for t in trials)
    assert any(any(lv.endswith("r") for lv in t["levels"]) for t in trials)
    assert golden["balanced/strict"]["winner"] is None


# --------------------------------------------------------------------------- cache keys and rule switches


def test_preset_cache_keys_are_the_literals_they_were():
    for name in WORKLOAD_NAMES:
        wl = get_workload(name)
        opts = SearchOptions()
        for objective in PRESETS:
            cons = default_constraints(wl, opts)
            assert opts.key(objective, cons, wl) == ({"ttft": "p95"} if objective == "balanced" else {})
    wl = get_workload("default")
    assert SearchOptions(ttft_percentile=50).key("balanced", default_constraints(wl, SearchOptions(
        ttft_percentile=50)), wl) == {}
    tight = Constraints(ttft_ceiling_ms=300.0, tpot_ceiling_ms=30.0, tok_s_floor_abs=200.0)
    assert SearchOptions().key("balanced", tight, wl) == {"ttft": "p95", "limits": "478e9ade84a1ef12"}
    assert SearchOptions().key("efficiency", Constraints(tok_s_floor_frac=0.7), wl) == {
        "limits": "09fcca11a9a38ea2"}
    spec = ModelSpec(hf_id="org/model")
    names = [profile_cache.profile_path("hw", spec, o, "default", options=SearchOptions().key(
        o, default_constraints(wl, SearchOptions()), wl)).name for o in PRESETS]
    assert names == ["throughput.json", "latency.json", "balanced-ttft-p95.json", "efficiency.json"]


class _Predictor:
    def __init__(self):
        self.ceilings: List[Optional[float]] = []

    def is_fitted(self, backend: str) -> bool:
        return True

    def best_level(self, model, cfg, prefill, decode, concurrencies, ceiling, **kwargs):
        self.ceilings.append(ceiling)
        return type("P", (), {"tok_s": 100.0})()


def _one(backend: str, tok_s: float, ttft: float) -> TrialResult:
    cfg = Config(backend=backend, quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    return TrialResult(config=cfg, stage="quant", metrics=TrialMetrics(
        tok_s=tok_s, ttft_ms=ttft, ttft_p95_ms=ttft, tpot_ms=10, joules_per_token=0.3, concurrency=64,
        requests=32, output_tokens=4096))


def rule_table() -> Dict[str, Dict[str, bool]]:
    out = {}
    for objective in PRESETS:
        cons = Constraints(ttft_ceiling_ms=400.0)
        predictor = _Predictor()
        search = StagedSearch(objective=objective, runner=None, constraints=cons, predictor=predictor,
                              models={"vllm": object()})
        search._predicted_tok_s(Config(backend="vllm", quant="bf16"))
        losing = search._losing_backends([_one("vllm", 1000.0, 100.0), _one("sglang", 300.0, 100.0)], {})
        out[objective] = {
            "early stop": live.enough_level(objective, cons) is not None,
            "close call": live.close_call_level(objective, cons) is not None,
            "drop losing engines": bool(losing),
            "predict under the ceiling": predictor.ceilings == [400.0],
            "idle tail": idle_tail_for(objective, SearchOptions()) > 0,
        }
    return out


def test_which_preset_switches_on_which_search_rule():
    assert rule_table() == {
        "throughput": {"early stop": True, "close call": False, "drop losing engines": True,
                       "predict under the ceiling": False, "idle tail": False},
        "latency": {"early stop": False, "close call": False, "drop losing engines": False,
                    "predict under the ceiling": False, "idle tail": False},
        "balanced": {"early stop": True, "close call": True, "drop losing engines": True,
                     "predict under the ceiling": True, "idle tail": False},
        "efficiency": {"early stop": False, "close call": False, "drop losing engines": False,
                       "predict under the ceiling": False, "idle tail": True},
    }


if __name__ == "__main__":
    GOLDEN.write_text(json.dumps(search_golden(), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    sys.exit(0)
