"""--kv-tier off (the default) changes nothing: every plan, launch argument, cache key and trial sequence is
the one rigtune produced before tiers existed.

tests/fixtures/kv_tier_off_golden.json was captured by running snapshot() below on the tree just before tiers
(session replay, hybrid layouts and SGLang speculation, no tiers), with the same engine probes pinned. It
covers a dense model (Qwen2.5-7B) and two hybrid ones (Qwen3.5-9B with linear attention, Gemma 4 with sliding
layers), on vLLM and SGLang, with and without per-request reuse reports, and one calibration of a session file
through the fake runner. snapshot() uses only what existed before tiers, so it runs unchanged on both trees.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "kv_tier_off_golden.json"


def _pin(monkeypatch) -> None:
    """Engine probes that read the installed engines, fixed so the snapshot is the same on every machine."""
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    from rigtune import speculative

    monkeypatch.setattr(vl, "_serve_cli", lambda: None)
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 29))
    monkeypatch.setattr(vl, "_attention_backend_flag", lambda: True)
    monkeypatch.setattr(vl, "_prompt_tokens_details_flag", lambda: True)
    monkeypatch.setattr(sg, "_cache_report_flag", lambda: True)
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    monkeypatch.setattr(sg, "sglang_spec_methods", lambda: frozenset())
    for name in ("draft", "ngram_gpu", "eagle3", "suffix"):
        monkeypatch.setattr(speculative, f"vllm_supports_{name}", lambda version=None: True)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def _args(spec) -> list:
    return [["PY" if a == sys.executable else a for a in spec.args], sorted(spec.env.items())]


def _digest(obj) -> dict:
    text = json.dumps(obj, separators=(",", ":"), sort_keys=True)
    return {"n": len(obj), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def snapshot(tmp_path) -> dict:
    from rigtune.backends import registry
    from rigtune.calibrate.workload import get_workload, workload_from_file
    from rigtune.memory import estimate, plan
    from rigtune.models import ModelSpec
    from rigtune.pipeline import PlanResult, SearchOptions, calibrate
    from tests.conftest import make_hw
    from tests.test_hybrid_memory import gemma4, qwen25, qwen35
    from tests.test_objectives_and_search import FakeRunner
    from tests.test_session_replay import _file

    hw = make_hw("rtx4090")
    out: dict = {}
    for backend in ("vllm", "sglang"):
        for mname, make in (("qwen2.5-7b", qwen25), ("qwen3.5-9b", qwen35), ("gemma-4", gemma4)):
            pm = make(backend)
            be, reporting = registry()[backend], registry(report_reuse=True)[backend]
            mm = be.memory_model(hw)
            grid = {c.key(): c for m in (0, 16384) for c in be.candidate_configs(hw, pm, min_ctx=m)}
            for c in list(grid.values()):
                for v in [c.model_copy(update={"kv_dtype": d}) for d in be.kv_dtypes(hw)] \
                        + [c.model_copy(update={"prefix_cache": False})] + be.prefill_variants(c):
                    grid[v.key()] = v
            configs = [grid[k] for k in sorted(grid)]
            rows = [[c.key(), *(lambda e: [e.weights, e.kv_cache, e.total, e.budget, e.feasible])(
                estimate(hw, pm, c, mm))] for c in configs]
            kept = plan(hw, pm, configs, mm)
            launches = [[c.key(), _args(b.launch_spec(c, pm, 8000))]
                        for c, _ in kept for b in (be, reporting)]
            out[f"{backend}/{mname}/plan"] = _digest(rows)
            out[f"{backend}/{mname}/launch"] = _digest(launches)
    objectives = ("throughput", "latency", "balanced", "efficiency")
    out["keys"] = _digest([SearchOptions().key(o) for o in objectives]
                          + [SearchOptions(report_reuse=True).key("balanced")])
    # One calibration of a session file, end to end through the fake runner.
    spec, reg = ModelSpec(hf_id="Qwen/Qwen2.5-7B-Instruct"), registry(report_reuse=True)
    wl = workload_from_file(_file(tmp_path), get_workload("chat"))
    feasible = {}
    for backend in ("vllm", "sglang"):
        be, pm = reg[backend], qwen25(backend)
        grid = be.candidate_configs(hw, pm, min_ctx=wl.min_ctx)
        feasible[backend] = plan(hw, pm, grid, be.memory_model(hw))
    result = PlanResult(hw=hw, spec=spec, candidates=["vllm", "sglang"],
                        prepared={b: qwen25(b) for b in ("vllm", "sglang")}, feasible=feasible)
    profile = calibrate(hw, spec, "balanced", result, reg, workload=wl, runner=FakeRunner(),
                        options=SearchOptions())
    out["sequence"] = _digest([[r.stage, r.config.key(), r.ok] for r in profile.calibration_table])
    launch_args = ["PY" if a == sys.executable else a for a in profile.launch_args]
    out["pick"] = _digest([profile.config.key(), launch_args, sorted(profile.launch_env.items()),
                           profile.options])
    return out


@pytest.mark.usefixtures("tmp_home")
def test_kv_tier_off_changes_no_plan_launch_key_or_trial(tmp_path, monkeypatch):
    _pin(monkeypatch)
    got = snapshot(tmp_path)
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert sorted(got) == sorted(golden)
    assert {k: v for k, v in got.items() if v != golden[k]} == {}


if __name__ == "__main__":  # capture: python -m tests.test_kv_tier_off <tmp dir>
    mp = pytest.MonkeyPatch()
    tmp = Path(sys.argv[1])
    mp.setenv("RIGTUNE_HOME", str(tmp / "home"))
    _pin(mp)
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(snapshot(tmp), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    mp.undo()
