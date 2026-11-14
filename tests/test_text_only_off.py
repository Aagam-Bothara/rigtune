"""The text-only retry changes nothing for a launch that starts: every launch argument, key and trial sequence of a
dense model and of multimodal architectures whose default launch works is the one rigtune produced before it.

tests/fixtures/text_only_off_golden.json was captured by running snapshot() below on 8503f2a, the tree just before
the retry, with the same engine probes pinned (tests/test_kv_tier_off.py _pin). It calibrates a dense model
(Qwen2.5-7B) and three multimodal architectures (Gemma 4, Qwen3.5-9B, Gemma 3) on vLLM through the fake runner,
whose launches all start, and records each planned launch. snapshot() uses only what existed before the retry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.test_kv_tier_off import _args, _digest, _pin

FIXTURE = Path(__file__).parent / "fixtures" / "text_only_off_golden.json"


def snapshot() -> dict:
    from rigtune.backends import registry
    from rigtune.calibrate.workload import get_workload
    from rigtune.memory import plan
    from rigtune.models import ModelSpec
    from rigtune.pipeline import PlanResult, SearchOptions, calibrate
    from tests.conftest import make_hw
    from tests.test_hybrid_memory import gemma3, gemma4, qwen25, qwen35
    from tests.test_objectives_and_search import FakeRunner

    hw = make_hw("rtx4090")
    reg = registry()
    be = reg["vllm"]
    out: dict = {}
    for mname, make in (("qwen2.5-7b", qwen25), ("gemma-4", gemma4), ("qwen3.5-9b", qwen35),
                        ("gemma-3", gemma3)):
        pm = make("vllm")
        feasible = plan(hw, pm, be.candidate_configs(hw, pm), be.memory_model(hw))
        out[f"{mname}/launch"] = _digest([[c.key(), _args(be.launch_spec(c, pm, 8000))] for c, _ in feasible])
        result = PlanResult(hw=hw, spec=ModelSpec(hf_id=pm.spec.hf_id), candidates=["vllm"], prepared={"vllm": pm},
                            feasible={"vllm": feasible})
        runner = FakeRunner()
        profile = calibrate(hw, result.spec, "balanced", result, reg, workload=get_workload("chat"), runner=runner,
                            options=SearchOptions())
        out[f"{mname}/sequence"] = _digest([[r.stage, r.config.key(), r.ok] for r in profile.calibration_table])
        out[f"{mname}/calls"] = _digest(runner.calls)
        launch_args = ["PY" if a == sys.executable else a for a in profile.launch_args]
        out[f"{mname}/pick"] = _digest([profile.config.key(), launch_args, sorted(profile.launch_env.items())])
    return out


@pytest.mark.usefixtures("tmp_home")
def test_text_only_retry_changes_no_launch_key_or_trial_that_starts(monkeypatch):
    _pin(monkeypatch)
    got = snapshot()
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert sorted(got) == sorted(golden)
    assert {k: v for k, v in got.items() if v != golden[k]} == {}


if __name__ == "__main__":  # capture: python -m tests.test_text_only_off <tmp dir>
    mp = pytest.MonkeyPatch()
    mp.setenv("RIGTUNE_HOME", str(Path(sys.argv[1]) / "home"))
    _pin(mp)
    FIXTURE.write_bytes((json.dumps(snapshot(), indent=1, sort_keys=True) + "\n").encode("utf-8"))
    mp.undo()
