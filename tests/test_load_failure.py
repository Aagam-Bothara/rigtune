"""An engine whose own model code fails at start-up is recorded as a load failure with a note, never retried.

The log line is SGLang 0.5.19's, from launching google/gemma-4-12B-it on an A40: a known SGLang bug for Gemma 4
unified models, fixed in SGLang 0.5.21.
"""

from __future__ import annotations

import pytest

from rigtune.backends import registry
from rigtune.calibrate.search import (
    LOAD_FAILED,
    _baseline_candidates,
    load_failure,
    mentions_oom,
    mm_profiling_failure,
    startup_cause,
)
from rigtune.calibrate.workload import get_workload
from rigtune.memcal import _is_oom
from rigtune.memory import plan
from rigtune.models import Config, ModelSpec, TrialMetrics, TrialResult
from rigtune.pipeline import PlanResult, SearchOptions, calibrate
from tests.conftest import make_hw
from tests.test_hybrid_memory import gemma4
from tests.test_kv_tier_off import _pin
from tests.test_objectives_and_search import FakeRunner
from tests.test_text_only import OOM_LOG, _runner_error

TIED = "AttributeError: 'Gemma4UnifiedForConditionalGeneration' object has no attribute 'lm_head_is_tied'"
SGLANG_LOG = "\n".join(["[2026-11-14] Load weight begin.", "Traceback (most recent call last):",
                        *["  frame"] * 25, TIED, "[2026-11-14] Received sigquit from a child process. Bloom."])


def test_classifier_reads_the_sglang_load_failure():
    assert load_failure(SGLANG_LOG) == TIED
    assert mm_profiling_failure(SGLANG_LOG) is None and not mentions_oom(SGLANG_LOG)
    assert load_failure(OOM_LOG) is None
    res = _runner_error(SGLANG_LOG)
    assert not res.launched and res.error.startswith(f"{LOAD_FAILED} ({TIED}); failed to start (rc=1)\n")
    assert startup_cause(res.error) == (LOAD_FAILED, TIED)
    assert not _is_oom(res.error)  # "Bloom" in the tail says "oom", and the log named the cause
    assert _runner_error(OOM_LOG).error.startswith("out of memory at start-up")


class SglangBroken(FakeRunner):
    def run(self, cfg: Config, stage: str) -> TrialResult:
        if cfg.backend == "sglang":
            self.calls.append(cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error=f"{LOAD_FAILED} ({TIED}); failed to start (rc=1)\n[...] Bloom")
        return super().run(cfg, stage)


@pytest.mark.usefixtures("tmp_home")
def test_a_load_failure_is_noted_once_and_not_retried(monkeypatch):
    import rigtune.backends.vllm as vl

    _pin(monkeypatch)
    monkeypatch.setattr(vl, "_language_model_only_flag", lambda: True)
    hw = make_hw("rtx4090")
    hw.backends["sglang"].version = "0.5.19"
    reg = registry()
    feasible, prepared = {}, {}
    for b in ("vllm", "sglang"):
        pm = prepared[b] = gemma4(b)
        feasible[b] = plan(hw, pm, reg[b].candidate_configs(hw, pm), reg[b].memory_model(hw))
    result = PlanResult(hw=hw, spec=ModelSpec(hf_id="google/gemma-4-12B-it"), candidates=["sglang", "vllm"],
                        prepared=prepared, feasible=feasible)
    runner = SglangBroken()
    profile = calibrate(hw, result.spec, "balanced", result, reg, workload=get_workload("chat"), runner=runner,
                        options=SearchOptions())
    assert profile.backend == "vllm" and not profile.config.text_only
    sglang = [r for r in profile.calibration_table if r.config.backend == "sglang"]
    quants = {r.config.quant for r in sglang}
    tried = [k for k in runner.calls if k.startswith("sglang/")]
    assert len(sglang) == len(quants) == len(tried) == 2  # one launch per precision: no step-down, no retry
    group = [c for c, _ in feasible["sglang"] if c.quant == "bf16"]
    assert len(_baseline_candidates(group)) > 1  # a step-down was there to take
    assert not any(r.config.text_only for r in sglang)
    note = ("google/gemma-4-12B-it could not start on SGLang 0.5.19: the engine's model code failed while loading it "
            f"({TIED}), a known SGLang bug for Gemma 4 unified models, fixed in SGLang 0.5.21 (sgl-project/sglang "
            "#38820, #42091); a load failure, not memory, so it is not retried")
    assert profile.notes.count(note) == 1


def test_an_unknown_load_failure_names_no_fix():
    from rigtune.pipeline import startup_failure_fn
    from tests.test_hybrid_memory import qwen25

    hw = make_hw("rtx4090")
    plan_ = PlanResult(hw=hw, spec=ModelSpec(hf_id="Qwen/Qwen2.5-7B-Instruct"), candidates=["sglang"],
                       prepared={"sglang": qwen25("sglang")}, feasible={"sglang": []})
    line = "AttributeError: 'Qwen2ForCausalLM' object has no attribute 'foo'"
    res = TrialResult(config=Config(backend="sglang", quant="bf16"), stage="quant", metrics=TrialMetrics(),
                      launched=False, error=f"{LOAD_FAILED} ({line}); failed to start (rc=1)\n")
    retry, note = startup_failure_fn(hw, registry(), plan_)(res.config, res)
    assert retry is None and note == ("Qwen/Qwen2.5-7B-Instruct could not start on SGLang 1.0: the engine's model "
                                      f"code failed while loading it ({line}); a load failure, not memory, so it is "
                                      "not retried")
