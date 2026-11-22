"""An engine refusing an EAGLE3 head or a DFlash or DSpark drafter at start-up: named from its own line, blocked for the
calibration, and remembered for the release, so the next calibration takes the next drafter the Hub search found."""

from __future__ import annotations

import pytest

from rigtune import headsearch
from rigtune.calibrate.search import (DRAFTER_REFUSED, METHOD_REFUSED, StagedSearch, SubprocessTrialRunner,
                                      drafter_startup_refusal, failure_kind, refusal_parts)
from rigtune.calibrate.workload import Workload
from rigtune.models import Config, TrialMetrics, TrialResult
from tests.test_text_only import _DeadBackend

DFLASH, DSPARK, HEAD = "z-lab/Qwen3-8B-DFlash-b16", "deepseek-ai/dspark_qwen3_8b_block7", "org/head"
FAILED = "; failed to start (rc=1)"
# The engines' own lines, as they print them (vLLM v0.33.0, SGLang v0.5.23).
VLLM_DFLASH_LAYERS = (
    "(EngineCore_DP0 pid=212) ValueError: DFlash drafter expects 20480 concatenated aux hidden features but received "
    "12288. This usually means the draft model's target_layer_ids reference layers that do not exist in the target "
    "model (incompatible draft/target pair).")  # model_executor/models/qwen3_dflash.py:841
VLLM_NO_INTERFACE = "(EngineCore_DP0 pid=212) RuntimeError: Model does not support EAGLE3 interface"  # eagle3_utils:29
VLLM_SPECULATORS = "pydantic_core._pydantic_core.ValidationError: Value error, Unsupported model_type llama4 for Eagle3 speculator"
SGLANG_LAYERS = ("Scheduler hit an exception: ValueError: DFLASH target_layer_ids contains an out-of-range layer id. "
                 "target_layer_ids[4]=40, target_num_layers=36.")  # speculative/dflash_utils.py:582
SGLANG_NO_CAPTURE = ("Scheduler hit an exception: ValueError: Model Gemma4ForCausalLM does not implement "
                     "set_eagle3_layers_to_capture, which EAGLE3 aux hidden-state capture requires.")


def _cfg(spec, backend="vllm"):
    return Config(backend=backend, quant="bf16", batch=64, spec_decode=spec, gpu_memory_utilization=0.9)


@pytest.mark.parametrize("backend, spec, log, label", [
    ("vllm", f"dflash:{DFLASH}:15", VLLM_DFLASH_LAYERS, DRAFTER_REFUSED),
    ("vllm", f"eagle3:{HEAD}:6", VLLM_SPECULATORS, DRAFTER_REFUSED),
    ("vllm", f"eagle3:{HEAD}:6", VLLM_NO_INTERFACE, METHOD_REFUSED),
    ("vllm", f"dspark:{DSPARK}:7", VLLM_NO_INTERFACE, METHOD_REFUSED),  # DSpark reads through the same interface
    ("sglang", f"dflash:{DFLASH}:16", SGLANG_LAYERS, DRAFTER_REFUSED),
    ("sglang", f"eagle3:{HEAD}:3", SGLANG_NO_CAPTURE, METHOD_REFUSED),
])
def test_each_engine_s_refusal_is_named_from_its_own_line(backend, spec, log, label):
    cause = drafter_startup_refusal(_cfg(spec, backend), "INFO starting\n" + log + "\nTraceback (most recent call last):")
    got = refusal_parts(cause + FAILED)
    assert got is not None and got[0] == label and got[2] == log.strip()[:240]


def test_a_method_refusal_says_every_kind_it_covers():
    cause = drafter_startup_refusal(_cfg(f"eagle3:{HEAD}:6"), VLLM_NO_INTERFACE)
    assert refusal_parts(cause + FAILED)[1] == ("eagle3", "dflash", "dspark")
    cause = drafter_startup_refusal(_cfg(f"eagle3:{HEAD}:3", "sglang"), SGLANG_NO_CAPTURE)
    assert refusal_parts(cause + FAILED)[1] == ("eagle3",)
    # SGLang words a DFlash trial's refusal so whatever the target's DSpark capture (DeepSeek V4 has only that).
    neither = ("ValueError: Model DeepseekV4ForCausalLM implements neither set_dspark_layers_to_capture nor "
               "set_dflash_layers_to_capture, one of which is required for DFLASH/DSPARK.")
    assert refusal_parts(drafter_startup_refusal(_cfg(f"dflash:{DFLASH}:16", "sglang"), neither) + FAILED)[1] == (
        "dflash",)
    assert refusal_parts(drafter_startup_refusal(_cfg(f"dspark:{DSPARK}:7", "sglang"), neither) + FAILED)[1] == (
        "dflash", "dspark")


def test_the_engine_s_whole_line_is_kept_whatever_it_holds():
    line = "ValueError: Qwen3-Omni DSpark draft hidden_size must match the target thinker hidden size (4096); got 2048."
    cause = drafter_startup_refusal(_cfg(f"dspark:{DSPARK}:7"), line)
    assert refusal_parts(cause + "; later calibrations on vllm 1.0 take the next drafter; failed to start")[2] == line
    # Its k, which rigtune chose, and a target it could not read are not the drafter's (config/speculative.py:200, 207).
    for other in ("ValueError: Qwen3-Omni DSpark requires num_speculative_tokens to match the trained block_size (7); "
                  "got 8.", "ValueError: Qwen3-Omni DSpark could not resolve the target thinker text_config."):
        assert drafter_startup_refusal(_cfg(f"dspark:{DSPARK}:7"), other) is None


@pytest.mark.parametrize("backend, spec, log", [
    ("vllm", f"eagle3:{HEAD}:6", VLLM_DFLASH_LAYERS),  # another kind's line: not this drafter's
    ("vllm", "mtp:3", VLLM_NO_INTERFACE),  # an MTP head is no EAGLE3 drafter
    ("vllm", "draft:org/draft:4", VLLM_SPECULATORS),  # a draft model is pairing's (draft_refused)
    ("sglang", f"dflash:{DFLASH}:16", VLLM_DFLASH_LAYERS),  # vLLM's wording on SGLang
    # Fire for the target, or for any draft: left to the other causes.
    ("vllm", f"eagle3:{HEAD}:6", "ValueError: Model architectures ['Eagle3Foo'] are not supported for now."),
    ("sglang", f"eagle3:{HEAD}:3", "ValueError: Warning: Target model's context_length (8192) is greater than the "
                                   "derived context_length (2048)."),
    ("vllm", f"eagle3:{HEAD}:6", "RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x4096 and 3072x3072)"),
])
def test_what_is_not_the_drafter_s_refusal_is_left_alone(backend, spec, log):
    assert drafter_startup_refusal(_cfg(spec, backend), log) is None


def _run(log, spec=f"dflash:{DFLASH}:15", model=None, hw=None):
    runner = SubprocessTrialRunner(backends={"vllm": _DeadBackend(log)}, models={"vllm": model}, hw=hw,
                                   workload=Workload())
    return runner.run(_cfg(spec), "spec")


def test_the_runner_names_the_refusal_beside_out_of_memory_text_and_records_it(prepared_vllm, hw_a100):
    prepared_vllm.spec_revisions[DFLASH] = "a" * 40
    res = _run("CUDA out of memory\n" + VLLM_DFLASH_LAYERS, model=prepared_vllm, hw=hw_a100)
    assert res.failure_kind == failure_kind(res) == "drafter_refused"
    assert res.error.startswith(DRAFTER_REFUSED) and "take the next drafter the search found" in res.error
    target = prepared_vllm.spec.hf_id
    entry = headsearch.refusal(DFLASH, "a" * 40, "vllm 1.0", target)
    assert entry is not None and "concatenated aux hidden features" in entry["reason"]
    assert headsearch.refusal(DFLASH, "b" * 40, "vllm 1.0", target) is None  # at that commit only
    assert headsearch.refusal(DFLASH, "a" * 40, "vllm 1.0", "Qwen/Qwen3-8B") is None  # for that model only


def test_a_drafter_that_started_in_this_calibration_is_not_recorded(prepared_vllm, hw_a100):
    runner = SubprocessTrialRunner(backends={"vllm": _DeadBackend(VLLM_DFLASH_LAYERS)}, models={"vllm": prepared_vllm},
                                   hw=hw_a100, workload=Workload())
    runner._drafters_started.add(("vllm", "dflash", DFLASH))
    res = runner.run(_cfg(f"dflash:{DFLASH}:15"), "prefill")
    assert res.failure_kind == "drafter_refused" and "take the next drafter" not in res.error
    assert not headsearch._refused_path().exists()


def test_a_kernel_refusal_in_the_same_log_wins_and_nothing_is_recorded(prepared_vllm, hw_a100, monkeypatch):
    import rigtune.calibrate.search as search

    monkeypatch.setattr(search, "kernel_startup_refusal", lambda cfg, log: "FLASHINFER is not valid")
    res = _run(VLLM_DFLASH_LAYERS, model=prepared_vllm, hw=hw_a100)
    assert res.failure_kind == "kernel_refused" and not headsearch._refused_path().exists()


def test_a_method_refusal_is_not_held_against_the_drafter(prepared_vllm, hw_a100):
    res = _run(VLLM_NO_INTERFACE, spec=f"eagle3:{HEAD}:6", model=prepared_vllm, hw=hw_a100)
    assert res.failure_kind == "drafter_refused" and res.error.startswith(METHOD_REFUSED)
    assert headsearch.refusal(HEAD, None, "vllm 1.0", prepared_vllm.spec.hf_id) is None  # another would fail too


def test_without_the_engine_s_release_nothing_is_recorded(prepared_vllm):
    assert _run(VLLM_DFLASH_LAYERS, model=prepared_vllm).failure_kind == "drafter_refused"
    assert not headsearch._refused_path().exists()


class _Refusing:
    """An engine that refuses one DFlash drafter, or every drafter reading the EAGLE3 interface."""

    def __init__(self, log, refused):
        self.log, self.refused, self.launches = log, refused, []

    def run(self, cfg, stage):
        self.launches.append((cfg.backend, cfg.spec_decode, cfg.batch))
        if cfg.backend == "vllm" and self.refused(cfg.spec_decode or ""):
            cause = drafter_startup_refusal(cfg, self.log)
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error=f"{cause}{FAILED}", failure_kind="drafter_refused")
        return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(tok_s=100, requests=20, output_tokens=100))


def test_a_refused_drafter_is_not_launched_again_and_the_others_are():
    runner = _Refusing(VLLM_DFLASH_LAYERS, lambda s: s.startswith("dflash:"))
    search = StagedSearch(objective="throughput", runner=runner)
    search._run(_cfg(f"dflash:{DFLASH}:15"), "spec")
    skipped = search._run(_cfg(f"dflash:{DFLASH}:15").model_copy(update={"batch": 128}), "prefill")
    assert skipped.failure_kind == "skipped_blocked" and "refused this drafter" in skipped.error
    assert search._run(_cfg(f"dspark:{DSPARK}:7"), "spec").ok
    assert search._run(_cfg(f"dflash:{DFLASH}:16", "sglang"), "spec").ok  # the other engine is its own
    assert len(runner.launches) == 3
    assert [n for n in search.notes if "refused" in n] == [
        f"vllm refused the dflash drafter {DFLASH} at start-up ({VLLM_DFLASH_LAYERS[:240]}); it is not launched "
        "again in this calibration"]


def test_a_method_refused_for_the_model_blocks_every_kind_it_covers_on_that_engine():
    runner = _Refusing(VLLM_NO_INTERFACE, lambda s: s.startswith(("eagle3:", "dflash:", "dspark:")))
    search = StagedSearch(objective="throughput", runner=runner)
    search._run(_cfg(f"eagle3:{HEAD}:6"), "spec")
    for spec in ("eagle3:other/head:6", f"dflash:{DFLASH}:15", f"dspark:{DSPARK}:7"):
        assert search._run(_cfg(spec), "spec").failure_kind == "skipped_blocked"
    assert search._run(_cfg("ngram_gpu:4"), "spec").ok and search._run(_cfg("mtp:3"), "spec").ok
    assert search._run(_cfg(f"eagle3:{HEAD}:3", "sglang"), "spec").ok
    assert len(runner.launches) == 4
    assert any(n.startswith("vllm refuses eagle3, dflash, dspark decoding for this model") for n in search.notes)


def test_a_drafter_that_started_earlier_is_not_blocked_by_a_later_refusal():
    runner = _Refusing(VLLM_DFLASH_LAYERS, lambda s: False)
    search = StagedSearch(objective="throughput", runner=runner)
    assert search._run(_cfg(f"dflash:{DFLASH}:15"), "spec").ok
    runner.refused = lambda s: s.startswith("dflash:")
    search._run(_cfg(f"dflash:{DFLASH}:15").model_copy(update={"prefill_budget": 1024}), "prefill")
    again = search._run(_cfg(f"dflash:{DFLASH}:15").model_copy(update={"prefill_budget": 8192}), "prefill")
    assert again.failure_kind == "drafter_refused"  # launched again, not skipped
    assert any("started earlier in this calibration" in n for n in search.notes)
