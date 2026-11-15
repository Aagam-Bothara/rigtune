"""A start-up failure Arctic Inference caused names its cause from the engine log, is noted, and is never retried;
after one, suffix decoding is not launched again in that calibration. Anything else on a suffix trial is quoted, not
blamed on suffix decoding.

The Arctic Inference lines are the ones vLLM 0.29.0 and arctic-inference 0.1.1 printed in the 14 Nov pod's image
(runpod/pytorch:1.4.1-cu1300-torch2130-ubuntu2404) on a CPU-only machine: vLLM's ImportError with arctic_inference
unimportable, the suffix tree built for Python 3.13 failing under 3.12, and the plugin's RuntimeError under
ARCTIC_INFERENCE_ENABLED=1. vLLM's warning for a package that fails to import is its utils/import_utils._has_module's."""

from __future__ import annotations

import traceback

import pytest

from rigtune.backends import registry
from rigtune.calibrate.search import (
    ARCTIC_PLUGIN_FAILED,
    LOAD_FAILED,
    MM_PROFILING_FAILED,
    SUFFIX_FAILED,
    SUFFIX_TRIAL_ERROR,
    SubprocessTrialRunner,
    arctic_cause,
    log_error,
    startup_cause,
)
from rigtune.calibrate.workload import Workload
from rigtune.memcal import _is_oom
from rigtune.models import Config, ModelSpec, TrialMetrics, TrialResult
from rigtune.pipeline import PlanResult, startup_failure_fn
from tests.conftest import make_hw
from tests.test_text_only import MM_LOG, OOM_LOG, VIDEO, _DeadBackend

MISSING = ("(APIServer pid=1) ImportError: Arctic Inference is required for suffix decoding. Install via "
           "`pip install arctic-inference==0.1.1`.")
PLUGIN = "(APIServer pid=1) RuntimeError: Arctic Inference plugin requires vllm==0.10.1 but found vllm==0.29.0!"
BUILD = "(EngineCore pid=9) ModuleNotFoundError: No module named 'arctic_inference.suffix_decoding._C'"
FRAMES = ["  File \"/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py\", line 1, in run"] * 30
WRAPPER = ("(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above. "
           "Failed core proc(s): {}")
# vLLM logs optional imports that fail, with their tracebacks, long before anything goes wrong.
TRITON = ["WARNING 11-14 [import_utils.py:407] Module triton_kernels was found but failed to import",
          "Traceback (most recent call last):",
          "ImportError: libtriton.so: cannot open shared object file: No such file or directory"]
KV = ("(EngineCore pid=9) ValueError: To serve at least one request with the models's max seq len (16384), (1.75 GiB "
      "KV cache is needed, which is larger than the available KV cache memory (0.9 GiB).")
SUFFIX = Config(backend="vllm", quant="bf16", ctx=16384, batch=64, gpu_memory_utilization=0.95,
                spec_decode="suffix:24")
PLAIN = SUFFIX.model_copy(update={"spec_decode": None})


def _log(*lines: str) -> str:
    return "\n".join(["INFO starting", "Traceback (most recent call last):", *FRAMES, *lines, *FRAMES, WRAPPER])


def _run(log: str, cfg: Config = SUFFIX) -> TrialResult:
    runner = SubprocessTrialRunner(backends={cfg.backend: _DeadBackend(log)}, models={cfg.backend: None}, hw=None,
                                   workload=Workload())
    return runner.run(cfg, "spec")


def _validation_error_tail() -> str:
    """A real pydantic ValidationError, raised the way vLLM's config dataclasses raise one from __post_init__."""
    from pydantic.dataclasses import dataclass

    @dataclass
    class SpeculativeConfig:
        num_speculative_tokens: int = 24

        def __post_init__(self):
            raise ValueError("suffix_decoding_max_tree_depth=0 must be >= 1")

    try:
        SpeculativeConfig()
    except Exception:
        return "\n".join(f"(APIServer pid=1) {line}" for line in traceback.format_exc().splitlines())
    raise AssertionError("no ValidationError")


@pytest.mark.parametrize("lines, kind, cause", [
    ([MISSING], SUFFIX_FAILED, "missing"),
    ([BUILD], SUFFIX_FAILED, "build"),
    ([*TRITON, BUILD], SUFFIX_FAILED, "build"),  # a benign earlier ImportError does not hide it
    (["ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by "
      "/usr/lib/python3/dist-packages/arctic_inference/suffix_decoding/_C.cpython-312-x86_64-linux-gnu.so)"],
     SUFFIX_FAILED, "build"),
    (["WARNING [import_utils.py] Module arctic_inference was found but failed to import", "Traceback",
      "ModuleNotFoundError: No module named 'grpc'", MISSING], SUFFIX_FAILED, "import"),
    ([PLUGIN], ARCTIC_PLUGIN_FAILED, "plugin"),
])
def test_arctic_inference_causes_are_read_from_the_log_even_above_the_tail(lines, kind, cause):
    res = _run(_log(*lines))
    got, line = startup_cause(res.error)
    assert not res.launched and got == kind and arctic_cause(line) == cause
    assert res.error.startswith(f"{kind} ({line}); failed to start (rc=1)\n")
    assert lines[-1] not in res.error.split("\n", 1)[1]  # 30 frames pushed it out of the 20-line tail
    assert not _is_oom(res.error)


def test_an_enabled_plugin_is_read_on_any_vllm_launch_and_only_there():
    assert startup_cause(_run(_log(PLUGIN), PLAIN).error)[0] == ARCTIC_PLUGIN_FAILED
    sglang = PLAIN.model_copy(update={"backend": "sglang"})
    assert _run(_log(PLUGIN), sglang).error.startswith("failed to start (rc=1)")
    assert _run(_log(MISSING), PLAIN).error.startswith("failed to start (rc=1)")  # no suffix: not read, as before


def test_anything_else_on_a_suffix_trial_is_quoted_not_blamed():
    res = _run(_log(*TRITON, KV))
    kind, line = startup_cause(res.error)
    assert kind == SUFFIX_TRIAL_ERROR and line == f"error in the log: {KV[:157]}..."  # not the benign triton line
    assert res.error.startswith("a suffix trial failed to start (error in the log: ")
    port = "(APIServer pid=1) OSError: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000)"
    assert startup_cause(_run(port).error) == (SUFFIX_TRIAL_ERROR, f"error in the log: {port}")
    tail = _validation_error_tail()
    quoted = log_error(tail + "\n" + WRAPPER)
    assert quoted.startswith("(APIServer pid=1) pydantic_core._pydantic_core.ValidationError: 1 validation error for "
                             "SpeculativeConfig Value error, suffix_decoding_max_tree_depth=0 must be >= 1")
    assert log_error(WRAPPER) is None and log_error("INFO loading\n(EngineCore pid=9) Killed") is None


def test_memory_is_still_memory_on_a_suffix_trial():
    assert _run(OOM_LOG).error.startswith("out of memory at start-up; failed to start")
    no_room = ("(EngineCore pid=9) ValueError: No available memory for the cache blocks. Try increasing "
               "`gpu_memory_utilization` when initializing the engine.")
    assert _run(_log(no_room)).error.startswith("failed to start (rc=1)")  # left to the memory report, as before
    assert _run("INFO loading\n(EngineCore pid=9) Killed").error.startswith("failed to start (rc=1)")


def test_a_load_or_multimodal_failure_on_a_suffix_trial_keeps_its_kind():
    tied = "AttributeError: 'Qwen2ForCausalLM' object has no attribute 'lm_head_is_tied'"
    assert startup_cause(_run(_log(tied)).error) == (LOAD_FAILED, tied)
    assert startup_cause(_run(MM_LOG).error) == (MM_PROFILING_FAILED, VIDEO)


def _plan():
    from tests.test_hybrid_memory import qwen25

    hw = make_hw("rtx4090")
    hw.backends["vllm"].version = "0.29.0"
    return hw, PlanResult(hw=hw, spec=ModelSpec(hf_id="Qwen/Qwen2.5-7B-Instruct"), candidates=["vllm"],
                          prepared={"vllm": qwen25("vllm")}, feasible={"vllm": []})


@pytest.mark.parametrize("lines, cfg, note", [
    ([MISSING], SUFFIX, "suffix decoding could not start on vLLM 0.29.0: arctic-inference is not installed where vLLM "
                        f"runs: {MISSING}; not retried, and not tried again in this calibration; install it in "
                        "vLLM's Python (see docs/usage.md#suffix-decoding), then run `rigtune recalibrate`"),
    ([BUILD], SUFFIX, "suffix decoding could not start on vLLM 0.29.0: arctic-inference's suffix tree did not load (a "
                      f"broken build): {BUILD}; not retried, and not tried again in this calibration; rebuild it in "
                      "vLLM's Python (see docs/usage.md#suffix-decoding), then run `rigtune recalibrate`"),
    ([PLUGIN], PLAIN, "vLLM could not start on vLLM 0.29.0: arctic-inference's vLLM plugin refused this vLLM: "
                      f"{PLUGIN}; unset ARCTIC_INFERENCE_ENABLED (vLLM's suffix decoding does not need the plugin); "
                      "not retried"),
    ([*TRITON, KV], SUFFIX, f"a suffix trial could not start on vLLM 0.29.0: error in the log: {KV[:157]}...; not "
                            "retried (the engine log has the rest)"),
])
def test_the_note_names_the_cause_and_what_to_do_and_nothing_is_retried(lines, cfg, note):
    hw, plan_ = _plan()
    res = _run(_log(*lines), cfg)
    retry, got = startup_failure_fn(hw, registry(), plan_)(res.config, res)
    assert retry is None and got == note
    assert "/usr/" not in got.split(": ", 2)[-1].split(";")[1]  # what to do names no interpreter path


# --------------------------------------------------------------------------- the rest of the calibration


class SuffixBroken:
    """The fake engine of tests/test_objectives_and_search.py, on which every suffix trial fails to start."""

    def __init__(self, error: str):
        from tests.test_objectives_and_search import FakeRunner

        self.inner, self.error, self.suffix_launches = FakeRunner(), error, []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        if (cfg.spec_decode or "").startswith("suffix:"):
            self.suffix_launches.append(cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=self.error)
        return self.inner.run(cfg, stage)


def _calibrate(monkeypatch, error: str):
    import rigtune.backends.vllm as vl
    from rigtune.calibrate.workload import get_workload
    from rigtune.memory import plan
    from rigtune.pipeline import SearchOptions, calibrate
    from tests.test_kv_tier_off import _pin

    _pin(monkeypatch)  # vLLM 0.29 with suffix decoding offered
    monkeypatch.setattr(vl, "_suffix_caveat", lambda: None)
    hw, result = _plan()
    reg = registry()
    pm = result.prepared["vllm"]
    result.feasible["vllm"] = plan(hw, pm, reg["vllm"].candidate_configs(hw, pm), reg["vllm"].memory_model(hw))
    runner = SuffixBroken(error)
    profile = calibrate(hw, result.spec, "balanced", result, reg, workload=get_workload("chat"), runner=runner,
                        options=SearchOptions())
    return profile, runner


BUILD_ERROR = f"{SUFFIX_FAILED} (arctic-inference's suffix tree did not load (a broken build): {BUILD}); failed to start"
KV_ERROR = f"{SUFFIX_TRIAL_ERROR} (error in the log: {KV}); failed to start (rc=1)"


@pytest.mark.parametrize("error, blocked", [(BUILD_ERROR, True), (KV_ERROR, False)])
def test_after_arctic_inference_failed_suffix_is_not_launched_again(error, blocked):
    from rigtune.calibrate.search import StagedSearch

    runner = SuffixBroken(error)
    search = StagedSearch(objective="throughput", runner=runner)
    first, second = SUFFIX, SUFFIX.model_copy(update={"batch": 16})
    assert search._run(first, "spec").error == error
    res = search._run(second, "spec")
    if blocked:  # a second suffix setting would only fail the same way
        assert runner.suffix_launches == [first.key()] and not res.launched
        assert res.error == "skipped: suffix decoding failed to start earlier in this calibration"
        assert search._run(PLAIN, "spec").error == "CUDA OOM"  # nothing else is held back: the fake ran it
    else:  # an unrelated failure says nothing about suffix decoding
        assert runner.suffix_launches == [first.key(), second.key()]


@pytest.mark.usefixtures("tmp_home")
def test_the_calibration_notes_the_cause_once_and_that_suffix_was_not_tried_again(monkeypatch):
    profile, runner = _calibrate(monkeypatch, BUILD_ERROR)
    assert len(runner.suffix_launches) == 1
    assert sum("suffix decoding could not start" in n for n in profile.notes) == 1
    assert any(n.startswith("suffix decoding failed to start (its cause is noted above) and was not launched "
                            "again in this calibration") for n in profile.notes)
    profile, runner = _calibrate(monkeypatch, KV_ERROR)
    assert runner.suffix_launches and not any("not launched again" in n for n in profile.notes)
