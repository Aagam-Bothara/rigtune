"""An engine's refusal of a kernel setting at start-up is its own failure kind, kernel_refused.

It is looked for only on a config with kernel settings, and only as the engine's whole phrase: a memory failure, or
vLLM's free-memory check, on such a trial keeps its own kind. The log lines are the format strings of vLLM v0.29.0 /
v0.33.0 and argparse, filled in; none has been read from a real failed launch.
"""

from __future__ import annotations

import pytest

from rigtune import ledger as lg
from rigtune.calibrate.search import (
    FAILURE_KINDS,
    KERNEL_REFUSED,
    SubprocessTrialRunner,
    failure_kind,
    kernel_startup_refusal,
)
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.ledger import Ledger
from rigtune.memcal import _is_oom
from rigtune.models import Config, TrialResult
from tests.test_load_failure import SGLANG_LOG
from tests.test_text_only import ENGINE, LOAD_LOG, MM_LOG, OOM_LOG, _DeadBackend

PLAIN = Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9)
ATTN = PLAIN.model_copy(update={"kernels": "attn=FLEX_ATTENTION"})
FULL = PLAIN.model_copy(update={"kernels": "cg=FULL"})
SGL = Config(backend="sglang", quant="bf16", gpu_memory_utilization=0.88, kernels="cg=breakable")

INVALID = ("ValueError: Selected backend AttentionBackendEnum.FLEX_ATTENTION is not valid for this configuration. "
           "Reason: ['kv_cache_dtype not supported']")
RAISED = ("ValueError: CUDAGraphMode.FULL is not supported with FlexAttentionBackend backend (support: "
          "AttentionCGSupport.NEVER); please try cudagraph_mode=PIECEWISE, and make sure compilation mode is "
          "VLLM_COMPILE")
WARNED = ("WARNING CUDAGraphMode.FULL is not supported with FlashInferBackend backend (support: "
          "AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=FULL_AND_PIECEWISE")
ARG = ("launch_server.py: error: argument --cuda-graph-backend-decode: invalid choice: 'breakable' "
       "(choose from 'full', 'disabled')")
OTHER_ARG = "launch_server.py: error: argument --tp-size: invalid int value: 'two'"
PRECHECK = ("ValueError: Free memory on device cuda:0 (21.3/24.0 GiB) on startup is less than desired GPU memory "
            "utilization (0.99, 23.76 GiB). Decrease GPU memory utilization or reduce GPU memory used by other "
            "processes.")


def _log(*lines: str) -> str:
    return "\n".join(lines + (ENGINE,))


def _run(cfg: Config, log: str) -> TrialResult:
    runner = SubprocessTrialRunner(backends={cfg.backend: _DeadBackend(log)}, models={cfg.backend: None}, hw=None,
                                   workload=Workload())
    return runner.run(cfg, "kernels")


def test_attention_refusal_is_kernel_refused():
    res = _run(ATTN, _log(INVALID))
    assert res.failure_kind == "kernel_refused" == failure_kind(res) and "kernel_refused" in FAILURE_KINDS
    assert res.error.startswith(f"{KERNEL_REFUSED} (Selected backend AttentionBackendEnum.FLEX_ATTENTION is not valid")
    assert "failed to start (rc=1)" in res.error and not res.launched
    stripped = res.model_copy(update={"failure_kind": None})
    assert failure_kind(stripped) == "kernel_refused"  # the text alone says the same


def test_resolver_refusal_and_argument_error():
    assert _run(FULL, _log(RAISED)).failure_kind == "kernel_refused"
    res = _run(SGL, _log(ARG))
    assert res.failure_kind == "kernel_refused" and "--cuda-graph-backend-decode: invalid choice" in res.error


def test_a_mode_the_engine_only_changed_is_not_a_refusal():
    """The resolver's warning opens like its error; it ends by naming the mode it set, and the engine went on."""
    assert kernel_startup_refusal(FULL, _log(WARNED)) is None
    assert _run(FULL, _log(WARNED, LOAD_LOG)).failure_kind == "startup_other"
    assert _run(FULL, _log(WARNED, RAISED)).failure_kind == "kernel_refused"  # the later one is the refusal


def test_an_argument_error_about_another_flag_is_not_a_refusal():
    assert kernel_startup_refusal(SGL, _log(OTHER_ARG)) is None
    assert _run(SGL, _log(OTHER_ARG)).failure_kind == "startup_other"
    assert kernel_startup_refusal(ATTN, _log(INVALID.replace("Selected backend", "Chosen backend"))) is None
    assert kernel_startup_refusal(FULL, _log(INVALID)) is None  # no attention setting was asked for


def test_refusal_with_memory_words_later_in_the_log_is_not_oom():
    res = _run(ATTN, _log(INVALID, "INFO shutting down: freed 2.1 GiB; no process ran out of memory"))
    assert res.failure_kind == "kernel_refused" and not _is_oom(res.error)


def test_oom_during_capture_stays_oom_startup_and_names_the_setting():
    res = _run(FULL, OOM_LOG)
    assert res.failure_kind == "oom_startup" == failure_kind(res.model_copy(update={"failure_kind": None}))
    assert res.error.startswith("out of memory at start-up; kernel settings cg=FULL; failed to start (rc=1)\n")
    assert _is_oom(res.error)  # the memory report still counts it


def test_free_memory_precheck_on_a_kernel_trial_is_not_kernel_refused():
    res = _run(FULL, _log(PRECHECK))
    assert kernel_startup_refusal(FULL, _log(PRECHECK)) is None
    assert res.failure_kind == "startup_other" and not res.error.startswith(KERNEL_REFUSED)


@pytest.mark.parametrize("log, kind", [(OOM_LOG, "oom_startup"), (MM_LOG, "mm_profiling"),
                                       (SGLANG_LOG, "load_failure"), (LOAD_LOG, "startup_other"),
                                       (_log(INVALID), "startup_other"), (_log(RAISED), "startup_other"),
                                       (_log(ARG), "startup_other")])
def test_finder_not_run_without_kernels(log, kind):
    """A config without kernel settings keeps every kind it had, even on a log that holds a refusal's words."""
    res = _run(PLAIN, log)
    assert res.failure_kind == kind and KERNEL_REFUSED not in res.error
    assert kernel_startup_refusal(PLAIN, log) is None
    if kind == "oom_startup":
        assert res.error.startswith("out of memory at start-up; failed to start (rc=1)\n")  # as it always was


def test_no_path_from_the_log_reaches_the_error():
    line = INVALID + " see /home/alice/venv/lib/python3.12/site-packages/vllm/platforms/cuda.py"
    res = _run(ATTN, _log(line))
    cause = res.error.split("; failed to start")[0]
    assert "alice" not in cause and "/home" not in cause and "cuda.py" in cause


def test_ledger_outcome_kind_is_kernel_refused(tmp_path, hw_a100, spec):
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    ledger.trial(_run(ATTN, _log(INVALID)))
    ledger.trial(_run(PLAIN, _log(INVALID)))
    kinds = [r["outcome"]["kind"] for r in lg.read(ledger.path) if r["type"] == "trial"]
    assert kinds == ["kernel_refused", "startup_other"]
