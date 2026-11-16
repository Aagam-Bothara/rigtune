"""A kernel trial's failure is its own: it blocks no second model, teaches the memory fit nothing, and is named
a refusal only for a flag its own settings rendered."""

from __future__ import annotations

import pytest

import rigtune.backends.sglang as sg
from rigtune import kernels
from rigtune.calibrate.search import KERNEL_REFUSED, StagedSearch, failure_kind, kernel_startup_refusal
from rigtune.memcal import observations_from_trials
from rigtune.models import Config, MemoryEstimate, MemoryObservation, MiB, TrialMetrics, TrialResult

DRAFT = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9,
               spec_decode="draft:Qwen/Qwen2.5-0.5B-Instruct:4")
OK = TrialMetrics(tok_s=100.0, ttft_ms=10.0, tpot_ms=5.0, requests=10, output_tokens=100)


class _Refusing:
    """Every config with kernel settings fails to start, as `error` says; every other one runs."""

    def __init__(self, error: str, kind: str):
        self.error, self.kind = error, kind

    def run(self, cfg, stage):
        if cfg.kernels:
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=self.error,
                               failure_kind=self.kind)
        return TrialResult(config=cfg, stage=stage, metrics=OK)


@pytest.mark.parametrize("error, kind", [
    (f"{KERNEL_REFUSED} (Selected backend TRITON_ATTN is not valid ...); failed to start (rc=1)\n", "kernel_refused"),
    ("out of memory at start-up; kernel settings cgmax=512; failed to start (rc=1)\n", "oom_startup"),
    ("failed to start (rc=1)\n", "startup_other"),
])
def test_a_failed_kernel_trial_blocks_no_second_model(error, kind):
    search = StagedSearch(objective="throughput", runner=_Refusing(error, kind), block_failed_pairs=True)
    refused = search._run(DRAFT.model_copy(update={"kernels": "attn=TRITON_ATTN"}), "kernels")
    assert not refused.launched and failure_kind(refused) == kind
    later = search._run(DRAFT.model_copy(update={"batch": 128}), "combine")
    assert later.launched and later.ok  # the pair had started on the leader: it is still tried
    assert not any("blocked" in note for note in search.notes)


def test_a_pair_that_fails_without_kernel_settings_is_still_blocked():
    class Dead:
        def run(self, cfg, stage):
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error="failed to start (rc=1)\n", failure_kind="startup_other")

    search = StagedSearch(objective="throughput", runner=Dead(), block_failed_pairs=True)
    search._run(DRAFT, "spec")
    assert failure_kind(search._run(DRAFT.model_copy(update={"batch": 128}), "combine")) == "skipped_blocked"


def test_an_argument_error_about_a_flag_the_settings_did_not_render_is_no_refusal():
    line = "vllm serve: error: argument --attention-backend: invalid choice: 'FLASHINFER'"
    graph_only = DRAFT.model_copy(update={"kernels": "cg=PIECEWISE", "kv_dtype": "fp8_e5m2"})
    assert kernel_startup_refusal(graph_only, line) is None  # the cache type's backend, not a kernel setting
    assert kernel_startup_refusal(DRAFT.model_copy(update={"kernels": "attn=FLASHINFER"}), line) is not None
    cc = "vllm serve: error: argument --compilation-config/-cc: invalid JSON"
    assert kernel_startup_refusal(graph_only, cc) is not None
    assert kernel_startup_refusal(DRAFT.model_copy(update={"kernels": "O=1"}), cc) is None


def _trial(kernels_setting):
    cfg = Config(backend="sglang", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.88,
                 kernels=kernels_setting)
    predicted = MemoryEstimate(config_key=cfg.key(), weights=6 * 1024 * MiB, kv_cache=1024 * MiB,
                               runtime_workspace=1024 * MiB, safety_margin=512 * MiB, total=9 * 1024 * MiB,
                               budget=20 * 1024 * MiB, feasible=True)
    return TrialResult(config=cfg, stage="batch", metrics=OK.model_copy(update={"peak_mem_mb": 9000.0}),
                       memory=MemoryObservation(predicted=predicted))


def test_kernel_trials_are_left_out_of_the_memory_fit():
    """Their extra CUDA-graph memory is planned apart; fitted into the workspace it would be counted twice."""
    plain, raised = _trial(None), _trial("cgmax=512")
    both = observations_from_trials([plain, raised], "hw")
    assert len(both) == len(observations_from_trials([plain], "hw"))
    assert observations_from_trials([raised], "hw") == []


def test_only_the_releases_read_have_a_known_default_cap():
    a40 = 46068.0
    assert kernels.sglang_default_decode_cap(a40, 16, 1, "0.5.19") == 32
    assert kernels.sglang_default_decode_cap(a40, 16, 1, "0.5.23") == 16
    for between in ("0.5.20", "0.5.21", "0.5.22"):  # where the clamp arrived is not known
        assert kernels.sglang_default_decode_cap(a40, 16, 1, between) is None
        assert kernels.sglang_same_as_default(kernels.parse("cgmax=32"), a40, 16, 1, between) is None
        assert kernels.sglang_graph_extra_bytes(kernels.parse("cgmax=128"), a40, 16, 1, between) == 128 * 2 * MiB


@pytest.mark.parametrize("spec, doubled", [("mtp:2", True), ("draft:org/small:4", True), ("assist:org/a:1", True),
                                           ("ngram:4", False), (None, False)])
def test_a_second_worker_doubles_the_graph_term(spec, doubled):
    cfg = Config(backend="sglang", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.88,
                 kernels="cgmax=64", spec_decode=spec)
    plain = cfg.model_copy(update={"spec_decode": None})
    total = int(46068 * MiB)
    one = sg.sglang_graph_bytes(plain, total, "0.5.23")
    assert one == 32 * 2 * MiB and sg.sglang_graph_bytes(cfg, total, "0.5.23") == (2 if doubled else 1) * one
