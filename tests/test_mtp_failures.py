"""A checkpoint's own MTP head setting whose start failed for a reason of its own is not launched again in the same
calibration, at other batches or prefill budgets; a memory shortage, another part of the configuration, a start that
ran out of time, a failure after a memory retry, or a start the same setting made earlier never blocks it. Blocks are
per setting (engine, weights, KV cache, tensor parallelism, spec, output-head quantization): a refused fp8 output
head, or a token count the engine refuses, says nothing about the head's other settings."""

from __future__ import annotations

import pytest

from rigtune.calibrate.search import (MEMORY_SHORT_CAUSE, MM_PROFILING_FAILED, StagedSearch, SubprocessTrialRunner,
                                      memory_short)
from rigtune.calibrate.workload import Workload
from rigtune.models import Config, TrialMetrics, TrialResult
from tests.test_text_only import _DeadBackend

OK = dict(metrics=TrialMetrics(tok_s=100, requests=20, output_tokens=100))
HEAD_LOAD = "failed to start (rc=1)\nKeyError: 'model.mtp.layers.0.weight'"


def _cfg(spec="mtp:3", **kw):
    return Config(**{"backend": "vllm", "quant": "bf16", "spec_decode": spec, **kw})


class _Runner:
    """Fails the (spec, head quant) settings in `fails` with `error` / `kind`, at the budgets in `only` if given;
    every other trial serves."""

    def __init__(self, fails, error=HEAD_LOAD, kind="startup_other", only=None):
        self.fails, self.error, self.kind, self.only, self.launches = fails, error, kind, only, []

    def run(self, cfg, stage):
        self.launches.append(cfg.key())
        failing = (cfg.spec_decode, cfg.spec_head_quant) in self.fails
        if failing and (self.only is None or cfg.prefill_budget in self.only):
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=self.error,
                               failure_kind=self.kind)
        return TrialResult(config=cfg, stage=stage, **OK)


def _search(runner, **kw):
    return StagedSearch(objective="throughput", runner=runner, **kw)


def test_a_head_setting_that_cannot_start_is_not_launched_again():
    runner = _Runner({("mtp:3", None)})
    search = _search(runner)
    assert search._run(_cfg(), "spec").failure_kind == "startup_other"
    for budget in (2048, 8192):
        assert search._run(_cfg(prefill_budget=budget), "prefill").failure_kind == "skipped_blocked"
    assert search._run(_cfg("mtp:1"), "spec").ok  # the head's other settings are not this one
    assert len(runner.launches) == 2
    assert any("the MTP head setting mtp:3 failed to start" in n and "not launched again" in n for n in search.notes)


def test_a_refused_fp8_output_head_does_not_block_the_plain_head():
    search = _search(_Runner({("mtp:1", "fp8")}))
    assert not search._run(_cfg("mtp:1", spec_head_quant="fp8"), "spec").ok
    assert search._run(_cfg("mtp:1"), "spec").ok and search._run(_cfg("mtp:1", batch=128), "batch").ok
    assert search._run(_cfg("mtp:1", spec_head_quant="fp8", batch=128), "batch").failure_kind == "skipped_blocked"


@pytest.mark.parametrize("error, kind", [
    ("out of memory at start-up; failed to start (rc=1)\ntorch.OutOfMemoryError: CUDA out of memory", "oom_startup"),
    ("failed to start (rc=1)\nRuntimeError: Not enough memory. Please try to increase --mem-fraction-static.",
     "startup_other"),  # SGLang's pool sizing
    ("failed to start (rc=1)\nValueError: No available memory for the cache blocks. Try increasing "
     "`gpu_memory_utilization` when initializing the engine.", "startup_other"),  # vLLM's KV-cache check
    ("failed to start (rc=1)\nValueError: Free memory on device cuda:0 (20.1/79.3 GiB) on startup is less than "
     "desired GPU memory utilization (0.95, 75.3 GiB).", "startup_other"),  # vLLM's free-memory check
    (f"{MEMORY_SHORT_CAUSE} (ValueError: ... larger than the available KV cache memory); failed to start (rc=1)",
     "startup_other"),  # found in the log above the tail
    (f"{MM_PROFILING_FAILED} (ValueError: image too large); failed to start (rc=1)", "mm_profiling"),
    ("failed to start (rc=None)\nloading the drafter", "startup_other"),  # ran out of time: a cold start
])
def test_a_shortage_another_settings_failure_or_a_timeout_blocks_nothing(error, kind):
    runner = _Runner({("mtp:3", None)}, error=error, kind=kind)
    search = _search(runner)
    search._run(_cfg(), "spec")
    search._run(_cfg(batch=128), "batch")
    assert len(runner.launches) == 2 and not search._head_blocked


def test_a_kernel_setting_trial_blocks_nothing():
    search = _search(_Runner({("mtp:3", None)}))
    search._run(_cfg(kernels="cgmax=64"), "kernels")
    assert not search._head_blocked


def test_a_setting_that_started_earlier_is_not_blocked_by_a_later_failure():
    runner = _Runner({("mtp:3", None)}, only={2048})  # serves, except with a 2,048-token prefill budget
    search = _search(runner)
    assert search._run(_cfg(), "spec").ok
    assert not search._run(_cfg(prefill_budget=2048), "prefill").ok
    assert not search._run(_cfg(prefill_budget=2048, batch=128), "prefill").ok
    assert search._run(_cfg(prefill_budget=8192), "prefill").ok
    assert not search._head_blocked
    assert sum("started earlier in this calibration" in n for n in search.notes) == 1  # one note, not one a trial


def test_other_speculation_is_untouched():
    runner = _Runner({("ngram_gpu:4", None)})
    search = _search(runner)
    search._run(_cfg("ngram_gpu:4"), "spec")
    search._run(_cfg("ngram_gpu:4", batch=128), "batch")
    assert len(runner.launches) == 2 and not search._head_blocked


def test_a_block_is_per_engine_and_per_target_setting():
    search = _search(_Runner({("mtp:3", None)}))
    search._run(_cfg(), "spec")
    for other in (dict(backend="sglang"), dict(kv_dtype="fp8"), dict(quant="fp8"), dict(tp=2)):
        assert search._run(_cfg(batch=128, **other), "explore").failure_kind != "skipped_blocked", other


def test_skips_reach_the_ledger_and_are_counted_at_the_end():
    skipped = []
    search = _search(_Runner({("mtp:3", None)}), on_skip=skipped.append)
    search._run(_cfg(), "spec")
    for budget in (2048, 8192):
        search._run(_cfg(prefill_budget=budget), "prefill")
    search._run(_cfg(prefill_budget=2048), "combinations")  # a later stage asks for a skipped key again
    assert [r.failure_kind for r in skipped] == ["skipped_blocked"] * 3
    assert {k: len(v) for k, v in search._head_blocked.items()} == {search._head_key(_cfg()): 2}  # distinct keys


class _OomThenShort(_Runner):
    """Out of memory at 0.95, then, with less reserved, a failure the log gives no memory words for."""

    def run(self, cfg, stage):
        self.launches.append(cfg.key())
        if cfg.spec_decode == "mtp:3" and cfg.batch == 64:
            if cfg.gpu_memory_utilization >= 0.95:
                return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                                   error="out of memory at start-up; failed to start (rc=1)",
                                   failure_kind="oom_startup")
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error="failed to start (rc=1)\nRuntimeError: Engine core initialization failed",
                               failure_kind="startup_other")
        return TrialResult(config=cfg, stage=stage, **OK)


def test_a_failure_after_a_memory_retry_blocks_nothing():
    search = _search(_OomThenShort(set()))
    assert not search._run(_cfg(gpu_memory_utilization=0.95), "spec").ok
    assert search._run(_cfg(batch=16, gpu_memory_utilization=0.95), "batch").ok
    assert not search._head_blocked


@pytest.mark.parametrize("error, short", [
    (HEAD_LOAD, False),
    ("failed to start (rc=1)\nValueError: To serve at least one request with the models's max seq len (8192), "
     "(1.2 GiB KV cache) is needed, which is larger than the available KV cache memory (0.4 GiB).", True),
    ("failed to start (rc=1)\nValueError: SWA pool (512 tokens) cannot hold even one request", True),  # SGLang
    ("failed to start (rc=1)\nINFO Available KV cache memory: 12.3 GiB\nRuntimeError: drafter compile failed",
     False),  # the line vLLM logs on every start is no shortage
])
def test_memory_short_reads_the_engines_words(error, short):
    res = TrialResult(config=_cfg(), stage="spec", metrics=TrialMetrics(), launched=False, error=error,
                      failure_kind="startup_other")
    assert memory_short(res) is short


# vLLM raises a memory shortage in its engine core; the API server's own traceback then ends the log, so the 20-line
# tail an error keeps does not hold it (benchmarks/sessions/2026-11/a40-gemma-4-12b holds such a recorded failure).
API_SERVER_TAIL = "\n".join(
    ["(APIServer pid=1) Traceback (most recent call last):"]
    + [f'(APIServer pid=1)   File "vllm/v1/engine/utils.py", line {n}, in wait_for_engine_startup' for n in range(20)]
    + ["(APIServer pid=1) RuntimeError: Engine core initialization failed. See root cause above."])
KV_SHORT = ("(EngineCore_DP0 pid=2) ValueError: To serve at least one request with the models's max seq len (8192), "
            "(1.20 GiB KV cache) is needed, which is larger than the available KV cache memory (0.41 GiB).")


def _launch(log, spec="mtp:3"):
    runner = SubprocessTrialRunner(backends={"vllm": _DeadBackend(log)}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    return runner.run(_cfg(spec, gpu_memory_utilization=0.9), "spec")


def test_a_shortage_above_the_tail_is_found_in_the_log():
    res = _launch(KV_SHORT + "\n" + API_SERVER_TAIL)
    assert "larger than the available" not in res.error.split("\n", 1)[1]  # out of the tail
    assert res.error.startswith(MEMORY_SHORT_CAUSE) and memory_short(res) and res.failure_kind == "startup_other"
    assert not StagedSearch._head_failed(res.config, res)


def test_other_trials_errors_are_worded_as_before():
    assert _launch(KV_SHORT + "\n" + API_SERVER_TAIL, spec="ngram_gpu:4").error.startswith("failed to start (rc=")
