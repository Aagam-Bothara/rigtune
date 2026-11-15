"""A failed trial's cause is kept in TrialResult.failure_kind beside its error text, which keeps its prefix.

The field is set where the runner writes the error, and read back from the error text for results recorded without
it. It is never written into a profile or a comparison's results, so every file those write stays as it was.
"""

from __future__ import annotations

import json

import pytest

import rigtune.calibrate.search as search_mod
from rigtune.calibrate.search import (
    FAILURE_KINDS,
    LOAD_FAILED,
    MM_PROFILING_FAILED,
    StagedSearch,
    SubprocessTrialRunner,
    failure_kind,
    startup_cause,
)
from rigtune.calibrate.workload import Workload
from rigtune.memcal import _is_oom
from rigtune.models import Config, Profile, TrialMetrics, TrialResult
from tests.conftest import make_hw
from tests.test_load_failure import SGLANG_LOG, TIED
from tests.test_objectives_and_search import FakeRunner
from tests.test_text_only import ENGINE, LOAD_LOG, MM_LOG, OOM_LOG, VIDEO, _DeadBackend, _runner_error

CFG = Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9)


def _stripped(res: TrialResult) -> TrialResult:
    """The result as a profile written before the field, or another runner, would hold it."""
    return res.model_copy(update={"failure_kind": None})


@pytest.mark.parametrize("log, kind, prefix", [
    (OOM_LOG, "oom_startup", "out of memory at start-up; failed to start (rc=1)\n"),
    (MM_LOG, "mm_profiling", f"{MM_PROFILING_FAILED} ({VIDEO}); failed to start (rc=1)\n"),
    (SGLANG_LOG, "load_failure", f"{LOAD_FAILED} ({TIED}); failed to start (rc=1)\n"),
    (LOAD_LOG, "startup_other", "failed to start (rc=1)\n"),
])
def test_the_runner_sets_the_kind_where_it_writes_the_error(log, kind, prefix):
    res = _runner_error(log)
    assert res.failure_kind == kind and failure_kind(res) == kind
    assert res.error.startswith(prefix)  # the prefix stays: startup_cause and memcal still read it
    assert failure_kind(_stripped(res)) == kind  # and the text alone says the same, for older results


def test_the_error_text_and_its_readers_are_unchanged():
    oom, load = _runner_error(OOM_LOG), _runner_error(SGLANG_LOG)
    assert _is_oom(oom.error) and not _is_oom(load.error)
    assert startup_cause(load.error) == (LOAD_FAILED, TIED)


class _Live:
    pid = 1

    def wait_ready(self, timeout: float) -> bool:
        return True

    def tail_log(self, n: int = 50) -> str:
        return ""

    def returncode(self) -> int:
        return 0

    def stop(self) -> None:
        pass


class _LiveBackend(_DeadBackend):
    def launch(self, cfg, model, port, log_path=None):
        return _Live()


def _live_run(cfg: Config) -> TrialResult:
    runner = SubprocessTrialRunner(backends={"vllm": _LiveBackend("")}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    return runner.run(cfg, "batch")


def test_a_power_setting_that_cannot_be_applied(monkeypatch):
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: pytest.fail("measured despite the power error"))
    res = _live_run(CFG.model_copy(update={"power_limit_w": 200}))
    assert res.launched and res.error.startswith("power setting requested")
    assert res.failure_kind == "power_unavailable" == failure_kind(_stripped(res))


def test_too_many_failed_requests(monkeypatch):
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: TrialMetrics(
        tok_s=10, requests=48, failed=12, output_tokens=100, errors=["HTTP 500"]))
    res = _live_run(CFG)
    assert res.error == "12/48 requests failed: HTTP 500"
    assert res.failure_kind == "request_failures" == failure_kind(_stripped(res))


def test_a_tolerated_failure_is_no_failure(monkeypatch):
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: TrialMetrics(
        tok_s=10, requests=48, failed=2, output_tokens=100))
    res = _live_run(CFG)
    assert res.ok and res.failure_kind is None and failure_kind(res) is None


def test_a_trial_that_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(search_mod, "run_trial", boom)
    res = _live_run(CFG)
    assert not res.launched and res.error == "connection reset"
    assert res.failure_kind == "crashed" == failure_kind(_stripped(res))


def test_the_search_marks_what_it_skips():
    t = [0.0]
    search = StagedSearch(objective="throughput", runner=FakeRunner(), budget_s=100, clock=lambda: t[0])
    first = search._run(CFG, "quant")
    assert first.ok and first.failure_kind is None
    t[0] = 200.0
    skipped = search._run(CFG.model_copy(update={"batch": 128}), "batch")
    assert skipped.error == "skipped: calibration budget reached"
    assert skipped.failure_kind == "skipped_budget" == failure_kind(_stripped(skipped))
    search = StagedSearch(objective="throughput", runner=FakeRunner())
    search._spec_blocked["suffix"] = []
    blocked = search._run(CFG.model_copy(update={"spec_decode": "suffix:24"}), "spec")
    assert blocked.failure_kind == "skipped_blocked" == failure_kind(_stripped(blocked))


def test_every_kind_is_listed():
    seen = {_runner_error(log).failure_kind for log in (OOM_LOG, MM_LOG, SGLANG_LOG, LOAD_LOG)}
    assert seen <= set(FAILURE_KINDS)
    assert set(search_mod._CAUSE_KINDS.values()) <= set(FAILURE_KINDS)


def test_never_written_into_a_profile_or_result_file():
    res = _runner_error(OOM_LOG)
    assert res.failure_kind == "oom_startup"
    assert "failure_kind" not in res.model_dump() and "failure_kind" not in json.loads(res.model_dump_json())
    hw = make_hw("rtx4090")
    profile = Profile(rigtune_version="0", hardware_hash="h", hardware=hw, model_id="m", objective="throughput",
                      backend="vllm", backend_version="1.0", config=CFG, launch_args=[], calibration_table=[res])
    text = profile.model_dump_json(indent=2)
    assert "failure_kind" not in text
    assert text == profile.model_copy(update={"calibration_table": [_stripped(res)]}).model_dump_json(indent=2)


def test_a_result_recorded_before_the_field_loads_and_is_read_from_its_text():
    old = {"config": CFG.model_dump(), "stage": "quant", "metrics": TrialMetrics().model_dump(), "launched": False,
           "error": f"out of memory at start-up; failed to start (rc=1)\n{ENGINE}"}
    res = TrialResult.model_validate(old)
    assert res.failure_kind is None and failure_kind(res) == "oom_startup"
    fake = FakeRunner().run(CFG.model_copy(update={"gpu_memory_utilization": 0.95}), "quant")  # "CUDA OOM"
    assert failure_kind(fake) == "crashed"  # no prefix the runner writes: not called a start-up OOM
