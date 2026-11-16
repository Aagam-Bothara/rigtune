"""Recording a start-up never costs a trial or leaves an engine running, and reads vLLM's lines as it prints them."""

from __future__ import annotations

import rigtune.calibrate.search as search_mod
from rigtune.calibrate.search import SubprocessTrialRunner
from rigtune.calibrate.workload import Workload
from rigtune.memlog import parse_startup
from rigtune.models import Config, TrialMetrics
from tests.test_failure_kind import _Live, _LiveBackend
from tests.test_startup_log import CAPTURE, MISS_033, _log

GOOD = TrialMetrics(requests=8, failed=0, output_tokens=800, tok_s=100.0, ttft_ms=50.0, tpot_ms=10.0)


class _Unreadable(_Live):
    started_at, ready_at = 5.0, 65.0
    stopped = 0
    order: list = []

    def tail_log(self, n: int = 50) -> str:
        if n > 1000:  # the whole-log read made for the start-up record
            type(self).order.append("read")
            raise MemoryError("log too big")
        return ""

    def stop(self) -> None:
        type(self).stopped += 1
        type(self).order.append("stop")


class _Backend(_LiveBackend):
    def launch(self, cfg, model, port, log_path=None):
        return _Unreadable()


def _run(backend_name: str, monkeypatch):
    _Unreadable.stopped, _Unreadable.order = 0, []
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: GOOD)
    runner = SubprocessTrialRunner(backends={backend_name: _Backend("")}, models={backend_name: None}, hw=None,
                                   workload=Workload())
    return runner.run(Config(backend=backend_name, quant="bf16", gpu_memory_utilization=0.9), "batch")


def test_a_log_that_cannot_be_read_costs_the_record_not_the_trial(monkeypatch):
    res = _run("vllm", monkeypatch)
    assert res.ok and _Unreadable.stopped == 1  # the trial stands and the engine was stopped
    assert _Unreadable.order == ["stop", "read"]  # stopped before anything that could fail
    assert res.startup.seconds == 60.0 and res.startup.parsed == []  # timed; nothing claimed from the log


def test_another_engines_log_is_not_read_at_all(monkeypatch):
    res = _run("sglang", monkeypatch)
    assert res.ok and _Unreadable.order == ["stop"] and res.startup.seconds == 60.0


def test_a_negative_size_still_gives_the_capture():
    """vLLM prints free memory before minus after, which can come out a little below zero."""
    got = parse_startup("vllm", _log(CAPTURE.replace("took 0.71 GiB", "took -0.01 GiB")))
    assert (got["capture_s"], got["cudagraph_gib"], got["parsed"]) == (23.0, -0.01, ["capture"])


def test_one_range_compiled_by_several_workers_is_counted_once():
    """0.33 prints the compile line once per tensor-parallel worker; they compile at the same time."""
    other = MISS_033.replace("(1, 8192)", "(8193, 16384)").replace("31.42", "8.00")
    got = parse_startup("vllm", _log(MISS_033, MISS_033.replace("31.42", "30.10"), other))
    assert got["compile_s"] == 39.42 and got["compile_cached"] is False


def test_the_other_wording_of_a_changed_mode_is_recorded():
    line = ("WARNING Piecewise compilation with empty splitting_ops does not contain piecewise cudagraph. "
            "Setting cudagraph_mode to NONE. Hint: If you are using attention backends that support cudagraph")
    got = parse_startup("vllm", _log(line))
    assert got["rewrites"] == ["cudagraph_mode=NONE"] and got["parsed"] == ["rewrites"]
    assert parse_startup("vllm", _log("INFO Setting cudagraph_mode to /home/alice/x")) == {"parsed": []}
