"""rigtune serve reports how each engine start went: seconds to ready and, for vLLM, whether its caches held."""

from __future__ import annotations

import logging

from rigtune.models import Config, StartupCost
from rigtune.serve.supervisor import Supervisor, startup_summary
from tests.test_startup_log import AOT_HIT_033, CAPTURE, MISS_033, WARMUP_COLD, _log

WARMUP_WARM = WARMUP_COLD.replace("81.98", "0.34")


class FakeProcess:
    pid = 4242

    def __init__(self, log: str, started: float = 100.0, ready: float = 188.0):
        self.log, self.started_at, self.ready_at = log, started, ready

    def wait_ready(self, timeout=0):
        return True

    def tail_log(self, n=40):
        return self.log

    def alive(self):
        return True

    def stop(self):
        pass


class FakeBackend:
    health_path = "/health"

    def __init__(self, name: str, proc: FakeProcess):
        self.name, self.proc = name, proc

    def launch(self, cfg, model, port, log_path=None):
        return self.proc


def _start(backend: str, log: str, **times) -> Supervisor:
    sup = Supervisor(FakeBackend(backend, FakeProcess(log, **times)),
                     Config(backend=backend, quant="bf16", ctx=8192, batch=64), model=None, port=8000)
    assert sup._launch()
    return sup


def test_a_warm_vllm_start_is_reported_with_its_cache_hit(caplog):
    with caplog.at_level(logging.INFO, logger="rigtune.serve.supervisor"):
        sup = _start("vllm", _log(AOT_HIT_033, WARMUP_WARM, CAPTURE))
    assert sup.startup.seconds == 88.0 and sup.startup.compile_cached is True and sup.startup.warmup_s == 0.34
    assert "vllm ready in 88 s: compile cache hit, kernel warm-up 0.3 s, CUDA graphs 23 s" in caplog.text
    status = sup.status()["startup"]
    assert status["seconds"] == 88.0 and status["compile_cached"] is True and status["warmup_s"] == 0.34


def test_a_cold_vllm_start_says_what_it_compiled(caplog):
    with caplog.at_level(logging.INFO, logger="rigtune.serve.supervisor"):
        sup = _start("vllm", _log(MISS_033, WARMUP_COLD), ready=324.0)
    assert sup.startup.compile_cached is False and sup.startup.warmup_s == 81.98
    assert "vllm ready in 224 s: compiled (31 s), kernel warm-up 82.0 s" in caplog.text


def test_another_engine_reports_its_seconds_only():
    sup = _start("sglang", "anything at all")
    assert sup.startup == StartupCost(seconds=88.0)
    assert sup.status()["startup"] == {"seconds": 88.0}


def test_an_unreadable_log_costs_the_report_not_the_server(caplog):
    class Broken(FakeProcess):
        def tail_log(self, n=40):
            raise OSError("gone")

    sup = Supervisor(FakeBackend("vllm", Broken("")), Config(backend="vllm", quant="bf16", ctx=8192, batch=64),
                     model=None, port=8000)
    with caplog.at_level(logging.WARNING, logger="rigtune.serve.supervisor"):
        assert sup._launch()
    assert sup.startup == StartupCost(seconds=88.0) and "start-up log not read: OSError" in caplog.text


def test_the_summary_says_only_what_the_log_showed():
    assert startup_summary("vllm", StartupCost()) == "vllm ready"
    assert startup_summary("vllm", StartupCost(seconds=61.4)) == "vllm ready in 61 s"
