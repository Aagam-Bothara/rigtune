"""Launch a real subprocess backend through Process / Supervisor / SubprocessTrialRunner."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List

import httpx
import pytest

from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, free_port
from rigtune.calibrate.search import SubprocessTrialRunner
from rigtune.calibrate.workload import Workload
from rigtune.memory import MemoryModel
from rigtune.models import Config
from rigtune.serve.supervisor import Supervisor

FAKE = Path(__file__).parent / "fake_server.py"


class FakeBackend(BaseBackend):
    name = "fake"

    def __init__(self, extra_args: List[str] | None = None):
        self.extra_args = extra_args or []

    def memory_model(self, hw):
        return MemoryModel(runtime_workspace=0, kv_tokens_fn=lambda c: 0, device="cpu")

    def launch_spec(self, cfg: Config, model, port: int) -> LaunchSpec:
        return LaunchSpec(args=[sys.executable, str(FAKE), "--port", str(port), *self.extra_args, *cfg.extra.get("args", [])])

    def workload_hooks(self, hw, model) -> LlmtraceHooks:
        return LlmtraceHooks(model_name="fake", gpu_ids=[], process_memory=True)


def _cfg(**extra) -> Config:
    return Config(backend="fake", quant="none", ctx=128, batch=1, extra=extra)


def test_process_starts_becomes_ready_and_stops(tmp_path):
    be = FakeBackend(["--startup-delay", "0.3"])
    port = free_port()
    proc = be.launch(_cfg(), None, port, log_path=tmp_path / "p.log")
    try:
        assert proc.alive()
        assert proc.wait_ready(timeout=15, poll=0.1)
        assert httpx.get(f"http://127.0.0.1:{port}/v1/models").json()["data"][0]["id"] == "fake"
    finally:
        proc.stop()
    assert not proc.alive()
    assert "listening" in proc.tail_log()


def test_process_reports_startup_failure(tmp_path):
    be = FakeBackend(["--fail"])
    proc = be.launch(_cfg(), None, free_port(), log_path=tmp_path / "f.log")
    try:
        assert not proc.wait_ready(timeout=10, poll=0.1)
        assert proc.returncode() == 3
        assert "refusing" in proc.tail_log()
    finally:
        proc.stop()


def test_supervisor_restarts_after_crash(tmp_path):
    # The test crashes the server itself (GET /crash) instead of using a timer: a 1 s die timer raced
    # wait_ready's 1 s poll, and on Linux the server was often dead before the first health check.
    be = FakeBackend(["--crashable"])
    sup = Supervisor(be, _cfg(), None, log_path=tmp_path / "s.log", startup_timeout=15, health_interval=0.3,
                     max_restarts=2)
    sup.start()
    try:
        first_pid = sup.process.pid
        assert sup.healthy()
        try:
            httpx.get(sup.base_url + "/crash", timeout=5)
        except httpx.HTTPError:
            pass  # the server exits mid-request
        deadline = time.time() + 20
        while time.time() < deadline and sup.restarts == 0:
            time.sleep(0.2)
        assert sup.restarts >= 1
        # Nothing crashes the restarted process, so recovery is deterministic.
        deadline = time.time() + 15
        while time.time() < deadline and not sup.healthy():
            time.sleep(0.1)
        assert sup.healthy()
        assert sup.process.pid != first_pid
        assert sup.status()["backend"] == "fake"
    finally:
        sup.stop()


def test_subprocess_trial_runner_end_to_end(tmp_path):
    be = FakeBackend()
    wl = Workload(n_prompts=3, prefill_tokens=16, decode_tokens=6, concurrencies=(1, 2))
    runner = SubprocessTrialRunner(backends={"fake": be}, models={"fake": None}, hw=None, workload=wl,
                                   log_dir=tmp_path, startup_timeout=15, request_timeout=30)
    res = runner.run(_cfg(), "quant")
    assert res.ok, res.error
    assert res.metrics.output_tokens == 3 * 6 * 2
    assert res.metrics.telemetry_source in ("psutil", "rapl", "none")
    assert list(tmp_path.glob("*.log"))

    bad = SubprocessTrialRunner(backends={"fake": FakeBackend(["--fail"])}, models={"fake": None}, hw=None,
                                workload=wl, log_dir=tmp_path, startup_timeout=10)
    res = bad.run(_cfg(), "quant")
    assert not res.ok and not res.launched and "failed to start" in res.error


@pytest.mark.skipif(sys.platform == "win32", reason="process-group kill is POSIX-only")
def test_process_group_is_used_on_posix():
    import os

    be = FakeBackend()
    proc = be.launch(_cfg(), None, free_port())
    try:
        assert os.getpgid(proc.pid) == proc.pid
    finally:
        proc.stop()


class FakeChild:
    """A started engine as the supervisor sees it, with no process behind it."""
    born = 0

    def __init__(self, ready=True, rc=None, log=""):
        FakeChild.born += 1
        self.pid = 1000 + FakeChild.born
        self._alive, self._ready, self.rc, self.log = True, ready, rc, log

    def wait_ready(self, timeout=None, **kwargs):
        return self._ready

    def alive(self):
        return self._alive

    def stop(self):
        self._alive = False

    def returncode(self):
        return self.rc

    def tail_log(self, n=30):
        return self.log


def test_shutdown_interrupts_restart_backoff(monkeypatch):
    """A stop during the third restart's eight-second backoff must not create a child."""
    stopping = threading.Event()

    class Dead:
        def alive(self):
            return False

        def stop(self):
            stopping.set()

    sup = Supervisor(FakeBackend(), _cfg(), None, port=12345, health_interval=0.001)
    sup.process = Dead()
    sup.restarts = 2
    launched = []
    monkeypatch.setattr(sup, "_launch", lambda: launched.append(True))
    sup._thread = threading.Thread(target=sup._watch, daemon=True)
    sup._thread.start()
    assert stopping.wait(2)
    sup.stop()
    assert not sup._thread.is_alive()
    assert launched == []


def test_shutdown_during_backoff_launches_nothing_through_the_real_launch_path(monkeypatch):
    """The same, through start(), _watch() and _launch() as they are: only the backend's launch is faked."""
    be = FakeBackend()
    children = []
    monkeypatch.setattr(be, "launch", lambda *a, **kw: children.append(FakeChild()) or children[-1])
    sup = Supervisor(be, _cfg(), None, port=12345, health_interval=0.01, max_restarts=5)
    sup.start()
    assert len(children) == 1 and sup.status()["ready"]
    sup.restarts = 2  # the next restart is the third: an eight-second backoff
    children[0].stop()  # the engine dies
    deadline = time.monotonic() + 5
    while sup.restarts < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert sup.restarts == 3 and not sup.status()["ready"]
    began = time.monotonic()
    sup.stop()
    assert time.monotonic() - began < 2  # the backoff did not hold shutdown for eight seconds
    assert not sup._thread.is_alive()  # joined, so nothing can launch from it any more
    assert len(children) == 1


def test_shutdown_during_startup_stops_the_created_child(monkeypatch):
    waiting, release = threading.Event(), threading.Event()

    class Starting:
        pid = 123
        stopped = False

        def wait_ready(self, **kwargs):
            waiting.set()
            assert release.wait(10)
            return True

        def stop(self):
            self.stopped = True
            release.set()

        def alive(self):
            return not self.stopped

    be = FakeBackend()
    child = Starting()
    monkeypatch.setattr(be, "launch", lambda *a, **kw: child)
    sup = Supervisor(be, _cfg(), None, port=12345)
    starter = threading.Thread(target=sup.start, daemon=True)
    starter.start()
    assert waiting.wait(2)
    sup.stop()
    starter.join(2)
    assert not starter.is_alive()
    assert child.stopped
    assert sup._thread is None or not sup._thread.is_alive()


def test_readiness_drops_only_after_as_many_missed_polls_as_a_restart_waits_for(monkeypatch):
    from fastapi.testclient import TestClient

    from rigtune.serve.proxy import create_app
    from rigtune.serve.supervisor import MISSES_TOLERATED

    be = FakeBackend()
    monkeypatch.setattr(be, "launch", lambda *a, **kw: FakeChild())
    sup = Supervisor(be, _cfg(), None, port=12345, health_interval=0.001, max_restarts=0)
    polls = iter([True] + [False] * MISSES_TOLERATED)
    seen = []

    def healthy():
        seen.append(sup.status()["ready"])  # what /health said after the previous poll
        return next(polls)

    monkeypatch.setattr(sup, "healthy", healthy)
    app = create_app("http://127.0.0.1:9", status_fn=sup.status)
    sup.start()
    sup._thread.join(5)  # with no restarts allowed, the third miss ends the watch
    assert seen == [True] * (1 + MISSES_TOLERATED)  # ready through the first misses...
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503  # ...and not after the third
    sup.stop()


def test_health_endpoint_reports_an_alive_but_unready_backend():
    from fastapi.testclient import TestClient

    from rigtune.serve.proxy import create_app

    with TestClient(create_app("http://127.0.0.1:9", status_fn=lambda: {"alive": True, "ready": False})) as c:
        assert c.get("/health").status_code == 503
    with TestClient(create_app("http://127.0.0.1:9", status_fn=lambda: {"alive": True})) as c:
        assert c.get("/health").status_code == 200  # a status without readiness falls back to liveness


def test_supervisor_does_not_mark_a_404_health_check_ready(monkeypatch):
    from types import SimpleNamespace

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return httpx.Response(404)

    sup = Supervisor(FakeBackend(), _cfg(), None, port=12345)
    sup.process = SimpleNamespace(alive=lambda: True)
    monkeypatch.setattr(httpx, "Client", lambda **kw: Client())
    assert not sup.healthy()

    # Startup must use the same readiness definition, before the periodic watch begins.
    from rigtune.backends.base import Process
    process = Process(LaunchSpec(args=[]), 12345, "http://127.0.0.1:12345/health")
    monkeypatch.setattr(process, "alive", lambda: True)
    assert not process.wait_ready(timeout=0.01, poll=0)


def test_disaggregated_shutdown_also_interrupts_restart_backoff(monkeypatch):
    from rigtune.disagg import DisaggSupervisor
    from tests.test_phases import _pd_spec
    stopping = threading.Event()

    class DeadPair:
        def alive(self):
            return False

        def stop(self):
            stopping.set()

    sup = DisaggSupervisor(FakeBackend(), _pd_spec(), None, health_interval=0.001)
    sup.pair = DeadPair()
    sup.restarts = 2
    launched = []
    monkeypatch.setattr(sup, "_launch", lambda: launched.append(True))
    sup._thread = threading.Thread(target=sup._watch, daemon=True)
    sup._thread.start()
    assert stopping.wait(2)
    sup.stop()
    assert not sup._thread.is_alive()
    assert launched == []


def test_a_disaggregated_pair_that_fails_to_start_reports_return_codes_and_its_log(monkeypatch):
    import rigtune.disagg as D
    from tests.test_phases import _pd_spec

    def launch(*args, wait_ready=True, **kwargs):
        assert not wait_ready  # the supervisor waits itself, so that stop() can reach a starting pair
        return D.PDProcesses(FakeChild(ready=False, rc=1, log="CUDA out of memory"), FakeChild(ready=False)), None

    monkeypatch.setattr(D, "launch_pair", launch)
    sup = D.DisaggSupervisor(FakeBackend(), _pd_spec(), None)
    with pytest.raises(RuntimeError) as failed:
        sup.start()
    assert "prefill rc=1, decode rc=None" in str(failed.value) and "CUDA out of memory" in str(failed.value)
    assert not sup.pair.alive() and not sup.status()["ready"]
