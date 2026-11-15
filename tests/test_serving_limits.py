"""Opt-in serving limits (rigtune.serve.limits): off by default, so `rigtune serve` serves exactly as before."""
import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner

from rigtune.serve.limits import ServingLimits, ServingMiddleware, measured_in_flight


async def echo(scope, receive, send):
    message = await receive()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": message.get("body", b"")})


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test")


# ---- the API key


async def test_a_key_protects_inference_and_the_profile_and_health_says_only_whether_the_backend_is_up():
    from rigtune.serve.proxy import create_app

    status = {"alive": True, "ready": True, "pid": 4242, "last_error": "secret detail"}
    app = ServingMiddleware(create_app("http://127.0.0.1:9", status_fn=lambda: status), ServingLimits(api_key="k"))
    async with _client(app) as c:
        assert (await c.post("/v1/completions", content="hi")).status_code == 401
        assert (await c.get("/rigtune/profile")).headers["www-authenticate"] == "Bearer"
        health = await c.get("/health")
        assert health.status_code == 200 and health.json() == {"status": "ok", "alive": True, "ready": True}
        status["ready"] = False
        health = await c.get("/health")
        assert health.status_code == 503 and health.json() == {"status": "degraded", "alive": True, "ready": False}
        assert "4242" not in health.text and "127.0.0.1" not in health.text and "secret" not in health.text
    authorised = ServingMiddleware(echo, ServingLimits(api_key="k"))
    async with _client(authorised) as c:
        assert (await c.post("/v1/completions", content="hi", headers={"authorization": "Bearer k"})).text == "hi"


# ---- the request body


async def test_a_body_cap_applies_with_and_without_content_length():
    reached = []

    async def downstream(scope, receive, send):
        reached.append(True)
        await echo(scope, receive, send)

    async def chunks():
        yield b"123"
        yield b"456"

    app = ServingMiddleware(downstream, ServingLimits(max_body_bytes=5))
    async with _client(app) as c:
        assert (await c.post("/v1/completions", content=chunks())).status_code == 413
        assert (await c.post("/v1/completions", content=b"123456")).status_code == 413
        assert not reached
        assert (await c.post("/v1/completions", content=b"12345")).content == b"12345"


async def test_a_slow_body_gets_408_not_504():
    async def slow():
        yield b"1"
        await asyncio.sleep(5)
        yield b"2"

    app = ServingMiddleware(echo, ServingLimits(body_timeout=0.05))
    async with _client(app) as c:
        response = await asyncio.wait_for(c.post("/v1/completions", content=slow()), 4)
    assert response.status_code == 408


# ---- admission


def _stream_app(release: asyncio.Event, started: list):
    async def app(scope, receive, send):
        await receive()
        started.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await release.wait()
        await send({"type": "http.response.body", "body": b"done"})
    return app


async def test_a_full_limit_without_a_queue_answers_429_and_keeps_the_running_request():
    release, started = asyncio.Event(), []
    app = ServingMiddleware(_stream_app(release, started), ServingLimits(max_in_flight=1, queue=0))
    async with _client(app) as c:
        first = asyncio.create_task(c.post("/v1/completions", content=b"a"))
        while not started:
            await asyncio.sleep(0.005)
        rejected = await c.post("/v1/completions", content=b"b")
        assert rejected.status_code == 429 and rejected.headers["retry-after"] == "1"
        assert app.in_flight == 1
        release.set()
        assert (await first).text == "done"
    assert app.in_flight == 0 and app.waiting == 0


async def test_a_queued_request_waits_for_a_slot_and_a_full_queue_is_refused():
    release, started = asyncio.Event(), []
    app = ServingMiddleware(_stream_app(release, started), ServingLimits(max_in_flight=1, queue=1))
    async with _client(app) as c:
        first = asyncio.create_task(c.post("/v1/completions", content=b"a"))
        while not started:
            await asyncio.sleep(0.005)
        queued = asyncio.create_task(c.post("/v1/completions", content=b"b"))
        while app.waiting == 0:
            await asyncio.sleep(0.005)
        assert (await c.post("/v1/completions", content=b"c")).status_code == 429  # the queue is full
        release.set()
        assert (await first).status_code == 200 and (await queued).status_code == 200
    assert app.in_flight == 0 and app.waiting == 0


async def test_a_queued_request_gives_up_after_the_queue_timeout():
    release, started = asyncio.Event(), []
    app = ServingMiddleware(_stream_app(release, started), ServingLimits(max_in_flight=1, queue_timeout=0.05))
    async with _client(app) as c:
        first = asyncio.create_task(c.post("/v1/completions", content=b"a"))
        while not started:
            await asyncio.sleep(0.005)
        late = await c.post("/v1/completions", content=b"b")
        assert late.status_code == 429 and "waited" in late.json()["error"]["message"]
        release.set()
        await first
    assert app.waiting == 0 and app.in_flight == 0


async def test_cancellation_releases_capacity():
    entered = asyncio.Event()

    async def hung(scope, receive, send):
        entered.set()
        await asyncio.Event().wait()

    app = ServingMiddleware(hung, ServingLimits(max_in_flight=1))
    async with _client(app) as c:
        pending = asyncio.create_task(c.post("/v1/completions", content=b"hi"))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert app.in_flight == 0


# ---- the idle timeout


async def test_a_stalled_stream_ends_with_an_sse_error_event_and_no_exception():
    async def stalls(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b'data: {"x": 1}\n\n', "more_body": True})
        await asyncio.Event().wait()

    app = ServingMiddleware(stalls, ServingLimits(idle_timeout=0.05, max_in_flight=2))
    async with _client(app) as c:
        response = await asyncio.wait_for(c.post("/v1/completions", content=b"hi"), 4)
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[0] == {"x": 1} and events[-1]["error"]["code"] == "idle_timeout"
    assert app.in_flight == 0


async def test_the_idle_timeout_never_cuts_a_slow_start_or_a_whole_reply():
    async def slow_start(scope, receive, send):
        await receive()
        await asyncio.sleep(0.2)  # a long prefill: no headers yet
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"whole"})

    app = ServingMiddleware(slow_start, ServingLimits(idle_timeout=0.05))
    async with _client(app) as c:
        assert (await c.post("/v1/completions", content=b"hi")).text == "whole"


async def test_the_wait_for_a_streams_first_token_after_its_headers_is_not_idle():
    """Engines send a stream's headers at once and its first token after queueing and prefill."""
    async def engine_like(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        await asyncio.sleep(0.3)
        await send({"type": "http.response.body", "body": b'data: {"x": 1}\n\n', "more_body": True})
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})

    app = ServingMiddleware(engine_like, ServingLimits(idle_timeout=0.1))
    async with _client(app) as c:
        response = await asyncio.wait_for(c.post("/v1/chat/completions", content=b"{}"), 4)
    assert "idle_timeout" not in response.text and response.text.endswith("data: [DONE]\n\n")


async def test_a_slow_reading_client_does_not_trip_the_idle_timeout():
    async def steady(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        for _ in range(3):
            await send({"type": "http.response.body", "body": b"data: {}\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def slow_send(message):
        if message["type"] == "http.response.body":
            await asyncio.sleep(0.3)  # each chunk takes three idle timeouts to leave
        sent.append(message)

    app = ServingMiddleware(steady, ServingLimits(idle_timeout=0.1))
    await asyncio.wait_for(app({"type": "http", "method": "POST", "path": "/v1/completions", "headers": []},
                               receive, slow_send), 5)
    bodies = [m.get("body", b"") for m in sent if m["type"] == "http.response.body"]
    assert len(bodies) == 4 and not any(b"idle_timeout" in b for b in bodies)


async def test_an_application_error_under_the_idle_timeout_is_raised_as_itself():
    async def boom(scope, receive, send):
        await receive()
        raise ValueError("app failed")

    app = ServingMiddleware(boom, ServingLimits(idle_timeout=5))
    async with _client(app) as c:
        with pytest.raises(ValueError, match="app failed"):
            await c.post("/v1/completions", content=b"x")


async def test_the_bearer_scheme_is_case_insensitive_and_the_key_is_not_passed_to_the_engine():
    seen = []

    async def engine(scope, receive, send):
        seen.append(dict(scope["headers"]).get(b"authorization"))
        await echo(scope, receive, send)

    app = ServingMiddleware(engine, ServingLimits(api_key="k"))
    async with _client(app) as c:
        for value in ("Bearer k", "bearer k", "BEARER k", "Bearer k "):
            assert (await c.post("/v1/x", content=b"hi", headers={"authorization": value})).status_code == 200
        for value in ("Bearer K", "Basic k", "Bearer", "k"):
            assert (await c.post("/v1/x", content=b"hi", headers={"authorization": value})).status_code == 401
    assert seen == [None] * 4  # only the accepted requests reached the engine, without rigtune's key
    unkeyed = ServingMiddleware(engine, ServingLimits(max_in_flight=2))
    async with _client(unkeyed) as c:
        await c.post("/v1/x", content=b"hi", headers={"authorization": "Bearer theirs"})
    assert seen[-1] == b"Bearer theirs"  # without a key of rigtune's, the header passes as it always did


# ---- what --max-in-flight measured uses


def test_measured_in_flight_is_the_met_level_capped_by_the_engines_batch(hw_a100, spec, prepared_vllm):
    from tests.test_cache_and_pipeline import _profile

    p = _profile(hw_a100, spec, prepared_vllm)  # batch 64
    for served, replicas, expected in [(8, 1, 8), (128, 1, 64), (None, 1, 64), (128, 2, 128), (None, 2, 128)]:
        p.served_concurrency, p.replicas = served, replicas
        assert measured_in_flight(p) == expected


def test_the_capacity_level_is_the_busiest_level_that_met_the_constraints():
    from rigtune.calibrate.objectives import Constraints, capacity_level
    from rigtune.models import Config, TrialMetrics, TrialResult

    levels = {str(c): TrialMetrics(tok_s=t, ttft_ms=tt, ttft_p95_ms=tt, concurrency=c, requests=32,
                                   output_tokens=4096) for c, t, tt in [(1, 60, 40), (8, 400, 120), (32, 1100, 650)]}
    r = TrialResult(config=Config(backend="vllm", quant="bf16"), stage="confirm",
                    metrics=levels["32"].model_copy(update={"by_concurrency": levels}))
    assert capacity_level(r, "balanced", Constraints(ttft_ceiling_ms=500)) == 8
    assert capacity_level(r, "balanced", Constraints(ttft_ceiling_ms=1000)) == 32
    assert capacity_level(r, "balanced", Constraints(ttft_ceiling_ms=10)) is None


# ---- the serve command


class _FakeSupervisor:
    base_url = "http://127.0.0.1:9"

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def status(self):
        return {"alive": True, "ready": True}


@pytest.fixture
def served(monkeypatch, hw_a100, spec, prepared_vllm):
    """Run `rigtune serve` up to uvicorn.run with a cached-looking profile and a fake engine; return the app."""
    import rigtune.backends
    import rigtune.cli as cli
    import rigtune.pipeline as pipeline
    import rigtune.serve
    import uvicorn
    from tests.test_cache_and_pipeline import _profile

    profile = _profile(hw_a100, spec, prepared_vllm)
    profile.served_concurrency = 8
    captured = {}
    monkeypatch.delenv("RIGTUNE_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "resolve_profile", lambda *a, **kw: profile)
    monkeypatch.setattr(rigtune.backends, "get_backend",
                        lambda *a, **kw: type("B", (), {"materialize": lambda self, *x: None})())
    monkeypatch.setattr(rigtune.serve, "Supervisor", _FakeSupervisor)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.setdefault("app", app))
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)

    def run(*args):
        result = CliRunner().invoke(cli.app, ["serve", "m", *args])
        assert result.exit_code == 0, result.output
        return captured.pop("app")

    return run


def test_serve_with_defaults_applies_no_limit_deadline_body_cap_or_key(served):
    app = served()
    assert [m.cls for m in app.user_middleware] == []  # the app as it was served before these limits existed


def test_serve_installs_only_the_limits_asked_for(served):
    app = served("--max-in-flight", "measured", "--idle-timeout", "30")
    (layer,) = [m for m in app.user_middleware if m.cls is ServingMiddleware]
    limits = layer.kwargs["limits"]
    assert limits.max_in_flight == 8 and limits.waiting_room == 16 and limits.idle_timeout == 30
    assert limits.max_body_bytes is None and limits.body_timeout is None and limits.api_key is None


@pytest.mark.parametrize("args", [["--idle-timeout", "nan"], ["--body-timeout", "inf"], ["--max-in-flight", "0"],
                                  ["--max-in-flight", "lots"], ["--max-body-mb", "-1"], ["--max-body-mb", "inf"],
                                  ["--max-body-mb", "nan"], ["--require-api-key"]])
def test_invalid_limits_or_a_missing_key_are_refused_before_calibration(args, monkeypatch):
    import rigtune.pipeline as pipeline
    from rigtune.cli import app

    called = []
    monkeypatch.delenv("RIGTUNE_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "resolve_profile", lambda *a, **kw: called.append(True))
    result = CliRunner().invoke(app, ["serve", "m", *args])
    assert result.exit_code == 2 and not called
