"""Observe SSE independently of network chunk boundaries and response interleaving."""
import json
import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from rigtune.drift import TrafficWatch
from rigtune.serve.proxy import create_app


def event(body):
    return b"data: " + json.dumps(body, ensure_ascii=False).encode() + b"\r\n\r\n"


@pytest.mark.parametrize("width", [1, 2, 7, 25, 1024])
def test_fragmented_usage_and_utf8_content_are_counted_once(width):
    from rigtune.serve.streaming import StreamObserver

    watch = TrafficWatch()
    observer = StreamObserver(watch, started=0, clock=lambda: 0.025)
    data = event({"choices": [{"delta": {"content": "café"}}]})
    data += event({"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 45}})
    for start in range(0, len(data), width):
        observer.feed(data[start:start + width])
    observer.finish()
    assert list(watch.prompt_tokens) == [123]
    assert list(watch.completion_tokens) == [45]
    assert list(watch.ttft_ms) == [25]


def test_interleaved_responses_do_not_share_a_parser_buffer():
    from rigtune.serve.streaming import StreamObserver

    watch = TrafficWatch()
    a, b = StreamObserver(watch, 0), StreamObserver(watch, 0)
    first = event({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
    second = event({"usage": {"prompt_tokens": 30, "completion_tokens": 40}})
    for i in range(max(len(first), len(second))):
        a.feed(first[i:i + 1])
        b.feed(second[i:i + 1])
    a.finish()
    b.finish()
    assert sorted(watch.prompt_tokens) == [10, 30]
    assert sorted(watch.completion_tokens) == [20, 40]


def test_cumulative_usage_is_counted_once_per_response():
    from rigtune.serve.streaming import StreamObserver
    watch = TrafficWatch()
    observer = StreamObserver(watch, 0)
    observer.feed(event({"usage": {"prompt_tokens": 5, "completion_tokens": 1}}))
    observer.feed(event({"usage": {"prompt_tokens": 5, "completion_tokens": 3}}))
    observer.finish()
    observer.finish()
    assert list(watch.prompt_tokens) == [5]
    assert list(watch.completion_tokens) == [3]


def test_ttft_ignores_role_heartbeats_and_empty_deltas():
    from rigtune.serve.streaming import StreamObserver

    watch = TrafficWatch()
    tick = [0.01]
    observer = StreamObserver(watch, 0, clock=lambda: tick[0])
    observer.feed(b": heartbeat\n\n" + event({"choices": [{"delta": {"role": "assistant", "content": ""}}]}))
    assert list(watch.ttft_ms) == []
    tick[0] = 0.05
    observer.feed(event({"choices": [{"delta": {"content": "answer"}}]}))
    assert list(watch.ttft_ms) == [50]


def test_a_bad_or_oversized_event_does_not_hide_the_next_usage():
    from rigtune.serve.streaming import StreamObserver

    watch = TrafficWatch()
    observer = StreamObserver(watch, 0)
    observer.feed(b"data: broken json\n\n" + b"data: " + b"x" * 200000 + b"\n\n")
    observer.feed(event({"usage": {"prompt_tokens": 5, "completion_tokens": 6}}))
    observer.finish()
    assert list(watch.completion_tokens) == [6]


def test_failed_stream_closes_upstream_and_balances_request_accounting(monkeypatch):
    closed = []

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield event({"choices": [{"delta": {"content": "hello"}}]})
            raise httpx.ReadError("stream broke")

        async def aclose(self):
            closed.append(True)

    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=Broken()))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))
    watch = TrafficWatch()
    with TestClient(create_app("http://engine", watch=watch), raise_server_exceptions=False) as client:
        client.post("/v1/completions", json={"prompt": "hi", "stream": True})
        assert watch.in_flight == 0
        assert closed


def test_non_streamed_usage_and_prefix_reuse_are_observed(monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"usage": {
        "prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 60}}}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))
    watch = TrafficWatch({"sessions": True})
    with TestClient(create_app("http://engine", watch=watch)) as client:
        assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 200
        assert list(watch.prompt_tokens) == [100]
        assert list(watch.completion_tokens) == [20]
        assert watch.live_reuse() == 0.6
        assert watch.in_flight == 0


@pytest.mark.parametrize("kind", ["replicas", "disaggregated"])
def test_distributed_stream_errors_close_upstreams_and_release_slots(kind, monkeypatch):
    from rigtune.layout import create_lb_app
    from rigtune.disagg import create_pd_app
    closed = []

    class Broken(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: hello\n\n"
            raise httpx.ReadError("stream broke")

        async def aclose(self):
            closed.append(True)

    def respond(request):
        if request.url.host == "prefill":
            return httpx.Response(200, json={"kv_transfer_params": {}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Broken())

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    app = create_lb_app(["http://decode"]) if kind == "replicas" else create_pd_app("http://prefill", "http://decode")
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/v1/completions", json={"prompt": "hi", "stream": True})
        assert closed
        if kind == "replicas":
            assert app.state.lb["inflight"] == [0]


@pytest.mark.parametrize("kind", ["single", "replicas", "disaggregated"])
async def test_cancellation_before_the_first_stream_chunk_still_closes_the_response(kind, monkeypatch):
    from rigtune.layout import create_lb_app
    from rigtune.disagg import create_pd_app
    closed = []

    class Quiet(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: hello\n\n"

        async def aclose(self):
            closed.append(True)

    def respond(request):
        if request.url.host == "prefill":
            return httpx.Response(200, json={"kv_transfer_params": {}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Quiet())

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    app = (create_lb_app(["http://decode"]) if kind == "replicas" else
           create_pd_app("http://prefill", "http://decode") if kind == "disaggregated" else
           create_app("http://decode"))
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/v1/completions", "query_string": b"",
             "headers": [(b"content-type", b"application/json")], "server": ("test", 80), "client": ("test", 1)}

    async def receive():
        return {"type": "http.request", "body": b'{"prompt":"hi","stream":true}', "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            raise asyncio.CancelledError

    async with app.router.lifespan_context(app):
        try:
            await app(scope, receive, send)
        except asyncio.CancelledError:
            pass
        assert closed
        if kind == "replicas":
            assert app.state.lb["inflight"] == [0]


ROLE = b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
TOKEN = b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
USAGE = b'data: {"id":"x","choices":[],"usage":{"prompt_tokens":123,"completion_tokens":45}}\n\n'
DONE = b"data: [DONE]\n\n"


def _observe(chunks):
    """Feed one chunk per second of a fake clock; return the watch."""
    from rigtune.serve.streaming import StreamObserver

    watch, now = TrafficWatch(), [0.0]
    observer = StreamObserver(watch, 0.0, clock=lambda: now[0])
    for chunk in chunks:
        now[0] += 1.0
        observer.feed(chunk)
    observer.finish()
    return watch


@pytest.mark.parametrize("name, chunks", [
    ("CR-only line endings", [(ROLE + TOKEN + USAGE + DONE).replace(b"\n", b"\r")]),
    ("CRLF split between CR and LF", [(ROLE + TOKEN).replace(b"\n", b"\r\n")[:-1],
                                      (ROLE + TOKEN).replace(b"\n", b"\r\n")[-1:] + USAGE]),
    ("final event without a blank line", [ROLE + TOKEN + USAGE.rstrip(b"\n")]),
    ("final event ending in one newline", [ROLE + TOKEN + USAGE.rstrip(b"\n") + b"\n"]),
    ("final event ending in a lone CR", [ROLE + TOKEN + USAGE.rstrip(b"\n") + b"\r"]),
    ("comments, event and id fields", [b": keepalive\n\n" + ROLE + b"event: message\nid: 7\n" + TOKEN + USAGE]),
    ("data without a space", [ROLE + TOKEN + USAGE.replace(b"data: ", b"data:")]),
])
def test_line_endings_and_field_forms_the_sse_specification_allows(name, chunks):
    watch = _observe(chunks)
    assert list(watch.prompt_tokens) == [123] and list(watch.completion_tokens) == [45], name
    assert len(watch.ttft_ms) == 1, name


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_is_generated_output_for_the_time_to_first_token(field):
    reason = b'data: {"choices":[{"index":0,"delta":{"%s":"think"}}]}\n\n' % field.encode()
    watch = _observe([ROLE, reason, TOKEN, USAGE])
    assert list(watch.ttft_ms) == [2000]  # the reasoning event, the second chunk, not the answer's first


def test_a_first_answer_event_over_the_size_cap_keeps_its_time_to_first_token():
    big = b'data: {"choices":[{"delta":{"content":"' + b"x" * 70000 + b'"}}]}\n\n'
    watch = _observe([ROLE, big[:30000], big[30000:], USAGE])
    assert list(watch.ttft_ms) == [2000]  # timed when the event began to arrive
    assert list(watch.completion_tokens) == [45]  # and the next event is still read
    role_only = b'data: {"choices":[{"delta":{"role":"assistant","content":""}}],"pad":"' + b"y" * 70000 + b'"}\n\n'
    assert list(_observe([role_only, USAGE]).ttft_ms) == []  # a large event without output is not an answer


def test_the_old_chunk_parser_is_gone_and_usage_without_timing_still_counts():
    """TrafficWatch.record_stream_chunk had the split-event bug and only tests called it; it is removed."""
    from rigtune.serve.streaming import StreamObserver

    watch = TrafficWatch()
    assert not hasattr(watch, "record_stream_chunk")
    observer = StreamObserver(watch, 0.0, time_first=False)
    observer.feed((ROLE + TOKEN + USAGE + DONE).replace(b"\n", b"\r"))
    observer.finish()
    assert list(watch.prompt_tokens) == [123] and list(watch.ttft_ms) == []


async def test_the_calibration_client_times_the_same_first_event_as_the_proxy():
    """measure.generated is the one rule: a reasoning delta starts the clock in calibration as it does live."""
    from rigtune.calibrate.measure import _one_request
    from rigtune.calibrate.tokens import TokenCounter

    body = ROLE + b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}\n\n' + TOKEN + USAGE + DONE
    transport = httpx.MockTransport(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"},
                                                             content=body))
    async with httpx.AsyncClient(transport=transport) as client:
        out = await _one_request(client, "http://engine/v1/chat/completions", {}, 30.0, TokenCounter(),
                                 keep_text=True)
    assert out.ok and out.tokens == 45 and out.text == "Hi"  # timing changes; the counted answer does not
