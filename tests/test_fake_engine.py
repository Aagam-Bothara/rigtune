"""rigtune/bench/fake_engine.py: the protocol the client self-check relies on. Nothing here times anything."""

from __future__ import annotations

import asyncio
import json

import httpx

from rigtune.bench.fake_engine import FakeEngine
from rigtune.calibrate.measure import _one_request
from rigtune.calibrate.tokens import TokenCounter


class Running:
    def __init__(self, **kw):
        self.engine = FakeEngine(**kw)

    async def __aenter__(self) -> str:
        self.server = await asyncio.start_server(self.engine.handle, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def __aexit__(self, *exc) -> None:
        self.server.close()


def _events(text: str) -> list:
    return [line for line in text.split("\n") if line]


async def test_health_models_and_errors():
    async with Running(model="m") as url, httpx.AsyncClient() as client:
        assert (await client.get(url + "/health")).json() == {"status": "ok"}
        assert (await client.get(url + "/v1/models")).json()["data"][0]["id"] == "m"
        assert (await client.get(url + "/nowhere")).status_code == 404
        assert (await client.get(url + "/v1/completions")).status_code == 405
        assert (await client.post(url + "/v1/completions", content=b"{not json")).status_code == 400


async def test_a_streamed_completion_sends_its_tokens_a_usage_chunk_and_its_own_timings():
    async with Running(ttft_ms=0, itl_ms=0, tokens=5) as url, httpx.AsyncClient() as client:
        r = await client.post(url + "/v1/completions", json={"prompt": "p", "stream": True})
        lines = _events(r.text)
    data = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    assert [d["choices"][0]["text"] for d in data[:5]] == ["x "] * 5
    assert data[5] == {"object": "text_completion", "model": "fake", "choices": [],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 5}}
    comment = [line for line in lines if line.startswith(": fake-engine ")]
    record = json.loads(comment[0][len(": fake-engine "):])
    assert sorted(record) == ["duration_s", "start_s", "tokens", "ttft_s"] and record["tokens"] == 5
    assert 0 <= record["ttft_s"] <= record["duration_s"]
    assert lines[-1] == "data: [DONE]"


async def test_chat_streams_deltas_and_a_plain_request_gets_one_answer():
    async with Running(ttft_ms=0, itl_ms=0, tokens=3) as url, httpx.AsyncClient() as client:
        r = await client.post(url + "/v1/chat/completions", json={"messages": [], "stream": True})
        deltas = [json.loads(line[6:]) for line in _events(r.text) if line.startswith("data: {")]
        assert [d["choices"][0]["delta"]["content"] for d in deltas[:3]] == ["x "] * 3
        plain = (await client.post(url + "/v1/completions", json={"prompt": "p"})).json()
        assert plain["choices"][0]["text"] == "x x x " and plain["usage"]["completion_tokens"] == 3


async def test_stats_hold_one_record_per_request_until_reset():
    async with Running(ttft_ms=0, itl_ms=0, tokens=2) as url, httpx.AsyncClient() as client:
        for _ in range(3):
            await client.post(url + "/v1/completions", json={"prompt": "p", "stream": True})
        stats = (await client.get(url + "/stats")).json()
        assert len(stats) == 3 and all(s["tokens"] == 2 for s in stats)
        await client.post(url + "/reset")
        assert (await client.get(url + "/stats")).json() == []


async def test_the_load_generator_reads_it_as_it_reads_an_engine():
    async with Running(ttft_ms=0, itl_ms=0, tokens=7) as url, httpx.AsyncClient() as client:
        body = {"prompt": "p", "stream": True, "stream_options": {"include_usage": True}}
        out = await _one_request(client, url + "/v1/completions", body, 30.0, TokenCounter())
    assert out.ok and out.tokens == 7 and out.token_source == "usage"


async def test_many_requests_share_one_keep_alive_connection():
    async with Running(ttft_ms=0, itl_ms=0, tokens=1) as url:
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1)) as client:
            for _ in range(5):
                r = await client.post(url + "/v1/completions", json={"prompt": "p", "stream": True})
                assert r.status_code == 200
