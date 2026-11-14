"""Above 100 users the load generator opens a connection per user; at 100 or fewer its client is unchanged.

httpx's default pool holds 100 connections, and a request's clock starts before it waits for one, so at the
high-concurrency preset's 128 users 28 requests counted their wait for a connection as time to first token. The
engine below counts the connections open at once and answers no request until `target` of them are in flight
(or a liveness timeout passes, which only a failing test reaches). Nothing here times anything.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import rigtune.calibrate.measure as M
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.workload import Workload

SSE = (f"data: {json.dumps({'choices': [{'text': 'x'}]})}\n\n"
       f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 1}})}\n\n"
       "data: [DONE]\n\n").encode()


class CountingEngine:
    """An HTTP/1.1 keep-alive server that records the peak of connections and of requests open at once."""

    def __init__(self, target: int, give_up_s: float = 20.0):
        self.target, self.give_up_s = target, give_up_s
        self.connections = self.peak_connections = 0
        self.requests = self.peak_requests = 0
        self.enough = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self.peak_connections = max(self.peak_connections, self.connections)
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n"):
                    name, _, value = line.partition(b":")
                    if name.strip().lower() == b"content-length":
                        length = int(value)
                await reader.readexactly(length)
                self.requests += 1
                self.peak_requests = max(self.peak_requests, self.requests)
                if self.requests >= self.target:
                    self.enough.set()
                try:
                    await asyncio.wait_for(self.enough.wait(), self.give_up_s)
                except asyncio.TimeoutError:
                    pass
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                             + f"Content-Length: {len(SSE)}\r\n\r\n".encode() + SSE)
                await writer.drain()
                self.requests -= 1
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.connections -= 1
            writer.close()

    async def __aenter__(self) -> str:
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def __aexit__(self, *exc) -> None:
        self.server.close()


def _workload(n: int) -> Workload:
    return Workload(n_prompts=n, prompts=[f"prompt {i}" for i in range(n)], decode_tokens=1, concurrencies=(n,))


def _client_kwargs(users: int, monkeypatch) -> dict:
    seen = []
    monkeypatch.setattr(M.httpx, "AsyncClient", lambda **kw: seen.append(kw))
    M._client(users)
    return seen[0]


@pytest.mark.parametrize("users", [1, 32, 64, 100])
def test_up_to_100_users_the_client_is_built_exactly_as_before(users, monkeypatch):
    assert _client_kwargs(users, monkeypatch) == {}  # httpx's own limits: 100 connections, 20 kept alive


@pytest.mark.parametrize("users", [101, 128, 512])
def test_above_100_users_each_user_has_a_connection_of_its_own(users, monkeypatch):
    limits = _client_kwargs(users, monkeypatch)["limits"]
    assert isinstance(limits, httpx.Limits)
    assert (limits.max_connections, limits.max_keepalive_connections) == (users, users)


@pytest.mark.parametrize("users", [64, 128])
async def test_every_user_is_in_flight_at_once(users):
    engine = CountingEngine(target=users)
    async with engine as url:
        out = await M._drive(url, LlmtraceHooks(model_name="m"), _workload(users), users, timeout=60.0)
    assert len(out) == users and all(o.ok for o in out)
    assert (engine.peak_connections, engine.peak_requests) == (users, users)


async def test_open_loop_arrivals_get_a_connection_each_too():
    engine = CountingEngine(target=128)
    async with engine as url:
        out, _, peak = await M._drive_open(url, LlmtraceHooks(model_name="m"), _workload(128), [0.0] * 128, 0.0,
                                           timeout=60.0)
    assert len(out) == 128 and all(o.ok for o in out) and peak == 128
    assert engine.peak_connections == 128
