"""Opt-in limits at the public ASGI boundary of `rigtune serve`: admission control with a bounded wait queue, an idle
timeout between streamed chunks, a request-body cap and read timeout, and a bearer API key.

Every limit is off unless asked for, and with all of them off `rigtune serve` does not install this middleware, so
it serves exactly as it did before they existed: no request is refused, delayed or cut short by rigtune itself."""
from __future__ import annotations

import hmac
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import anyio
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_FACTOR = 2  # --queue defaults to this many waiting requests per admitted one
DEFAULT_QUEUE_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ServingLimits:
    max_in_flight: Optional[int] = None  # inference requests (POST /v1/*) served at once; None = no limit
    queue: Optional[int] = None  # requests that may wait for a slot; None = DEFAULT_QUEUE_FACTOR x max_in_flight
    queue_timeout: float = DEFAULT_QUEUE_TIMEOUT_S  # how long a queued request waits before a 429
    idle_timeout: Optional[float] = None  # seconds a streamed reply may go without a chunk; None = no limit
    max_body_bytes: Optional[int] = None  # None = no cap
    body_timeout: Optional[float] = None  # seconds to receive the whole request body; None = no limit
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("max_in_flight", "max_body_bytes"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name.replace('_', '-')} must be at least 1")
        if self.queue is not None and self.queue < 0:
            raise ValueError("queue must not be negative")
        for name in ("queue_timeout", "idle_timeout", "body_timeout"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name.replace('_', '-')} must be a finite number of seconds above 0")

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.max_in_flight, self.idle_timeout, self.max_body_bytes,
                                           self.body_timeout, self.api_key or None))

    @property
    def waiting_room(self) -> int:
        if self.max_in_flight is None:
            return 0
        return self.queue if self.queue is not None else DEFAULT_QUEUE_FACTOR * self.max_in_flight

    def describe(self) -> List[str]:
        out = []
        if self.max_in_flight is not None:
            out.append(f"{self.max_in_flight} inference requests at once, {self.waiting_room} more waiting up to "
                       f"{self.queue_timeout:g} s, then 429")
        if self.idle_timeout is not None:
            out.append(f"streams closed after {self.idle_timeout:g} s without a chunk")
        if self.max_body_bytes is not None:
            out.append(f"request bodies up to {self.max_body_bytes / 2**20:g} MiB")
        if self.body_timeout is not None:
            out.append(f"request bodies received within {self.body_timeout:g} s")
        if self.api_key:
            out.append("a bearer API key on every route but /health")
        return out


def measured_in_flight(profile: Any) -> int:
    """--max-in-flight measured: the busiest concurrency at which the calibrated pick met its constraints
    (Profile.served_concurrency), capped by what the engines can batch (each engine's batch x engines), since a
    request beyond that waits inside the engine anyway. Without a measured level, the batch capacity."""
    engines = max(1, int(getattr(profile, "replicas", 1) or 1))
    disagg = getattr(profile, "disagg", None)
    batch = (disagg.decode.batch if disagg is not None else profile.config.batch) or 1
    capacity = batch * engines
    measured = getattr(profile, "served_concurrency", None)
    return max(1, min(measured, capacity)) if measured else capacity


class ServingMiddleware:
    def __init__(self, app: ASGIApp, limits: ServingLimits, clock: Callable[[], float] = time.monotonic):
        self.app = app
        self.limits = limits
        self.clock = clock
        self.in_flight = 0
        self.waiting = 0
        self._slots: Optional[anyio.Semaphore] = None  # made on the server's event loop, at the first request

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        limits = self.limits
        if limits.api_key:
            if path == "/health":  # open for probes, and saying only whether the backend is up
                await self._minimal_health(scope, receive, send)
                return
            headers = list(scope.get("headers", []))
            given = dict(headers).get(b"authorization", b"")
            scheme, _, credentials = given.strip().partition(b" ")
            # The scheme is case-insensitive (RFC 7235); the key itself is compared exactly, in constant time.
            if scheme.lower() != b"bearer" or not hmac.compare_digest(credentials.strip(),
                                                                     limits.api_key.encode()):
                await _error(scope, receive, send, 401, "a valid bearer API key is required",
                             {"www-authenticate": "Bearer"})
                return
            # rigtune's key is for rigtune: the engines behind it are launched without one, so it is not passed on
            # to them, where it could only end up in their logs.
            scope = {**scope, "headers": [(k, v) for k, v in headers if k.lower() != b"authorization"]}
        if limits.max_body_bytes is not None or limits.body_timeout is not None:
            body = await self._read_body(scope, receive, send)
            if body is None:
                return
            receive = _replay(body, receive)
        if scope.get("method") == "POST" and path.startswith("/v1/") and limits.max_in_flight is not None:
            await self._admitted(scope, receive, send)
        else:
            await self._serve(scope, receive, send)

    # ---- admission

    async def _admitted(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._slots is None:
            self._slots = anyio.Semaphore(self.limits.max_in_flight or 1)
        slots = self._slots
        if slots.value == 0:
            if self.waiting >= self.limits.waiting_room:
                await _error(scope, receive, send, 429, "inference capacity and its queue are full; retry later",
                             {"retry-after": "1"})
                return
            self.waiting += 1
            acquired = False
            try:
                with anyio.move_on_after(self.limits.queue_timeout):
                    await slots.acquire()
                    acquired = True
            finally:
                self.waiting -= 1
            if not acquired:
                await _error(scope, receive, send, 429, f"waited {self.limits.queue_timeout:g} s for inference "
                             "capacity; retry later", {"retry-after": "1"})
                return
        else:
            await slots.acquire()
        self.in_flight += 1
        try:
            await self._serve(scope, receive, send)
        finally:
            self.in_flight -= 1
            slots.release()

    # ---- the request body

    async def _read_body(self, scope: Scope, receive: Receive, send: Send) -> Optional[bytes]:
        cap = self.limits.max_body_bytes
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared = -1
        if declared < 0:
            await _error(scope, receive, send, 400, "invalid Content-Length")
            return None
        if cap is not None and declared > cap:
            await _error(scope, receive, send, 413, "request body exceeds the configured size limit")
            return None
        body = bytearray()
        try:
            with anyio.fail_after(self.limits.body_timeout):  # None: no limit
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return None
                    body.extend(message.get("body", b""))
                    if cap is not None and len(body) > cap:  # a chunked upload declares no length
                        await _error(scope, receive, send, 413, "request body exceeds the configured size limit")
                        return None
                    if not message.get("more_body", False):
                        return bytes(body)
        except TimeoutError:
            await _error(scope, receive, send, 408, f"the request body did not arrive within "
                         f"{self.limits.body_timeout:g} s")
            return None

    # ---- the response

    async def _serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        idle = self.limits.idle_timeout
        if idle is None:
            await self.app(scope, receive, send)
            return
        # The clock starts at a stream's first chunk of body, not at its headers: engines send headers at once and
        # the first token after queueing and prefill, which can take as long as a prompt needs and is not a stall.
        # It is stamped once a chunk has been sent, and not judged while a send is under way, so a client that reads
        # slowly does not count against the engine.
        state: Dict[str, Any] = {"stream": False, "armed": False, "done": False, "sending": False,
                                 "last": self.clock()}

        async def tracked(message: Message) -> None:
            if message["type"] == "http.response.start":
                kinds = [v for k, v in message.get("headers", []) if k.lower() == b"content-type"]
                state["stream"] = any(b"text/event-stream" in v for v in kinds)
            state["sending"] = True
            try:
                await send(message)
            finally:
                state["sending"] = False
            if message["type"] == "http.response.body":
                if not message.get("more_body", False):
                    state["done"] = True
                elif message.get("body"):
                    state["armed"] = True
            state["last"] = self.clock()

        timed_out = False
        error: Optional[Exception] = None
        async with anyio.create_task_group() as tg:
            async def run() -> None:
                nonlocal error
                try:
                    await self.app(scope, receive, tracked)
                except Exception as exc:  # re-raised below as itself, not wrapped in the task group's exception group
                    error = exc
                finally:
                    tg.cancel_scope.cancel()

            async def watch() -> None:
                nonlocal timed_out
                while True:
                    # Only between a stream's chunks: a whole reply, or the wait for a stream's first chunk, has no
                    # idle time to judge, since a long prefill or a long answer is not a stalled one.
                    judged = state["stream"] and state["armed"] and not state["done"] and not state["sending"]
                    wait = state["last"] + idle - self.clock() if judged else idle
                    if judged and wait <= 0:
                        timed_out = True
                        tg.cancel_scope.cancel()
                        return
                    await anyio.sleep(max(wait, 0.01))

            tg.start_soon(run)
            tg.start_soon(watch)
        if error is not None:
            raise error
        if timed_out and not state["done"]:
            logger.warning("closed a stream that sent nothing for %g s (--idle-timeout)", idle)
            event = {"error": {"message": f"the stream sent nothing for {idle:g} s and was closed",
                               "type": "timeout", "code": "idle_timeout"}}
            await send({"type": "http.response.body", "body": b"data: " + json.dumps(event).encode() + b"\n\n",
                        "more_body": False})

    async def _minimal_health(self, scope: Scope, receive: Receive, send: Send) -> None:
        """/health with an API key configured: whether the backend is alive and ready, and nothing else (no pid,
        upstream address or error text, which the full body carries for a local operator)."""
        status, chunks = 200, []

        async def capture(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await self.app(scope, receive, capture)
        try:
            body = json.loads(b"".join(chunks))
        except ValueError:
            body = {}
        found = body.get("backend") if isinstance(body, dict) else None
        backend: Dict[str, Any] = found if isinstance(found, dict) else {}
        alive = bool(backend.get("alive", status == 200))
        minimal = {"status": "ok" if status == 200 else "degraded", "alive": alive,
                   "ready": bool(backend.get("ready", alive)) and status == 200}
        await JSONResponse(minimal, status_code=status)(scope, receive, send)


def _replay(body: bytes, receive: Receive) -> Receive:
    delivered = False

    async def replay() -> Message:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


async def _error(scope: Scope, receive: Receive, send: Send, status: int, message: str,
                 headers: Optional[Dict[str, str]] = None) -> None:
    response = JSONResponse({"error": {"message": message, "type": "serving_limit_error"}}, status_code=status,
                            headers=headers)
    await response(scope, receive, send)
