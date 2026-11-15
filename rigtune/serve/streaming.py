"""Read streamed replies event by event for the drift report, and close every upstream response however it ends.

Forwarding never waits on parsing: bytes go to the client as they arrive and the observer reads a copy. A network
chunk is not an event: one event can be split anywhere, several can share a chunk, and the SSE specification lets a
line end in CRLF, LF or a lone CR."""
from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

import anyio
import httpx
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from rigtune.calibrate.measure import generated

if TYPE_CHECKING:
    from rigtune.drift import TrafficWatch

# An event larger than this is not parsed. Usage, role and heartbeat events are a few hundred bytes; the cap bounds
# what one response can make the proxy hold.
MAX_EVENT_BYTES = 65536
_EOL = re.compile(rb"\r\n|\r|\n")
# Generated output in the first bytes of an event too large to parse: a non-empty text field, or a tool call.
_OUTPUT = re.compile(rb'"(?:content|text|reasoning|reasoning_content)"\s*:\s*"(?!")|"tool_calls"\s*:\s*\[\s*\{')


class StreamObserver:
    """One streamed response's SSE events: times the first event carrying generated output (measure.generated,
    the rule the calibration client uses), and keeps the last usage block, recorded once when the response ends.

    `time_first=False` records usage only, for a caller that has whole events but no clock of its own."""

    def __init__(self, watch: "TrafficWatch", started: float, clock: Callable[[], float] = time.perf_counter,
                 time_first: bool = True):
        self.watch = watch
        self.started = started
        self.clock = clock
        self._timed = not time_first
        self._buffer = b""  # the unfinished line, ending in a held-back CR when its chunk ended in one
        self._line_t = 0.0  # when the unfinished line began to arrive
        self._data: List[bytes] = []
        self._size = 0
        self._event_t: Optional[float] = None  # when the current event's first data line began to arrive
        self._discard = False  # the current event is over the cap: its data is dropped
        self._skip_line = False  # the unfinished line is over the cap: drop it up to its end
        self._usage: Optional[Dict[str, Any]] = None
        self._finished = False

    def feed(self, chunk: bytes) -> None:
        if not chunk or self._finished:
            return
        now = self.clock()
        if not self._buffer:
            self._line_t = now
        data = self._buffer + chunk
        # A CR at the end may be the first half of a CRLF split across chunks: hold it until the next byte.
        held = data.endswith(b"\r")
        lines = _EOL.split(data[:-1] if held else data)
        rest = lines.pop()
        for line in lines:
            self._line(line, self._line_t)
            self._line_t = now
        self._buffer = rest + (b"\r" if held else b"")
        if len(self._buffer) > MAX_EVENT_BYTES:
            if not self._skip_line and self._buffer.startswith(b"data"):
                self._too_large(self._buffer, self._line_t)
            self._buffer = b"\r" if held else b""
            self._skip_line = True

    def finish(self) -> None:
        """The response ended: a last line or event without its terminator still counts. Records usage once."""
        if self._finished:
            return
        if self._buffer.rstrip(b"\r"):
            self._line(self._buffer.rstrip(b"\r"), self._line_t)
        self._buffer = b""
        self._dispatch()
        self._finished = True
        if self._usage is not None:
            self.watch.record_response(json.dumps({"usage": self._usage}).encode(), "application/json")

    def _line(self, line: bytes, began: float) -> None:
        if self._skip_line:  # the end of a line already dropped for its size
            self._skip_line = False
            return
        if not line:
            self._dispatch()
            return
        name, _, value = line.partition(b":")
        if name != b"data":  # comments (":"), event, id and retry fields carry nothing measured
            return
        if value.startswith(b" "):
            value = value[1:]
        if self._event_t is None:
            self._event_t = began
        if self._discard:
            return
        self._size += len(value) + 1
        if self._size > MAX_EVENT_BYTES:
            self._too_large(b"\n".join([*self._data, value]), self._event_t)
            return
        self._data.append(value)

    def _too_large(self, head: bytes, began: float) -> None:
        """Drop an event over the cap, but keep its time if it is the first generated output: an answer's first
        event can be that large (a long first delta, or logprobs), and dropping it would lose the TTFT."""
        self._data.clear()
        self._discard = True
        if self._event_t is None:
            self._event_t = began
        if not self._timed and _OUTPUT.search(head):
            self._time(self._event_t)

    def _dispatch(self) -> None:
        data, discard, began = self._data, self._discard, self._event_t
        self._data, self._size, self._discard, self._event_t = [], 0, False, None
        if discard or not data:
            return
        payload = b"\n".join(data)
        if payload.strip() == b"[DONE]":
            return
        try:
            body = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(body, dict):
            return
        choices = body.get("choices")
        if not self._timed and isinstance(choices, list) and any(generated(c) for c in choices):
            self._time(began if began is not None else self.clock())
        # Some clients ask for cumulative usage in every event: keep the last, recorded once per response.
        if isinstance(body.get("usage"), dict):
            self._usage = body["usage"]

    def _time(self, at: float) -> None:
        self._timed = True
        self.watch.record_ttft((at - self.started) * 1000)


class ClosingStreamingResponse(StreamingResponse):
    """A streamed response that runs its cleanup however it ends, even when sending the headers is cancelled
    before the body iterator (whose own `finally` would close the upstream) has started."""

    def __init__(self, content: Any, close: Callable[[], Awaitable[None]], **kwargs: Any):
        super().__init__(content, **kwargs)
        self.close_upstream = close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.close_upstream()


class ManagedResponse:
    """An upstream response that is closed exactly once, and whose slot is released, when it is read, when its
    stream ends or fails, or when ASGI cancels the request."""

    def __init__(self, response: httpx.Response, on_close: Optional[Callable[[], None]] = None):
        self.response = response
        self.on_close = on_close
        self.closed = False

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            with anyio.CancelScope(shield=True):  # cleanup must survive the cancellation that triggered it
                await self.response.aclose()
        finally:
            if self.on_close is not None:
                self.on_close()

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        try:
            # Decoded: the proxies drop Content-Encoding from what they forward.
            async for chunk in self.response.aiter_bytes():
                yield chunk
        finally:
            await self.close()

    async def read(self) -> bytes:
        try:
            return await self.response.aread()
        finally:
            await self.close()
