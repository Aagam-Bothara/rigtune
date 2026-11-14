"""A fake OpenAI-compatible engine with a fixed time to first token and a fixed pace, for checking the client.

    python -I rigtune/bench/fake_engine.py --port 8123 [--ttft-ms 20] [--itl-ms 10] [--tokens 128]

(It imports nothing outside the standard library, so it is run by path: `python -m rigtune.bench.fake_engine`
works too, but imports rigtune first, which takes seconds.)

Every completion streams `tokens` chunks: the first `ttft_ms` after the request has been read, then one every
`itl_ms`, then a usage chunk and a comment line with the engine's own timings. It is built on asyncio streams
alone, with no web framework, so that it is cheaper per request than the client it checks, and it runs as its
own process so that the two do not share an event loop. GET /stats returns each request's timings since the
last POST /reset, measured on the engine's own clock: `start_s` when the request had been read, `ttft_s` from
then to writing its first chunk, `duration_s` to writing its last, and `tokens`. `--itl-ms 0` sends as fast as the socket takes it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

STATUS = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed"}


class FakeEngine:
    def __init__(self, ttft_ms: float = 20.0, itl_ms: float = 10.0, tokens: int = 128, model: str = "fake"):
        self.ttft_s, self.itl_s, self.tokens, self.model = ttft_ms / 1000, itl_ms / 1000, tokens, model
        self.stats: List[Dict[str, Any]] = []

    # ---- HTTP/1.1, keep-alive

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                request = await self._read(reader)
                if request is None:
                    break
                await self._route(writer, *request)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    @staticmethod
    async def _read(reader: asyncio.StreamReader) -> Optional[Tuple[str, str, bytes, float]]:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError:
            return None
        lines = head.decode("latin-1").split("\r\n")
        method, path = (lines[0].split(" ") + ["", ""])[:2]
        length = 0
        for line in lines[1:]:
            name, _, value = line.partition(":")
            if name.strip().lower() == "content-length":
                length = int(value.strip() or 0)
        body = await reader.readexactly(length) if length else b""
        return method, path, body, time.perf_counter()

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        writer.write(f"HTTP/1.1 {status} {STATUS[status]}\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        await writer.drain()

    async def _route(self, writer: asyncio.StreamWriter, method: str, path: str, body: bytes, t0: float) -> None:
        if method == "GET" and path == "/health":
            await self._send(writer, 200, {"status": "ok"})
        elif method == "GET" and path == "/v1/models":
            await self._send(writer, 200, {"object": "list", "data": [{"id": self.model, "object": "model"}]})
        elif method == "GET" and path == "/stats":
            await self._send(writer, 200, self.stats)
        elif method == "POST" and path == "/reset":
            self.stats = []
            await self._send(writer, 200, {"reset": True})
        elif method == "POST" and path in ("/v1/completions", "/v1/chat/completions"):
            try:
                req = json.loads(body or b"{}")
            except json.JSONDecodeError:
                await self._send(writer, 400, {"error": "body is not JSON"})
                return
            await self._complete(writer, req, chat=path.endswith("chat/completions"), t0=t0)
        elif path in ("/health", "/v1/models", "/stats", "/reset", "/v1/completions", "/v1/chat/completions"):
            await self._send(writer, 405, {"error": f"{method} not allowed on {path}"})
        else:
            await self._send(writer, 404, {"error": f"no {path}"})

    # ---- completions

    def _chunk(self, chat: bool) -> Dict[str, Any]:
        choice = {"index": 0, "delta": {"content": "x "}} if chat else {"index": 0, "text": "x "}
        return {"object": "chat.completion.chunk" if chat else "text_completion", "model": self.model,
                "choices": [choice]}

    async def _complete(self, writer: asyncio.StreamWriter, req: Dict[str, Any], chat: bool, t0: float) -> None:
        n = self.tokens
        if not req.get("stream"):
            await asyncio.sleep(self.ttft_s + self.itl_s * (n - 1))
            text = {"message": {"role": "assistant", "content": "x " * n}} if chat else {"text": "x " * n}
            await self._send(writer, 200, {"model": self.model, "choices": [dict(index=0, **text)],
                                           "usage": {"prompt_tokens": 1, "completion_tokens": n}})
            done = time.perf_counter() - t0
            self.stats.append({"start_s": t0, "ttft_s": done, "duration_s": done, "tokens": n})
            return
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")

        async def event(data: str) -> None:
            raw = data.encode()
            writer.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
            await writer.drain()

        line = "data: " + json.dumps(self._chunk(chat)) + "\n\n"
        await asyncio.sleep(max(0.0, t0 + self.ttft_s - time.perf_counter()))
        await event(line)
        first = time.perf_counter()
        for i in range(1, n):
            if self.itl_s:
                # Paced from the first chunk, not from the previous write, so a slow write does not slow the pace.
                await asyncio.sleep(max(0.0, first + i * self.itl_s - time.perf_counter()))
            await event(line)
        last = time.perf_counter()
        usage = {"object": "text_completion", "model": self.model, "choices": [],
                 "usage": {"prompt_tokens": 1, "completion_tokens": n}}
        record = {"start_s": t0, "ttft_s": first - t0, "duration_s": last - t0, "tokens": n}
        await event("data: " + json.dumps(usage) + "\n\n" + ": fake-engine " + json.dumps(record) + "\n\n"
                    + "data: [DONE]\n\n")
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        self.stats.append(record)


async def serve(port: int, engine: FakeEngine, host: str = "127.0.0.1") -> None:
    """Listen, then print "ready <port>" (the port it got, for --port 0) as the only line it ever prints."""
    server = await asyncio.start_server(engine.handle, host, port, backlog=1024)
    print(f"ready {server.sockets[0].getsockname()[1]}", flush=True)
    async with server:
        await server.serve_forever()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, required=True, help="0 picks a free one")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ttft-ms", type=float, default=20.0)
    ap.add_argument("--itl-ms", type=float, default=10.0)
    ap.add_argument("--tokens", type=int, default=128)
    args = ap.parse_args(argv)
    try:
        asyncio.run(serve(args.port, FakeEngine(args.ttft_ms, args.itl_ms, max(1, args.tokens)), args.host))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
