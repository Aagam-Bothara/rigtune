"""Thin OpenAI-compatible proxy in front of the supervised backend.

Exposes /v1/chat/completions, /v1/completions, /v1/models (and any other /v1/* the backend
serves), plus /rigtune/profile and /health. Streaming responses are passed through byte-for-byte.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from rigtune.models import Profile

logger = logging.getLogger(__name__)

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "content-length", "content-encoding", "host",
}


def _filtered(headers: httpx.Headers) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def _ask_for_usage(raw: bytes) -> bytes:
    """Add `stream_options: {"include_usage": true}` to a streaming request, leaving everything else alone.

    A streamed reply carries no token counts unless it was asked for, and most OpenAI clients stream, so without
    this the drift report would only ever see the non-streaming minority. The extra final chunk carries usage and
    an empty `choices`, which the OpenAI clients accept; a request that already set stream_options is untouched.
    """
    if not raw:
        return raw
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return raw
    if not isinstance(body, dict) or not body.get("stream"):
        return raw
    options = body.get("stream_options")
    if isinstance(options, dict) and "include_usage" in options:
        return raw
    body["stream_options"] = {**(options if isinstance(options, dict) else {}), "include_usage": True}
    try:
        return json.dumps(body).encode()
    except (TypeError, ValueError):
        return raw


# Chat content parts that carry an image, audio or video (OpenAI's, and vLLM's own extensions).
MEDIA_PARTS = frozenset({"image_url", "input_image", "image", "input_audio", "audio_url", "video_url", "video"})


def media_refusal(profile: Optional[Profile], method: str, path: str, raw: bytes) -> Optional[JSONResponse]:
    """HTTP 400 for a chat request with image, audio or video content when the profile is served text-only
    (Config.text_only: vLLM --language-model-only), rather than let the engine fail it obscurely; else None."""
    if profile is None or not profile.config.text_only or method != "POST" or path.strip("/") != "chat/completions":
        return None
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    messages = body.get("messages") if isinstance(body, dict) else None
    for m in messages if isinstance(messages, list) else []:
        content = m.get("content") if isinstance(m, dict) else None
        for part in content if isinstance(content, list) else []:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in MEDIA_PARTS:
                msg = (f"{profile.model_id} is served text-only (vLLM --language-model-only, since its multimodal "
                       f"start-up failed): it accepts no image, audio or video input, and this request has a "
                       f"{kind!r} part")
                return JSONResponse({"error": {"message": msg, "type": "invalid_request_error",
                                               "code": "text_only"}}, status_code=400)
    return None


def create_app(
    upstream_base_url: str,
    profile: Optional[Profile] = None,
    status_fn=None,
    request_timeout: Optional[float] = None,
    watch=None,
) -> FastAPI:
    state: Dict[str, Any] = {"client": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["client"] = httpx.AsyncClient(base_url=upstream_base_url, timeout=request_timeout)
        try:
            yield
        finally:
            await state["client"].aclose()

    app = FastAPI(title="rigtune", version=profile.rigtune_version if profile else "dev", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        body: Dict[str, Any] = {"status": "ok", "upstream": upstream_base_url}
        if status_fn is not None:
            body["backend"] = status_fn()
            if not body["backend"].get("alive", True):
                body["status"] = "degraded"
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    @app.get("/rigtune/profile")
    async def get_profile() -> JSONResponse:
        if profile is None:
            return JSONResponse({"error": "no profile (running with defaults)"}, status_code=404)
        data = profile.model_dump(mode="json")
        # Keep the wire payload small: the full hardware descriptor and trial table are large.
        data["calibration_table"] = [
            {
                "stage": t["stage"],
                "config": t["config"],
                "ok": t["error"] is None and t["launched"],
                "tok_s": t["metrics"]["tok_s"],
                "ttft_ms": t["metrics"]["ttft_ms"],
                "tpot_ms": t["metrics"]["tpot_ms"],
                "peak_mem_mb": t["metrics"]["peak_mem_mb"],
                "power_w": t["metrics"]["power_w"],
                "joules_per_token": t["metrics"]["joules_per_token"],
                "error": t["error"],
            }
            for t in data.get("calibration_table", [])
        ]
        if status_fn is not None:
            data["runtime"] = status_fn()
        return JSONResponse(data)

    @app.get("/rigtune/drift")
    async def drift() -> JSONResponse:
        """Whether live traffic still looks like the workload this profile was calibrated on."""
        if watch is None:
            return JSONResponse({"error": "no traffic watch (serving without a profile)"}, status_code=404)
        return JSONResponse(watch.report())

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "OPTIONS"])
    async def passthrough(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = state["client"]
        body = await request.body()
        refused = media_refusal(profile, request.method, path, body)
        if refused is not None:
            return refused
        if watch is not None:
            body = _ask_for_usage(body)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        upstream_req = client.build_request(
            request.method,
            f"/v1/{path}",
            content=body,
            headers=headers,
            params=request.query_params,
        )
        started = time.perf_counter()
        if watch is not None:
            watch.began()
        try:
            upstream = await client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("upstream error: %s", exc)
            if watch is not None:
                watch.ended()
            return JSONResponse({"error": {"message": f"backend unavailable: {exc}", "type": "upstream_error"}},
                                status_code=502)
        media = upstream.headers.get("content-type", "")
        if "text/event-stream" in media:
            # Streamed bytes are never parsed, so a streamed reply counts as a request and toward concurrency,
            # but carries no token counts; the request is only over once the client has drained it.
            async def _finish_stream() -> None:
                await upstream.aclose()
                if watch is not None:
                    watch.ended()

            async def _tee():
                """Pass the bytes through untouched, timing the first chunk and reading the usage one."""
                first = True
                async for chunk in upstream.aiter_raw():
                    if watch is not None:
                        if first:
                            watch.record_ttft((time.perf_counter() - started) * 1000)
                        watch.record_stream_chunk(chunk)
                    first = False
                    yield chunk

            return StreamingResponse(
                _tee(),
                status_code=upstream.status_code,
                headers=_filtered(upstream.headers),
                media_type="text/event-stream",
                background=BackgroundTask(_finish_stream),
            )
        content = await upstream.aread()
        await upstream.aclose()
        if watch is not None:
            watch.record_response(content, media)
            watch.ended()
            if watch.should_warn():
                logger.warning("traffic no longer looks like the calibrated workload: %s; "
                               "`rigtune recalibrate` would re-tune for it", "; ".join(watch.findings()))
        return Response(content=content, status_code=upstream.status_code, headers=_filtered(upstream.headers))

    return app
