"""A text-only pick (vLLM --language-model-only) is served with image, audio and video chat input refused with
HTTP 400 by rigtune's proxy and its replicas load balancer; every other request, and every other profile, passes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from rigtune.layout import create_lb_app
from rigtune.serve.proxy import create_app, media_refusal
from tests.test_cache_and_pipeline import _profile
from tests.test_measure_and_proxy import _Server


def _engine() -> FastAPI:
    app = FastAPI()
    app.state.seen = []

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> dict:
        app.state.seen.append(await request.json())
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    return app


def _chat(*parts) -> dict:
    return {"model": "m", "messages": [{"role": "system", "content": "be brief"},
                                       {"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                                                    *parts]}]}


IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
AUDIO = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
VIDEO = {"type": "video_url", "video_url": {"url": "https://example.invalid/v.mp4"}}


SERVERS = [lambda url, p: create_app(url, profile=p), lambda url, p: create_lb_app([url], profile=p)]


@pytest.fixture(scope="module")
def engine():
    app = _engine()
    with _Server(app) as srv:
        yield srv, app


@pytest.mark.parametrize("make", SERVERS, ids=["proxy", "replicas"])
def test_text_only_profile_refuses_media_and_passes_text(engine, make, hw_a100, spec, prepared_vllm):
    srv, app = engine
    p = _profile(hw_a100, spec, prepared_vllm)
    p.config = p.config.model_copy(update={"text_only": True})
    with TestClient(make(srv.url, p)) as client:
        for part in (IMAGE, AUDIO, VIDEO):
            before = len(app.state.seen)
            r = client.post("/v1/chat/completions", json=_chat(part))
            assert r.status_code == 400 and len(app.state.seen) == before  # never reached the engine
            err = r.json()["error"]
            assert err["code"] == "text_only" and err["type"] == "invalid_request_error"
            assert "served text-only" in err["message"] and repr(part["type"]) in err["message"]
        assert client.post("/v1/chat/completions", json=_chat()).status_code == 200  # text parts only
        assert client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}) \
            .status_code == 200
    assert media_refusal(p, "POST", "chat/completions", b"not json") is None  # the engine answers that
    assert media_refusal(p, "POST", "completions", b'{"prompt": "hi"}') is None


@pytest.mark.parametrize("make", SERVERS, ids=["proxy", "replicas"])
def test_other_profiles_pass_media_through(engine, make, hw_a100, spec, prepared_vllm):
    srv, app = engine
    for p in (_profile(hw_a100, spec, prepared_vllm), None):
        with TestClient(make(srv.url, p)) as client:
            before = len(app.state.seen)
            assert client.post("/v1/chat/completions", json=_chat(IMAGE)).status_code == 200
            assert app.state.seen[before]["messages"][1]["content"][1] == IMAGE
