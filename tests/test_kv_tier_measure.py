"""A tier trial records where its cached tokens came from and time to first token on returning steps; every
other trial's requests and metrics are what they were."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import rigtune.backends.sglang as sg
import rigtune.calibrate.measure as M
from rigtune.backends import get_backend
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import get_workload, workload_from_file
from rigtune.models import TrialMetrics
from tests.test_measure_and_proxy import _Server
from tests.test_session_replay import _file, _level
from tests.test_session_reuse import _turn


def _tier_engine() -> FastAPI:
    """SGLang 0.5.19's stream when asked for details: content, a chunk with no choices carrying
    sglext.cached_tokens_details, then the usage chunk."""
    app = FastAPI()
    app.state.bodies = []

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        app.state.bodies.append(body)
        turns = sum(m["role"] == "user" for m in body["messages"])
        details = {"device": 0, "host": 0} if turns == 1 else {"device": 8, "host": 24, "storage": 0,
                                                                "storage_backend": "file"}

        async def gen():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': 'ok '}}]})}\n\n"
            if body.get("return_cached_tokens_details"):
                yield f"data: {json.dumps({'choices': [], 'sglext': {'cached_tokens_details': details}})}\n\n"
            usage = {"prompt_tokens": 40 * turns, "completion_tokens": 1,
                     "prompt_tokens_details": {"cached_tokens": details["device"] + details["host"]}}
            yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _replay(app, hooks):
    with _Server(app) as s:
        level = _level(n=1, c=1, budget_per_slot=50)
        return asyncio.run(M._drive(s.url, hooks, level, 1, 30.0, TokenCounter()))


def test_a_tier_trial_asks_sglang_for_its_split_and_reads_it():
    app = _tier_engine()
    hooks = LlmtraceHooks(model_name="m", tier_report=True, tier_details_field="return_cached_tokens_details")
    out = [o for o in _replay(app, hooks) if o.skipped is None]
    assert all(b["return_cached_tokens_details"] is True for b in app.state.bodies)
    assert [o.cached_by_source for o in out] == [{"device": 0, "host": 0}] + [{"device": 8, "host": 24,
                                                                               "storage": 0}] * 2
    assert [o.cached_tokens for o in out] == [0, 32, 32]  # usage still read as before


def test_other_trials_send_exactly_what_they_did():
    app = _tier_engine()
    field = "return_cached_tokens_details"
    _replay(app, LlmtraceHooks(model_name="m", tier_details_field=field))  # not a tier trial
    _replay(app, LlmtraceHooks(model_name="m", tier_report=True))  # a tier, but the engine has no such field
    assert app.state.bodies and not any("return_cached_tokens_details" in b for b in app.state.bodies)


def test_a_tier_level_splits_cached_tokens_and_times_returning_steps():
    def step(sid, turn, ttft, src=None, cached=None):
        o = _turn(sid, turn, 100 * turn, cached=cached, ttft=ttft)
        o.cached_by_source = src or {}
        return o

    none = {"device": 0, "host": 0}
    outcomes = [step("a", 1, 0.30, none), step("a", 2, 0.05, {"device": 90, "host": 0}, 90),
                step("a", 3, 0.10, {"device": 20, "host": 200}, 220), step("b", 1, 0.32, none),
                step("b", 2, 0.40, none, 0), step("b", 3, 0.12, {"device": 0, "host": 250}, 250)]
    m = M._metrics_from(outcomes, 2.0, 2, tier=True)
    assert m.cached_by_source == {"device": 110, "host": 450}
    assert m.ttft_returning_ms == 110.0 and m.ttft_returning_p95_ms == 400.0  # turns 2 and 3 only
    assert m.ttft_by_hit_ms == {"device": 50.0, "host": 110.0, "miss": 400.0}
    plain = M._metrics_from(outcomes, 2.0, 2)  # the same level, not a tier trial
    assert plain.cached_by_source == {} and plain.ttft_returning_ms is None and plain.ttft_by_hit_ms == {}
    assert plain.model_dump() == M._metrics_from(outcomes, 2.0, 2).model_dump()


def test_without_a_split_hits_and_misses_are_still_told_apart():
    outcomes = [_turn("a", 1, 100, 0, ttft=0.3), _turn("a", 2, 200, 150, ttft=0.05),
                _turn("b", 1, 100, None, ttft=0.3), _turn("b", 2, 200, None, ttft=0.25)]  # None: zero
    m = M._metrics_from(outcomes, 2.0, 2, tier=True)
    assert m.cached_by_source == {} and m.ttft_by_hit_ms == {"hit": 50.0, "miss": 250.0}
    unknown = M._metrics_from([_turn("a", 1, 100), _turn("a", 2, 200, ttft=0.2)], 1.0, 1, tier=True)
    assert unknown.ttft_returning_ms == 200.0 and unknown.ttft_by_hit_ms == {}  # no count at all: no split


def _by_source(local: float, external: float):
    labels = 'engine="0",model_name="m",source="{}"'
    return {("vllm:prompt_tokens_by_source", labels.format("local_compute")): 5000.0,
            ("vllm:prompt_tokens_by_source", labels.format("local_cache_hit")): local,
            ("vllm:prompt_tokens_by_source", labels.format("external_kv_transfer")): external,
            (M.PREFIX_QUERIES, 'engine="0"'): 100.0, (M.PREFIX_HITS, 'engine="0"'): 10.0}


def test_vllm_tier_trials_split_from_the_engines_counters(tmp_path, monkeypatch):
    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        return [_turn(s.id, t, 100 * t, cached=50) for s in workload.session_list[:2] for t in (1, 2)]

    def run(hooks):
        reads = iter([_by_source(1000.0, 0.0), _by_source(1300.0, 2000.0)])
        monkeypatch.setattr(M, "_prefix_counters", lambda base_url, h: next(reads))
        return M.run_trial("http://x", hooks, wl, warmup=False, counter=TokenCounter())

    monkeypatch.setattr(M, "_drive", fake_drive)
    wl = replace(workload_from_file(_file(tmp_path), get_workload("chat")), concurrencies=(2,))
    m = run(LlmtraceHooks(tier_report=True))
    assert m.by_concurrency["2"].cached_by_source == {"device": 300, "tier": 2000}
    assert m.cached_by_source == {"device": 300, "tier": 2000} and m.ttft_returning_ms == 100.0
    assert m.ttft_by_hit_ms == {"hit": 100.0}
    plain = run(LlmtraceHooks())  # not a tier trial: nothing new recorded
    assert plain.cached_by_source == {} and plain.ttft_returning_ms is None


def test_counters_never_overwrite_the_engines_own_split():
    m = TrialMetrics(cached_by_source={"device": 1, "host": 2})
    M._engine_sources(m, _by_source(0.0, 0.0), _by_source(10.0, 10.0))
    assert m.cached_by_source == {"device": 1, "host": 2}
    m = TrialMetrics()
    M._engine_sources(m, {}, {})  # an engine without the counter
    assert m.cached_by_source == {}


def test_old_trial_metrics_still_load():
    old = TrialMetrics.model_validate({"tok_s": 10.0, "requests": 4, "output_tokens": 40})
    assert old.cached_by_source == {} and old.ttft_returning_ms is None and old.ttft_by_hit_ms == {}


def test_the_details_field_is_offered_only_where_the_sglang_source_has_it(tmp_path, monkeypatch):
    pkg = tmp_path / "sglang"
    (pkg / "srt" / "entrypoints" / "openai").mkdir(parents=True)
    protocol = pkg / "srt" / "entrypoints" / "openai" / "protocol.py"
    protocol.write_text("    return_cached_tokens_details: bool = False\n", encoding="utf-8")
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    monkeypatch.setattr(sg, "_sglang_package_dir", lambda python: pkg)
    sg._tier_details_flag.cache_clear()
    try:
        assert get_backend("sglang").tier_details_field() == "return_cached_tokens_details"
        protocol.write_text("nothing\n", encoding="utf-8")
        sg._tier_details_flag.cache_clear()
        assert get_backend("sglang").tier_details_field() is None
    finally:
        sg._tier_details_flag.cache_clear()
    assert get_backend("vllm").tier_details_field() is None  # vLLM: counters only
