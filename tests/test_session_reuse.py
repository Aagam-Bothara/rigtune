"""What prefix reuse a level of session replay achieved, from the engine, beside what its structure allowed."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
import rigtune.calibrate.measure as M
from rigtune.backends import get_backend
from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import get_workload, workload_from_file
from rigtune.models import Config, TrialMetrics
from tests.test_session_replay import _file


def _turn(sid, turn, prompt, cached=None, tokens=10, ttft=0.1, reusable=0, chars=100, seq=0, distance=None, live=2):
    return M.RequestOutcome(ok=True, ttft_s=ttft, duration_s=0.5, tokens=tokens, token_source="usage",
                            prompt_tokens=prompt, cached_tokens=cached, session=sid, turn=turn, seq=seq,
                            distance=distance, live=live, request_chars=chars, reusable_chars=reusable)


def test_achieved_and_expected_reuse_per_level():
    outcomes = [_turn("a", 1, 100, 0, reusable=10), _turn("a", 2, 200, 150, ttft=0.04, reusable=80, distance=1),
                _turn("b", 1, 100, None, reusable=10), _turn("b", 2, 200, 0, ttft=0.2, reusable=80, distance=1),
                M.RequestOutcome(ok=False, skipped="cut", session="b", turn=3),
                M.RequestOutcome(ok=False, skipped="truncated", session="c", turn=2)]
    m = M._metrics_from(outcomes, 2.0, 2)
    assert m.requests == 4 and m.sessions == 3 and m.sessions_cut == 1 and m.sessions_truncated == 1
    # b's first reply carried no count while others in the level did: SGLang's way of saying zero.
    assert m.prompt_tokens_total == 600 and m.cached_tokens_total == 150 and m.reuse_source == "usage"
    assert m.prefix_reuse == round(150 / 600, 4)
    assert m.reuse_expected == round((10 + 160 + 10 + 160) / 600, 4)  # each step's tokens x its reusable share
    assert m.mean_in_flight == 1.0 and m.effective_pool == 2.0 and m.reuse_distance == 1
    assert m.ttft_by_turn_ms == {"1": 100.0, "2": 120.0}


def test_reuse_is_split_into_first_and_later_steps_with_gaps_and_working_set():
    outcomes = [_turn("a", 1, 100, 20, reusable=20), _turn("a", 2, 200, 180, reusable=90, distance=3),
                _turn("b", 1, 100, 20, reusable=20), _turn("b", 2, 300, 150, reusable=80, distance=5)]
    outcomes[1].idle_gap, outcomes[3].idle_gap = 1, 3
    outcomes[3].working_set = 700
    m = M._metrics_from(outcomes, 2.0, 2)
    assert m.prefix_reuse_first == 0.2 and m.prefix_reuse_later == round(330 / 500, 4)
    assert m.reuse_expected_first == 0.2 and m.reuse_expected_later == round((180 + 240) / 500, 4)
    assert m.reuse_distance == 4 and m.idle_gap == 2 and m.working_set_tokens == 700


def test_a_flat_level_with_report_reuse_records_what_was_cached():
    flat = [M.RequestOutcome(ok=True, ttft_s=0.1, duration_s=0.5, tokens=8, prompt_tokens=100, cached_tokens=64),
            M.RequestOutcome(ok=True, ttft_s=0.1, duration_s=0.5, tokens=8, prompt_tokens=100)]  # SGLang: zero
    m = M._metrics_from(flat, 1.0, 2)
    assert (m.prompt_tokens_total, m.cached_tokens_total, m.prefix_reuse, m.reuse_source) == (200, 64, 0.32, "usage")
    assert m.sessions == 0 and m.reuse_expected is None


def test_report_reuse_is_opt_in_for_flat_workloads_and_keys_the_cache(hw_a100, spec):
    from rigtune.pipeline import SearchOptions, select

    assert "reuse" not in SearchOptions().key() and SearchOptions(report_reuse=True).key()["reuse"] == "on"
    _, off = select(hw_a100, spec, workload=get_workload("chat"))
    _, on = select(hw_a100, spec, workload=get_workload("chat"), report_reuse=True)
    assert not any(b.report_reuse for b in off.values()) and all(b.report_reuse for b in on.values())


def test_a_level_with_no_count_at_all_is_unknown_not_zero():
    m = M._metrics_from([_turn("a", 1, 100), _turn("a", 2, 200)], 1.0, 1)
    assert m.prefix_reuse is None and m.cached_tokens_total == 0 and m.reuse_source == "none"


def test_flat_requests_record_no_reuse():
    m = M._metrics_from([M.RequestOutcome(ok=True, ttft_s=0.1, duration_s=0.5, tokens=8, prompt_tokens=50)], 1.0, 1)
    assert m.sessions == 0 and m.prefix_reuse is None and m.reuse_source == "none" and m.ttft_by_turn_ms == {}
    assert m.mean_in_flight is None


def test_cached_tokens_are_read_from_each_reply():
    import asyncio

    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    from tests.test_measure_and_proxy import _Server

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        async def gen():
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            yield ('data: {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 1, '
                   '"prompt_tokens_details": {"cached_tokens": 32}}}\n\n')
        return StreamingResponse(gen(), media_type="text/event-stream")

    from tests.test_session_replay import _level

    with _Server(app) as s:
        out = asyncio.run(M._drive(s.url, LlmtraceHooks(model_name="m"), _level(n=1, c=1, budget_per_slot=50), 1,
                                   30.0, TokenCounter()))
    sent = [o for o in out if o.skipped is None]
    assert [o.cached_tokens for o in sent] == [32, 32, 32] and M._metrics_from(out, 1.0, 1).prefix_reuse == 0.8


def test_levels_fall_back_to_the_engines_counters(tmp_path, monkeypatch):
    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None):
        return [_turn(s.id, t, 100 * t) for s in workload.session_list[:2] for t in (1, 2)]

    def counters(queries, hits):
        return {(M.PREFIX_QUERIES, 'engine="0"'): queries, (M.PREFIX_HITS, 'engine="0"'): hits}

    reads = iter([counters(1000.0, 100.0), counters(1800.0, 500.0), counters(1800.0, 500.0),
                  counters(2000.0, 500.0)])
    monkeypatch.setattr(M, "_drive", fake_drive)
    monkeypatch.setattr(M, "_prefix_counters", lambda base_url, hooks: next(reads))
    wl = replace(workload_from_file(_file(tmp_path), get_workload("chat")), concurrencies=(1, 2))
    m = M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter())
    by = m.by_concurrency
    assert by["1"].reuse_source == "metrics" and by["1"].prefix_reuse == 0.5  # 400 hits of 800 queried
    assert by["2"].prefix_reuse == 0.0 and by["2"].ttft_by_turn_ms == {"1": 100.0, "2": 100.0}
    best = max(by.values(), key=lambda x: x.tok_s)
    assert m.prefix_reuse == best.prefix_reuse and m.sessions == best.sessions > 0


def test_prometheus_counters_are_summed_over_engines(monkeypatch):
    text = ("# HELP vllm:prefix_cache_queries_total x\n"
            'vllm:prefix_cache_queries_total{engine="0",model_name="m"} 120.0\n'
            'vllm:prefix_cache_queries_total{engine="1",model_name="m"} 80.0\n'
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 50.0\n'
            'vllm:external_prefix_cache_hits_total{engine="0",model_name="m"} 7.0\n')

    class _R:
        def __init__(self, t):
            self.text = t

    monkeypatch.setattr(M.httpx, "get", lambda url, timeout: _R(text))
    got = M._prefix_counters("http://x", LlmtraceHooks(metrics_path="/metrics"))
    assert got[("vllm:external_prefix_cache_hits", 'engine="0",model_name="m"')] == 7.0  # every counter, by labels
    assert M._counter_sum(got, M.PREFIX_QUERIES) == 200.0 and M._counter_sum(got, M.PREFIX_HITS) == 50.0
    assert M._counter_sum(got, "vllm:no_such_counter") is None
    assert M._prefix_counters("http://x", LlmtraceHooks()) is None


def test_report_flags_only_for_session_workloads_and_only_where_the_engine_has_them(prepared_vllm, monkeypatch):
    cfg, sg_cfg = Config(backend="vllm", quant="bf16"), Config(backend="sglang", quant="bf16")
    monkeypatch.setattr(vl, "_prompt_tokens_details_flag", lambda: True)
    monkeypatch.setattr(sg, "_cache_report_flag", lambda: True)
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    assert "--enable-prompt-tokens-details" not in get_backend("vllm").launch_spec(cfg, prepared_vllm, 1).args
    assert "--enable-cache-report" not in get_backend("sglang").launch_spec(sg_cfg, prepared_vllm, 1).args
    vllm, sglang = get_backend("vllm", report_reuse=True), get_backend("sglang", report_reuse=True)
    assert vllm.launch_spec(cfg, prepared_vllm, 1).args[-1] == "--enable-prompt-tokens-details"
    assert sglang.launch_spec(sg_cfg, prepared_vllm, 1).args[-1] == "--enable-cache-report"
    monkeypatch.setattr(vl, "_prompt_tokens_details_flag", lambda: False)  # an engine without the flag
    monkeypatch.setattr(sg, "_cache_report_flag", lambda: False)
    assert "--enable-prompt-tokens-details" not in vllm.launch_spec(cfg, prepared_vllm, 1).args
    assert "--enable-cache-report" not in sglang.launch_spec(sg_cfg, prepared_vllm, 1).args


def test_select_builds_reporting_backends_only_for_a_session_workload(hw_a100, spec, tmp_path):
    from rigtune.pipeline import select

    _, flat = select(hw_a100, spec, workload=get_workload("chat"))
    _, sessions = select(hw_a100, spec, workload=workload_from_file(_file(tmp_path)))
    assert not any(b.report_reuse for b in flat.values()) and all(b.report_reuse for b in sessions.values())


def test_the_flag_is_read_from_the_vllm_that_will_run(tmp_path, monkeypatch):
    """A `vllm` on PATH from another environment: its own source decides, not the one importable here."""
    env = tmp_path / "other-env"
    pkg = env / "lib" / "python3.12" / "site-packages" / "vllm" / "entrypoints" / "launchers"
    pkg.mkdir(parents=True)
    (pkg / "cli_args.py").write_text("enable_prompt_tokens_details: bool = False\n", encoding="utf-8")
    (env / "bin").mkdir()
    cli = env / "bin" / "vllm"
    cli.write_text("", encoding="utf-8")
    monkeypatch.setattr(vl, "_serve_cli", lambda: str(cli))
    for fn in (vl._vllm_package, vl._prompt_tokens_details_flag):
        fn.cache_clear()
    try:
        assert vl._vllm_package() == Path(env / "lib" / "python3.12" / "site-packages" / "vllm").resolve()
        assert vl._prompt_tokens_details_flag()
        (pkg / "cli_args.py").write_text("nothing here\n", encoding="utf-8")
        vl._prompt_tokens_details_flag.cache_clear()
        assert not vl._prompt_tokens_details_flag()
        (env / "lib").rename(env / "elsewhere")  # an environment whose source cannot be found: no flag
        vl._vllm_package.cache_clear()
        vl._prompt_tokens_details_flag.cache_clear()
        assert vl._vllm_package() is None and not vl._prompt_tokens_details_flag()
    finally:
        for fn in (vl._vllm_package, vl._prompt_tokens_details_flag):
            fn.cache_clear()


def test_old_trial_metrics_still_load():
    old = TrialMetrics.model_validate({"tok_s": 10.0, "requests": 4, "output_tokens": 40})
    assert old.reuse_source == "none" and old.prefix_reuse is None and old.sessions == 0
