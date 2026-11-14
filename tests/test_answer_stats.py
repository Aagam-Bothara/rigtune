"""Answer statistics beside every compared run: lengths, finish reasons and repetition, from what the load
generator already receives, so a row cannot look fast by answering longer or looping unseen."""

from __future__ import annotations

import asyncio
import importlib
import json

import httpx
import pytest

import rigtune.calibrate.measure as M
from rigtune.backends.base import LlmtraceHooks
from rigtune.bench.compare import ComparisonResult, ComparisonRow, combine_runs, run_record
from rigtune.bench.verdict import NA, SPEED_ONLY, answers_of, score, to_markdown
from rigtune.calibrate.tokens import TokenCounter
from rigtune.calibrate.workload import Workload
from rigtune.models import AnswerStats, Config, TrialMetrics

PROSE = ("The quick brown fox jumps over the lazy dog while a cautious cat watches from the warm windowsill "
         "and the morning light moves slowly across the kitchen floor")
LOOP = "I am sure that this is right. " * 8


def test_repetition_is_the_share_of_word_4grams_seen_earlier_in_the_answer():
    assert M.repeated_share(PROSE) == 0.0
    # 8 sentences of 7 words: 53 4-grams, of which only the first 7 are new
    assert M.repeated_share(LOOP) == pytest.approx(46 / 53)
    assert M.repeated_share("too short to judge") is None


def _out(tokens, text="", finish="stop", ok=True, **kw):
    return M.RequestOutcome(ok=ok, ttft_s=0.01, duration_s=0.1, tokens=tokens, token_source="usage",
                            text=text, finish_reason=finish, **kw)


def test_a_levels_answers_are_summarised_from_the_measured_requests():
    stats = M.answer_stats([_out(10, PROSE), _out(30, LOOP, finish="length"), _out(20, PROSE, finish=None),
                            _out(0, ok=False), _out(99, LOOP, primer=True), _out(99, LOOP, skipped="cut")])
    assert (stats.answers, stats.mean_tokens, stats.median_tokens) == (3, 20.0, 20.0)
    assert stats.finish_reasons == {"stop": 1, "length": 1, "none": 1}
    assert stats.judged_for_repetition == 3 and stats.looping_answers == 1
    assert stats.repeated_4gram_share == pytest.approx((0 + 46 / 53 + 0) / 3)
    no_text = M.answer_stats([_out(10)])
    assert no_text.repeated_4gram_share is None and no_text.judged_for_repetition == 0


def _sse(*chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_the_last_finish_reason_the_stream_sends_is_kept():
    body = _sse({"choices": [{"text": "Hello", "finish_reason": None}]},
                {"choices": [{"text": " there", "finish_reason": "length"}]},
                {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 3}})

    async def go(keep):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        async with httpx.AsyncClient(transport=transport) as client:
            return await M._one_request(client, "http://x/v1/completions", {}, 5.0, TokenCounter(),
                                        keep_text=keep)

    kept, dropped = asyncio.run(go(True)), asyncio.run(go(False))
    assert (kept.finish_reason, kept.text, kept.tokens) == ("length", "Hello there", 2)
    assert (dropped.finish_reason, dropped.text) == ("length", "")


def test_run_trial_records_answers_only_when_asked(monkeypatch):
    calls = []

    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None, **kw):
        calls.append(kw)
        return [_out(12, LOOP if kw.get("keep_text") else "")] * 4

    monkeypatch.setattr(M, "_drive", fake_drive)
    wl = Workload(n_prompts=4, prefill_tokens=16, decode_tokens=4, concurrencies=(1, 4))
    plain = M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter())
    assert calls == [{}, {}] and all(lv.answers is None for lv in plain.by_concurrency.values())
    assert "answers" not in json.loads(plain.model_dump_json())["by_concurrency"]["1"]  # written as before
    calls.clear()
    m = M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(), answers=True)
    assert calls == [{"keep_text": True}] * 2
    level = m.by_concurrency["4"].answers
    assert level.answers == 4 and level.looping_answers == 4 and level.finish_reasons == {"stop": 4}
    assert TrialMetrics.model_validate_json(m.model_dump_json()).by_concurrency["4"].answers == level


def test_a_close_calls_re_measure_is_left_out_of_the_levels_answers(monkeypatch):
    """Every configuration is sent the same prompts at a level; a close call's second batch is its own, so the
    level's answer statistics leave it out and stay comparable across rows."""
    batches = iter([[_out(10, PROSE)] * 4, [_out(99, LOOP, finish="length")] * 4])

    async def fake_drive(base_url, hooks, workload, concurrency, timeout, counter=None, **kw):
        return next(batches)

    monkeypatch.setattr(M, "_drive", fake_drive)
    wl = Workload(n_prompts=4, prefill_tokens=16, decode_tokens=4, concurrencies=(4,))
    m = M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(), answers=True,
                    close_call=lambda metrics: True)
    level = m.by_concurrency["4"]
    assert level.resampled and level.requests == 8
    assert (level.answers.answers, level.answers.mean_tokens, level.answers.looping_answers) == (4, 10.0, 0)


def test_several_client_processes_keep_the_answers_text_when_asked(monkeypatch):
    calls = []

    def fake_clients(base_url, hooks, workload, concurrency, timeout, clients, **kw):
        calls.append(kw)
        return [_out(12, LOOP if kw.get("keep_text") else "")] * 4, 1.0

    monkeypatch.setattr(M, "_drive_clients", fake_clients)
    wl = Workload(n_prompts=4, prefill_tokens=16, decode_tokens=4, concurrencies=(4,))
    M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(), clients=2)
    m = M.run_trial("http://x", LlmtraceHooks(), wl, warmup=False, counter=TokenCounter(), clients=2,
                    answers=True)
    assert calls == [{}, {"keep_text": True}] and m.by_concurrency["4"].answers.looping_answers == 4


def test_a_replicas_runner_records_answers_through_its_balancer(tmp_path):
    """Through the fake engine and rigtune's own balancer, driven by several client processes: the answers'
    text comes back from each process, so this pick is judged on its answers like the rest."""
    from rigtune.layout import LayoutTrialRunner
    from tests.conftest import make_hw
    from tests.test_supervisor import FakeBackend

    wl = Workload(n_prompts=4, prefill_tokens=8, decode_tokens=12, concurrencies=(2,))
    cfg = Config(backend="fake", quant="none", ctx=128, batch=1)
    rep = LayoutTrialRunner(FakeBackend(), None, make_hw("cpu"), wl, log_dir=tmp_path, startup_timeout=15,
                            answers=True).run_replicas(cfg, [0, 1], "layout")
    assert rep.ok, rep.error
    level = rep.metrics.by_concurrency["2"].answers
    assert level.answers == 4 and level.judged_for_repetition == 4  # "x x x ...": twelve words, one 4-gram


def test_a_disaggregated_runner_asks_for_answers_through_its_router(monkeypatch):
    from rigtune import disagg as D
    from tests.conftest import make_hw
    from tests.test_phases import FakePDBackend, _pd_spec

    asked = []

    class Pair:
        prefill_url, decode_url = "http://p", "http://d"

        def stop(self):
            pass

    class Router:
        url = "http://router"

        def __init__(self, *args, **kw):
            pass

        def start(self):
            return self

        def stop(self):
            pass

    def run_trial(url, hooks, workload, **kw):
        asked.append((url, kw.get("answers"), kw.get("clients")))
        return TrialMetrics(tok_s=10.0, requests=2, output_tokens=8, concurrency=1)

    monkeypatch.setattr(D, "launch_pair", lambda *a, **kw: (Pair(), None))
    monkeypatch.setattr(D, "ProcessServer", Router)
    monkeypatch.setattr(D, "run_trial", run_trial)
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=12, concurrencies=(1,))
    D.DisaggTrialRunner(FakePDBackend(), None, make_hw("cpu"), wl).run(_pd_spec(), "pd")
    res = D.DisaggTrialRunner(FakePDBackend(), None, make_hw("cpu"), wl, answers=True).run(_pd_spec(), "pd")
    assert res.ok, res.error
    assert asked == [("http://router", None, 2), ("http://router", True, 2)]


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize("layout", ["disaggregated", "replicas"])
def test_compare_asks_a_disaggregated_or_replicas_pick_for_answers_too(hw_a100, spec, monkeypatch, layout):
    import rigtune.disagg as D
    import rigtune.layout as L
    from rigtune.bench import compare
    from rigtune.models import DisaggSpec, TrialResult
    from rigtune.pipeline import calibrate, prepare_and_plan, select
    from tests.test_compare import CompareRunner
    from tests.test_objectives_and_search import FakeRunner

    built = []

    class Runner:
        def __init__(self, *args, **kw):
            built.append(kw)

        def _result(self, cfg):
            stats = AnswerStats(answers=4, mean_tokens=10.0, finish_reasons={"stop": 4})
            level = dict(tok_s=500.0, ttft_ms=50.0, tpot_ms=5.0, requests=4, output_tokens=40, concurrency=8)
            m = TrialMetrics(**level, by_concurrency={"8": TrialMetrics(**level, answers=stats)})
            return TrialResult(config=cfg, stage="compare", metrics=m)

        def run(self, spec, stage):
            return self._result(spec.decode)

        def run_replicas(self, cfg, gpus, stage):
            return self._result(cfg)

    monkeypatch.setattr(D, "DisaggTrialRunner", Runner)
    monkeypatch.setattr(L, "LayoutTrialRunner", Runner)
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    if layout == "replicas":
        profile = profile.model_copy(update={"replicas": 2})
    else:
        pair = DisaggSpec(prefill=profile.config, decode=profile.config)
        profile = profile.model_copy(update={"disagg": pair})
    for asked in (False, True):
        result = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(), answers=asked,
                         record_runs=True)
    assert [kw.get("answers") for kw in built] == [None, True]  # passed only when asked
    assert result.rigtune_row.run_records[0].level_answers["8"].answers == 4


@pytest.mark.usefixtures("no_network")
def test_compare_asks_its_runner_for_answers_only_when_told(hw_a100, spec, monkeypatch):
    from rigtune.bench import compare
    from rigtune.pipeline import calibrate, prepare_and_plan, select
    from tests.test_compare import CompareRunner
    from tests.test_objectives_and_search import FakeRunner

    cmp = importlib.import_module("rigtune.bench.compare")
    built = []
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kw: built.append(kw) or CompareRunner())
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    compare(hw_a100, spec, profile, planned.prepared, reg)
    compare(hw_a100, spec, profile, planned.prepared, reg, answers=True)
    assert "answers" not in built[0] and built[1]["answers"] is True


def _stats(mean_tokens, looping=0, cut=1, n=10, judged=None):
    judged = n if judged is None else judged
    return AnswerStats(answers=n, mean_tokens=mean_tokens, median_tokens=mean_tokens,
                       finish_reasons={"stop": n - cut, "length": cut},
                       repeated_4gram_share=(0.6 if looping else 0.05) if judged else None,
                       looping_answers=looping, judged_for_repetition=judged)


def _row(label, runtime, mean_tokens, looping=0, runs=(100.0, 101.0, 102.0), cut=1, level=8, n=10,
         judged=None):
    """A row whose every run answered at 4 users and at `level` with these statistics, scored at `level`."""
    each = []
    for v in runs:
        stats = _stats(mean_tokens, looping, cut, n, judged)
        levels = {str(c): TrialMetrics(tok_s=v * c / level, concurrency=c, answers=stats) for c in (4, level)}
        each.append(ComparisonRow(label=label, runtime=runtime, config=Config(backend=runtime, quant="bf16"),
                                  config_key=label, ok=True, scored_concurrency=level, scored_tok_s=v,
                                  meets_slo=True, metrics=TrialMetrics(tok_s=v, by_concurrency=levels)))
    row = combine_runs(each)
    row.run_records = [run_record(r, i, 1) for i, r in enumerate(each, 1)]
    return row


def _res(*rows):
    return ComparisonResult(rigtune_version="x", hardware_hash="h", gpu="A40", cpu="c", model_id="m",
                            objective="balanced", workload="chat", workload_spec={}, ttft_ceiling_ms=500,
                            rows=list(rows))


def test_rigtunes_answers_changed_on_the_workload_mean_no_row_counts():
    """393 tokens against 173 on the same prompts, 12 of 30 answers looping: the answers changed, so rigtune's
    speed is not the same work and no row counts, whatever the margin."""
    res = _res(_row("rigtune", "vllm", 393, looping=4, runs=(493.0, 482.3, 591.4)),
               _row("vllm-default", "vllm", 173), _row("sglang-default", "sglang", 180))
    assert answers_of(res.rows[0], 8)["answers"] == 30 and answers_of(res.rows[0], 8)["looping_answers"] == 12
    v = score(res)
    assert v.rigtune["answers"]["mean_tokens"] == 393
    want = ("at 8 users, on the same prompts as vllm-default: they average 393 tokens against 173 for "
            "vllm-default (2.27x); 12 of 30 look like loops (40%) against 0%")
    assert v.rigtune["workload_answers"] == f"changed: {want}"
    assert [(r.label, r.outcome, r.reason) for r in v.rows] == [
        ("vllm-default", NA, f"rigtune's answers changed on the workload {want}"),
        ("sglang-default", NA, f"rigtune's answers changed on the workload {want}")]
    assert v.overall["outcome"] == NA
    assert v.overall["text"] == f"rigtune's answers changed on the workload ({want}), so no row counts"
    assert v.rows[0].answers["finish_reasons"] == {"length": 3, "stop": 27}
    md = to_markdown(v)
    assert f"| rigtune | 30 | 393 | 27 | 3 | 60% | 12 of 30 | changed: {want} |" in md
    assert "| vllm-default | 30 | 173 | 27 | 3 | 5% | 0 of 30 | same |" in md


def test_rigtune_looping_840_of_900_answers_at_twice_the_length_is_not_applicable_everywhere():
    """The critic's probe K: rigtune at 2.13x the stock row's speed, 840 of its 900 workload answers looping
    and every one cut at max tokens. Speed alone would call it a win; the workload rule rules out every
    row."""
    def runs(label, tok, stats):
        each = [ComparisonRow(label=label, runtime="vllm", config=Config(backend="vllm", quant="bf16"),
                              config_key=label, ok=True, meets_slo=True, scored_tok_s=v, scored_concurrency=8)
                for v in tok]
        row = combine_runs(each)
        row.run_records = [run_record(x, i, 1) for i, x in enumerate(each, 1)]
        for rec in row.run_records:
            rec.answers = stats  # only the scored level's kept: still the level rigtune is scored at
        return row

    loop = AnswerStats(answers=300, mean_tokens=256, finish_reasons={"length": 300}, repeated_4gram_share=0.9,
                       looping_answers=280, judged_for_repetition=300)
    sane = AnswerStats(answers=300, mean_tokens=120, finish_reasons={"stop": 290, "length": 10},
                       repeated_4gram_share=0.02, looping_answers=0, judged_for_repetition=300)
    res = _res(runs("rigtune", (256.0, 257.0, 258.0), loop),
               runs("vllm-default", (120.0, 121.0, 122.0), sane))
    v = score(res)
    assert [r.outcome for r in v.rows] == [NA] and v.overall["outcome"] == NA
    assert v.rows[0].reason == (
        "rigtune's answers changed on the workload at 8 users, on the same prompts as vllm-default: they "
        "average 256 tokens against 120 for vllm-default (2.13x); 100% were cut at max tokens against 3%; "
        "840 of 900 look like loops (93%) against 0%")


@pytest.mark.parametrize("change, expected", [
    ({"mean_tokens": 125}, None),  # +25.0%: within objectives.OUTPUT_CHANGE
    ({"mean_tokens": 126}, "they average 126 tokens against 100 for vllm-default (1.26x)"),
    ({"cut": 2}, None),  # 20% against 10%: 10 points, not more
    ({"cut": 3}, "30% were cut at max tokens against 10%"),
    ({"looping": 1}, "3 of 30 look like loops (10%) against 0%"),  # 10 points over the stock row's 0%
])
def test_each_workload_rule_has_its_own_bar(change, expected):
    mean = change.pop("mean_tokens", 100)
    rig = _row("rigtune", "vllm", mean, runs=(130.0, 131.0, 132.0), **change)
    v = score(_res(rig, _row("vllm-default", "vllm", 100)))
    if expected is None:
        assert v.rigtune["workload_answers"] == "same" and v.rows[0].outcome == SPEED_ONLY
        assert v.rows[0].reason.endswith("; rigtune's answers were not graded")
    else:
        assert v.rigtune["workload_answers"].endswith(expected) and v.rows[0].outcome == NA


def test_another_rows_changed_answers_rule_out_that_row_only():
    res = _res(_row("rigtune", "vllm", 100, runs=(130.0, 131.0, 132.0)), _row("vllm-default", "vllm", 100),
               _row("vllm-expert-kv", "vllm", 40, runs=(200.0, 201.0, 202.0)))
    res.rows[2].kind = "expert"
    got = {r.label: r for r in score(res).rows}
    assert got["vllm-default"].outcome == SPEED_ONLY  # rigtune's answers match its stock row's, ungraded
    assert got["vllm-expert-kv"].outcome == NA  # faster than rigtune, but on shorter answers: not a loss
    assert got["vllm-expert-kv"].reason.startswith("answers changed on the workload at 8 users, on the same "
                                                   "prompts as vllm-default: they average 40 tokens")


def test_answers_are_compared_at_rigtunes_scored_level_and_unknown_without_a_stock_row():
    rig = _row("rigtune", "vllm", 100, runs=(130.0, 131.0, 132.0))
    stock = _row("vllm-default", "vllm", 100, level=4)  # scored at 4; it answered at 4 only
    v = score(_res(rig, stock))
    assert v.rigtune["workload_answers"] == ("unknown: vllm-default, whose answers its own are compared "
                                             "with, recorded no answers at 8 users")
    assert v.rows[0].outcome == SPEED_ONLY and "nor matched against its stock row's" in v.rows[0].reason
    alone = score(_res(_row("rigtune", "sglang", 100, runs=(130.0, 131.0, 132.0)),
                       _row("vllm-default", "vllm", 100)))
    assert alone.rigtune["workload_answers"] == ("unknown: sglang-default, whose answers its own are "
                                                 "compared with, is not in this comparison")


def test_a_verdict_says_when_no_answers_were_recorded():
    from tests.test_verdict import _result, _row as plain_row

    v = score(_result(plain_row("rigtune", [130.0, 131.0, 132.0]),
                      plain_row("vllm-default", [100.0, 101.0, 102.0])))
    assert any(n.startswith("no answer statistics were recorded") for n in v.notes)
    assert "Answers, over the runs that count" not in to_markdown(v)


@pytest.mark.parametrize("mine, stock", [
    ({"cut": 4}, {"cut": 3}),  # 40% against 30%: 0.4 - 0.3 is 0.10000000000000003, exactly the 10-point bar
    ({"looping": 13, "n": 20}, {"looping": 12, "n": 20}),  # 65% against 60%: exactly 5 points
    ({"mean_tokens": 75}, {}),  # 0.75x: exactly 25% shorter
    ({"mean_tokens": 125}, {}),  # 1.25x: exactly 25% longer
])
def test_answers_exactly_on_a_bar_are_unchanged(mine, stock):
    rig = _row("rigtune", "vllm", mine.pop("mean_tokens", 100), runs=(130.0, 131.0, 132.0), **mine)
    v = score(_res(rig, _row("vllm-default", "vllm", 100, **stock)))
    assert v.rigtune["workload_answers"] == "same"


def test_missing_answer_data_reads_unchecked_never_unchanged():
    """Session replay keeps no text, so neither row's repetition is known; a stock row averaging 0 tokens
    gives no length to compare. Either way the rule cannot say the answers are unchanged, so no win."""
    rig = _row("rigtune", "vllm", 100, runs=(130.0, 131.0, 132.0), judged=0)
    v = score(_res(rig, _row("vllm-default", "vllm", 100, judged=0)))
    assert v.rigtune["workload_answers"] == ("unknown: repetition was measured on neither rigtune nor "
                                             "vllm-default")
    assert v.rows[0].outcome == SPEED_ONLY
    v = score(_res(_row("rigtune", "vllm", 100, runs=(130.0, 131.0, 132.0)), _row("vllm-default", "vllm", 0)))
    assert v.rigtune["workload_answers"] == ("unknown: vllm-default's answers average 0 tokens, so a change "
                                             "in length cannot be measured")
    assert v.rows[0].outcome == SPEED_ONLY
    # A rule that does trip still rules the row out, whatever another could not measure.
    v = score(_res(_row("rigtune", "vllm", 100, runs=(130.0, 131.0, 132.0), cut=5, judged=0),
                   _row("vllm-default", "vllm", 100, judged=0)))
    assert v.rigtune["workload_answers"].startswith("changed: ") and v.rows[0].outcome == NA
