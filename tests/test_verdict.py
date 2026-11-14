"""`rigtune verdict`: a comparison scored win, tie or loss against every row."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rigtune.bench.compare import ComparisonResult, ComparisonRow, SkippedRow, combine_runs, run_record
from rigtune.bench.verdict import (BEHIND, BEHIND_CLEAR, LOSS, NA, PROVISIONAL, SPEED_ONLY, TIE, WIN,
                                   Thresholds, score, to_markdown, write)
from rigtune.models import Config

FIXTURES = Path(__file__).parent / "fixtures"
NOV14 = FIXTURES / "compare-2026-11-14-qwen2.5-7b-a40.json"


def test_the_14_november_qwen_verdict_is_reproduced_from_its_per_repeat_figures():
    """The fixture holds the A40 run's per-repeat tok/s as the session transcript printed them (the results
    file was lost). The pod's verdict: (a) stock vLLM +100.4%, rigtune 482-591 vs 239-250; (b) stock SGLang
    +104.6%, 240-247; (c) auto_tune's pick +109.5%, 235-242; every range apart, every criterion a pass. Its
    results predate per-run limit status, so the verdict counts each successful run of a row whose median run
    met the limits, and says so. Nobody checked the answers, so every speed criterion passes and it is still
    not a win: ahead on speed, answers unchecked."""
    v = score(ComparisonResult.model_validate_json(NOV14.read_text(encoding="utf-8")))
    assert v.rigtune["median_tok_s"] == 493.0 and v.rigtune["range_tok_s"] == (482.3, 591.4)
    got = {r.label: r for r in v.rows}
    assert list(got) == ["vllm-default", "sglang-default", "autotune-vllm"]
    expected = {"vllm-default": ("stock", 0.20, 246.0, (239.0, 249.7), 100.4, 100.4),
                "sglang-default": ("stock", 0.20, 241.0, (239.8, 247.3), 104.6, 101.1),
                "autotune-vllm": ("tuner", 0.10, 235.4, (235.3, 242.4), 109.4, 104.9)}
    for label, (kind, bar, med, rng, pct, worst) in expected.items():
        r = got[label]
        assert (r.kind, r.threshold, r.median_tok_s, r.range_tok_s) == (kind, bar, med, rng)
        assert r.counted_runs == 3 and r.outcome == SPEED_ONLY and r.overlap is False
        assert r.reason.endswith(", ranges apart; rigtune's answers were not checked")
        assert f"{r.gain * 100:+.1f}%" == f"+{pct}%"
        assert f"{r.worst_paired_gain * 100:+.1f}%" == f"+{worst}%"
    assert v.overall["outcome"] == SPEED_ONLY and v.overall["wins"] == 0
    assert v.overall["ahead_answers_unchecked"] == 3
    # The pod printed +109.5% for (c), from the unrounded runs; the transcript's figures are rounded to 0.1
    # tok/s, which give +109.4%. The recorded figure lies inside what those roundings allow.
    lo, hi = 492.95 / 235.45 - 1, 493.05 / 235.35 - 1
    assert lo <= 1.0945 <= hi + 1e-9 and round(hi * 100, 1) == 109.5
    md = to_markdown(v)
    assert ("| vllm-default (`vllm/bf16/ctx32768/b256/gmu0.90`) | stock | +20% | 246.0 | "
            "239.0-249.7 (3 of 3) | +100.4% | +100.4% | apart | **ahead on speed, answers unchecked**") in md
    assert ("**Overall: ahead on speed on every row (3 of 3), with rigtune's answers unchecked against 3 "
            "rows: not a win**") in md
    assert v.notes[0].startswith("these results do not record each run's limit status")
    assert any(n.startswith("RECONSTRUCTED, not a recorded results file") for n in v.notes)


def _row(label, runs, meets=None, kind=None, error="engine died"):
    """A row as compare builds it: one run per value (None: the run failed), combined by compare's own
    combine_runs, with each run's RunRecord."""
    meets = meets or [True] * len(runs)
    each = [ComparisonRow(label=label, runtime="vllm", config=Config(backend="vllm", quant="bf16"),
                          config_key=label, ok=v is not None, error=None if v is not None else error,
                          scored_concurrency=8, scored_tok_s=v, meets_slo=m if v is not None else None)
            for v, m in zip(runs, meets)]
    row = combine_runs(each)
    row.run_records = [run_record(r, i, 1) for i, r in enumerate(each, 1)]
    row.kind = kind
    return row


def _result(*rows, skipped=()):
    return ComparisonResult(rigtune_version="x", hardware_hash="h", gpu="A40", cpu="c", model_id="m",
                            objective="balanced", workload="chat", workload_spec={}, ttft_ceiling_ms=500,
                            rows=list(rows), skipped=list(skipped))


def _outcomes(res, t=None):
    return {r.label: (r.outcome, r.reason) for r in score(res, t).rows}


@pytest.fixture
def answers_checked(monkeypatch):
    """As if every answer had been checked and found unchanged, to test the speed rule's wins on their own."""
    import rigtune.bench.verdict as vd

    monkeypatch.setattr(vd, "answers_unchecked", lambda result, row: None)


def test_each_row_is_held_to_its_kinds_threshold():
    rig = _row("rigtune", [115.0, 116.0, 117.0])
    res = _result(rig, _row("vllm-default", [99.0, 100.0, 101.0]),  # +16%, apart: under the stock bar
                  _row("autotune-vllm", [99.0, 100.0, 101.0]),  # the same runs as a tuner: over +10%
                  _row("vllm-expert-x", [104.0, 105.0, 106.0], kind="expert"))  # +10.5%, worst round +10.4%
    got = _outcomes(res)
    assert got["vllm-default"] == (TIE, "+16.0%, worst paired round +15.8%: the median is under the +20% bar")
    assert got["autotune-vllm"] == (SPEED_ONLY, "+16.0%, worst paired round +15.8%, both at least +10%, "
                                                "ranges apart; rigtune's answers were not checked")
    assert got["vllm-expert-x"][0] == SPEED_ONLY
    assert score(res).overall["text"] == ("no row ahead of rigtune: 0 wins, 1 tie, 2 ahead on speed with "
                                          "answers unchecked")
    looser = Thresholds(stock=0.15)
    assert _outcomes(res, looser)["vllm-default"][0] == SPEED_ONLY
    assert score(res, looser).overall["outcome"] == SPEED_ONLY


@pytest.mark.usefixtures("answers_checked")
def test_with_the_answers_checked_the_same_speed_is_a_win():
    rig = _row("rigtune", [115.0, 116.0, 117.0])
    res = _result(rig, _row("vllm-default", [99.0, 100.0, 101.0]),
                  _row("autotune-vllm", [99.0, 100.0, 101.0]))
    got = _outcomes(res)
    assert got["autotune-vllm"] == (WIN, "+16.0%, worst paired round +15.8%, both at least +10%, ranges "
                                         "apart")
    assert score(res).overall["text"] == "no row ahead of rigtune: 1 win, 1 tie"
    looser = Thresholds(stock=0.15)
    assert score(res, looser).overall == {
        "outcome": WIN, "text": "win against every row (2 of 2)", "wins": 2, "ahead_answers_unchecked": 0,
        "ties": 0, "behind": 0, "behind_beyond_spread": 0, "losses": 0, "provisional": 0, "not_applicable": 0}
    few = _result(_row("rigtune", [130.0, None, 131.0]), _row("vllm-default", [100.0, 101.0, 102.0]))
    row = score(few).rows[0]
    assert (row.outcome, row.lean) == (PROVISIONAL, WIN) and row.reason.startswith("would be a win: +28.7%")


def test_the_worst_paired_round_must_clear_the_bar_too():
    rig = _row("rigtune", [125.0, 150.0, 160.0])
    base = _row("vllm-default", [110.0, 100.0, 103.0])  # round 1: 125 against 110, +13.6%
    got = _outcomes(_result(rig, base))
    assert got["vllm-default"] == (TIE, "+45.6%, worst paired round +13.6%: the worst paired round is under "
                                        "the +20% bar")


def test_behind_by_more_than_the_rows_spread_is_behind_even_with_the_ranges_overlapping():
    rig = _row("rigtune", [100.0, 101.0, 260.0])  # rigtune's median half of stock's, one outlier run
    res = _result(rig, _row("vllm-default", [200.0, 205.0, 210.0]))
    assert _outcomes(res)["vllm-default"] == (
        BEHIND_CLEAR, "-50.7%, worst paired round -50.7%: behind by more than this row's own spread (10.0 "
                      "tok/s) in the median and in 2 of 3 rounds; not a loss, ranges overlap")
    overall = score(res).overall
    assert overall["outcome"] == BEHIND_CLEAR and "no loss" not in overall["text"]
    assert overall["text"] == "behind on 1 row by more than its own spread; 0 wins, 0 ties"


def test_behind_within_the_rows_spread_is_behind_within_noise_and_never_no_loss():
    rig = _row("rigtune", [195.0, 199.0, 230.0])  # median 6 tok/s behind a row whose runs span 10
    res = _result(rig, _row("vllm-default", [200.0, 205.0, 210.0]))
    assert _outcomes(res)["vllm-default"] == (BEHIND, "-2.9%, worst paired round -2.9%, ranges overlap")
    overall = score(res).overall
    assert overall["outcome"] == BEHIND and "no loss" not in overall["text"]
    assert overall["text"] == "behind on 1 row, within noise; 0 wins, 0 ties"
    # Every paired round behind by more than the spread, but with fewer than 3 runs a side it stays noise.
    few = _result(_row("rigtune", [100.0, 101.0, 260.0]), _row("vllm-default", [200.0, 205.0]))
    assert _outcomes(few)["vllm-default"][0] == BEHIND


def test_a_row_ahead_in_every_round_with_its_ranges_apart_is_a_loss():
    rig = _row("rigtune", [100.0, 101.0, 102.0])
    res = _result(rig, _row("vllm-expert-x", [103.0, 104.0, 105.0], kind="expert"),
                  _row("sglang-default", [101.5, 103.0, 104.0]))
    got = _outcomes(res)
    assert got["vllm-expert-x"] == (LOSS, "-2.9%, worst paired round -2.9%: this row is ahead in every round "
                                          "and the ranges are apart")
    assert got["sglang-default"] == (BEHIND, "-1.9%, worst paired round -1.9%, ranges overlap")
    overall = score(res).overall
    assert (overall["outcome"], overall["text"]) == (LOSS, "1 loss")
    assert (overall["losses"], overall["behind"]) == (1, 1)
    margin = Thresholds(loss_margin=0.05)
    assert _outcomes(res, margin)["vllm-expert-x"] == (
        BEHIND_CLEAR, "-2.9%, worst paired round -2.9%: behind by more than this row's own spread (2.0 "
                      "tok/s) in the median and in 3 of 3 rounds; not a loss, under the loss margin")


def test_a_win_or_a_loss_needs_three_runs_that_met_the_limits_on_each_side():
    base = _row("vllm-default", [100.0, 101.0, 102.0])
    failed = _row("rigtune", [130.0, None, 131.0])  # rigtune's failed run counts against it
    got = _outcomes(_result(failed, base))
    assert got["vllm-default"][0] == PROVISIONAL
    # Answers unchecked: even with enough runs it would be speed alone, and the reason says so.
    assert got["vllm-default"][1].startswith("would be ahead on speed, answers unchecked, not a win: +28.7%, "
                                             "worst paired round +28.4%")
    assert "but rigtune met the limits in 2 runs, fewer than 3; rigtune's answers were not" in \
        got["vllm-default"][1]
    assert score(_result(failed, base)).rows[0].lean == SPEED_ONLY
    assert score(_result(failed, base)).overall["outcome"] == PROVISIONAL
    # A run that missed the limits neither counts nor widens the range; its tok/s is kept as None.
    missed = _row("rigtune", [130.0, 300.0, 131.0], meets=[True, False, True])
    v = score(_result(missed, base))
    assert v.rigtune["range_tok_s"] == (130.0, 131.0) and v.rigtune["runs_tok_s"] == [130.0, None, 131.0]
    assert v.rows[0].outcome == PROVISIONAL
    # The median is of the runs that count: [130 missed, 300, 90] stands for 90, not the missed run.
    odd = score(_result(_row("rigtune", [130.0, 300.0, 90.0], meets=[False, True, True]), base))
    assert odd.rigtune["median_tok_s"] == 90.0


def test_a_baseline_that_could_not_start_is_not_applicable_never_a_win():
    rig = _row("rigtune", [200.0, 210.0, 220.0])
    res = _result(rig, _row("vllm-default", [None, None, None], error="CUDA out of memory\nmore"),
                  _row("sglang-default", [50.0, 55.0, 60.0], meets=[False] * 3),
                  _row("llamacpp-cuda-default", [0.0, 0.0, 0.0]),
                  skipped=[SkippedRow(label="autotune-vllm", kind="tuner", reason="no pick")])
    got = _outcomes(res)
    assert got["vllm-default"] == (NA, "could not start or failed: CUDA out of memory")
    assert got["sglang-default"] == (NA, "never met the limits at any level")
    assert got["llamacpp-cuda-default"] == (NA, "measured 0 tok/s")  # no division by zero
    assert got["autotune-vllm"] == (NA, "not run: no pick")
    assert score(res).overall == {"outcome": NA, "text": "no row could be compared", "wins": 0,
                                  "ahead_answers_unchecked": 0, "ties": 0, "behind": 0,
                                  "behind_beyond_spread": 0, "losses": 0, "provisional": 0,
                                  "not_applicable": 4}


def test_ahead_with_a_row_not_applicable_says_so():
    rig = _row("rigtune", [130.0, 131.0, 132.0])
    res = _result(rig, _row("vllm-default", [None, None, None]),
                  _row("autotune-vllm", [100.0, 101.0, 102.0]))
    overall = score(res).overall
    assert (overall["outcome"], overall["text"]) == (
        SPEED_ONLY, "ahead on speed on the 1 row that could be compared; 1 not applicable, with rigtune's "
                    "answers unchecked against 1 row: not a win")


@pytest.mark.usefixtures("answers_checked")
def test_a_win_with_a_row_not_applicable_is_its_own_outcome():
    rig = _row("rigtune", [130.0, 131.0, 132.0])
    res = _result(rig, _row("vllm-default", [None, None, None]),
                  _row("autotune-vllm", [100.0, 101.0, 102.0]))
    overall = score(res).overall
    assert overall["outcome"] == "win on the rows that could be compared"
    assert overall["text"] == ("win on the 1 row that could be compared; 1 not applicable, so not a win "
                               "against every row")


def test_when_rigtune_meets_the_limits_in_no_run_a_row_that_does_is_a_loss():
    rig = _row("rigtune", [None, None, None])
    res = _result(rig, _row("vllm-default", [99.0, 100.0, 101.0]), _row("sglang-default", [99.0, None, None]))
    got = _outcomes(res)
    assert got["vllm-default"] == (LOSS, "rigtune's row met the limits in no run (could not start or failed: "
                                         "engine died); this row met them in 3")
    assert got["sglang-default"][0] == PROVISIONAL


def test_runs_scored_at_different_levels_are_noted():
    rig = _row("rigtune", [130.0, 131.0, 132.0])
    rig.run_records[1].scored_concurrency = 4
    v = score(_result(rig, _row("vllm-default", [100.0, 101.0, 102.0])))
    assert ("rigtune's counted runs were scored at different levels (4, 8 users): each run counts at the "
            "busiest level at which it met the limits") in v.notes


def test_results_without_run_records_are_read_and_said_to_be():
    legacy = _row("rigtune", [130.0, 131.0, 132.0])
    legacy.run_records = []
    base = _row("vllm-default", [100.0, 101.0, 102.0])
    base.run_records = []
    v = score(_result(legacy, base))
    assert v.rows[0].outcome == SPEED_ONLY and v.notes[0].startswith("these results do not record each run's")


def test_the_verdict_is_written_beside_the_results(tmp_path):
    results = tmp_path / "h__Qwen__Qwen2.5-7B-Instruct__chat__balanced.json"
    results.write_bytes(NOV14.read_bytes())
    v, js, md = write(results)
    assert js.name == "h__Qwen__Qwen2.5-7B-Instruct__chat__balanced.verdict.json" and md.parent == tmp_path
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["overall"]["outcome"] == SPEED_ONLY and data["thresholds"]["stock"] == 0.2
    assert [r["outcome"] for r in data["rows"]] == [SPEED_ONLY] * 3 and data["results"] == results.name
    assert md.read_text(encoding="utf-8") == to_markdown(v)


def test_rigtune_verdict_writes_both_files_and_takes_thresholds_in_percent(tmp_path):
    from rigtune.cli import app

    results = tmp_path / "r.json"
    results.write_bytes(NOV14.read_bytes())
    r = CliRunner().invoke(app, ["verdict", str(results), "--stock-gain", "150", "--tuner-gain", "10"])
    assert r.exit_code == 0, r.output
    data = json.loads((tmp_path / "r.verdict.json").read_text(encoding="utf-8"))
    assert data["thresholds"]["stock"] == 1.5 and data["overall"]["outcome"] == TIE
    assert [r["outcome"] for r in data["rows"]] == [TIE, TIE, SPEED_ONLY]
    assert (tmp_path / "r.verdict.md").exists()
    bad = CliRunner().invoke(app, ["verdict", str(tmp_path / "missing.json")])
    assert bad.exit_code == 2


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_compare_with_verdict_keeps_every_run_and_writes_the_verdict_beside_the_results(
        tmp_path, monkeypatch, hw_a100):
    from rigtune.cli import app
    from tests.test_extra_rows import MODEL, _cli

    _cli(monkeypatch, hw_a100)
    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--repeats", "2", "--verdict",
                                 "--out", str(out)])
    assert r.exit_code == 0, r.output
    names = sorted(p.name for p in out.iterdir())
    stem = names[0].removesuffix(".json")
    assert names == [f"{stem}.json", f"{stem}.runs.jsonl", f"{stem}.verdict.json", f"{stem}.verdict.md"]
    res = ComparisonResult.model_validate_json((out / f"{stem}.json").read_text(encoding="utf-8"))
    for i, row in enumerate(res.rows, 1):
        assert [(x.round, x.position) for x in row.run_records] == [(1, i), (2, i)]
        assert all(x.started_at <= x.ended_at for x in row.run_records)
    data = json.loads((out / f"{stem}.verdict.json").read_text(encoding="utf-8"))
    assert [row["label"] for row in data["rows"]] == ["vllm-default", "vllm-fp8-default"]
    assert data["results"] == f"{stem}.json" and "**Overall:" in r.output
