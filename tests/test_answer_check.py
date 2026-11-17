"""`compare --check-answers`: every row graded on labelled items against its engine's stock row, so that no
row wins on answers that got worse."""

from __future__ import annotations

import importlib
import json

import pytest
from typer.testing import CliRunner

from rigtune.bench import compare
from rigtune.bench.answers import KeyedGate, RowGrader, check
from rigtune.bench.compare import ComparisonResult, read_runs
from rigtune.bench.verdict import (NA, RIG_WORSE, RIG_WORSE_ONLY, ROW_WORSE, ROW_WORSE_ONLY, SPEED_ONLY, WIN,
                                   score, to_markdown)
from rigtune.evalset import EvalItem
from rigtune.models import Config
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner
from tests.test_verdict import _result, _row

ITEMS = [EvalItem(messages=[{"role": "user", "content": f"What is {i} + {i}?"}], answer=str(2 * i),
                  grader="number") for i in range(200)]


def _grader(**grades):
    g = RowGrader(items=ITEMS, set_name="sums", sha256="abc")
    g.grades.update(grades)
    return g


def test_each_configuration_is_graded_once_on_its_warm_engine():
    asked = []

    def answer(base_url, model, items, concurrency, max_tokens, wall_s):
        asked.append((base_url, concurrency))
        said = [f"The answer is {2 * i}." if i % 2 else "I do not know." for i in range(len(items) - 1)]
        return said + [None]

    g = RowGrader(items=ITEMS, set_name="sums", sha256="abc", answer=answer)
    cfg = Config(backend="vllm", quant="bf16", batch=256)
    g.record("http://a", "m", cfg)
    g.record("http://b", "m", cfg)  # the same configuration's next run: already graded
    g.record("http://c", "m", Config(backend="vllm", quant="bf16", batch=1))
    assert asked == [("http://a", 64), ("http://c", 8)]  # 8 to 64 at a time, by its batch
    outcomes = g.grades[cfg.key()]
    assert len(outcomes) == 200 and outcomes.count("1") == 99 and outcomes.endswith("-")
    silent = RowGrader(items=ITEMS, set_name="sums", sha256="abc", answer=lambda *a: [None] * 200)
    silent.record("http://a", "m", cfg)
    assert silent.grades == {}  # a pass that answered nothing is not a grade: the next run tries again


def test_every_graded_answer_is_kept_with_how_it_ended(tmp_path):
    """A failed check must be readable answer by answer: was it wrong, or cut off before its last line?"""

    def answer(base_url, model, items, concurrency, max_tokens, wall_s, details):
        for i in range(len(items) - 1):
            details[i] = {"finish_reason": "length" if i == 1 else "stop", "completion_tokens": 7,
                          "reasoning_chars": 0}
        return [f"The answer is {2 * i}." if i != 1 else "Let me think" for i in range(len(items) - 1)] + [None]

    kept = tmp_path / "x.answers.jsonl"
    g = RowGrader(items=ITEMS[:3], set_name="sums", sha256="abc", answer=answer, answers_file=kept)
    cfg = Config(backend="vllm", quant="bf16", batch=8)
    g.twice = {cfg.key()}
    g.record("http://a", "m", cfg)
    lines = [json.loads(x) for x in kept.read_text(encoding="utf-8").splitlines()]
    assert [(x["pass"], x["item"], x["right"]) for x in lines] == [(1, 0, True), (1, 1, False), (1, 2, None),
                                                                    (2, 0, True), (2, 1, False), (2, 2, None)]
    assert lines[1] == {"key": cfg.key(), "pass": 1, "item": 1, "label": "2", "right": False,
                        "response": "Let me think", "finish_reason": "length", "completion_tokens": 7,
                        "reasoning_chars": 0}
    assert lines[2]["response"] is None and "finish_reason" not in lines[2]  # unanswered: nothing to say
    assert g.grades[cfg.key()] == "10-"  # keeping the answers changes no grade


def test_a_row_is_checked_against_its_engines_stock_row_with_the_gates_bound():
    ref, lost4 = "1" * 200, "0000" + "1" * 196
    g = _grader(ref=ref, cand=lost4)
    keys = {"vllm-default": "ref", "rigtune": "cand", "sglang-default": "none"}
    assert check("vllm-default", "vllm", "ref", g, keys, 0.03).passed is True
    # 4 lost and none gained of 200: "not significant" by McNemar, but the bound is -5.0 points.
    strict = check("rigtune", "vllm", "cand", g, keys, 0.03)
    assert (strict.lost, strict.gained, strict.paired, strict.passed) == (4, 0, 200, False)
    assert strict.reason == ("fails: 4 lost and 0 gained of 200 items against vllm-default; the lower bound "
                             "of the difference is -5.0 points, against -3")
    assert check("rigtune", "vllm", "cand", g, keys, 0.06).passed is True
    assert check("sglang-expert-x", "sglang", "cand", g, keys, 0.03).reason == (
        "not checked: sglang-default was not graded, so there is no reference on sglang")
    assert check("x", "vllm", "missing", g, keys, 0.03).reason == (
        "not graded: its engine never answered the items")


@pytest.mark.usefixtures("no_network")
def test_compare_records_each_rows_check_and_keeps_it_in_the_runs_file(tmp_path, hw_a100, spec, monkeypatch):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "balanced", planned, reg, runner=FakeRunner())
    stock = "1" * 200
    keys = {"vllm-default": None, "vllm-fp8-default": None}
    from rigtune.bench.references import reference_configs

    refs = reference_configs(hw_a100, planned.prepared, reg)
    keys = {label: refs[label].key() for label in keys}
    g = _grader(**{keys["vllm-default"]: stock, keys["vllm-fp8-default"]: "0" * 10 + "1" * 190,
                   profile.config.key(): stock})
    path = tmp_path / "x.runs.jsonl"
    result = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(), grader=g,
                     runs_path=path, answer_tolerance=0.03)
    checks = {r.label: r.answer_check for r in result.rows}
    assert checks["vllm-default"].passed and checks["rigtune"].passed
    assert checks["vllm-fp8-default"].passed is False
    assert {r.label: r.answer_check for r in read_runs(path).rows} == checks
    assert '"type": "answers"' in path.read_text(encoding="utf-8")
    # The runner it builds itself gets the grader as its gate, and only when there is one.
    cmp = importlib.import_module("rigtune.bench.compare")
    built = []
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kw: built.append(kw) or CompareRunner())
    compare(hw_a100, spec, profile, planned.prepared, reg, grader=g)
    compare(hw_a100, spec, profile, planned.prepared, reg)
    assert built[0]["gate"] is g and "gate" not in built[1]


def _checked(label, runs, passed, reason="fails: 9 lost", mean=100, check=True):
    """A row with answer statistics at rigtune's level (the same for every row unless `mean` differs) and,
    unless check is False, an AnswerCheck."""
    from rigtune.bench.compare import AnswerCheck
    from tests.test_answer_stats import _row as answered

    row = answered(label, "vllm", mean, runs=runs)
    if check:
        row.answer_check = AnswerCheck(eval_set="sums", sha256="abc", items=200, tolerance=0.03, correct=150,
                                       answered=200, reference="vllm-default", lost=9, gained=1, paired=200,
                                       lower_bound=-0.07, passed=passed, reason=reason)
    return row


def _stock(passed=True, reason="the reference for its engine", **kw):
    return _checked("vllm-default", [100.0, 101.0, 102.0], passed, reason, **kw)


def test_a_win_is_faster_with_answers_graded_and_unchanged_on_the_workload():
    v = score(_result(_checked("rigtune", [130.0, 131.0, 132.0], True, "holds"), _stock()))
    assert [(r.outcome, r.reason) for r in v.rows] == [
        (WIN, "+29.7%, worst paired round +29.4%, both at least +20%, ranges apart")]
    assert v.overall["outcome"] == WIN and v.overall["text"] == "win against every row (1 of 1)"
    assert v.rigtune["workload_answers"] == "same" and v.rigtune["graded"]["passed"] is True


def test_graded_answers_without_workload_statistics_are_not_a_win():
    rig, stock = _row("rigtune", [130.0, 131.0, 132.0]), _row("vllm-default", [100.0, 101.0, 102.0])
    rig.answer_check = _checked("x", [1.0], True, "holds").answer_check
    stock.answer_check = _checked("x", [1.0], True, "the reference for its engine").answer_check
    v = score(_result(rig, stock))
    assert v.rows[0].outcome == SPEED_ONLY and v.rows[0].reason.endswith(
        "; rigtune's answers on the workload were not matched against its stock row's (rigtune recorded no "
        "answers at 8 users)")
    ungraded = score(_result(_checked("rigtune", [130.0, 131.0, 132.0], None, check=False),
                             _stock(check=False)))
    assert ungraded.rows[0].outcome == SPEED_ONLY
    assert ungraded.rows[0].reason.endswith("; rigtune's answers were not graded")
    assert ("answers were not graded (compare --check-answers), so no row is a win: rigtune ahead of a row "
            "is ahead on speed, answers unchecked") in ungraded.notes


def test_a_row_whose_answers_fail_is_not_applicable_never_a_win():
    res = _result(_checked("rigtune", [130.0, 131.0, 132.0], True, "holds"), _stock(),
                  _checked("vllm-expert-x", [10.0, 11.0, 12.0], False))
    v = score(res)
    got = {r.label: (r.outcome, r.reason) for r in v.rows}
    assert got["vllm-default"][0] == WIN  # slower and worse: no claim about speed, only about answers
    assert got["vllm-expert-x"] == (ROW_WORSE_ONLY, "answers differ: fails: 9 lost")
    assert v.overall["outcome"] == "win on the rows that could be compared"
    md = to_markdown(v)
    assert "Answers graded on `sums` (200 items, greedy)" in md and "| vllm-expert-x | 150 of 200 |" in md
    assert "**fails**" in md and not any(n.startswith("answers were not graded") for n in v.notes)


def test_a_row_ahead_on_speed_whose_answers_fail_has_its_own_outcome():
    res = _result(_checked("rigtune", [130.0, 131.0, 132.0], True, "holds"), _stock(),
                  _checked("vllm-expert-kv", [200.0, 201.0, 202.0], False))
    v = score(res)
    row = v.rows[1]
    assert row.outcome == ROW_WORSE and row.reason == (
        "-34.8%, worst paired round -35.0%: this row is ahead in every round and the ranges are apart; but "
        "its answers are worse: fails: 9 lost")
    assert v.overall["outcome"] == "win on the rows that could be compared"
    assert v.overall["text"] == ("win on the 1 row that could be compared; 1 ahead of rigtune on worse "
                                 "answers, so not a win against every row")
    assert v.overall["row_ahead_on_worse_answers"] == 1


def test_when_rigtunes_own_answers_fail_no_row_counts():
    res = _result(_checked("rigtune", [130.0, 131.0, 132.0], False, "fails: 12 lost"), _stock(),
                  _checked("vllm-expert-x", [200.0, 201.0, 202.0], True, "holds"))
    v = score(res)
    assert [(r.outcome, r.reason) for r in v.rows] == [
        (RIG_WORSE, "+29.7%, worst paired round +29.4%, both at least +20%, ranges apart; but rigtune's "
                    "answers differ: fails: 12 lost"),
        (RIG_WORSE_ONLY, "rigtune's answers differ: fails: 12 lost")]  # behind: no claim about speed
    assert v.overall["outcome"] == RIG_WORSE
    assert v.overall["text"] == "rigtune's answers differ (fails: 12 lost), so no row counts"


def test_under_check_answers_unchecked_answers_count_for_nothing():
    """The critic's probes I and J: grading was asked for, and rigtune's answers were not checked (its engine
    never answered, or its stock row was not graded): no row counts, as if they had failed. Another row left
    unchecked is not applicable."""
    reason = "not graded: its engine never answered the items"  # probe I
    v = score(_result(_checked("rigtune", [130.0, 131.0, 132.0], None, reason), _stock()))
    assert [(r.outcome, r.reason) for r in v.rows] == [(NA, f"rigtune's answers were not checked: {reason}")]
    assert (v.overall["outcome"], v.overall["text"]) == (
        NA, f"rigtune's answers were not checked ({reason}), so no row counts")
    reason = "not checked: vllm-default was not graded, so there is no reference on vllm"  # probe J
    v = score(_result(_checked("rigtune", [130.0, 131.0, 132.0], None, reason), _stock(None, "not graded")))
    assert [(r.outcome, r.reason) for r in v.rows] == [(NA, "answers not checked: not graded")]
    assert (v.overall["outcome"], v.overall["text"]) == (
        NA, f"rigtune's answers were not checked ({reason}), so no row counts")
    res = _result(_checked("rigtune", [130.0, 131.0, 132.0], True, "holds"), _stock(),
                  _checked("sglang-x", [100.0, 101.0, 102.0], None,
                           "not checked: sglang-default was not graded"),
                  _checked("vllm-expert-y", [100.0, 101.0, 102.0], None, check=False))
    got = {r.label: (r.outcome, r.reason) for r in score(res).rows}
    assert got["sglang-x"] == (NA, "answers not checked: not checked: sglang-default was not graded")
    assert got["vllm-expert-y"] == (NA, "answers not checked: it has no answer check")
    assert got["vllm-default"][0] == WIN


def test_the_verdict_says_when_answers_were_not_checked():
    res = _result(_checked("rigtune", [130.0, 131.0, 132.0], None, "not checked: no reference"),
                  _row("vllm-default", [100.0, 101.0, 102.0]))
    assert "rigtune's answers were not checked: not checked: no reference" in score(res).notes


def test_a_stock_row_answers_twice_and_every_check_reports_its_floor():
    asked = []

    def answer(base_url, model, items, concurrency, max_tokens, wall_s):
        asked.append(base_url)
        first = len(asked) == 1  # the stock row's two passes differ on the first 3 items
        return [("I do not know." if first and i < 3 else f"The answer is {2 * i}.")
                for i in range(len(items))]

    g = RowGrader(items=ITEMS, set_name="sums", sha256="abc", answer=answer)
    stock, rig = Config(backend="vllm", quant="bf16", batch=64), Config(backend="vllm", quant="fp8", batch=64)
    g.twice = {stock.key()}
    g.record("http://stock", "m", stock)
    g.record("http://stock-next-run", "m", stock)  # already graded twice
    g.record("http://rig", "m", rig)
    assert asked == ["http://stock", "http://stock", "http://rig"]
    keys = {"vllm-default": stock.key(), "rigtune": rig.key()}
    ref = check("vllm-default", "vllm", stock.key(), g, keys, 0.03)
    assert (ref.noise_flips, ref.noise_paired, ref.passed) == (3, 200, True)
    assert ref.reason == ("the reference for its engine; vllm-default itself differed on 3 of 200 items "
                          "between two passes")
    mine = check("rigtune", "vllm", rig.key(), g, keys, 0.03)
    assert (mine.noise_flips, mine.gained) == (3, 3) and mine.reason.endswith(
        "against -3; vllm-default itself differed on 3 of 200 items between two passes")
    md = to_markdown(score(_result(_checked("rigtune", [130.0, 131.0, 132.0], True, "holds"), _stock())))
    assert "| Row | Right | Against | Lost | Gained | Lower bound | Floor | Check |" in md
    keyed = KeyedGate(g, "pd[...]")
    keyed.record("http://router", "m", rig)
    assert "pd[...]" in g.grades and asked[-1] == "http://router"


@pytest.mark.usefixtures("no_network")
def test_a_real_runner_grades_through_its_gate_on_the_warm_engine(tmp_path):
    """SubprocessTrialRunner, the fake engine and rigtune's own chat client: the gate is asked once the
    engine has served the workload, and a stock key answers twice."""
    from rigtune.calibrate.search import SubprocessTrialRunner
    from rigtune.calibrate.workload import Workload
    from tests.conftest import make_hw
    from tests.test_power import _fake_model
    from tests.test_supervisor import FakeBackend

    g = RowGrader(items=ITEMS[:6], set_name="sums", sha256="abc", max_tokens=3)
    cfg = Config(backend="fake", quant="none", ctx=128, batch=1)
    g.twice = {cfg.key()}
    wl = Workload(n_prompts=2, prefill_tokens=8, decode_tokens=4, concurrencies=(1,))
    runner = SubprocessTrialRunner(backends={"fake": FakeBackend()}, models={"fake": _fake_model()},
                                   hw=make_hw("cpu"), workload=wl, log_dir=tmp_path, startup_timeout=15,
                                   gate=g)
    res = runner.run(cfg, "compare:vllm-default")
    assert res.ok, res.error
    assert g.grades[cfg.key()] == "000000" and g.again[cfg.key()] == "000000"  # "x x x": no number in it


@pytest.mark.usefixtures("no_network")
def test_check_answers_keeps_rigtunes_stock_row_when_include_leaves_it_out(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "balanced", planned, reg, runner=FakeRunner())
    plain = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(),
                    include=["vllm-fp8-default"])
    assert [r.label for r in plain.rows] == ["rigtune", "vllm-fp8-default"]
    graded = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(),
                     include=["vllm-fp8-default"], grader=_grader())
    assert [r.label for r in graded.rows] == ["rigtune", "vllm-default", "vllm-fp8-default"]
    assert ("vllm-default is measured although --include leaves it out: --check-answers grades rigtune's "
            "answers against it, and without it they could not be checked") in graded.notes


class StockFails(CompareRunner):
    def run(self, cfg, stage):
        if stage == "compare:vllm-default":
            from rigtune.models import TrialMetrics, TrialResult

            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error="OOM")
        return super().run(cfg, stage)


@pytest.mark.usefixtures("no_network")
def test_when_the_stock_row_cannot_start_rigtunes_answers_are_unchecked_and_nothing_counts(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "balanced", planned, reg, runner=FakeRunner())
    g = _grader(**{profile.config.key(): "1" * 200})  # rigtune answered; its stock row never started
    result = compare(hw_a100, spec, profile, planned.prepared, reg, runner=StockFails(), grader=g)
    assert result.rigtune_row.answer_check.reason == (
        "not checked: vllm-default was not graded, so there is no reference on vllm")
    v = score(result)
    assert {r.outcome for r in v.rows} == {NA} and v.overall["text"].startswith(
        "rigtune's answers were not checked (not checked: vllm-default was not graded")


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize("layout", ["disaggregated", "replicas"])
def test_a_disaggregated_or_replicas_pick_is_graded_through_its_router(hw_a100, spec, monkeypatch, layout):
    import rigtune.disagg as D
    import rigtune.layout as L
    from rigtune.bench.references import reference_configs
    from rigtune.models import DisaggSpec, TrialMetrics, TrialResult

    built = []

    class Runner:
        def __init__(self, *args, gate=None, **kw):
            self.gate = gate
            built.append(gate)

        def _result(self, cfg):
            self.gate.record("http://router", "m", cfg)  # as the real runners do once the pick has served
            m = TrialMetrics(tok_s=500.0, ttft_ms=50.0, tpot_ms=5.0, requests=4, output_tokens=40,
                             concurrency=8)
            return TrialResult(config=cfg, stage="compare", metrics=m)

        def run(self, pair, stage):
            return self._result(pair.decode)

        def run_replicas(self, cfg, gpus, stage):
            return self._result(cfg)

    monkeypatch.setattr(D, "DisaggTrialRunner", Runner)
    monkeypatch.setattr(L, "LayoutTrialRunner", Runner)
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    if layout == "replicas":
        profile = profile.model_copy(update={"replicas": 2})
        key = f"{profile.config.key()} x2"
    else:
        pair = DisaggSpec(prefill=profile.config, decode=profile.config)
        profile = profile.model_copy(update={"disagg": pair})
        key = profile.disagg.key()
    stock = reference_configs(hw_a100, planned.prepared, reg)
    g = RowGrader(items=ITEMS, set_name="sums", sha256="abc",
                  answer=lambda url, *a: [f"The answer is {2 * i}." for i in range(200)])
    g.grades.update({stock["vllm-default"].key(): "1" * 200, stock["vllm-fp8-default"].key(): "1" * 200})
    result = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(), grader=g)
    assert isinstance(built[0], KeyedGate) and built[0].key == key
    mine = result.rigtune_row.answer_check
    assert g.grades[key] == "1" * 200 and mine.passed is True and mine.reference == "vllm-default"
    assert stock["vllm-default"].key() in g.twice


def _cli(monkeypatch, hw):
    from tests.test_extra_rows import _cli as cli

    calls = cli(monkeypatch, hw)
    cmp = importlib.import_module("rigtune.bench.compare")
    built = []
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kw: built.append(kw) or CompareRunner())
    return calls, built


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_check_answers_reads_its_set_before_calibrating_and_reaches_the_runner(tmp_path, monkeypatch,
                                                                               hw_a100):
    from rigtune.cli import app
    from tests.test_extra_rows import MODEL

    calls, built = _cli(monkeypatch, hw_a100)
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "q"}\n', encoding="utf-8")
    r = CliRunner().invoke(app, ["compare", MODEL, "--check-answers", str(bad)])
    assert r.exit_code == 2 and calls == []
    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--check-answers", "gsm8k",
                                 "--check-items", "20", "--max-accuracy-loss", "5", "--out", str(out)])
    assert r.exit_code == 0, r.output
    grader = built[0]["gate"]
    assert isinstance(grader, RowGrader) and len(grader.items) == 20 and grader.set_name == "gsm8k"
    (path,) = out.iterdir()
    res = ComparisonResult.model_validate_json(path.read_text(encoding="utf-8"))
    # Nothing answered (a fake runner launches no engine), so every row says so rather than passing.
    assert {r.answer_check.reason for r in res.rows} == {"not graded: its engine never answered the items"}
    assert res.rows[0].answer_check.tolerance == 0.05
    assert json.loads(path.read_text())["rows"][0]["answer_check"]["eval_set"] == "gsm8k"
    # By default the whole set is graded, whatever --eval-items says (it sizes calibration's gate).
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--check-answers", "gsm8k",
                                 "--eval-items", "20", "--out", str(tmp_path / "all")])
    assert r.exit_code == 0, r.output
    assert len(built[1]["gate"].items) == 1319
    assert built[1]["gate"].answers_file is None  # without --verdict there is no evidence folder to keep them in
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--check-answers", "gsm8k",
                                 "--check-items", "20", "--verdict", "--out", str(tmp_path / "kept")])
    assert r.exit_code == 0, r.output
    kept = built[2]["gate"].answers_file
    assert kept.name.endswith(".answers.jsonl") and kept.with_name(
        kept.name.removesuffix(".answers.jsonl") + ".json").exists()  # beside the results it grades


def test_answers_worse_claims_speed_only_where_the_speed_rule_does():
    """Within noise, neither side is called ahead: a row whose answers fail reads row's answers worse, and
    rigtune's failing answers read rigtune's answers worse, with no claim about speed."""
    near = _result(_checked("rigtune", [100.0, 101.0, 102.0], True, "holds"), _stock(),
                   _checked("vllm-expert-x", [101.5, 103.0, 104.0], False))  # 1.9% ahead, ranges overlap
    assert score(near).rows[1].outcome == ROW_WORSE_ONLY
    tied = score(_result(_checked("rigtune", [115.0, 116.0, 117.0], False, "fails: 12 lost"), _stock()))
    assert tied.rows[0].outcome == RIG_WORSE_ONLY  # +16%: under the stock bar, so not ahead on speed
    assert (tied.overall["outcome"], tied.overall["text"]) == (
        RIG_WORSE_ONLY, "rigtune's answers differ (fails: 12 lost), so no row counts")
