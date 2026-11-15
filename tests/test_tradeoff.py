"""The trade-off table (rigtune/tradeoff.py): a pure reading of recorded trials, shown after a calibration and by
`rigtune tradeoffs`, never stored in the profile."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rigtune import cli
from rigtune.calibrate.objectives import CUSTOM, Constraints, pick
from rigtune.goals import rescore_goal
from rigtune.models import Profile
from rigtune.tradeoff import PARETO_ROWS, _dominates, profile_constraints, to_lines, tradeoffs
from rigtune.tradeoff import _key as tradeoff_key
from tests.test_goals import _trial
from tests.test_preset_pins import PRESETS, _recorded_tables, _synthetic

ROOT = Path(__file__).resolve().parent.parent
L4 = ROOT / "benchmarks/strategies/results-l4/p95/profiles/balanced-sharegpt-ttft-p95.json"
runner = CliRunner()


def _l4() -> Profile:
    return Profile.model_validate_json(L4.read_text(encoding="utf-8"))


def _rows(t, why):
    return [r for r in t.rows if r.why.startswith(why)]


def test_a_profile_written_before_constraints_were_recorded_is_scored_at_its_workloads_ceilings():
    p = _l4()
    assert p.calibration_constraints == {}
    cons = profile_constraints(p)
    assert (cons.ttft_ceiling_ms, cons.tpot_ceiling_ms, cons.ttft_percentile) == (1000.0, 100.0, 95)


def test_the_l4_table_shows_the_trial_printed_at_695_that_counted_at_219():
    p = _l4()
    t = tradeoffs(p.calibration_table, p.objective, profile_constraints(p), winner=cli._winner_key(p))
    (served,) = _rows(t, "the pick")
    assert (served.point.result.config.key(), served.point.metrics.concurrency) == (p.config.key(), 32)
    assert served.why == "the pick" and not served.point.misses
    # The 16k prefill budget: 694.9 tok/s at 32 users broke the 1000 ms p95 ceiling (1007 ms); it met it at 8.
    budget = [r.point for r in t.rows if r.point.result.config.prefill_budget == 16384]
    assert {(pt.metrics.concurrency, round(pt.values["tok_s"], 1)) for pt in budget} >= {(8, 219.5), (32, 694.9)}
    assert [pt.misses for pt in budget if pt.metrics.concurrency == 32] == [["ttft_p95_ms<=1000"]]
    (ceiling,) = _rows(t, "without ttft_p95_ms<=1000")
    assert ceiling.why.endswith(": +1.5% more tok_s") and ceiling.within_noise
    lines = to_lines(t)
    assert lines[0] == "Trade-offs under balanced: max tok_s; ttft_p95_ms<=1000; tpot_ms<=100"
    assert any(line.startswith("| best p95 TTFT ms, limits ignored | vllm/fp8/ctx8192/b64/gmu0.95/pb16384 | 8 |")
               for line in lines)
    assert "- recorded points only: configurations the search never tried are absent, so each cost is a floor" in lines


def test_the_table_reads_the_trials_and_changes_nothing():
    p = _l4()
    before = [r.model_dump() for r in p.calibration_table]
    chosen = pick(p.calibration_table, p.objective, profile_constraints(p))[0]
    tradeoffs(p.calibration_table, p.objective, profile_constraints(p), winner=cli._winner_key(p))
    tradeoffs(p.calibration_table, CUSTOM, Constraints(goal="min cost_per_mtok; gpu_hourly_usd=1; kwh_usd=0.2"))
    assert [r.model_dump() for r in p.calibration_table] == before
    assert pick(p.calibration_table, p.objective, profile_constraints(p))[0] is chosen


def test_a_limits_cost_is_claimed_only_for_a_better_point_that_misses_that_limit_alone_on_any_recorded_table():
    checked = 0
    for table in _recorded_tables().values():
        for objective in PRESETS:
            cons = Constraints(ttft_ceiling_ms=500.0, tpot_ceiling_ms=60.0)
            t = tradeoffs(table, objective, cons)
            (lead,) = _rows(t, "the pick")
            if lead.point.misses:
                continue
            for row in _rows(t, "without "):
                text = row.why.split(":")[0].removeprefix("without ")
                assert row.point.score < lead.point.score and row.point.misses == [text]
                assert tradeoff_key(row.point.result) != tradeoff_key(lead.point.result)
                checked += 1
    assert checked > 10


def test_another_run_of_the_served_configuration_is_never_credited_to_a_limit():
    """On the 14B profile the pick is the combine run at 380.9 tok/s; the same configuration's confirm run, at
    385.2, misses no limit, so neither limit cost anything that was measured."""
    path = ROOT / "benchmarks/strategies/results-14b/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95.json"
    p = Profile.model_validate_json(path.read_text(encoding="utf-8"))
    t = tradeoffs(p.calibration_table, p.objective, profile_constraints(p), winner=cli._winner_key(p))
    (served,) = _rows(t, "the pick")
    assert (served.point.result.stage, round(served.point.values["tok_s"], 1)) == ("combine", 380.9)
    assert _rows(t, "without ") == []
    assert "without ttft_p95_ms<=500: no recorded configuration does better on tok_s" in t.notes
    assert "without tpot_ms<=50: no recorded configuration does better on tok_s" in t.notes
    front = _rows(t, "meets the limits")
    assert [r.point for r in front if tradeoff_key(r.point.result) == cli._winner_key(p)] == [served.point]


def test_the_points_meeting_the_limits_that_nothing_beats_are_capped_and_counted():
    # 12 configurations trading tok/s for TTFT one for one: none beats another on both, so all are on the front.
    trials = [_trial(c, [(8, 100.0 + c, 50.0 + c, 0.2, 200.0)]) for c in range(1, 13)]
    t = tradeoffs(trials, CUSTOM, Constraints(goal="max tok_s; ttft_p95_ms<=1000"))
    front = _rows(t, "meets the limits")
    assert len(front) == PARETO_ROWS and [r.point.result.config.batch for r in front] == list(range(12, 4, -1))
    assert f"{12 - PARETO_ROWS} more points meet the limits and are unbeaten on tok_s, ttft_p95_ms; not shown" \
        in t.notes


def test_front_rows_meet_the_limits_and_none_beats_another_on_synthetic_trials():
    for seed in range(30):
        t = tradeoffs(_synthetic(seed), "balanced", Constraints(ttft_ceiling_ms=200.0))
        front = [r.point for r in _rows(t, "meets the limits")]
        names = ["tok_s", "ttft_p95_ms"]
        assert all(not p.misses for p in front)
        assert not any(_dominates(a, b, names) for a in front for b in front)


def test_the_served_configuration_is_the_pick_and_another_goal_says_it_is_its_own():
    a = _trial(16, [(8, 900.0, 100.0, 0.3, 200.0)])
    b = _trial(64, [(8, 905.0, 100.0, 0.3, 200.0)])  # the best-ranked, within noise
    served = tradeoffs([a, b], "throughput", Constraints(), winner=a.config.key())
    assert _rows(served, "the pick")[0].point.result is a and _rows(served, "the pick")[0].why == "the pick"
    unserved = tradeoffs([a, b], "throughput", Constraints())
    assert _rows(unserved, "the pick")[0].why == "the pick under this goal"


def test_an_arrival_rate_pick_is_not_a_level_and_gets_a_note_instead():
    t = tradeoffs(_synthetic(1), "efficiency", Constraints(arrival_rate=0.5))
    assert t.rows == [] and t.notes[0].startswith("not shown at an arrival rate")


def test_prices_add_a_cost_column_and_an_accuracy_note_is_kept():
    trial = _trial(64, [(8, 1000.0, 100.0, 0.3, 200.0)])
    t = tradeoffs([trial], "throughput", Constraints(), accuracy=["accuracy gate: refusing x cost 2x"],
                  prices=(2.0, None))
    assert t.columns[-1][1] == "$/Mtok" and t.rows[0].point.values["cost_per_mtok"] == 2.0 / 3600 / 1000 * 1e6
    assert t.notes[-1] == "accuracy gate: refusing x cost 2x"
    assert "cost_per_mtok" not in [c[0] for c in tradeoffs([trial], "throughput", Constraints()).columns]


@pytest.mark.parametrize("argv, first", [
    ([], "Trade-offs under balanced: max tok_s; ttft_p95_ms<=1000; tpot_ms<=100"),
    (["--objective", "throughput"], "Trade-offs under throughput: max tok_s; tpot_ms<=100"),
    (["--optimise", "joules_per_token", "--limit", "ttft_p95_ms<=1000"],
     "Trade-offs under min joules_per_token; ttft_p95_ms<=1000; tpot_ms<=100"),
])
def test_rigtune_tradeoffs_rescores_a_profile_without_launching_anything(argv, first, monkeypatch):
    monkeypatch.setattr(cli, "console", cli.Console(width=80))
    r = runner.invoke(cli.app, ["tradeoffs", str(L4), *argv])
    assert r.exit_code == 0, r.output
    lines = r.output.splitlines()
    assert lines[0] == first and all(len(line) > 80 for line in lines if line.startswith("| the pick"))
    assert ("re-scored: a search for max tok_s; ttft_p95_ms<=1000; tpot_ms<=100 chose which configurations to "
            "measure" in r.output) == bool(argv)


def test_rigtune_tradeoffs_refuses_a_goal_it_cannot_score():
    r = runner.invoke(cli.app, ["tradeoffs", str(L4), "--kwh-usd", "0.2"])
    assert r.exit_code == 2 and "--gpu-hourly-usd, which is missing" in r.output
    r = runner.invoke(cli.app, ["tradeoffs", str(L4), "--gpu-hourly-usd", "0.8"])
    assert r.exit_code == 0 and "$/Mtok" in r.output


def test_bench_and_recalibrate_print_the_table_under_the_goal_they_calibrated(capsys, monkeypatch):
    p = _l4()
    monkeypatch.setattr(cli, "console", cli.Console(width=80))
    cli._print_tradeoffs(p, p.objective, profile_constraints(p))
    out = capsys.readouterr().out
    assert out.startswith("Trade-offs under balanced") and "| the pick | vllm/fp8/ctx8192/b64/gmu0.95/pb8192 |" in out
    cli._print_tradeoffs(p.model_copy(update={"calibration_table": []}), p.objective, profile_constraints(p))
    assert capsys.readouterr().out == ""  # served without a search: nothing to show


def test_a_limit_alone_rescores_the_profiles_own_goal_plus_that_limit():
    p = _l4()
    own = profile_constraints(p)
    objective, cons, _, note = rescore_goal(p.objective, own, None, None, ["ttft_p95_ms<=800"])
    assert (objective, cons) == ("balanced", dataclasses.replace(own, ttft_ceiling_ms=800.0)) and note
    objective, cons, _, _ = rescore_goal(p.objective, own, None, None, ["power_w<=70"])
    assert (objective, cons.goal) == (CUSTOM, "max tok_s; power_w<=70; ttft_p95_ms<=1000")
    # A custom profile's own goal, with the stated limit in place of its own on that metric.
    mine = Constraints(goal="min joules_per_token; power_w<=250; ttft_p95_ms<=800", tpot_ceiling_ms=100.0)
    objective, cons, _, _ = rescore_goal(CUSTOM, mine, None, None, ["power_w<=200"])
    assert (objective, cons.goal) == (CUSTOM, "min joules_per_token; power_w<=200; ttft_p95_ms<=800")
    assert rescore_goal(CUSTOM, mine, None, None, []) == (CUSTOM, mine, None, None)


def test_a_table_that_cannot_be_drawn_is_said_and_never_stops_the_command(capsys, monkeypatch):
    import rigtune.tradeoff

    def broken(*args, **kwargs):
        raise ValueError("no such level")

    monkeypatch.setattr(rigtune.tradeoff, "tradeoffs", broken)
    p = _l4()
    cli._print_tradeoffs(p, p.objective, profile_constraints(p))
    assert "trade-off table not shown: no such level" in capsys.readouterr().err
