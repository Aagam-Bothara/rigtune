"""Goals stated on the command line (--optimise, --limit; rigtune/goals.py): refused before anything is probed,
the same preset (and cached profile) as the older flags whenever a preset says the same thing, and the same
profile for serve as for bench."""

from __future__ import annotations

import dataclasses
import re

import pytest
import typer
from typer.testing import CliRunner

from rigtune import cli
from rigtune.calibrate.objectives import CUSTOM, Constraints
from rigtune.calibrate.workload import get_workload
from rigtune.goals import GoalError, parse_limit, recorded_constraints, resolve
from rigtune.pipeline import SearchOptions
from tests.test_cli import _unboxed

runner = CliRunner()
WL = get_workload("chat")  # TTFT ceiling 500 ms, TPOT ceiling 50 ms


def _stated(objective=None, optimise=None, limits=(), ttft_ceiling=None, ttft_percentile=95, tok_s_floor=None,
            tpot_ceiling=None):
    return cli._goal(objective, optimise, list(limits), ttft_ceiling, ttft_percentile, tok_s_floor, tpot_ceiling,
                     WL, None, False)


def _key(objective, cons, pct):
    return SearchOptions(ttft_percentile=pct).key(objective, cons, WL)


@pytest.mark.parametrize("stated, legacy", [
    (dict(optimise="tok_s"), dict(objective="throughput")),
    (dict(optimise="tok_s", limits=["ttft_p95_ms<=300"]), dict(objective="balanced", ttft_ceiling=300.0)),
    (dict(optimise="tok_s", limits=["ttft_p50_ms <= 0.3s"]),
     dict(objective="balanced", ttft_ceiling=300.0, ttft_percentile=50)),
    (dict(optimise="joules_per_token", limits=["tok_s>=50%best"]), dict(objective="efficiency")),
    (dict(optimise="request_latency_ms", limits=["tok_s>=200"]), dict(objective="latency", tok_s_floor=200.0)),
    (dict(optimise="request_latency_ms", tok_s_floor=200.0), dict(objective="latency", tok_s_floor=200.0)),
    (dict(objective="balanced", limits=["ttft_p95_ms<=300"]), dict(objective="balanced", ttft_ceiling=300.0)),
    (dict(limits=["tpot_ms<=40"]), dict(tpot_ceiling=40.0)),
    (dict(optimise="tok_s", tpot_ceiling=40.0), dict(objective="throughput", tpot_ceiling=40.0)),
    (dict(objective="efficiency", limits=["tok_s>=50%best", "tpot_ms<=50"]), dict(objective="efficiency")),
])
def test_a_goal_a_preset_says_is_that_preset_with_its_profile(stated, legacy):
    new, old = _stated(**stated), _stated(**legacy)
    assert new == old and _key(*new[:3]) == _key(*old[:3])


def test_no_new_option_is_the_balanced_default_exactly_as_before():
    objective, cons, pct, loss = _stated()
    assert (objective, pct, loss) == ("balanced", 95, 3.0)
    assert cons == cli._constraints(None, None, WL, None, 95, None, strict=False)
    assert _key(objective, cons, pct) == {"ttft": "p95"}


def test_a_share_of_the_best_throughput_other_than_half_is_a_preset_with_its_own_floor():
    objective, cons, _, _ = _stated(optimise="joules_per_token", limits=["tok_s>=70%best"])
    assert objective == "efficiency" and cons.tok_s_floor_frac == 0.7 and cons.goal is None
    assert "limits" in _key(objective, cons, 95)


def test_any_other_goal_is_custom_with_its_canonical_text():
    objective, cons, pct, _ = _stated(optimise="joules_per_token", limits=["ttft_p95_ms<=0.8s", "tok_s>=200"])
    assert objective == CUSTOM and cons.goal == "min joules_per_token; tok_s>=200; ttft_p95_ms<=800"
    assert (cons.tpot_ceiling_ms, cons.ttft_ceiling_ms, pct) == (50.0, 500.0, 95)  # the workload's
    key = _key(objective, cons, pct)
    assert key["optimise"] == "joules_per_token" and set(key) == {"optimise", "limits"}
    # Added to a preset: balanced's own TTFT ceiling stays, plus the limit.
    objective, cons, _, _ = _stated(objective="balanced", limits=["mem_mb<=20GiB"])
    assert (objective, cons.goal) == (CUSTOM, "max tok_s; mem_mb<=20480; ttft_p95_ms<=500")


def test_the_line_before_calibration_says_the_goal_and_whether_a_preset_says_it():
    r = resolve(None, "tok_s", ["ttft_p95_ms<=300"], ttft_ceiling=None, ttft_percentile=95, tok_s_floor=None,
                tpot_ceiling=None, workload_ttft_ms=500.0, workload_tpot_ms=50.0)
    assert r.line == ("goal: max tok_s; tpot_ms<=50; ttft_p95_ms<=300 (the TPOT ceiling is the workload's) = the "
                      "balanced preset")
    r = resolve(None, "joules_per_token", ["power_w<=250"], ttft_ceiling=None, ttft_percentile=95,
                tok_s_floor=None, tpot_ceiling=40.0, workload_ttft_ms=500.0, workload_tpot_ms=None)
    assert r.line == "goal: min joules_per_token; power_w<=250; tpot_ms<=40 = a custom goal, cached under its own name"
    r = resolve("efficiency", None, ["tok_s>=50%best"], ttft_ceiling=None, ttft_percentile=95, tok_s_floor=None,
                tpot_ceiling=None, workload_ttft_ms=500.0, workload_tpot_ms=None)
    assert r.line.endswith("= the efficiency preset (without energy readings it orders by tok/s)")


@pytest.mark.parametrize("text, message", [
    ("tok_s", "has no bound: quote the limit"),
    ("ttft_p95_ms<300", "a limit is <= (it includes its bound), not <"),
    ("tok_s>200", "a limit is >= (it includes its bound), not >"),
    ("tok_s=200", "not ="),
    ("tok_s 200", "not a bare number"),
    ("tok_s<=200", "tok_s is better higher, so its limit is >="),
    ("ttft_p95_ms>=200", "better lower, so its limit is <="),
    ("ttft_p95_ms<=50%best", "%best is a share of the best tok/s measured, for tok_s only"),
    ("power_w<=200ms", "power_w does not take 'ms' (units: W)"),
    ("joules_per_token<=0.4w", "does not take 'w' (units: none)"),
    ("vibes<=3", "unknown metric 'vibes'"),
    ("<=3", "is not a limit"),
    ("tok_s>=fast", "the bound must be a number"),
])
def test_a_limit_that_cannot_mean_anything_says_why(text, message):
    with pytest.raises(GoalError) as exc:
        parse_limit(text)
    assert message in str(exc.value)


def test_limits_take_units_and_the_reversed_operators():
    assert parse_limit("ttft_p95_ms=<1.5s").bound == 1500.0
    assert parse_limit("mem_mb<=20GiB").bound == 20480.0 and parse_limit("power_w<=250W").bound == 250.0
    assert parse_limit("TOK_S >= 200").metric == "tok_s"


@pytest.mark.parametrize("stated, message", [
    (dict(objective="balanced", optimise="tok_s"), "leave out --objective"),
    (dict(optimise="power_w"), "power_w is a limit only"),
    (dict(optimise="vibes"), "unknown metric 'vibes'"),
    (dict(optimise="tok_s", limits=["power_w<=200", "power_w<=250"]), "power_w is limited twice"),
    (dict(optimise="tok_s", limits=["ttft_p95_ms<=200"], ttft_ceiling=300.0), "--limit and --ttft-ceiling"),
    (dict(optimise="tok_s", limits=["tok_s>=200"], tok_s_floor=300.0), "--limit and --tok-s-floor"),
    (dict(optimise="tok_s", limits=["tpot_ms<=20"], tpot_ceiling=30.0), "--limit and --tpot-ceiling"),
])
def test_a_goal_stated_two_ways_or_impossibly_is_refused(stated, message):
    import typer

    with pytest.raises(typer.BadParameter, match=message):
        _stated(**stated)


@pytest.fixture
def never_probed(monkeypatch):
    import rigtune.hardware
    import rigtune.pipeline

    def probed(*args, **kwargs):
        raise AssertionError("probed before the goal was checked")

    monkeypatch.setattr(rigtune.hardware, "probe", probed)
    monkeypatch.setattr(rigtune.pipeline, "probe", probed)
    monkeypatch.setattr(rigtune.pipeline, "resolve_profile", probed)


@pytest.mark.parametrize("command", ["bench", "recalibrate", "compare", "serve"])
@pytest.mark.parametrize("argv, message", [
    (["--limit", "tok_s"], "has no bound: quote the limit"),
    (["--optimise", "power_w"], "power_w is a limit only"),
    (["--objective", "balanced", "--optimise", "tok_s"], "leave out --objective"),
    (["--optimise", "tok_s", "--limit", "ttft_p95_ms<=50%best"], "%best is a share"),
    (["--optimise", "tok_s", "--limit", "power_w<=1", "--arrival-rate", "1"], "applies to --objective efficiency"),
])
def test_every_goal_refusal_comes_before_anything_is_probed(never_probed, command, argv, message):
    r = runner.invoke(cli.app, [command, "x/y", *argv])
    assert r.exit_code == 2 and message in _unboxed(r.output), r.output


def test_serve_looks_up_the_profile_bench_saved_for_the_same_goal(monkeypatch, hw_a100):
    import rigtune.hardware
    import rigtune.pipeline

    seen = {}

    class Stop(Exception):
        pass

    def calibrate(hw, spec, objective, plan, reg, workload=None, constraints=None, options=None, **kw):
        seen["bench"] = options.key(objective, constraints, workload)
        raise Stop

    def resolve_profile(spec, objective, workload=None, constraints=None, options=None, **kw):
        seen["serve"] = options.key(objective, constraints, workload)
        raise Stop

    plan = type("Plan", (), {"all_feasible": [], "total_considered": 0})()
    monkeypatch.setattr(rigtune.hardware, "probe", lambda: hw_a100)
    monkeypatch.setattr(rigtune.pipeline, "select", lambda *a, **k: (["vllm"], {}))
    monkeypatch.setattr(rigtune.pipeline, "prepare_and_plan", lambda *a, **k: plan)
    monkeypatch.setattr(rigtune.pipeline, "calibrate", calibrate)
    monkeypatch.setattr(rigtune.pipeline, "resolve_profile", resolve_profile)
    monkeypatch.setattr(rigtune.pipeline, "power_reading_problem", lambda hw: None)  # as on a GPU that reads power
    argv = ["x/y", "--optimise", "joules_per_token", "--limit", "power_w<=250", "--limit", "ttft_p50_ms<=0.4s"]
    out = runner.invoke(cli.app, ["bench", *argv])
    assert isinstance(out.exception, Stop)
    assert "goal: min joules_per_token; power_w<=250; tpot_ms<=100; ttft_p50_ms<=400" in out.output
    assert isinstance(runner.invoke(cli.app, ["serve", *argv]).exception, Stop)
    assert seen["bench"] == seen["serve"] and seen["bench"]["optimise"] == "joules_per_token"
    # And with no goal options at all, nothing new is said.
    out = runner.invoke(cli.app, ["bench", "x/y"])
    assert isinstance(out.exception, Stop) and "goal:" not in out.output


def test_a_custom_profile_prints_and_redraws_its_goal(tmp_path, capsys, monkeypatch, hw_a100):
    from rigtune.models import Config, Profile, TrialMetrics, TrialResult

    cons = dataclasses.replace(Constraints(), goal="max tok_s; power_w<=250")
    trial = TrialResult(config=Config(backend="vllm", quant="bf16"), stage="quant", metrics=TrialMetrics(
        tok_s=500.0, ttft_ms=50.0, power_w=200.0, requests=8, output_tokens=800, concurrency=4))
    profile = Profile(rigtune_version="0", hardware_hash="hw", hardware=hw_a100, model_id="org/model",
                      objective=CUSTOM, backend="vllm", backend_version="1.0", config=trial.config, launch_args=["vllm"], calibration_table=[trial],
                      calibration_constraints=dataclasses.asdict(cons))
    assert recorded_constraints(profile.calibration_constraints) == cons
    assert recorded_constraints({**profile.calibration_constraints, "a_field_since_removed": 1}) == cons
    monkeypatch.setattr(cli, "console", cli.Console(width=200))
    cli._print_profile(profile)
    assert "goal:      max tok_s; power_w<=250" in capsys.readouterr().out
    path = tmp_path / "profile.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    cli._trace(str(path))  # ranks under the recorded goal; without it the custom objective has none
    assert "vllm/bf16" in capsys.readouterr().out


# --------------------------------------------------------------------------- a limit added to a named preset


@pytest.mark.parametrize("stated, legacy", [
    # The older flags keep the meaning they have for the preset; the limit is the preset's own TPOT ceiling.
    (dict(objective="latency", ttft_ceiling=300.0, limits=["tpot_ms<=40"]),
     dict(objective="latency", ttft_ceiling=300.0, tpot_ceiling=40.0)),
    (dict(objective="throughput", ttft_ceiling=300.0, limits=["tpot_ms<=40"]),
     dict(objective="throughput", ttft_ceiling=300.0, tpot_ceiling=40.0)),
    (dict(objective="balanced", tok_s_floor=100.0, limits=["tpot_ms<=40"]),
     dict(objective="balanced", tok_s_floor=100.0, tpot_ceiling=40.0)),
    (dict(objective="latency", limits=["tok_s>=200"]), dict(objective="latency", tok_s_floor=200.0)),
    (dict(objective="balanced", ttft_percentile=50, limits=["ttft_p50_ms<=300"]),
     dict(objective="balanced", ttft_percentile=50, ttft_ceiling=300.0)),
    (dict(objective="balanced", ttft_ceiling=300.0, limits=["ttft_p95_ms<=300"]),
     dict(objective="balanced", ttft_ceiling=300.0)),
])
def test_a_limit_a_named_preset_has_a_setting_for_is_that_setting(stated, legacy):
    new, old = _stated(**stated), _stated(**legacy)
    assert new == old and _key(*new[:3]) == _key(*old[:3])


def _rated(objective, limits=(), ttft_ceiling=None):
    return resolve(objective, None, list(limits), ttft_ceiling=ttft_ceiling, ttft_percentile=95, tok_s_floor=None,
                   tpot_ceiling=None, workload_ttft_ms=500.0, workload_tpot_ms=50.0, arrival_rate=0.5)


def test_efficiency_at_a_rate_keeps_its_ttft_ceiling_from_either_flag():
    """At a rate, efficiency judges latency against the TTFT ceiling (objectives._rank_at_rate)."""
    legacy = cli._constraints(800.0, None, WL, None, 95, 0.5, strict=False)
    for r in (_rated("efficiency", ["tpot_ms<=50"], ttft_ceiling=800.0), _rated("efficiency", ["ttft_p95_ms<=800"])):
        assert r.objective == "efficiency" and r.constraints == legacy


@pytest.mark.parametrize("objective, limits, message", [
    ("efficiency", ["power_w<=200"], "--objective efficiency at --arrival-rate has no setting for power_w<=200"),
    ("efficiency", ["tok_s>=200"], "--objective efficiency at --arrival-rate has no setting for tok_s>=200"),
    ("latency", ["power_w<=200"], "--arrival-rate applies to --objective efficiency only, not latency"),
])
def test_a_rate_with_a_limit_no_preset_setting_holds_is_refused_naming_the_preset(objective, limits, message):
    with pytest.raises(GoalError, match=re.escape(message)):
        _rated(objective, limits)


@pytest.mark.parametrize("objective, limits, other", [
    ("throughput", ["ttft_p95_ms<=300"], "balanced"),
    ("throughput", ["ttft_p50_ms<=300"], "balanced"),
])
def test_a_named_preset_never_turns_into_another(objective, limits, other):
    with pytest.raises(typer.BadParameter, match=f"is what --objective {other} calibrates: ask for that preset"):
        _stated(objective=objective, limits=limits)


@pytest.mark.parametrize("objective, limits, goal", [
    ("latency", ["ttft_p95_ms<=300"], "min request_latency_ms; tok_s>=50%best; ttft_p95_ms<=300"),
    ("balanced", ["tok_s>=100"], "max tok_s; tok_s>=100; ttft_p95_ms<=500"),
    ("efficiency", ["ttft_p95_ms<=800"], "min joules_per_token; tok_s>=50%best; ttft_p95_ms<=800"),
    ("throughput", ["power_w<=250"], "max tok_s; power_w<=250"),
    ("balanced", ["ttft_p50_ms<=200", "ttft_p95_ms<=600"], "max tok_s; ttft_p50_ms<=200; ttft_p95_ms<=600"),
])
def test_a_limit_a_named_preset_has_no_setting_for_makes_a_custom_goal_said_in_one_line(objective, limits, goal):
    r = resolve(objective, None, limits, ttft_ceiling=None, ttft_percentile=95, tok_s_floor=None,
                tpot_ceiling=None, workload_ttft_ms=500.0, workload_tpot_ms=None)
    assert (r.objective, r.constraints.goal) == (CUSTOM, goal)
    assert f"= a custom goal: --objective {objective} has no setting for " in r.line


@pytest.mark.parametrize("stated", [
    dict(objective="balanced", limits=["ttft_p95_ms<=300"]),
    dict(optimise="tok_s", limits=["ttft_p95_ms<=300"]),
])
def test_a_median_percentile_with_a_p95_limit_is_refused_not_dropped(stated):
    with pytest.raises(typer.BadParameter, match="--ttft-percentile 50 judges the median"):
        _stated(ttft_percentile=50, **stated)


def test_optimise_reads_its_metric_in_any_case_as_limit_does():
    assert _stated(optimise="TOK_S") == _stated(optimise="tok_s") == _stated(objective="throughput")


def test_a_bound_written_in_another_unit_is_the_same_bound():
    for v in range(1, 5001):
        assert parse_limit(f"ttft_p95_ms<={v / 1000}s").bound == parse_limit(f"ttft_p95_ms<={v}ms").bound == v
    for v in range(1, 2049):
        assert parse_limit(f"mem_mb<={v / 1024}GiB").bound == v


@pytest.mark.parametrize("text, message", [
    ("power_w<=" + "9" * 400, "the bound is not a finite number"),
    ("tok_s>=0%best", "above 0% and at most 100%"),
    ("tok_s>=150%best", "above 0% and at most 100%"),
])
def test_a_bound_that_cannot_be_written_back_is_refused(text, message):
    with pytest.raises(GoalError, match=re.escape(message)):
        parse_limit(text)


def test_efficiency_at_a_rate_with_a_ttft_limit_reaches_calibration(never_probed):
    for extra in (["--ttft-ceiling", "800", "--limit", "tpot_ms<=50"], ["--limit", "ttft_p95_ms<=800"]):
        r = runner.invoke(cli.app, ["bench", "x/y", "--objective", "efficiency", "--arrival-rate", "0.5", *extra])
        assert "probed before the goal was checked" in str(r.exception), r.output
