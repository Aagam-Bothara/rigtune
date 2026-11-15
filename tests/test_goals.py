"""Goals: one metric to optimise within limits (calibrate/objectives.py). The presets are goals built from the
constraints' fields; tests/test_preset_pins.py pins that they rank exactly as before."""

from __future__ import annotations

import dataclasses
import json
import math
import re

import pytest

from rigtune import cache, pipeline
from rigtune.calibrate.objectives import (
    CUSTOM,
    Constraints,
    Goal,
    Limit,
    close_call_level,
    enough_level,
    format_goal,
    parse_goal,
    pick,
    rank,
    resolve_goal,
    strict_refusal,
)
from rigtune.calibrate.workload import get_workload
from rigtune.models import Config, ModelSpec, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions, constraint_changes, default_constraints


def test_each_preset_is_the_goal_its_table_row_says():
    cons = Constraints(ttft_ceiling_ms=300.0, tpot_ceiling_ms=40.0)
    tpot = Limit("tpot_ms", "<=", 40.0, missing="zero")
    half = Limit("tok_s", ">=", 0.5, of_best=True)
    assert resolve_goal("throughput", cons) == Goal("tok_s", (tpot,))
    assert resolve_goal("balanced", cons) == Goal("tok_s", (Limit("ttft_p95_ms", "<=", 300.0), tpot))
    assert resolve_goal("latency", cons) == Goal("request_latency_ms", (half, tpot))
    assert resolve_goal("efficiency", cons) == Goal("joules_per_token", (half, tpot), energy_fallback=True)
    # The median ceiling, an absolute floor, and no TPOT ceiling at all.
    other = Constraints(ttft_percentile=50, tok_s_floor_abs=200.0)
    assert resolve_goal("balanced", other).limits == (Limit("ttft_p50_ms", "<=", 500.0),)
    assert resolve_goal("efficiency", other).limits == (Limit("tok_s", ">=", 200.0),)
    with pytest.raises(ValueError, match="unknown objective"):
        resolve_goal("vibes", cons)


def test_a_missing_value_misses_a_limit_unless_it_counts_as_zero():
    m = TrialMetrics(tok_s=100.0, requests=1, output_tokens=1)  # no TTFT, no TPOT
    assert Limit("ttft_p95_ms", "<=", 300.0).excess(m, 300.0) == math.inf
    assert Limit("tpot_ms", "<=", 40.0, missing="zero").excess(m, 40.0) == 0.0
    assert Limit("tok_s", ">=", 150.0).excess(m, 150.0) == 50.0
    assert Goal("joules_per_token").score(m) == math.inf
    assert Goal("joules_per_token", energy_fallback=True).score(m) == 1e9 - 100.0


def test_the_search_rules_follow_what_the_goal_optimises_and_limits():
    cons = Constraints(ttft_ceiling_ms=300.0, ttft_percentile=50)
    rules = {o: (g.improves_with_load, g.scores_tok_s, g.ttft_limit) for o in ("throughput", "latency", "balanced",
                                                                             "efficiency")
             for g in [resolve_goal(o, cons)]}
    assert rules == {"throughput": (True, True, None), "latency": (False, False, None),
                     "balanced": (True, True, Limit("ttft_p50_ms", "<=", 300.0)),
                     "efficiency": (False, False, None)}


# --------------------------------------------------------------------------- custom goals


def _trial(batch: int, levels, quant: str = "bf16") -> TrialResult:
    """A trial measured at (concurrency, tok/s, p95 TTFT, J/tok, W) levels."""
    cfg = Config(backend="vllm", quant=quant, ctx=4096, batch=batch, gpu_memory_utilization=0.9)
    by = {str(c): TrialMetrics(tok_s=t, ttft_ms=ttft / 2, ttft_p95_ms=ttft, joules_per_token=j, power_w=w,
                               tpot_ms=10.0, concurrency=c, requests=32, output_tokens=4096)
          for c, t, ttft, j, w in levels}
    best = max(by.values(), key=lambda m: m.tok_s)
    return TrialResult(config=cfg, stage="t", metrics=best.model_copy(update={"by_concurrency": by}))


@pytest.mark.parametrize("text", [
    "max tok_s",
    "min joules_per_token; tok_s>=50%best; ttft_p95_ms<=800",
    "min request_latency_ms; mem_mb<=20000; power_w<=250.5; tok_s>=7%best",
    "min ttft_p50_ms; tok_s>=1e+20",
])
def test_a_goal_reads_back_from_its_own_text(text):
    goal = parse_goal(text)
    assert format_goal(goal) == text and parse_goal(format_goal(goal)) == goal and goal.custom


def test_one_goal_is_one_text_however_it_was_written():
    assert format_goal(parse_goal("min joules_per_token;ttft_p95_ms <= 800.0;  tok_s>=50%best")) == \
        "min joules_per_token; tok_s>=50%best; ttft_p95_ms<=800"
    assert parse_goal("max tok_s; tok_s>=7%best").limits == (Limit("tok_s", ">=", 0.07, of_best=True),)


def test_a_whole_number_bound_given_as_an_int_still_writes_as_one():
    assert format_goal(Goal("tok_s", (Limit("tok_s", ">=", 200),))) == "max tok_s; tok_s>=200"


@pytest.mark.parametrize("text, message", [
    ("tok_s", "starts with `min <metric>`"),
    ("max vibes", "cannot optimise 'vibes'"),
    ("min power_w", "cannot optimise 'power_w'"),
    ("min tok_s", "`max tok_s`"),
    ("max tok_s; ttft_p95_ms>=300", "its limit is <=, not >="),
    ("max tok_s; tok_s<=300", "its limit is >=, not <="),
    ("max tok_s; ttft_p95_ms<300", "not a limit"),
    ("max tok_s; ttft_p95_ms<=50%best", "%best is a share of the best tok/s"),
    ("max tok_s; power_w<=200; power_w<=250", "limited twice"),
    ("max tok_s; tpot_ms<=40", "Constraints.tpot_ceiling_ms"),
    ("max tok_s; vibes<=3", "unknown limit 'vibes'"),
    ("max tok_s; power_w<=" + "9" * 400, "not a finite number"),
    ("max tok_s; tok_s>=0%best", "above 0% and at most 100%"),
    ("max tok_s; tok_s>=150%best", "above 0% and at most 100%"),
])
def test_a_goal_that_does_not_parse_says_why(text, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_goal(text)


def test_a_custom_goal_comes_from_its_text_with_the_workloads_tpot_ceiling_last():
    cons = Constraints(goal="min joules_per_token; power_w<=250", tpot_ceiling_ms=40.0)
    assert resolve_goal(CUSTOM, cons).limits == (Limit("power_w", "<=", 250.0),
                                                 Limit("tpot_ms", "<=", 40.0, missing="zero"))
    with pytest.raises(ValueError, match="needs a goal"):
        resolve_goal(CUSTOM, Constraints())
    with pytest.raises(ValueError, match="custom objective only"):
        resolve_goal("balanced", cons)


def test_a_custom_goal_ranks_by_limits_missed_then_how_far_relative_to_each():
    # Two levels miss limits: one misses the power cap by 100 W (40%), one misses both by 5%. Fewer missed first.
    one = _trial(16, [(8, 900.0, 400.0, 0.3, 350.0)])
    two = _trial(64, [(8, 950.0, 840.0, 0.3, 262.5)])
    cons = Constraints(goal="max tok_s; power_w<=250; ttft_p95_ms<=800")
    ranked = rank([one, two], CUSTOM, cons)
    assert [x.result for x in ranked] == [one, two] and not ranked[0].feasible
    assert ranked[0].violation_key == (1.0, 0.4) and ranked[1].violation_key == (2.0, 0.05 + 0.05)
    winner, notes = pick([one, two], CUSTOM, cons)
    assert winner is one and notes[0] == ("no config met every limit of the goal max tok_s; power_w<=250; "
                                          "ttft_p95_ms<=800; picked the one that missed the least (missed 1 "
                                          "limit, 40% past it)")
    strict = dataclasses.replace(cons, strict=True)
    assert strict_refusal(one, [one, two], CUSTOM, strict) == (
        "strict constraints: no measured configuration met every limit of the goal max tok_s; power_w<=250; "
        "ttft_p95_ms<=800 (the closest missed 1 limit, 40% past it); refusing to serve the one that missed the "
        "least")


def test_a_custom_goal_scores_each_trial_at_its_best_level_and_never_falls_back():
    # Lowest J/tok among levels drawing at most 250 W; the trial without energy readings is never chosen on tok/s.
    a = _trial(16, [(1, 100.0, 50.0, 0.9, 120.0), (8, 600.0, 120.0, 0.25, 240.0), (32, 900.0, 300.0, 0.2, 300.0)])
    b = _trial(64, [(8, 1500.0, 90.0, None, 200.0)])
    ranked = rank([a, b], CUSTOM, Constraints(goal="min joules_per_token; power_w<=250"))
    assert ranked[0].result is a and ranked[0].concurrency == 8 and ranked[0].score == 0.25
    assert ranked[1].score == math.inf and ranked[1].feasible
    # A limit on a metric the level did not measure is missed there.
    c = _trial(64, [(8, 1500.0, 90.0, 0.2, None)])
    assert not rank([c], CUSTOM, Constraints(goal="min joules_per_token; power_w<=250"))[0].feasible


def test_only_a_custom_goal_that_optimises_tok_s_stops_at_the_busiest_level_meeting_its_limits():
    tok = enough_level(CUSTOM, Constraints(goal="max tok_s; power_w<=250"))
    assert tok is not None
    assert tok(TrialMetrics(tok_s=1.0, power_w=250.0, requests=1, output_tokens=1))
    assert not tok(TrialMetrics(tok_s=1.0, power_w=251.0, requests=1, output_tokens=1))
    assert not tok(TrialMetrics(tok_s=1.0, requests=1, output_tokens=1))  # not measured: missed
    assert enough_level(CUSTOM, Constraints(goal="min joules_per_token; power_w<=250")) is None
    assert close_call_level(CUSTOM, Constraints(goal="max tok_s; power_w<=250")) is None
    assert close_call_level(CUSTOM, Constraints(goal="min joules_per_token; ttft_p50_ms<=300")) is not None


def test_a_custom_goal_has_its_own_windows_safe_profile_named_by_what_it_optimises():
    wl = get_workload("default")
    opts = SearchOptions()
    one = dataclasses.replace(default_constraints(wl, opts), goal="min joules_per_token; tok_s>=200")
    two = dataclasses.replace(one, goal="min joules_per_token; tok_s>=300")
    key = opts.key(CUSTOM, one, wl)
    assert key["optimise"] == "joules_per_token" and set(key) == {"optimise", "limits"}
    assert opts.key(CUSTOM, two, wl)["limits"] != key["limits"]
    name = cache.profile_path("hw", ModelSpec(hf_id="org/model"), CUSTOM, "default", options=key).name
    assert name == f"custom-limits-{key['limits']}-optimise-joules_per_token.json"
    assert not set('<>:"|?*;') & set(name)
    with pytest.raises(ValueError, match="needs a goal"):
        opts.key(CUSTOM, None, wl)


def test_a_custom_goal_recorded_as_json_is_the_goal_asked_for():
    wl = get_workload("default")
    cons = dataclasses.replace(default_constraints(wl, SearchOptions()), goal="min joules_per_token; tok_s>=200")
    record = json.loads(json.dumps(dataclasses.asdict(cons)))
    assert constraint_changes(record, cons) == {}
    assert constraint_changes(record, dataclasses.replace(cons, goal="min joules_per_token; tok_s>=300")) == {
        "goal": "min joules_per_token; tok_s>=200"}


@pytest.mark.usefixtures("no_network")
def test_a_custom_calibration_records_its_goal_and_only_the_ttft_ceiling_it_was_scored_against(hw_a100, spec,
                                                                                               monkeypatch):
    from tests.test_objectives_and_search import FakeRunner

    monkeypatch.setattr(pipeline, "resolve_revision", lambda s: None)
    _, reg = pipeline.select(hw_a100, spec, force="vllm")
    wl = get_workload("default")
    plan = pipeline.prepare_and_plan(hw_a100, spec, ["vllm"], reg, workload=wl)
    for goal, ttft in (("min joules_per_token; power_w<=300", None), ("max tok_s; ttft_p95_ms<=400", 400.0)):
        cons = dataclasses.replace(default_constraints(wl, SearchOptions()), goal=goal)
        profile = pipeline.calibrate(hw_a100, spec, CUSTOM, plan, reg, workload=wl, constraints=cons,
                                     runner=FakeRunner())
        assert profile.objective == CUSTOM and profile.calibration_constraints["goal"] == goal
        assert profile.workload_spec["ttft_ceiling_ms"] == ttft
        assert profile.options["optimise"] == parse_goal(goal).optimise
        assert pipeline._usable(profile, SearchOptions(), None, cons, wl)
