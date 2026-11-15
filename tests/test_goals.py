"""Goals: one metric to optimise within limits (calibrate/objectives.py). The presets are goals built from the
constraints' fields; tests/test_preset_pins.py pins that they rank exactly as before."""

from __future__ import annotations

import math

import pytest

from rigtune.calibrate.objectives import Constraints, Goal, Limit, resolve_goal
from rigtune.models import TrialMetrics


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
