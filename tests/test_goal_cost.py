"""cost_per_mtok: dollars per million output tokens at the GPU's hourly price, and its power at an energy price."""

from __future__ import annotations

import dataclasses

import pytest

from rigtune.calibrate.objectives import (
    CUSTOM,
    Constraints,
    cost_per_mtok,
    enough_level,
    format_goal,
    parse_goal,
    pick,
    rank,
)
from rigtune.calibrate.workload import get_workload
from rigtune.goals import GoalError, resolve
from rigtune.models import TrialMetrics
from rigtune.pipeline import OptionRefused, SearchOptions, check_goal_measurable, default_constraints
from tests.test_goals import _trial
from tests.test_preset_pins import _recorded_tables


def _level(tok_s=1000.0, power_w=300.0) -> TrialMetrics:
    return TrialMetrics(tok_s=tok_s, power_w=power_w, requests=8, output_tokens=800)


def test_the_cost_of_a_million_output_tokens():
    assert cost_per_mtok(_level(), 2.0, None) == 2.0 / (1000.0 * 3600) * 1e6
    assert cost_per_mtok(_level(), 2.0, 0.1) == (2.0 + 0.1 * 300.0 / 1000) / (1000.0 * 3600) * 1e6
    assert cost_per_mtok(_level(power_w=None), 2.0, None) is not None  # no energy price: power is not needed
    assert cost_per_mtok(_level(power_w=None), 2.0, 0.1) is None
    assert cost_per_mtok(_level(tok_s=0.0), 2.0, None) is None
    assert cost_per_mtok(_level(), None, None) is None  # unpriced


def test_a_priced_goal_reads_back_from_its_text_with_its_prices():
    text = "min cost_per_mtok; ttft_p95_ms<=800; gpu_hourly_usd=2.5; kwh_usd=0.15"
    goal = parse_goal(text)
    assert (goal.gpu_hourly_usd, goal.kwh_usd) == (2.5, 0.15) and format_goal(goal) == text
    assert parse_goal("max tok_s; cost_per_mtok<=0.4; gpu_hourly_usd=2").limits[0].metric == "cost_per_mtok"


@pytest.mark.parametrize("text, message", [
    ("min cost_per_mtok", "needs the GPU's price per hour"),
    ("min cost_per_mtok; kwh_usd=0.1", "needs the GPU's price per hour"),
    ("max tok_s; gpu_hourly_usd=2", "which the goal does not name"),
    ("min cost_per_mtok; gpu_hourly_usd=2; gpu_hourly_usd=3", "given twice"),
    ("min cost_per_mtok; gpu_hourly_usd=" + "9" * 400, "a finite price above 0"),
    ("min cost_per_mtok; gpu_hourly_usd=0", "a finite price above 0"),
])
def test_a_price_that_prices_nothing_or_a_cost_without_one_is_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_goal(text)


def test_at_an_hourly_price_alone_cost_picks_what_throughput_picks_on_every_recorded_table():
    cons = Constraints(tpot_ceiling_ms=100.0)
    priced = dataclasses.replace(cons, goal="min cost_per_mtok; gpu_hourly_usd=2")
    for table in _recorded_tables().values():
        assert pick(table, CUSTOM, priced)[0] is pick(table, "throughput", cons)[0]
        mine = [(x.result, x.metrics) for x in rank(table, CUSTOM, priced)][:1]
        assert mine == [(x.result, x.metrics) for x in rank(table, "throughput", cons)][:1]
    assert enough_level(CUSTOM, priced) is not None
    # An energy price makes power count, which load does not order: every level is measured.
    assert enough_level(CUSTOM, dataclasses.replace(cons, goal="min cost_per_mtok; gpu_hourly_usd=2; kwh_usd=0.1")) \
        is None


def test_an_energy_price_can_move_the_pick_to_a_level_drawing_less():
    # (concurrency, tok/s, p95 TTFT, J/tok, W): the busier level is 11% faster and draws 2.7x the power.
    trial = _trial(64, [(4, 900.0, 100.0, 0.3, 150.0), (8, 1000.0, 200.0, 0.4, 400.0)])
    hourly = Constraints(goal="min cost_per_mtok; gpu_hourly_usd=0.01")
    energy = Constraints(goal="min cost_per_mtok; gpu_hourly_usd=0.01; kwh_usd=10")
    assert rank([trial], CUSTOM, hourly)[0].concurrency == 8
    assert rank([trial], CUSTOM, energy)[0].concurrency == 4


def test_prices_are_part_of_the_profile_fingerprint():
    wl = get_workload("default")
    base = default_constraints(wl, SearchOptions())
    keys = [SearchOptions().key(CUSTOM, dataclasses.replace(base, goal=g), wl)["limits"]
            for g in ("min cost_per_mtok; gpu_hourly_usd=2", "min cost_per_mtok; gpu_hourly_usd=3",
                      "min cost_per_mtok; gpu_hourly_usd=2; kwh_usd=0.1")]
    assert len(set(keys)) == 3


def _resolve(optimise=None, limits=(), gpu=None, kwh=None):
    return resolve(None, optimise, list(limits), ttft_ceiling=None, ttft_percentile=95, tok_s_floor=None,
                   tpot_ceiling=None, workload_ttft_ms=500.0, workload_tpot_ms=None, gpu_hourly_usd=gpu,
                   kwh_usd=kwh)


def test_the_command_line_prices_a_goal_that_names_cost_and_only_one():
    r = _resolve("cost_per_mtok", ["ttft_p95_ms<=800"], gpu=2.5, kwh=0.15)
    assert r.objective == CUSTOM
    assert r.constraints.goal == "min cost_per_mtok; ttft_p95_ms<=800; gpu_hourly_usd=2.5; kwh_usd=0.15"
    assert _resolve("tok_s", ["cost_per_mtok<=0.5"], gpu=2.0).constraints.goal == \
        "max tok_s; cost_per_mtok<=0.5; gpu_hourly_usd=2"


@pytest.mark.parametrize("kwargs, message", [
    (dict(optimise="cost_per_mtok"), "needs --gpu-hourly-usd"),
    (dict(optimise="cost_per_mtok", kwh=0.1), "needs --gpu-hourly-usd"),
    (dict(optimise="tok_s", gpu=2.0), "price cost_per_mtok: name it"),
    (dict(limits=["tpot_ms<=40"], kwh=0.1), "price cost_per_mtok: name it"),
    (dict(optimise="cost_per_mtok", gpu=0.0), "--gpu-hourly-usd is a price above 0"),
    (dict(optimise="cost_per_mtok", gpu=1.0, kwh=0.0), "--kwh-usd is a price above 0"),
    (dict(optimise="cost_per_mtok", gpu=float("inf")), "--gpu-hourly-usd is a price above 0"),
    (dict(limits=["tpot_ms<=40"], gpu=2.0), "price cost_per_mtok: name it"),
    (dict(optimise="tok_s", limits=["cost_per_mtok<=0.5$"], gpu=1.0), "the bound must be a number"),
])
def test_a_price_on_the_command_line_that_prices_nothing_is_refused(kwargs, message):
    with pytest.raises(GoalError, match=message):
        _resolve(**kwargs)


def test_only_an_energy_price_makes_a_cost_goal_read_power(hw_a100):
    def no_power(hw):
        return "no NVIDIA GPU is visible"

    hourly = Constraints(goal="min cost_per_mtok; gpu_hourly_usd=2")
    check_goal_measurable(CUSTOM, hourly, SearchOptions(), hw_a100, power_probe=no_power)
    energy = Constraints(goal="min cost_per_mtok; gpu_hourly_usd=2; kwh_usd=0.1")
    with pytest.raises(OptionRefused, match=r"reads the GPU's power \(cost_per_mtok\)"):
        check_goal_measurable(CUSTOM, energy, SearchOptions(), hw_a100, power_probe=no_power)
