"""A scored comparison starts each round one row later, and keeps each run's place in its round."""

from __future__ import annotations

import pytest

from rigtune.bench import compare
from rigtune.calibrate.workload import get_workload
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare_repeats import DriftingRunner
from tests.test_objectives_and_search import FakeRunner

ROWS = ["rigtune", "vllm-default", "vllm-fp8-default"]


@pytest.mark.usefixtures("no_network")
def test_each_round_starts_one_row_later_and_records_the_order(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    runner = DriftingRunner()
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                     runner=runner, repeats=3, rotate=True, record_runs=True)
    order = [s.split(":", 1)[1] for s in runner.stages]
    assert order == ROWS + ROWS[1:] + ROWS[:1] + ROWS[2:] + ROWS[:2]
    assert [r.label for r in result.rows] == ROWS  # the results keep the planned order
    rig = result.rigtune_row
    assert [(x.round, x.position) for x in rig.run_records] == [(1, 1), (2, 3), (3, 2)]
    # Rounds still pair rigtune's run with each row's run of the same round.
    assert [x.round for x in result.rows[1].run_records] == [1, 2, 3]


@pytest.mark.usefixtures("no_network")
def test_without_rotation_every_round_keeps_the_planned_order(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    runner = DriftingRunner()
    compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"), runner=runner,
            repeats=3)
    assert [s.split(":", 1)[1] for s in runner.stages] == ROWS * 3
