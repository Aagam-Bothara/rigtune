"""A comparison under a custom goal: its own results file per goal, the goal and each row's value of its metric
recorded (left out of a preset comparison's file), and no verdict, which weighs tok/s, for a goal optimising
anything else."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from rigtune import cli
from rigtune.bench.compare import ComparisonResult, ComparisonRow, _score_row, results_path, save, to_markdown
from rigtune.bench.verdict import score
from rigtune.calibrate.objectives import CUSTOM, Constraints
from rigtune.models import Config, ModelSpec, TrialMetrics
from tests.test_cli import _unboxed

SPEC = ModelSpec(hf_id="org/model")
ENERGY = "min joules_per_token; power_w<=250"


def _result(objective="balanced", goal=None, rows=()):
    return ComparisonResult(rigtune_version="x", hardware_hash="h", gpu="A40", cpu="c", model_id="org/model",
                            objective=objective, workload="chat", workload_spec={}, ttft_ceiling_ms=500,
                            tpot_ceiling_ms=50.0, rows=list(rows), goal=goal)


def _row():
    m = TrialMetrics(tok_s=900.0, ttft_ms=80.0, ttft_p95_ms=120.0, tpot_ms=20.0, joules_per_token=0.25,
                     power_w=220.0, requests=32, output_tokens=4096, concurrency=8)
    return ComparisonRow(label="rigtune", runtime="vllm", config=Config(backend="vllm", quant="bf16"),
                         config_key="vllm/bf16", ok=True, metrics=m)


def test_each_custom_goal_has_its_own_results_file_and_a_preset_keeps_its_name():
    assert results_path("h", SPEC, "chat", "balanced").name == "h__org__model__chat__balanced.json"
    one = results_path("h", SPEC, "chat", CUSTOM, goal=ENERGY).name
    two = results_path("h", SPEC, "chat", CUSTOM, goal="min joules_per_token; power_w<=200").name
    assert one.startswith("h__org__model__chat__custom-joules_per_token-") and one.endswith(".json")
    assert one != two and not set('<>:"|?*;') & set(one + two)


def test_two_custom_goals_write_two_files(tmp_path):
    a = save(_result(CUSTOM, ENERGY), tmp_path)
    b = save(_result(CUSTOM, "max tok_s; power_w<=250"), tmp_path)
    assert a != b and sorted(p.name for p in tmp_path.iterdir()) == sorted([a.name, b.name])
    assert ComparisonResult.model_validate_json(a.read_text(encoding="utf-8")).goal == ENERGY


def test_a_preset_comparison_is_written_without_the_goal_fields():
    row = _row()
    _score_row(row, "balanced", Constraints())
    data = json.loads(_result(rows=[row]).model_dump_json())
    assert "goal" not in data and "scored_objective" not in data["rows"][0]
    assert row.scored_objective is None and row.meets_slo


def test_a_custom_comparison_records_the_goal_and_each_rows_value_of_its_metric():
    row = _row()
    _score_row(row, CUSTOM, Constraints(goal=ENERGY))
    assert row.scored_objective == 0.25 and row.meets_slo
    data = json.loads(_result(CUSTOM, ENERGY, [row]).model_dump_json())
    assert data["goal"] == ENERGY and data["rows"][0]["scored_objective"] == 0.25
    priced = _row()
    _score_row(priced, CUSTOM, Constraints(goal="min cost_per_mtok; gpu_hourly_usd=3.6"))
    assert priced.scored_objective == pytest.approx(3.6 / (900.0 * 3600) * 1e6)


def test_a_custom_comparison_names_its_goal_not_the_workloads_ttft_ceiling():
    header = to_markdown(_result(CUSTOM, ENERGY)).splitlines()[0]
    assert header.endswith(f"objective `custom` `{ENERGY}` (TPOT ≤ 50 ms)") and "TTFT ≤" not in header
    assert to_markdown(_result()).splitlines()[0].endswith("objective `balanced` (TTFT ≤ 500 ms, TPOT ≤ 50 ms)")


def test_a_verdict_weighs_tok_s_so_only_a_goal_optimising_it_gets_one():
    with pytest.raises(ValueError, match="optimises joules_per_token"):
        score(_result(CUSTOM, ENERGY))
    with pytest.raises(ValueError, match="no rigtune row"):  # past the goal check, to the usual one
        score(_result(CUSTOM, "max tok_s; power_w<=250"))


def test_compare_refuses_a_verdict_on_such_a_goal_before_anything_is_probed(monkeypatch):
    import rigtune.hardware

    def probed(*args, **kwargs):
        raise AssertionError("probed")

    monkeypatch.setattr(rigtune.hardware, "probe", probed)
    r = CliRunner().invoke(cli.app, ["compare", "x/y", "--verdict", "--optimise", "joules_per_token",
                                     "--limit", "power_w<=250"])
    assert r.exit_code == 2 and "a verdict weighs tok/s, and this goal optimises joules_per_token" in \
        _unboxed(r.output)
