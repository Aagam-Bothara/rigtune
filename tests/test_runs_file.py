"""Every run of a scored comparison is on disk the moment it is measured, so a crash keeps what was done."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rigtune.bench import compare
from rigtune.bench.compare import free_results_path, read_runs, runs_path_for
from rigtune.bench.verdict import INCOMPLETE, NA, score, verdict_paths, write
from rigtune.calibrate.workload import get_workload
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner

ROWS = ["rigtune", "vllm-default", "vllm-fp8-default"]


class MachineLost(Exception):
    pass


class DiesMidRun(CompareRunner):
    """Measures `alive` runs, then the machine goes away in the middle of the next one."""

    def __init__(self, alive: int):
        super().__init__()
        self.alive, self.stages = alive, []

    def run(self, cfg, stage):
        if len(self.stages) == self.alive:
            raise MachineLost("the pod deleted itself")
        self.stages.append(stage)
        return super().run(cfg, stage)


def _setup(hw, spec):
    candidates, reg = select(hw, spec, force="vllm")
    planned = prepare_and_plan(hw, spec, candidates, reg)
    return planned, reg, calibrate(hw, spec, "balanced", planned, reg, runner=FakeRunner())


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _crash(hw, spec, path, alive, repeats=3):
    planned, reg, profile = _setup(hw, spec)
    with pytest.raises(MachineLost):
        compare(hw, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                runner=DiesMidRun(alive), repeats=repeats, runs_path=path)
    return profile


@pytest.mark.usefixtures("no_network")
def test_a_crash_mid_comparison_keeps_every_run_measured_before_it(tmp_path, hw_a100, spec, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd)))
    path = tmp_path / "x.runs.jsonl"
    profile = _crash(hw_a100, spec, path, alive=4)  # round one (3 rows) and the first row of round two
    records = _records(path)
    assert [r["type"] for r in records] == ["comparison", "run", "run", "run", "run"]
    assert len({r["run_id"] for r in records}) == 1 and len(synced) == len(records)  # each record on disk
    head = records[0]
    assert [p["label"] for p in head["planned"]] == ROWS and head["repeats"] == 3
    assert [p["kind"] for p in head["planned"]] == ["rigtune", "stock", "stock"]
    assert [(r["label"], r["record"]["round"], r["record"]["position"]) for r in records[1:]] == [
        ("rigtune", 1, 1), ("vllm-default", 1, 2), ("vllm-fp8-default", 1, 3), ("rigtune", 2, 1)]

    result = read_runs(path)
    assert [(r.label, len(r.run_records)) for r in result.rows] == [
        ("rigtune", 2), ("vllm-default", 1), ("vllm-fp8-default", 1)]
    assert result.rigtune_row.calibration_trials == profile.calibration_trials
    assert result.incomplete == ("the comparison stopped before it finished: runs measured, rigtune 2 of 3, "
                                 "vllm-default 1 of 3, vllm-fp8-default 1 of 3")
    v = score(result)  # what was measured can still be read, and the verdict settles nothing
    assert v.overall["outcome"] == INCOMPLETE and v.overall["text"].startswith("incomplete: the comparison")


@pytest.mark.usefixtures("no_network")
def test_rows_the_comparison_never_reached_are_not_applicable(tmp_path, hw_a100, spec):
    path = tmp_path / "x.runs.jsonl"
    _crash(hw_a100, spec, path, alive=2)  # rigtune and stock vLLM once; stock vLLM fp8 never
    result = read_runs(path)
    assert [r.label for r in result.rows] == ["rigtune", "vllm-default"]
    assert [(s.label, s.kind, s.reason) for s in result.skipped] == [
        ("vllm-fp8-default", "stock", "the comparison stopped before it reached this row")]
    v = score(result)
    assert {r.label: r.outcome for r in v.rows}["vllm-fp8-default"] == NA
    assert v.overall["outcome"] == INCOMPLETE


@pytest.mark.usefixtures("no_network")
def test_a_comparison_that_stopped_before_rigtunes_row_has_nothing_to_score(tmp_path, hw_a100, spec):
    path = tmp_path / "x.runs.jsonl"
    _crash(hw_a100, spec, path, alive=0)
    with pytest.raises(ValueError, match="stopped before rigtune's row was measured"):
        score(read_runs(path))


@pytest.mark.usefixtures("no_network")
def test_a_finished_comparison_reads_back_as_its_results_and_is_complete(tmp_path, hw_a100, spec):
    planned, reg, profile = _setup(hw_a100, spec)
    path = tmp_path / "x.runs.jsonl"
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                     runner=CompareRunner(), repeats=2, runs_path=path, record_runs=True)
    assert _records(path)[-1]["type"] == "end" and _records(path)[-1]["runs"] == 6
    back = read_runs(path)
    assert back.incomplete is None and not back.skipped
    assert [r.model_dump(exclude={"run_records"}) for r in back.rows] == [
        r.model_dump(exclude={"run_records"}) for r in result.rows]
    assert [[(x.round, x.position) for x in r.run_records] for r in back.rows] == [
        [(x.round, x.position) for x in r.run_records] for r in result.rows]
    assert back.notes[-1] == "rebuilt from x.runs.jsonl: 6 runs of 3 rows"
    assert score(back).overall["outcome"] != INCOMPLETE


@pytest.mark.usefixtures("no_network")
def test_only_the_last_comparisons_records_are_read_after_a_torn_line(tmp_path, hw_a100, spec):
    """The critic's case: a torn last line with no newline, then a second comparison into the same file. The
    second header starts on a line of its own, and nothing of the first comparison is mixed into it."""
    planned, reg, profile = _setup(hw_a100, spec)
    path = tmp_path / "x.runs.jsonl"
    _crash(hw_a100, spec, path, alive=4)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"type": "run", "label": "rig')  # cut off as the machine went away
    first = path.read_text(encoding="utf-8")
    compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(), repeats=1, runs_path=path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith(first + "\n")  # appended only, after a newline of its own
    back = read_runs(path)
    assert [len(r.runs_tok_s) or 1 for r in back.rows] == [1, 1, 1] and back.incomplete is None
    assert back.notes[-1] == "rebuilt from x.runs.jsonl: 3 runs of 3 rows; 1 unreadable line skipped"


@pytest.mark.usefixtures("no_network")
def test_records_of_another_comparison_in_the_same_file_are_ignored(tmp_path, hw_a100, spec):
    path = tmp_path / "x.runs.jsonl"
    _crash(hw_a100, spec, path, alive=4)
    records = _records(path)
    stray = dict(records[1], run_id="someone-else")  # a record from a run with another id, after ours
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(stray) + "\n")
    assert [len(r.run_records) for r in read_runs(path).rows] == [2, 1, 1]


def test_an_evidence_run_never_writes_over_earlier_files(tmp_path):
    base = tmp_path / "h__m__chat__balanced.json"
    assert free_results_path(base) == base
    (tmp_path / "h__m__chat__balanced.runs.jsonl").write_text("", encoding="utf-8")
    assert free_results_path(base) == tmp_path / "h__m__chat__balanced-2.json"
    assert runs_path_for(Path("r/h__m__chat__balanced.json")) == Path("r/h__m__chat__balanced.runs.jsonl")
    (tmp_path / "r.json").write_text("{}", encoding="utf-8")
    assert verdict_paths(tmp_path / "r.json") == (tmp_path / "r.verdict.json", tmp_path / "r.verdict.md")
    (tmp_path / "r.verdict.md").write_text("", encoding="utf-8")
    assert verdict_paths(tmp_path / "r.runs.jsonl") == (tmp_path / "r.verdict-2.json",
                                                        tmp_path / "r.verdict-2.md")


@pytest.mark.usefixtures("no_network")
def test_a_second_verdict_of_one_file_is_written_beside_the_first(tmp_path, hw_a100, spec):
    planned, reg, profile = _setup(hw_a100, spec)
    path = tmp_path / "x.runs.jsonl"
    compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(), repeats=1, runs_path=path)
    _, first, _ = write(path)
    _, second, _ = write(path)
    assert (first.name, second.name) == ("x.verdict.json", "x.verdict-2.json") and first.exists()


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_compare_verdict_streams_runs_and_a_crash_can_still_be_read(tmp_path, monkeypatch, hw_a100):
    import importlib

    from rigtune.cli import app
    from tests.test_extra_rows import MODEL, _cli

    _cli(monkeypatch, hw_a100)
    cmp = importlib.import_module("rigtune.bench.compare")
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kwargs: DiesMidRun(alive=4))
    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--repeats", "3", "--verdict",
                                 "--out", str(out)])
    assert isinstance(r.exception, MachineLost)
    (runs,) = out.iterdir()  # no results file: the comparison never finished
    assert runs.name.endswith(".runs.jsonl") and len(_records(runs)) == 5
    r = CliRunner().invoke(app, ["verdict", str(runs)])
    assert r.exit_code == 0, r.output
    stem = runs.name.removesuffix(".runs.jsonl")
    data = json.loads((out / f"{stem}.verdict.json").read_text(encoding="utf-8"))
    assert data["results"] == runs.name and data["overall"]["outcome"] == INCOMPLETE

    # Run again into the same folder: a new name, so the crashed run's files are all kept.
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kwargs: CompareRunner())
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--repeats", "2", "--verdict",
                                 "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"{stem}.runs.jsonl", f"{stem}.verdict.json", f"{stem}.verdict.md", f"{stem}-2.json",
         f"{stem}-2.runs.jsonl", f"{stem}-2.verdict.json", f"{stem}-2.verdict.md"])
    assert _records(runs)[-1]["type"] == "run"  # the first run's file is untouched


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_verdict_on_a_runs_file_with_nothing_to_score_is_a_clean_error(tmp_path, hw_a100, spec):
    from rigtune.cli import app

    path = tmp_path / "x.runs.jsonl"
    _crash(hw_a100, spec, path, alive=0)
    r = CliRunner().invoke(app, ["verdict", str(path)])
    assert r.exit_code == 2 and "stopped before rigtune's row was measured" in " ".join(r.output.split())
