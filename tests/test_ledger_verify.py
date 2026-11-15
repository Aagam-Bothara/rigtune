"""A profile rebuilt from the ledger is the profile the calibration wrote, and `rigtune ledger verify` says so.

Checked as exact equality, the whole profile's JSON and its cache path, on fake-runner calibrations under several
search options and on every recorded profile under benchmarks/strategies, written to a ledger through the same
code a calibration uses. The pick re-derived from the table alone is the profile's configuration in each of them,
all 25 recorded profiles included, so a change to how the search ends that verify does not follow fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from typer.testing import CliRunner

from rigtune import cache as profile_cache
from rigtune import ledger as lg
from rigtune.calibrate.search import StagedSearch
from rigtune.calibrate.workload import get_workload
from rigtune.ledger import Ledger, LedgerError, LedgerRunner
from rigtune.models import Config, ModelSpec, Profile, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions, calibrate
from rigtune.power import PowerSetting
from tests.test_ledger import RECORDED, ROOT, _plan
from tests.test_objectives_and_search import FakeRunner


def _cache_path(p: Profile) -> Path:
    """Where cache.save puts a profile: the cache key, as a path."""
    requested = p.prepared.spec.revision if p.prepared is not None else None
    return profile_cache.profile_path(p.hardware_hash, ModelSpec(hf_id=p.model_id, revision=requested), p.objective,
                                      p.workload, p.power_mode, p.phases, p.options)


def _assert_same(rebuilt: Profile, original: Profile) -> None:
    assert rebuilt.model_dump_json(indent=2) == original.model_dump_json(indent=2)
    assert _cache_path(rebuilt) == _cache_path(original) and rebuilt.options == original.options
    assert [r.failure_kind for r in rebuilt.calibration_table] == [r.failure_kind for r in original.calibration_table]


class Noisy(FakeRunner):
    """The fake engine, a little slower on every second launch, so confirm runs differ from the search's."""

    def run(self, cfg, stage):
        res = super().run(cfg, stage)
        if res.ok and len(self.calls) % 2 == 0:
            res.metrics.tok_s *= 0.97
        return res


class Powered(Noisy):
    def sweep(self, base, settings, stage, progress=None):
        from rigtune.power import with_power

        out = []
        for i, s in enumerate(settings):
            res = super().run(with_power(base, s), stage)
            res.metrics.joules_per_token = (res.metrics.joules_per_token or 1.0) * (1 - 0.01 * i)
            out.append(res)
        return out


OPTIONS = {
    "default": (SearchOptions(), "throughput", Noisy),
    "no-confirm": (SearchOptions(confirm=False), "throughput", Noisy),
    "three-rounds": (SearchOptions(confirm_rounds=3), "balanced", Noisy),
    "latency": (SearchOptions(), "latency", Noisy),
    "efficiency": (SearchOptions(), "efficiency", Noisy),
}


@pytest.mark.usefixtures("no_network")
@pytest.mark.parametrize("name", sorted(OPTIONS))
def test_a_calibration_rebuilds_exactly_and_verifies(name, tmp_path, hw_a100, spec):
    opts, objective, runner = OPTIONS[name]
    plan, reg = _plan(hw_a100, spec)
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), objective, {}, base=tmp_path)
    profile = calibrate(hw_a100, spec, objective, plan, reg, runner=runner(), options=opts, ledger=ledger)
    rebuilt = lg.rebuild(lg.find(profile.calibration_id, tmp_path))
    _assert_same(rebuilt, profile)
    loaded = Profile.model_validate_json(profile.model_dump_json(indent=2))  # as the cache would hand it back
    v = lg.verify(loaded, base=tmp_path)
    assert v.ok, v.findings
    winner, score, how = lg.rederive_pick(rebuilt)
    assert winner.config.key() == profile.config.key()
    assert ("confirmed" in how) == opts.confirm


@pytest.mark.usefixtures("no_network")
def test_a_calibration_with_a_power_stage(tmp_path, hw_a100, spec):
    plan, reg = _plan(hw_a100, spec)
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "efficiency", {}, base=tmp_path)
    points = [PowerSetting(), PowerSetting(power_limit_w=250), PowerSetting(power_limit_w=200)]
    profile = calibrate(hw_a100, spec, "efficiency", plan, reg, runner=Powered(), ledger=ledger,
                        power_points=points, power_mode="cap")
    assert any(r.stage == "power" for r in profile.calibration_table)
    _assert_same(lg.rebuild(lg.find(profile.calibration_id, tmp_path)), profile)
    assert lg.verify(profile, base=tmp_path).ok


@pytest.mark.usefixtures("no_network")
def test_a_budgeted_calibration_rebuilds_without_its_skipped_trials(tmp_path, hw_a100, spec, monkeypatch):
    import rigtune.pipeline as pl

    t = [0.0]

    class Slow(Noisy):
        def run(self, cfg, stage):
            t[0] += 60.0
            return super().run(cfg, stage)

    monkeypatch.setattr(pl, "StagedSearch", lambda **kw: StagedSearch(clock=lambda: t[0], **kw))
    plan, reg = _plan(hw_a100, spec)
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    profile = calibrate(hw_a100, spec, "throughput", plan, reg, runner=Slow(), ledger=ledger,
                        options=SearchOptions(budget_s=600))
    cal = lg.find(profile.calibration_id, tmp_path)
    assert any(r["outcome"]["kind"] == "skipped_budget" for r in cal.trials)
    _assert_same(lg.rebuild(cal), profile)
    assert lg.verify(profile, base=tmp_path).ok


def _record(profile: Profile, base: Path) -> Profile:
    """Write a recorded profile's calibration to a ledger as a calibration would have: each trial through the
    ledger's runner as the search ran it, then the profile. Returns the profile with its calibration id."""
    table = list(profile.calibration_table)

    class Replay:
        def run(self, cfg, stage):
            return table.pop(0)

    spec = ModelSpec(hf_id=profile.model_id)
    ledger = Ledger.for_calibration(profile.hardware, spec, get_workload("default"), profile.objective,
                                    profile.options, base=base)
    runner = LedgerRunner(Replay(), ledger)
    for res in list(profile.calibration_table):
        assert runner.run(res.config, res.stage) is res
    profile.calibration_id = ledger.calibration_id
    ledger.finish(profile)
    return profile


@pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_recorded_profile_rebuilds_exactly(path, tmp_path):
    original = Profile.model_validate_json(path.read_text(encoding="utf-8"))
    profile = _record(original.model_copy(deep=True), tmp_path)
    rebuilt = lg.rebuild(lg.find(profile.calibration_id, tmp_path))
    _assert_same(rebuilt, profile)
    # Apart from the id that names it in the ledger, the rebuilt profile is the recorded file's own content.
    assert rebuilt.model_copy(update={"calibration_id": None}).model_dump_json() == original.model_dump_json()
    # And the pick re-derived from the recorded table alone is the configuration each one serves. The one recorded
    # with --layout auto is a layout profile: its layout's own choice is reported as not checked.
    v = lg.verify(profile, base=tmp_path)
    layout = profile.layout != "single"
    assert lg.DIFFERS not in _statuses(v) and v.exit_code == (lg.EXIT_NOT_CHECKED if layout else lg.EXIT_OK), \
        v.findings




def _statuses(v) -> List[str]:
    return [status for status, _ in v.findings]


def test_what_verify_finds_when_something_is_wrong(tmp_path, hw_a100):
    path = RECORDED[0]
    profile = _record(Profile.model_validate_json(path.read_text(encoding="utf-8")), tmp_path)
    assert lg.verify(profile, base=tmp_path).exit_code == lg.EXIT_OK  # the clean case is checked above too
    # A profile edited after the fact.
    edited = profile.model_copy(update={"config": profile.calibration_table[0].config})
    v = lg.verify(edited, base=tmp_path)
    assert v.exit_code == lg.EXIT_DIFFERS
    assert any(status == lg.DIFFERS and "config" in text for status, text in v.findings)
    # A table trial whose line was changed in the ledger.
    file = lg.ledger_path(tmp_path, profile.hardware_hash, profile.model_id)
    lines = file.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["trial"]["metrics"]["tok_s"] = 1e9
    file.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n", encoding="utf-8", newline="\n")
    v = lg.verify(profile, base=tmp_path)
    assert v.exit_code == lg.EXIT_NOT_FOUND and "not in the ledger" in v.findings[-1][1]
    with pytest.raises(LedgerError, match="not in the ledger"):
        lg.rebuild(lg.find(profile.calibration_id, tmp_path))
    # No calibration id, and an unknown one.
    assert lg.verify(profile.model_copy(update={"calibration_id": None}), base=tmp_path).exit_code == 2
    v = lg.verify(profile.model_copy(update={"calibration_id": "0" * 32}), base=tmp_path)
    assert v.exit_code == 2 and "no calibration" in v.findings[-1][1]


def _layout_of(profile: Profile) -> Profile:
    """What layout.calibrate_layout makes of a calibration's profile: a copy, with the layout's own trials added to
    the table, its count and its choice among them."""
    extra = profile.calibration_table[-1].model_copy(update={"replicas": 2, "stage": "layout"})
    return profile.model_copy(deep=True, update={
        "layout": "replicas", "replicas": 2, "options": {**profile.options, "layout": "replicas"},
        "calibration_table": list(profile.calibration_table) + [extra],
        "calibration_trials": profile.calibration_trials + 1, "config": extra.config,
        "launch_args": profile.launch_args + ["--a-layout-flag"]})


def test_a_layout_profile_is_checked_through_the_calibration_it_was_built_on(tmp_path):
    profile = _record(Profile.model_validate_json(RECORDED[0].read_text(encoding="utf-8")), tmp_path)
    layout = _layout_of(profile)
    v = lg.verify(layout, base=tmp_path)
    assert lg.DIFFERS not in _statuses(v), v.findings
    assert any("built on calibration" in text for _, text in v.findings)
    # The layout's own choice cannot be checked against the ledger: said, never passed, and a distinct exit code.
    assert _statuses(v).count(lg.NOT_CHECKED) == 1 and not v.ok and v.exit_code == lg.EXIT_NOT_CHECKED


@pytest.mark.parametrize("edit", ["forged", "quant", "launch_args", "empty table", "trial count", "model",
                                  "another calibration's trial"])
def test_an_edited_layout_profile_does_not_verify(edit, tmp_path):
    """`profile = rebuilt` once threw the layout profile away, so any edit to one verified with exit 0."""
    profile = _record(Profile.model_validate_json(RECORDED[1].read_text(encoding="utf-8")), tmp_path)
    layout = _layout_of(profile)
    edits = {
        "forged": {"replicas": 8, "config": Config(backend="vllm", quant="awq", gpu_memory_utilization=0.99, batch=4096),
                   "launch_args": ["--anything", "--trust-remote-code"], "calibration_table": [],
                   "calibration_trials": 999},
        "quant": {"config": layout.config.model_copy(update={"quant": "awq"})},
        "launch_args": {"launch_env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}},
        "empty table": {"calibration_table": []},
        "trial count": {"calibration_trials": 999},
        "model": {"model_id": "someone/else"},
        "another calibration's trial": {"calibration_table": [
            layout.calibration_table[0].model_copy(update={"stage": "forged"}), *layout.calibration_table[1:]]},
    }
    v = lg.verify(layout.model_copy(update=edits[edit]), base=tmp_path)
    assert v.exit_code == lg.EXIT_DIFFERS, v.findings


def test_a_pick_that_cannot_be_rederived_is_not_passed(tmp_path):
    from rigtune.models import EvalSetInfo

    profile = Profile.model_validate_json(RECORDED[0].read_text(encoding="utf-8"))
    profile.eval_set = EvalSetInfo(name="gsm8k", path="gsm8k.jsonl", items=10, sha256="0" * 64, max_accuracy_loss=3)
    profile = _record(profile, tmp_path)
    v = lg.verify(profile, base=tmp_path)
    assert (v.findings[-1][0], v.exit_code) == (lg.NOT_CHECKED, lg.EXIT_NOT_CHECKED) and not v.ok
    assert lg.DIFFERS not in _statuses(v)


def test_a_calibration_without_its_profile_record(tmp_path, hw_a100, spec):
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    runner = LedgerRunner(FakeRunner(), ledger)
    runner.run(Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9), "quant")
    cal = lg.find(ledger.calibration_id[:8], tmp_path)
    assert cal.calibration_id == ledger.calibration_id and cal.record is None and len(cal.trials) == 1
    with pytest.raises(LedgerError, match="failed or was stopped"):
        lg.rebuild(cal)
    with pytest.raises(LedgerError, match="at least the first 6"):
        lg.find("abc", tmp_path)


def test_a_prefix_shared_across_files_is_ambiguous(tmp_path):
    """find() once returned the first file's match and never looked at the others."""
    for cid, where in (("abcdef111", "hw1/m1.jsonl"), ("abcdef222", "hw2/m2.jsonl")):
        rec = {"schema": 1, "type": "trial", "calibration_id": cid, "seq": 0}
        rec["id"] = lg.record_id(rec)
        lg.append(tmp_path / where, rec)
    with pytest.raises(LedgerError, match="start of 2"):
        lg.find("abcdef", tmp_path)
    assert lg.find("abcdef2", tmp_path).calibration_id == "abcdef222"


# --------------------------------------------------------------------------- the commands


def _wide(monkeypatch) -> None:
    """Consoles wide enough that no table cell is folded across lines."""
    from rich.console import Console

    monkeypatch.setattr("rigtune.cli.console", Console(width=400))
    monkeypatch.setattr("rigtune.cli.err", Console(width=400, stderr=True))


def _all_output(result) -> str:
    try:
        return result.output + result.stderr
    except ValueError:  # an older click mixes stderr into output
        return result.output


@pytest.mark.usefixtures("no_network")
def test_the_commands(ledger_on, hw_a100, spec, monkeypatch):
    from rigtune.cli import app

    _wide(monkeypatch)
    plan, reg = _plan(hw_a100, spec)
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {})
    assert ledger.path.is_relative_to(ledger_on)
    profile = calibrate(hw_a100, spec, "throughput", plan, reg, runner=Noisy(), ledger=ledger)
    saved = profile_cache.save(profile)
    cli = CliRunner()
    out = cli.invoke(app, ["ledger", "show"])
    assert out.exit_code == 0, out.output
    assert profile.calibration_id[:12] in out.output and "Llama-3.2-3B-Instruct" in out.output
    out = cli.invoke(app, ["ledger", "show", "--calibration", profile.calibration_id[:8]])
    assert out.exit_code == 0 and "confirm" in out.output and "quant" in out.output
    out = cli.invoke(app, ["ledger", "verify", str(saved)])
    assert out.exit_code == 0 and "DIFFERS" not in out.output and "NOT CHECKED" not in out.output, out.output
    target = ledger_on.parent / "rebuilt.json"
    out = cli.invoke(app, ["ledger", "rebuild", profile.calibration_id, "--out", str(target)])
    assert out.exit_code == 0, out.output
    assert target.read_bytes() == saved.read_bytes()
    saved.write_text(profile.model_copy(update={"calibration_trials": 1}).model_dump_json(indent=2), encoding="utf-8")
    out = cli.invoke(app, ["ledger", "verify", str(saved)])
    assert out.exit_code == 1 and "DIFFERS" in out.output
    saved.write_text(_layout_of(profile).model_dump_json(indent=2), encoding="utf-8")
    out = cli.invoke(app, ["ledger", "verify", str(saved)])
    assert out.exit_code == 3 and "NOT CHECKED" in out.output and "DIFFERS" not in out.output, out.output


def test_what_the_commands_print_has_no_paths(ledger_on, hw_a100, monkeypatch):
    from rigtune.cli import app

    _wide(monkeypatch)
    for local in ("/home/someone/models/my-llama", "C:\\Users\\someone\\models\\my-llama"):
        ledger = Ledger.for_calibration(hw_a100, ModelSpec(hf_id=local), get_workload("default"), "throughput", {})
        assert "someone" not in ledger.path.name  # the file is named by the folder's last part
        ledger.trial(TrialResult(config=Config(backend="vllm", quant="bf16"), stage="quant", metrics=TrialMetrics(),
                                 launched=False, error="FileNotFoundError: /home/someone/weights/x.bin"))
        with ledger.path.open("ab") as fh:  # a crash cut a line short: `show` names the file it found it in
            fh.write(b'{"schema": 1, "type": "tr')
        assert "someone" in ledger.path.read_text(encoding="utf-8")  # kept in the ledger, as in a profile
    cli = CliRunner()
    out = cli.invoke(app, ["ledger", "show"])
    text = _all_output(out)
    assert out.exit_code == 0 and "someone" not in text and "my-llama" in text and "unreadable line" in text
    out = cli.invoke(app, ["ledger", "show", "--calibration", ledger.calibration_id])
    assert out.exit_code == 0 and "someone" not in _all_output(out) and "<path>/x.bin" in out.output
    # rebuild names the model it wrote.
    profile = Profile.model_validate_json(RECORDED[0].read_text(encoding="utf-8"))
    profile.model_id = "/home/someone/models/my-llama"
    profile = _record(profile, ledger_on)
    out = cli.invoke(app, ["ledger", "rebuild", profile.calibration_id, "--out", str(ledger_on.parent / "p.json")])
    assert out.exit_code == 0 and "someone" not in _all_output(out) and "my-llama" in out.output
