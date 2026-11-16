"""Each launched trial's start-up time is recorded in the trial ledger, and nowhere else.

The record sits beside the trial in its ledger line, so a profile, a comparison result and a profile rebuilt from the
ledger are byte for byte what they were. A value that was not observed is None, never 0.
"""

from __future__ import annotations

import pytest

import rigtune.calibrate.search as search_mod
from rigtune import ledger as lg
from rigtune.calibrate.search import SubprocessTrialRunner
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.ledger import Ledger
from rigtune.models import Config, Profile, StartupCost, TrialMetrics, TrialResult
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_failure_kind import _Live, _LiveBackend
from tests.test_objectives_and_search import FakeRunner
from tests.test_text_only import LOAD_LOG, _DeadBackend

CFG = Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9)
GOOD = TrialMetrics(requests=8, failed=0, output_tokens=800, tok_s=100.0, ttft_ms=50.0, tpot_ms=10.0)


class _Stamped(_Live):
    started_at, ready_at = 100.0, 142.5


class _StampedBackend(_LiveBackend):
    def launch(self, cfg, model, port, log_path=None):
        return _Stamped()


def _run(backend, monkeypatch) -> TrialResult:
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: GOOD)
    runner = SubprocessTrialRunner(backends={"vllm": backend}, models={"vllm": None}, hw=None, workload=Workload())
    return runner.run(CFG, "batch")


def test_seconds_are_ready_minus_started(monkeypatch):
    res = _run(_StampedBackend(""), monkeypatch)
    assert res.ok and res.startup == StartupCost(seconds=42.5)


def test_never_ready_is_none_not_zero(monkeypatch):
    res = _run(_DeadBackend(LOAD_LOG), monkeypatch)  # the engine exited before it answered
    assert not res.launched and res.startup is not None and res.startup.seconds is None


def test_a_process_without_stamps_is_none_not_zero(monkeypatch):
    res = _run(_LiveBackend(""), monkeypatch)  # another runner's process object: nothing to subtract
    assert res.ok and res.startup == StartupCost(seconds=None)


def test_skipped_trial_has_no_startup():
    """A result the search writes without launching (a budget skip, a blocked pair) started no engine."""
    assert TrialResult(config=CFG, stage="batch", metrics=TrialMetrics(), launched=False, error="skipped").startup is None


def test_one_trials_startup_is_not_the_next_ones(monkeypatch):
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: GOOD)
    backends = {"vllm": _StampedBackend("")}
    runner = SubprocessTrialRunner(backends=backends, models={"vllm": None}, hw=None, workload=Workload())
    assert runner.run(CFG, "batch").startup.seconds == 42.5

    def refuse(cfg, model, port, log_path=None):
        raise RuntimeError("no launch")

    backends["vllm"].launch = refuse
    assert runner.run(CFG, "batch").startup is None  # nothing was started: the earlier stamps are not reused


def test_profile_dump_has_no_startup_key(monkeypatch):
    res = _run(_StampedBackend(""), monkeypatch)
    plain = res.model_copy(update={"startup": None})
    assert res.model_dump_json() == plain.model_dump_json() and "startup" not in res.model_dump(mode="json")


class _Costed(FakeRunner):
    """The fake runner, with a start-up record on every trial it launches."""

    def run(self, cfg, stage):
        res = super().run(cfg, stage)
        if res.launched:
            res.startup = StartupCost(seconds=12.0 if res.ok else None)
        return res


def _calibrate(hw, spec, runner, ledger) -> Profile:
    candidates, reg = select(hw, spec, force="vllm")
    plan = prepare_and_plan(hw, spec, candidates, reg)
    return calibrate(hw, spec, "throughput", plan, reg, runner=runner, ledger=ledger)


def _ledger(hw, spec, base) -> Ledger:
    return Ledger.for_calibration(hw, spec, get_workload("default"), "throughput", {}, base=base)


@pytest.mark.usefixtures("no_network")
def test_ledger_trial_record_carries_startup_seconds(tmp_path, hw_a100, spec):
    ledger = _ledger(hw_a100, spec, tmp_path / "with")
    _calibrate(hw_a100, spec, _Costed(), ledger)
    assert tmp_path in ledger.path.parents  # never the real home
    trials = [r for r in lg.read(ledger.path) if r["type"] == "trial"]
    launched = [r for r in trials if r["outcome"]["launched"]]
    assert launched and all("startup" in r for r in launched)
    assert {r["startup"]["seconds"] for r in launched if r["outcome"]["kind"] == "ok"} == {12.0}
    assert all("startup" not in r["trial"] for r in trials)  # beside the trial, never inside it

    plain = _ledger(hw_a100, spec, tmp_path / "without")
    _calibrate(hw_a100, spec, FakeRunner(), plain)
    assert all("startup" not in r for r in lg.read(plain.path))  # a runner that records none writes no key


@pytest.mark.usefixtures("no_network")
def test_ledger_rebuild_is_byte_identical_with_startup_recorded(tmp_path, hw_a100, spec):
    profile = _calibrate(hw_a100, spec, _Costed(), _ledger(hw_a100, spec, tmp_path / "a"))
    plain = _calibrate(hw_a100, spec, FakeRunner(), _ledger(hw_a100, spec, tmp_path / "b"))
    rebuilt = lg.rebuild(lg.find(profile.calibration_id, tmp_path / "a"))
    assert rebuilt.model_dump_json(indent=2) == profile.model_dump_json(indent=2)
    assert lg.verify(profile, base=tmp_path / "a").ok
    same = profile.model_copy(update={"calibration_id": plain.calibration_id, "created_at": plain.created_at})
    assert [r.config.key() for r in same.calibration_table] == [r.config.key() for r in plain.calibration_table]
    assert same.config.key() == plain.config.key()  # recording a cost changes no trial and no pick
