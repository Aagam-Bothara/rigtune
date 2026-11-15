"""The trial ledger: one line per trial, appended and never rewritten, under rigtune's home.

It records and changes nothing: a calibration with it makes the same trials and the same pick as one without, and
a profile written without it is byte for byte what it was. Concurrent writers, a crash part-way through a line, a
corrupt or newer line and a full disk are each handled without losing the other records or failing a trial.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from rigtune import ledger as lg
from rigtune.calibrate.search import StagedSearch
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.ledger import Ledger, LedgerRunner, Scan
from rigtune.models import Config, Profile, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions, calibrate, prepare_and_plan, select
from tests.test_objectives_and_search import FakeRunner

ROOT = Path(__file__).resolve().parent.parent
RECORDED = sorted((ROOT / "benchmarks" / "strategies").glob("**/profiles/**/*.json"), key=lambda p: p.as_posix())


def _plan(hw, spec):
    candidates, reg = select(hw, spec, force="vllm")
    return prepare_and_plan(hw, spec, candidates, reg), reg


def _calibrate(hw, spec, ledger=None, runner=None, **kw) -> Profile:
    plan, reg = _plan(hw, spec)
    return calibrate(hw, spec, "throughput", plan, reg, runner=runner or FakeRunner(), ledger=ledger, **kw)


def _records(path: Path, scan=None) -> List[dict]:
    return list(lg.read(path, scan))


def _sequence(results):
    return [(r.stage, r.config.key(), r.error) for r in results]


# --------------------------------------------------------------------------- what is written


@pytest.mark.usefixtures("no_network")
def test_one_line_per_trial_then_the_calibration(tmp_path, hw_a100, spec):
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    profile = _calibrate(hw_a100, spec, ledger)
    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(x) for x in lines]
    assert ledger.path.parent.parent == tmp_path and ledger.path.name == "meta-llama__Llama-3.2-3B-Instruct.jsonl"
    trials, last = records[:-1], records[-1]
    assert [r["type"] for r in trials] == ["trial"] * len(profile.calibration_table)
    assert last["type"] == "calibration" and len(lines) == len(profile.calibration_table) + 1
    assert [r["seq"] for r in records] == list(range(len(records)))
    assert {r["calibration_id"] for r in records} == {profile.calibration_id} == {ledger.calibration_id}
    assert last["trials"] == [r["id"] for r in trials]  # the table, in order
    assert [(r["stage"], r["config_key"]) for r in trials] == [(t.stage, t.config.key())
                                                               for t in profile.calibration_table]
    for rec, res in zip(trials, profile.calibration_table):
        want = json.loads(res.model_dump_json())
        want["failure_kind"] = res.failure_kind
        assert rec["trial"] == want
        assert rec["outcome"] == {"kind": "ok" if res.ok else "crashed", "launched": res.launched}  # "CUDA OOM"
        assert rec["engine"] == {"name": "vllm", "version": "1.0", "libs": {}}
        assert rec["hardware_hash"] == profile.hardware_hash and rec["model"]["id"] == spec.hf_id
        assert rec["objective"] == {"name": "throughput", "options": {}}
        assert rec["workload"]["levels"] == [1, 4, 8] and rec["workload"]["sessions"] is False
    assert last["profile"] == json.loads(profile.model_dump_json(exclude={"calibration_table"}))
    assert all(r["id"] == lg.record_id(r) for r in records)


@pytest.mark.usefixtures("no_network")
def test_the_ledger_changes_no_trial_and_no_pick(tmp_path, hw_a100, spec):
    without = _calibrate(hw_a100, spec)
    with_ = _calibrate(hw_a100, spec, Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput",
                                                             {}, base=tmp_path))
    assert _sequence(with_.calibration_table) == _sequence(without.calibration_table)
    assert with_.config == without.config and with_.options == without.options
    assert without.calibration_id is None and with_.calibration_id


@pytest.mark.usefixtures("no_network")
def test_a_profile_without_the_ledger_is_written_as_before(hw_a100, spec):
    profile = _calibrate(hw_a100, spec)
    assert "calibration_id" not in profile.model_dump_json() and "calibration_id" not in profile.model_dump()


def _kept(recorded, dumped, where="") -> List[str]:
    """Where `dumped` does not hold every value of `recorded`, recursively (fields added since may be extra)."""
    if isinstance(recorded, dict) and isinstance(dumped, dict):
        return [x for k in recorded for x in (_kept(recorded[k], dumped[k], f"{where}/{k}") if k in dumped
                                               else [f"{where}/{k} missing"])]
    if isinstance(recorded, list) and isinstance(dumped, list) and len(recorded) == len(dumped):
        return [x for i, (a, b) in enumerate(zip(recorded, dumped)) for x in _kept(a, b, f"{where}[{i}]")]
    return [] if recorded == dumped else [f"{where}: {recorded!r} != {dumped!r}"]


# Profile fields added after the recorded profiles were written; each is dumped at its default.
ADDED_SINCE = {"accuracy", "calibration_constraints", "eval_set", "model_revision", "served_concurrency"}


@pytest.mark.parametrize("path", RECORDED, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_recorded_profiles_load_and_dump_as_they_did(path):
    recorded = json.loads(path.read_text(encoding="utf-8"))
    recorded["rigtune_version"] = recorded.pop("polyserve_version", recorded.get("rigtune_version"))  # the rename
    profile = Profile.model_validate(recorded)
    dumped = json.loads(profile.model_dump_json())
    assert profile.calibration_id is None
    assert _kept(recorded, dumped) == []  # every recorded value is written back unchanged
    assert set(dumped) - set(recorded) <= ADDED_SINCE  # and the only new keys are older fields' defaults


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_an_injected_runner_writes_nothing_unless_given_a_ledger(tmp_home, hw_a100, spec):
    _calibrate(hw_a100, spec)  # tests and benchmarks/replay_search.py replay trials: nothing new was measured
    assert not (tmp_home / "ledger").exists()


@pytest.mark.usefixtures("no_network", "ledger_on")
def test_a_calibration_that_launches_engines_records_by_default(tmp_home, hw_a100, spec, monkeypatch):
    import rigtune.pipeline as pl

    fake = FakeRunner()

    class Launching:
        def __init__(self, **kwargs):
            pass

        def run(self, cfg, stage, arrival_rate=None):
            return fake.run(cfg, stage)

    monkeypatch.setattr(pl, "SubprocessTrialRunner", Launching)
    plan, reg = _plan(hw_a100, spec)
    profile = calibrate(hw_a100, spec, "throughput", plan, reg)
    found = lg.files(tmp_home / "ledger")
    assert len(found) == 1 and found[0].parent.name == profile.hardware_hash
    records = _records(found[0])
    assert records[-1]["type"] == "calibration" and records[-1]["calibration_id"] == profile.calibration_id
    monkeypatch.setenv("RIGTUNE_LEDGER", "off")
    profile = calibrate(hw_a100, spec, "throughput", plan, reg)
    assert profile.calibration_id is None and len(_records(found[0])) == len(records)


def test_where_it_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGTUNE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("RIGTUNE_LEDGER", raising=False)
    monkeypatch.delenv("POLYSERVE_LEDGER", raising=False)
    assert lg.root() == tmp_path / "home" / "ledger"
    monkeypatch.setenv("POLYSERVE_LEDGER", str(tmp_path / "old"))  # the former name, as every variable
    assert lg.root() == tmp_path / "old"
    monkeypatch.setenv("RIGTUNE_LEDGER", str(tmp_path / "new"))
    assert lg.root() == tmp_path / "new"
    for off in ("off", "OFF", "0", "false", "No", "none", "NONE", "disabled", " Disabled "):
        monkeypatch.setenv("RIGTUNE_LEDGER", off)
        assert lg.root() is None  # never a directory called "none" under the working directory
    monkeypatch.setenv("RIGTUNE_LEDGER", "")
    monkeypatch.delenv("POLYSERVE_LEDGER")
    assert lg.root() == tmp_path / "home" / "ledger"  # empty is unset, not off


@pytest.mark.usefixtures("no_network")
def test_the_suite_never_writes_to_the_real_home(tmp_path, hw_a100, spec, monkeypatch):
    """tests/test_idle_tail.py drives calibrate() through the real runner, and appended to the developer's own ledger
    until tests/conftest.py turned the ledger off for every test. Run it with the home moved here: were the ledger
    on, its calibration would land under this home's .rigtune/ledger. (conftest.py also fails the whole run if the
    real home's ledger changed while it ran.)"""
    from tests.test_idle_tail import _calibrate as through_the_real_runner

    assert lg.root() is None
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.delenv("RIGTUNE_HOME", raising=False)
    monkeypatch.delenv("POLYSERVE_HOME", raising=False)
    through_the_real_runner("throughput", hw_a100, spec, monkeypatch)
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_a_model_id_cannot_leave_the_ledger_directory(tmp_path):
    assert lg.file_name("Qwen/Qwen2.5-7B-Instruct") == "Qwen__Qwen2.5-7B-Instruct.jsonl"
    for local in ("C:\\Users\\me\\models\\llama", "/home/me/models/llama", "~/models/llama", "\\\\srv\\me\\llama",
                  "../../etc/x"):
        name = lg.file_name(local)
        path = lg.ledger_path(tmp_path, "abc", local)
        assert path.parent.parent == tmp_path and path.name == name and "/" not in name and "\\" not in name
        assert "me" not in name.replace("-", "_").split("_")  # a local folder is named by its last part only
    assert lg.file_name("/home/me/models/llama").startswith("local_llama-")
    assert lg.file_name("/a/b_c") != lg.file_name("/a_b/c")  # cut to the same characters, told apart by the hash


@pytest.mark.usefixtures("no_network")
def test_no_prompt_text_is_recorded(tmp_path, hw_a100, spec):
    marker = "zqxv-prompt-text-marker"
    wl = Workload(name="marked", shared_prefix_tokens=16, prefix_text=f"{marker} " * 8,
                  prompts=[f"{marker} {i}" for i in range(16)])
    ledger = Ledger.for_calibration(hw_a100, spec, wl, "throughput", {}, base=tmp_path)
    _calibrate(hw_a100, spec, ledger, workload=wl)
    text = ledger.path.read_text(encoding="utf-8")
    assert marker not in text
    other = Workload(name="marked", shared_prefix_tokens=16, prefix_text="other text " * 8)
    assert lg.workload_summary(other)["fingerprint"] != lg.workload_summary(wl)["fingerprint"]


def test_a_file_workload_is_fingerprinted_without_its_path():
    a = Workload(name="file-x-1234", source="file:/home/me/a/prompts.jsonl")
    b = Workload(name="file-x-1234", source="file:C:\\Users\\other\\prompts.jsonl")
    assert lg.workload_summary(a) == lg.workload_summary(b)
    assert "home" not in json.dumps(lg.workload_summary(a))


@pytest.mark.usefixtures("no_network")
def test_budget_skipped_trials_are_recorded_and_left_out_of_the_table(tmp_path, hw_a100, spec, monkeypatch):
    import rigtune.pipeline as pl

    t = [0.0]  # every trial takes 60 s on a fake clock

    class Slow(FakeRunner):
        def run(self, cfg, stage):
            t[0] += 60.0
            return super().run(cfg, stage)

    monkeypatch.setattr(pl, "StagedSearch", lambda **kw: StagedSearch(clock=lambda: t[0], **kw))
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {"budget": "300s"},
                                    base=tmp_path)
    plan, reg = _plan(hw_a100, spec)
    profile = calibrate(hw_a100, spec, "throughput", plan, reg, runner=Slow(), ledger=ledger,
                        options=SearchOptions(budget_s=300))
    records = _records(ledger.path)
    skipped = [r for r in records if r["type"] == "trial" and r["outcome"]["kind"] == "skipped_budget"]
    assert skipped and any("calibration budget" in n for n in profile.notes)
    table_ids = set(records[-1]["trials"])
    assert not table_ids & {r["id"] for r in skipped}
    assert len(table_ids) == len(profile.calibration_table)
    assert all(r["trial"]["error"] == "skipped: calibration budget reached" for r in skipped)


def test_the_runner_wrapper_keeps_what_the_runner_has(tmp_path, hw_a100, spec):
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    plain = LedgerRunner(FakeRunner(), ledger)
    assert getattr(plain, "sweep", None) is None and getattr(plain, "run_open_loop", None) is None
    assert plain.calls == []  # anything else is the wrapped runner's

    class Full(FakeRunner):
        gate = None

        def sweep(self, base, settings, stage, progress=None):
            return [self.run(base, stage) for _ in settings]

        def run_open_loop(self, cfg, rate, stage="rate-check"):
            return self.run(cfg, stage)

    full = LedgerRunner(Full(), ledger)
    cfg = Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9)
    res = full.run(cfg, "quant")
    assert full.sweep(cfg, [1, 2], "power") and full.run_open_loop(cfg, 1.0).stage == "rate-check"
    assert full.gate is None
    stages = [r["stage"] for r in _records(ledger.path)]
    assert stages == ["quant", "power", "power", "rate-check"]
    assert res.failure_kind is None and isinstance(res, TrialResult)


# --------------------------------------------------------------------------- robustness

_WRITER = """
import sys
from pathlib import Path
from rigtune.ledger import append, record_id
path, who, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
for i in range(n):
    rec = {"schema": 1, "type": "trial", "calibration_id": who, "seq": i, "pad": who * 4000}
    rec["id"] = record_id(rec)
    append(path, rec)
"""


def test_concurrent_writers_never_interleave(tmp_path):
    """Twelve writers at once. With Windows' blocking lock (one try a second, ten tries) three appends raised and
    their records were lost; polling with a deadline of a minute loses none, however loaded the machine."""
    path = tmp_path / "l.jsonl"
    writers, n = 12, 15
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(path), f"w{k}", str(n)], cwd=str(ROOT))
             for k in range(writers)]
    assert [p.wait(timeout=600) for p in procs] == [0] * writers
    scan = Scan()
    records = _records(path, scan)
    assert scan.unreadable == 0 and len(records) == writers * n
    for k in range(writers):
        assert [r["seq"] for r in records if r["calibration_id"] == f"w{k}"] == list(range(n))
    assert len(path.read_bytes().splitlines()) == writers * n


_HOLDER = """
import sys, time
from pathlib import Path
from rigtune.ledger import _locked
with _locked(Path(sys.argv[1])):
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
"""


def _hold(path: Path, seconds: float) -> subprocess.Popen:
    """Another process holding the ledger's lock for `seconds`; returns once it holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(path), str(seconds)], cwd=str(ROOT),
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_an_append_waits_out_a_lock_held_past_ten_seconds(tmp_path):
    """Windows' LK_LOCK gave up after ten one-second tries, so a lock held for eleven seconds made the append raise
    and its record was lost. It waits for up to LOCK_TIMEOUT_S now."""
    path = tmp_path / "l.jsonl"
    holder = _hold(path, 11)
    _write(path, 1)
    assert holder.wait(timeout=60) == 0
    assert [r["seq"] for r in _records(path)] == [0]


def test_an_append_gives_up_at_its_deadline(tmp_path, monkeypatch):
    """A writer that hung holding the lock costs the others their record, logged by the ledger, not the calibration."""
    path = tmp_path / "l.jsonl"
    monkeypatch.setattr(lg, "LOCK_TIMEOUT_S", 0.2)
    holder = _hold(path, 30)
    try:
        with pytest.raises(TimeoutError, match="not granted"):
            _write(path, 1)
    finally:
        holder.kill()
        holder.wait(timeout=60)
    assert _records(path) == []


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX file modes")
def test_the_ledgers_folders_and_files_are_this_users_only(tmp_path):
    base = tmp_path / "ledger"
    path = lg.ledger_path(base, "abc", "Qwen/Qwen2.5-7B-Instruct")
    _write(path, 1)
    for p in (base, path.parent, path, path.with_name(path.name + ".lock")):
        assert p.stat().st_mode & 0o077 == 0, p


def _write(path: Path, n: int, who: str = "c1") -> None:
    for i in range(n):
        rec = {"schema": 1, "type": "trial", "calibration_id": who, "seq": i}
        rec["id"] = lg.record_id(rec)
        lg.append(path, rec)


def test_a_line_cut_short_costs_only_itself(tmp_path):
    path = tmp_path / "l.jsonl"
    _write(path, 3)
    with path.open("ab") as fh:
        fh.write(b'{"schema": 1, "type": "trial", "calibration_id": "c1", "se')  # a crash mid-write
    _write(path, 2, who="c2")
    scan = Scan()
    records = _records(path, scan)
    assert [(r["calibration_id"], r["seq"]) for r in records] == [("c1", 0), ("c1", 1), ("c1", 2), ("c2", 0),
                                                                   ("c2", 1)]
    assert scan.unreadable == 1 and scan.problems()


def test_a_truncated_last_line_is_skipped(tmp_path):
    path = tmp_path / "l.jsonl"
    _write(path, 3)
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    scan = Scan()
    assert [r["seq"] for r in _records(path, scan)] == [0, 1] and scan.unreadable == 1


def test_a_changed_record_and_a_newer_schema_are_skipped(tmp_path):
    path = tmp_path / "l.jsonl"
    _write(path, 2)
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = json.loads(lines[1])
    changed["seq"] = 7  # still valid JSON, but no longer what was written
    newer = {"schema": lg.SCHEMA + 1, "type": "trial", "calibration_id": "c1", "seq": 2, "new": True}
    newer["id"] = lg.record_id(newer)
    invalid = []
    for schema in (0, -5, True):  # no schema below 1 was ever written: never read as the current one
        rec = {"schema": schema, "type": "trial", "calibration_id": "c1", "seq": 3}
        rec["id"] = lg.record_id(rec)
        invalid.append(json.dumps(rec))
    path.write_text("\n".join([lines[0], json.dumps(changed), json.dumps(newer), "[1, 2]", "", "{}", *invalid])
                    + "\n", encoding="utf-8", newline="\n")
    scan = Scan()
    assert [r["seq"] for r in _records(path, scan)] == [0]
    assert (scan.unreadable, scan.newer) == (6, 1)


def test_reading_one_calibration_skips_the_others(tmp_path):
    path = tmp_path / "l.jsonl"
    _write(path, 3, who="aaaa")
    _write(path, 2, who="bbbb")
    assert [r["seq"] for r in lg.read(path, calibration_id="bbbb")] == [0, 1]
    assert list(lg.read(tmp_path / "absent.jsonl")) == []


@pytest.mark.usefixtures("no_network")
def test_a_write_that_fails_never_fails_a_trial(tmp_path, hw_a100, spec, monkeypatch, caplog):
    def full(path, record):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(lg, "append", full)
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    with caplog.at_level(logging.WARNING, logger="rigtune.ledger"):
        profile = _calibrate(hw_a100, spec, ledger)
    assert profile.config.quant == "fp8" and len(profile.calibration_table) > 3
    warnings = [r for r in caplog.records if r.name == "rigtune.ledger"]
    assert len(warnings) == 1 and "No space left" in warnings[0].getMessage()  # said once, not per trial
    assert ledger.failed_writes == len(profile.calibration_table) + 1
    # No record names it, so the profile names no calibration: it is written exactly as without the ledger.
    assert profile.calibration_id is None and "calibration_id" not in profile.model_dump_json()


@pytest.mark.usefixtures("no_network")
def test_a_profile_whose_record_was_not_written_names_no_calibration(tmp_path, hw_a100, spec, monkeypatch):
    real = lg.append

    def no_profile_record(path, record):
        if record["type"] == "calibration":
            raise PermissionError(13, "read-only home")
        real(path, record)

    monkeypatch.setattr(lg, "append", no_profile_record)
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    profile = _calibrate(hw_a100, spec, ledger)
    assert [r["type"] for r in _records(ledger.path)] == ["trial"] * len(profile.calibration_table)
    assert profile.calibration_id is None and "calibration_id" not in profile.model_dump_json()


def test_a_skip_hook_is_optional():
    search = StagedSearch(objective="throughput", runner=FakeRunner())
    assert search.on_skip is None
    seen: List[TrialResult] = []
    t = [0.0]
    search = StagedSearch(objective="throughput", runner=FakeRunner(), budget_s=10, clock=lambda: t[0],
                          on_skip=seen.append)
    cfg = Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9)
    search._run(cfg, "quant")
    t[0] = 100.0
    res = search._run(cfg.model_copy(update={"batch": 8}), "batch")
    assert seen == [res] and res.failure_kind == "skipped_budget" and search.results[-1] is not res
    assert search._remeasure(cfg) is None and seen[-1].stage == "confirm"
    assert seen[-1].failure_kind == "skipped_budget" and seen[-1].metrics == TrialMetrics()
