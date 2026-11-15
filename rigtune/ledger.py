"""The trial ledger: every calibration trial, appended to a file under rigtune's home as soon as it ends.

A profile keeps the pick and the trials behind it, and the next calibration of the same request replaces it. The
ledger keeps every trial of every calibration, the failed and budget-skipped ones included, so that a pick can be
audited later, and a profile that was lost or replaced can be rebuilt from it exactly.

Where: <home>/ledger/<hardware_hash>/<model>.jsonl, a local model by its folder's last part. RIGTUNE_LEDGER set to off,
0, false, no, none or disabled (in any case) turns it off, and RIGTUNE_LEDGER=<dir> moves it (POLYSERVE_LEDGER is read
too, as every rigtune variable is). Only calibrations that launch engines write to it: a caller that passes its own
runner (tests, benchmarks/replay_search.py replaying recorded trials) measures nothing new and writes nothing unless
it passes a Ledger. The test suite turns it off for every test (tests/conftest.py).

Format: one JSON object per line, appended and never rewritten. Each record carries `schema` (SCHEMA) and `id`, the
sha256 of the record without its id, so a line cut short or changed by accident is found and skipped. The id is no
defence against tampering: anyone who can write the file can rewrite a record and its id together. A reader skips
records of a later schema, and of a schema below 1, and says how many it skipped. Two types:
- "trial", after each trial (and each one the search skipped for its budget): the TrialResult as the profile holds
  it plus its failure_kind, its place in the calibration (`seq`), and the hardware, model, engine, workload and
  objective it was measured under, so trials from different calibrations can be compared.
- "calibration", once the pick is made: the profile without its table, and the ids of the trial records that make up
  the table, in order. A calibration that failed or was stopped leaves its trials and no such record.

Writing: one write per record, under an exclusive lock on <file>.lock, then fsync. The lock is needed because two
calibrations can run at once, and on Windows O_APPEND is the C runtime's seek-then-write, which two processes can
interleave. The lock is polled every LOCK_POLL_S for up to LOCK_TIMEOUT_S. A line a crash left without its
newline is closed off before the next record, so it costs only itself. A write that fails (a full disk, a lock not
granted) is logged once and never fails a trial: calibration does not need the ledger, and losing the measurement to
its audit trail would be the wrong way round. Files are created 0600 and the ledger's folders 0700 on POSIX.

What it holds is what the profile cache holds already (the hardware, the model and engine, launch arguments and the
launch environment rigtune sets, each trial's metrics and error text) and nothing more: no prompt or answer text, no
variable from the process environment, no host name. Nothing rigtune publishes reads it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from rigtune import __version__, paths
from rigtune.models import HardwareDescriptor, ModelSpec, Profile, TrialResult

logger = logging.getLogger(__name__)

SCHEMA = 1
OFF = ("off", "0", "false", "no", "none", "disabled")  # RIGTUNE_LEDGER values that turn it off, in any case
# How long an append waits for another writer's lock before it gives up (the trial goes on without its record), and
# how often it tries meanwhile. One append holds the lock for one write and an fsync: milliseconds.
LOCK_TIMEOUT_S = 60.0
LOCK_POLL_S = 0.02


def root() -> Optional[Path]:
    """The ledger's directory, or None when RIGTUNE_LEDGER turns it off. A value that turns it off is never read
    as a directory of that name."""
    given = (paths.env("LEDGER") or "").strip()
    if given.lower() in OFF:
        return None
    return Path(given) if given else paths.home() / "ledger"


_LOCAL = re.compile(r"^(?:[/~\\]|[A-Za-z]:)|\\")  # a POSIX, home-relative, Windows or UNC path, as an id may be


def file_name(model_id: str) -> str:
    """The model's file name: its Hub id with `/` as `__`, as the profile cache names it. An id that is a local
    directory is named by its last part, since the folders above it can carry a user name; that, or an id holding
    anything a file name cannot, is cut to safe characters with a hash of the whole id after it, so it can neither
    leave the ledger's directory nor share a file with another model."""
    safe = ModelSpec(hf_id=model_id).safe_id
    local = bool(_LOCAL.search(model_id))
    if local:
        safe = "local_" + ([x for x in re.split(r"[\\/]+", model_id) if x] or ["model"])[-1]
    clean = re.sub(r"[^A-Za-z0-9._=+-]+", "_", safe).strip("._")[:80] or "model"
    if local or clean != safe:
        clean += "-" + hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
    return clean + ".jsonl"


def ledger_path(base: Path, hardware_hash: str, model_id: str) -> Path:
    return base / re.sub(r"[^A-Za-z0-9]+", "_", hardware_hash) / file_name(model_id)


def _canonical(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def record_id(record: Dict[str, Any]) -> str:
    """sha256 of the record without its id, over its canonical JSON (keys sorted), so it does not depend on the order
    the record was built in and survives a read and a write unchanged."""
    return hashlib.sha256(_canonical({k: v for k, v in record.items() if k != "id"}).encode("utf-8")).hexdigest()


def _make_dirs(folder: Path) -> None:
    """The file's folder and the ledger folder above it, created readable by this user only (0700 on POSIX, less any
    umask; Windows ignores the mode). Folders above those, such as rigtune's home, are created with the default mode."""
    folder.parent.parent.mkdir(parents=True, exist_ok=True)
    for d in (folder.parent, folder):
        d.mkdir(mode=0o700, exist_ok=True)


def _acquire(try_lock) -> None:
    """Call try_lock (which raises OSError while another holds the lock) every LOCK_POLL_S until it succeeds, or
    raise TimeoutError after LOCK_TIMEOUT_S. Windows' blocking LK_LOCK retries once a second and gives up after ten
    tries: with twelve writers, three appends raised and their records were lost. A blocking flock on POSIX would
    instead wait forever on a writer that hung while holding it. Polling often, with one generous deadline, is the
    same on both."""
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            try_lock()
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"the ledger's lock was not granted within {LOCK_TIMEOUT_S:g} s") from None
            time.sleep(LOCK_POLL_S)


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """An exclusive lock on <path>.lock for one append, between processes and threads alike."""
    fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            _acquire(lambda: msvcrt.locking(fd, msvcrt.LK_NBLCK, 1))  # one byte from offset 0
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            _acquire(lambda: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB))
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def append(path: Path, record: Dict[str, Any]) -> None:
    """Append one record as one line: one write under the lock, then fsync. Raises OSError on failure."""
    # Keys in the order they were built, not sorted: a profile's dicts (its engines, its options) keep their order,
    # so the profile rebuilt from the record is written byte for byte as it was. The id does not depend on it.
    data = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
    _make_dirs(path.parent)
    with _locked(path):
        fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        try:
            end = os.lseek(fd, 0, os.SEEK_END)
            if end:
                os.lseek(fd, end - 1, os.SEEK_SET)
                if os.read(fd, 1) != b"\n":  # a crash cut the last record short: keep it on a line of its own
                    data = b"\n" + data
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)


# --------------------------------------------------------------------------- reading


@dataclass
class Scan:
    """What a read of a ledger file skipped, beside the records it gave."""

    unreadable: int = 0  # not a JSON object, no schema, or an id that does not match the content
    newer: int = 0  # records of a later schema than this rigtune reads

    def problems(self) -> List[str]:
        out = []
        if self.unreadable:
            out.append(f"{self.unreadable} unreadable line{'s' if self.unreadable != 1 else ''} skipped (cut short "
                       "or changed after it was written)")
        if self.newer:
            out.append(f"{self.newer} record{'s' if self.newer != 1 else ''} of a later schema skipped (written by "
                       "a newer rigtune)")
        return out


def read(path: Path, scan: Optional[Scan] = None, calibration_id: Optional[str] = None,
         contains: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """The intact records of a ledger file, in file order, one line at a time, so a file of any size is read in
    constant memory. With `calibration_id`, only that calibration's; with `contains`, only lines holding that text.
    Lines without the text asked for are not even parsed, so `scan` then counts only the problems among the rest."""
    scan = scan if scan is not None else Scan()
    if not path.exists():
        return
    text = calibration_id or contains
    needle = text.encode("utf-8") if text else None
    with path.open("rb") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or (needle is not None and needle not in line):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                scan.unreadable += 1
                continue
            schema = rec.get("schema") if isinstance(rec, dict) else None
            if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:  # schema 1 is the first
                scan.unreadable += 1
            elif schema > SCHEMA:
                scan.newer += 1
            elif rec.get("id") != record_id(rec):
                scan.unreadable += 1
            elif calibration_id is None or rec.get("calibration_id") == calibration_id:
                yield rec


def files(base: Optional[Path] = None) -> List[Path]:
    """Every ledger file under `base` (the ledger's directory by default), in a fixed order."""
    base = base if base is not None else root()
    if base is None or not base.is_dir():
        return []
    return sorted(base.glob("*/*.jsonl"), key=lambda p: p.as_posix())


# --------------------------------------------------------------------------- writing


_TEXT_SETTINGS = ("prefix_text", "session_system")  # text a workload sends: in the fingerprint by hash only
_STATE = ("fitted", "prefix_fixed")  # whether prompts were sized yet, not what was asked for


def workload_summary(workload: Any) -> Dict[str, Any]:
    """The workload's shape without its text: name, levels, whether it replays sessions, answer length, and a
    fingerprint over its scalar settings. A file's path is cut to its name; a shared prefix or a replacing system
    prompt counts by its hash.

    The prompts' own digests are not in it, as the plan had them, because they are not there to read: a file or
    named-dataset workload loads its prompts one concurrency level at a time inside the trial, and loading them here
    would change what the first level measures (calibration's quality probe was once caught doing exactly that). What
    the fingerprint covers pins them down all the same: a file workload's name carries a hash of the file's bytes
    (the first 8 hex digits of its sha256, calibrate/workload.py), synthetic prompts follow from the seed and the settings, and a named dataset's sample
    from its name, seed and offset, for as long as the dataset itself does not change."""
    settings: Dict[str, Any] = {"concurrencies": list(workload.concurrencies)}
    for f in dataclasses.fields(workload):
        value = getattr(workload, f.name)
        if f.name in _TEXT_SETTINGS:
            value = hashlib.sha256(value.encode("utf-8")).hexdigest() if value else ""
        elif f.name == "source":
            value = re.split(r"[\\/]", value)[-1]
        if f.name not in _STATE and (value is None or isinstance(value, (bool, int, float, str))):
            settings[f.name] = value
    return {"name": workload.name, "levels": list(workload.concurrencies), "sessions": bool(workload.sessions),
            "decode_tokens": workload.decode_tokens, "natural_stop": bool(workload.natural_stop),
            "fingerprint": hashlib.sha256(_canonical(settings).encode("utf-8")).hexdigest()}


def _card(hw: HardwareDescriptor) -> Dict[str, Any]:
    return {"gpus": [{"name": g.name, "cc": list(g.compute_capability) if g.compute_capability else None,
                      "vram_mib": g.vram_total_bytes // 2**20, "driver": g.driver_version, "cuda": g.cuda_version}
                     for g in hw.gpus],
            "cpu": hw.cpu.model_name, "ram_gib": round(hw.cpu.ram_total_bytes / 2**30)}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Ledger:
    """One calibration's writer: its id, the file, and what every trial record of it shares."""

    path: Path
    hw: HardwareDescriptor
    context: Dict[str, Any] = field(default_factory=dict)
    calibration_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seq: int = 0
    failed_writes: int = 0
    _recorded: List[Tuple[TrialResult, Optional[str]]] = field(default_factory=list)  # matched by identity

    @classmethod
    def for_calibration(cls, hw: HardwareDescriptor, spec: ModelSpec, workload: Any, objective: str,
                        options: Optional[Dict[str, str]], base: Optional[Path] = None) -> Optional["Ledger"]:
        """A ledger for a calibration about to start, under `base` (by default root()), or None when it is off."""
        from rigtune.hardware import hardware_hash

        base = base if base is not None else root()
        if base is None:
            return None
        hh = hardware_hash(hw)
        context = {"rigtune": __version__, "hardware_hash": hh, "card": _card(hw),
                   "model": {"id": spec.hf_id, "revision": spec.revision},
                   "workload": workload_summary(workload),
                   "objective": {"name": objective, "options": dict(sorted((options or {}).items()))}}
        return cls(path=ledger_path(base, hh, spec.hf_id), hw=hw, context=context)

    def _append(self, record: Dict[str, Any]) -> Optional[str]:
        record = {"schema": SCHEMA, "calibration_id": self.calibration_id, "seq": self.seq, **record}
        self.seq += 1
        record["id"] = record_id(record)
        try:
            append(self.path, record)
        except Exception as exc:  # a full disk, a lock not granted, a read-only home: never the trial's problem
            self.failed_writes += 1
            if self.failed_writes == 1:
                logger.warning("trial ledger %s could not be written (%s); the calibration goes on without it",
                               self.path, exc)
            return None
        return record["id"]

    def trial(self, res: TrialResult, duration_s: Optional[float] = None) -> None:
        """Record one trial as it ended (or as the search skipped it)."""
        from rigtune.calibrate.search import failure_kind

        backend = self.hw.backends.get(res.config.backend)
        kind = failure_kind(res)
        try:
            trial = res.model_dump(mode="json")
        except Exception as exc:  # never the trial's problem either
            logger.warning("trial %s not recorded in the ledger: %s", res.config.key(), exc)
            return
        trial["failure_kind"] = res.failure_kind
        rid = self._append({
            "type": "trial", "recorded_at": _now(),
            "duration_s": round(duration_s, 3) if duration_s is not None else None,
            **self.context,
            "engine": {"name": res.config.backend, "version": backend.version if backend else None,
                       "libs": dict(sorted(backend.libs.items())) if backend else {}},
            "stage": res.stage, "config_key": res.config.key(),
            "outcome": {"kind": kind or "ok", "launched": res.launched},
            "trial": trial,
        })
        self._recorded.append((res, rid))

    def _seen(self, res: TrialResult) -> Tuple[bool, Optional[str]]:
        """(whether this result object was passed to trial(), the id of its record or None if it was not written)."""
        return next(((True, rid) for r, rid in self._recorded if r is res), (False, None))

    def finish(self, profile: Profile) -> bool:
        """Record the calibration's pick: the profile without its table, and the trial records that make up the
        table, in order (None for one whose write failed). A table entry the ledger never saw (it should see every
        one) is recorded now. The profile is given this calibration's id only once that record is written, so a
        profile whose record could not be written names no calibration and is written exactly as without the
        ledger. Returns whether it was written."""
        ids: List[Optional[str]] = []
        for res in profile.calibration_table:
            if not self._seen(res)[0]:
                self.trial(res)
            ids.append(self._seen(res)[1])
        profile.calibration_id = self.calibration_id
        rid = self._append({"type": "calibration", "recorded_at": _now(), "rigtune": __version__,
                            "hardware_hash": profile.hardware_hash, "model": {"id": profile.model_id},
                            "trials": ids, "profile": profile.model_dump(mode="json", exclude={"calibration_table"})})
        if rid is None:
            profile.calibration_id = None
        return rid is not None


class LedgerRunner:
    """A TrialRunner that records each trial in `ledger` as it ends. Everything else is the wrapped runner's own: the
    results are returned untouched, and a runner without a power sweep or an open-loop run still has none."""

    def __init__(self, inner: Any, ledger: Ledger):
        self.inner = inner
        self.ledger = ledger
        if hasattr(inner, "sweep"):
            self.sweep = self._sweep
        if hasattr(inner, "run_open_loop"):
            self.run_open_loop = self._run_open_loop

    def __getattr__(self, name: str) -> Any:  # only for what this class does not define itself
        if name in ("inner", "ledger"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def run(self, cfg, stage: str, **kwargs) -> TrialResult:
        t0 = time.monotonic()
        res = self.inner.run(cfg, stage, **kwargs)
        self.ledger.trial(res, time.monotonic() - t0)
        return res

    def _run_open_loop(self, cfg, rate: float, stage: str = "rate-check") -> TrialResult:
        t0 = time.monotonic()
        res = self.inner.run_open_loop(cfg, rate, stage)
        self.ledger.trial(res, time.monotonic() - t0)
        return res

    def _sweep(self, base, settings, stage: str, progress=None) -> List[TrialResult]:
        results = self.inner.sweep(base, settings, stage, progress=progress)
        for res in results:  # one launch for every point: no duration of its own
            self.ledger.trial(res)
        return results


# --------------------------------------------------------------------------- calibrations: show, verify, rebuild


class LedgerError(ValueError):
    """A calibration the ledger cannot give: not found, ambiguous, or without its profile record."""


@dataclass
class Calibration:
    """One calibration's records: its trials in the order they ran, and its calibration record if it has one."""

    calibration_id: str
    path: Path
    trials: List[Dict[str, Any]] = field(default_factory=list)
    record: Optional[Dict[str, Any]] = None
    scan: Scan = field(default_factory=Scan)

    def table_records(self) -> List[Dict[str, Any]]:
        """The trial records that make up the profile's table, in its order. Raises LedgerError for a calibration
        without its profile record or with any of those trials missing."""
        if self.record is None:
            raise LedgerError(f"calibration {self.calibration_id} has no profile record: it failed or was stopped "
                              f"before its pick ({len(self.trials)} trials recorded)")
        by_id = {r["id"]: r for r in self.trials}
        ids = self.record.get("trials") or []
        missing = sum(1 for i in ids if i not in by_id)
        if missing:
            raise LedgerError(f"calibration {self.calibration_id}: {missing} of the {len(ids)} trials in its table "
                              "are not in the ledger (a write failed, or the line was cut short or changed)"
                              + (f"; {'; '.join(self.scan.problems())}" if self.scan.problems() else ""))
        return [by_id[i] for i in ids]


def calibrations(path: Path, scan: Optional[Scan] = None) -> List[Calibration]:
    """Every calibration in one ledger file, in the order each began, read in one pass."""
    scan = scan if scan is not None else Scan()
    found: Dict[str, Calibration] = {}
    for rec in read(path, scan):
        cid = rec.get("calibration_id")
        if not isinstance(cid, str):
            scan.unreadable += 1
            continue
        cal = found.setdefault(cid, Calibration(calibration_id=cid, path=path, scan=scan))
        if rec.get("type") == "calibration":
            cal.record = rec
        elif rec.get("type") == "trial":
            cal.trials.append(rec)
    for cal in found.values():
        cal.trials.sort(key=lambda r: r.get("seq", 0))
    return list(found.values())


def find(calibration_id: str, base: Optional[Path] = None) -> Calibration:
    """The calibration whose id is `calibration_id`, or starts with it (at least 6 characters), looked for in every
    ledger file under `base`. Raises LedgerError when there is none, or more than one, in whichever files they are."""
    if len(calibration_id) < 6:
        raise LedgerError("give at least the first 6 characters of a calibration id")
    found: Dict[Tuple[str, Path], Calibration] = {}
    for path in files(base):
        scan = Scan()
        for rec in read(path, scan, contains=calibration_id):
            cid = rec.get("calibration_id")
            if isinstance(cid, str) and cid.startswith(calibration_id):
                cal = found.setdefault((cid, path), Calibration(calibration_id=cid, path=path, scan=scan))
                if rec.get("type") == "calibration":
                    cal.record = rec
                elif rec.get("type") == "trial":
                    cal.trials.append(rec)
    if len(found) > 1:
        raise LedgerError(f"{calibration_id} is the start of {len(found)} calibrations' ids; give more of it")
    if not found:
        raise LedgerError(f"no calibration {calibration_id} in the ledger")
    cal = next(iter(found.values()))
    cal.trials.sort(key=lambda r: r.get("seq", 0))
    return cal


def rebuild(cal: Calibration) -> Profile:
    """The profile the calibration wrote, from its profile record and its trials: the same profile, field for
    field, with the same cache path, as long as its records are intact."""
    data = dict(cal.record or {}).get("profile")
    table = [r["trial"] for r in cal.table_records()]
    return Profile.model_validate({**data, "calibration_table": table})


def _median_runs(runs: List[TrialResult], objective: str, cons: Any) -> Optional[TrialResult]:
    """A configuration's representative among its confirm runs, as search.stage_confirm takes it: the middle of its
    successful runs best first, the worse of the middle two for an even count."""
    from rigtune.calibrate.objectives import rank

    order = rank(runs, objective, cons)
    return order[len(order) // 2].result if order else None


def rederive_pick(profile: Profile) -> Tuple[Optional[TrialResult], Optional[float], str]:
    """(the trial the pick comes from, its score, how it was chosen) from the profile's own table, objective and
    constraints, by the rule the search ends with: objectives.pick over the confirm stage's representative run of
    each re-measured configuration (with the power stage's runs after it), or over every trial when nothing was
    re-measured; precisions an accuracy gate refused are never eligible. (None, None, why) when the table alone
    cannot settle it: a gated calibration chooses among graded configurations, which the table does not show."""
    from rigtune.calibrate.objectives import pick, rank
    from rigtune.tradeoff import profile_constraints

    if profile.eval_set is not None:
        return None, None, "an accuracy-gated calibration chooses among graded configurations; not re-derived"
    cons = profile_constraints(profile)
    refused = {(a.backend, a.quant) for a in profile.accuracy if a.admitted is False}
    eligible = [r for r in profile.calibration_table if (r.config.backend, r.config.quant) not in refused]
    confirm: Dict[str, List[TrialResult]] = {}
    for r in eligible:
        if r.stage == "confirm":
            confirm.setdefault(r.config.key(), []).append(r)
    if confirm:
        pool = [m for runs in confirm.values() if (m := _median_runs(runs, profile.objective, cons)) is not None]
        pool += [r for r in eligible if r.stage == "power"]
        how = f"the median re-measured run of each of {len(confirm)} confirmed configurations"
    else:
        pool, how = eligible, f"every one of the {len(eligible)} trials"
    winner, _ = pick(pool, profile.objective, cons)
    if winner is None:
        return None, None, "no successful trial to pick from"
    score = next(x.score for x in rank(pool, profile.objective, cons) if x.result is winner)
    return winner, score, how


OK, DIFFERS, NOT_CHECKED = "ok", "DIFFERS", "NOT CHECKED"
# `rigtune ledger verify` exits with: 0 every check passed, 1 something differs, 2 nothing to check against (no
# calibration id, the ledger off, the calibration not found or incomplete), 3 nothing differs but something could not
# be checked. A check that cannot be made is never reported as passed.
EXIT_OK, EXIT_DIFFERS, EXIT_NOT_FOUND, EXIT_NOT_CHECKED = 0, 1, 2, 3
# What --layout (layout.calibrate_layout) and --phases (disagg.calibrate_disaggregated, disagg._fallback) leave as the
# single-engine calibration had it, so a profile they built must still hold it.
_KEPT_BY_DERIVED = ("rigtune_version", "hardware_hash", "hardware", "model_id", "model_revision",
                    "calibration_constraints", "objective", "workload", "workload_spec", "backend", "backend_version",
                    "prepared", "prepared_all", "launch_env", "llmtrace_version", "created_at", "accuracy", "eval_set",
                    "calibration_id")
_DERIVED_OPTIONS = ("layout", "tier")  # options a layout adds to the calibration's (pipeline._resolve_layout)


@dataclass
class Verification:
    calibration_id: Optional[str]
    findings: List[Tuple[str, str]] = field(default_factory=list)  # (OK | DIFFERS | NOT_CHECKED, what)
    found: bool = True  # whether there was a calibration to check against

    @property
    def ok(self) -> bool:
        return bool(self.findings) and all(status == OK for status, _ in self.findings)

    @property
    def exit_code(self) -> int:
        if not self.found:
            return EXIT_NOT_FOUND
        if any(status == DIFFERS for status, _ in self.findings):
            return EXIT_DIFFERS
        return EXIT_OK if self.ok else EXIT_NOT_CHECKED

    def add(self, passed: bool, text: str) -> None:
        self.findings.append((OK if passed else DIFFERS, text))

    def unchecked(self, text: str) -> None:
        self.findings.append((NOT_CHECKED, text))

    def missing(self, text: str) -> "Verification":
        self.found = False
        self.findings.append((DIFFERS, text))
        return self


def _same_trials(mine: List[TrialResult], theirs: List[TrialResult]) -> Optional[int]:
    """None when `mine` starts with `theirs`, trial for trial; else the 1-based trial where it stops doing so."""
    for i, b in enumerate(theirs):
        if i >= len(mine) or mine[i].model_dump_json() != b.model_dump_json():
            return i + 1
    return None


def verify(profile: Profile, base: Optional[Path] = None) -> Verification:
    """Check a profile against the ledger: its calibration's records are intact and complete, the profile they
    rebuild is this one (its table trial for trial), the trial count matches, and the pick re-derived from the
    table is the profile's configuration. A --layout or --phases profile is checked against the single-engine
    calibration it was built on, as far as that can be (_verify_derived). What cannot be checked is reported as
    NOT_CHECKED, never as passed."""
    out = Verification(calibration_id=profile.calibration_id)
    if profile.calibration_id is None:
        return out.missing("the profile names no calibration: it was calibrated without the ledger, before it "
                           "existed, or its record could not be written")
    base = base if base is not None else root()
    if base is None:
        return out.missing("the ledger is off (RIGTUNE_LEDGER=off)")
    try:
        cal = find(profile.calibration_id, base)
        rebuilt = rebuild(cal)
    except LedgerError as exc:
        return out.missing(str(exc))
    for problem in cal.scan.problems():
        out.add(True, f"elsewhere in its ledger file: {problem}")
    out.add(True, f"{len(cal.trials)} trial records, the profile record, every id matches its content")
    theirs = rebuilt.calibration_table
    trials = sum(1 for r in theirs if not r.metrics.open_loop)
    if profile.layout != "single" or profile.phases != "unified" or profile.disagg is not None:
        _verify_derived(out, profile, rebuilt, cal.calibration_id)
    else:
        bad = _same_trials(profile.calibration_table, theirs)
        if bad is None and len(profile.calibration_table) == len(theirs):
            out.add(True, f"the table: {len(theirs)} trials, each identical to its record")
        else:
            out.add(False, f"the table differs from the ledger's ({len(profile.calibration_table)} trials against "
                           f"{len(theirs)}), first at trial {bad or len(theirs) + 1}")
        a, b = profile.model_dump(exclude={"calibration_table"}), rebuilt.model_dump(exclude={"calibration_table"})
        fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        out.add(not fields, "the rest of the profile is the one the calibration recorded" if not fields else
                f"the profile differs from the one the calibration recorded in: {', '.join(fields)}")
        out.add(trials == profile.calibration_trials,
                f"trial count {profile.calibration_trials}, and the table has {trials} (the rate check not counted)")
    winner, score, how = rederive_pick(rebuilt)
    if winner is None:
        out.unchecked(f"the calibration's pick: {how}")
    else:
        same_pick = winner.config.key() == rebuilt.config.key()
        out.add(same_pick, f"the calibration's pick re-derived from {how} under {rebuilt.objective}: "
                           f"{winner.config.key()}, score {score:.6g}"
                           + ("" if same_pick else f"; its profile serves {rebuilt.config.key()}"))
    return out


def _verify_derived(out: Verification, profile: Profile, rebuilt: Profile, calibration_id: str) -> None:
    """A --layout or --phases profile: a copy of its calibration's profile with the layout's own trials added to the
    table and its choice among them (layout.py, disagg.py). The ledger holds only the calibration, so check what
    must still be the calibration's, and that what the layout chose is among its trials; the choice itself, and the
    launch arguments built for it, cannot be checked."""
    theirs, mine = rebuilt.calibration_table, profile.calibration_table
    out.add(True, f"built on calibration {calibration_id} by --layout or --phases; checked against that calibration")
    bad = _same_trials(mine, theirs)
    out.add(bad is None, f"its table starts with the calibration's {len(theirs)} trials, each identical to its record"
            if bad is None else f"its table does not start with the calibration's {len(theirs)} trials (first "
            f"differs at trial {bad})")
    added = len(mine) - len(theirs)
    want = rebuilt.calibration_trials + added
    out.add(added >= 0 and profile.calibration_trials == want,
            f"trial count {profile.calibration_trials}: the calibration's {rebuilt.calibration_trials} and the "
            f"{max(added, 0)} trials the layout added" + ("" if profile.calibration_trials == want else
                                                          f" make {want}"))
    if profile.disagg is not None:
        served = any(r.disagg is not None and r.disagg.key() == profile.disagg.key() for r in mine)
        what = profile.disagg.key()
    else:
        served = any(r.config.key() == profile.config.key() and r.replicas == profile.replicas and r.disagg is None
                     for r in mine)
        what = profile.config.key() + (f" x{profile.replicas}" if profile.replicas > 1 else "")
    out.add(served, f"what it serves, {what}, is among its trials" if served else
            f"what it serves, {what}, is not among its trials")
    a, b = profile.model_dump(include=set(_KEPT_BY_DERIVED)), rebuilt.model_dump(include=set(_KEPT_BY_DERIVED))
    fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    strip = {k: v for k, v in profile.options.items() if k not in _DERIVED_OPTIONS}
    if strip != {k: v for k, v in rebuilt.options.items() if k not in _DERIVED_OPTIONS}:
        fields.append("options")
    out.add(not fields, "the fields a layout leaves alone are the calibration's" if not fields else
            f"it differs from its calibration where a layout changes nothing: {', '.join(fields)}")
    out.unchecked("the layout's choice among its own trials, and the launch arguments built for it: the ledger does "
                  "not hold a layout's trials")


def _scrub(text: str) -> str:
    """Paths cut to their last part, as compare --verdict cuts them from what it publishes."""
    from rigtune.bench.compare import scrub_text

    return scrub_text(text)


def _model_name(model_id: str) -> str:
    from rigtune.bench.compare import name_only

    return name_only(model_id)


def calibration_rows(base: Optional[Path] = None, model: Optional[str] = None) -> Tuple[List[List[str]], List[str]]:
    """One row per calibration in the ledger (oldest first), and notes on what a read skipped: id, model, when it
    began, objective, trials, failed, skipped, and its pick, or that it has none. Paths are cut out."""
    from collections import Counter

    rows: List[Tuple[str, str, List[str]]] = []
    notes: List[str] = []
    for path in files(base):
        scan = Scan()
        label = "a ledger file"  # a file is named by the model in it, cut to a name, never by its file name
        for cal in calibrations(path, scan):
            first = cal.trials[0] if cal.trials else (cal.record or {})
            mid = str((first.get("model") or {}).get("id") or "?")
            label = _model_name(mid)
            if model and model.lower() not in mid.lower():
                continue
            kinds = Counter(str((r.get("outcome") or {}).get("kind")) for r in cal.trials)
            skipped = sum(n for k, n in kinds.items() if k.startswith("skipped_"))
            failed = sum(n for k, n in kinds.items() if k != "ok") - skipped
            objective = str((first.get("objective") or {}).get("name") or "?")
            pick = "no pick (failed or stopped)"
            if cal.record is not None:
                try:
                    pick = _scrub(Profile.model_validate({**cal.record["profile"], "calibration_table": []}).config.key())
                except Exception:
                    pick = "unreadable profile record"
            when = str(first.get("recorded_at") or "")
            rows.append((when, cal.calibration_id, [cal.calibration_id[:12], _model_name(mid), when, objective,
                                                    str(len(cal.trials) - skipped), str(failed), str(skipped), pick]))
        notes += [f"the ledger of {label} (hardware {path.parent.name[:8]}): {p}" for p in scan.problems()]
    return [r for _, _, r in sorted(rows, key=lambda x: (x[0], x[1]))], notes


def trial_rows(cal: Calibration) -> List[List[str]]:
    """One row per trial of a calibration, in the order they ran: seq, stage, configuration, outcome, tok/s, p95 TTFT,
    the objective's score among the calibration's trials (lower is better), whether it is the pick's configuration,
    and the first line of its error. Paths are cut out."""
    from rigtune.calibrate.objectives import rank
    from rigtune.tradeoff import profile_constraints

    results: List[Optional[TrialResult]] = []
    for r in cal.trials:
        try:
            results.append(TrialResult.model_validate(r["trial"]))
        except Exception:
            results.append(None)
    scores: Dict[int, float] = {}
    pick = None
    if cal.record is not None:
        try:
            profile = Profile.model_validate({**cal.record["profile"], "calibration_table": []})
            pick = profile.config.key()
            ok = [x for x in results if x is not None]
            scores = {id(x.result): x.score for x in rank(ok, profile.objective, profile_constraints(profile))}
        except Exception:
            pass

    def num(x: Optional[float], nd: int = 1) -> str:
        return "-" if x is None else f"{x:.{nd}f}"

    rows = []
    for rec, res in zip(cal.trials, results):
        error = ((res.error if res is not None else None) or "").strip().splitlines()
        rows.append([str(rec.get("seq")), str(rec.get("stage")), _scrub(str(rec.get("config_key"))),
                     str((rec.get("outcome") or {}).get("kind")),
                     num(res.metrics.tok_s) if res is not None and res.ok else "-",
                     num(res.metrics.ttft_p95_ms, 0) if res is not None and res.ok else "-",
                     f"{scores[id(res)]:.4g}" if res is not None and id(res) in scores else "-",
                     "pick" if pick is not None and rec.get("config_key") == pick else "",
                     _scrub(error[0])[:80] if error else ""])
    return rows


def sizes(base: Optional[Path] = None) -> Tuple[int, int]:
    """(ledger files, bytes in them)."""
    found = files(base)
    return len(found), sum(p.stat().st_size for p in found)
