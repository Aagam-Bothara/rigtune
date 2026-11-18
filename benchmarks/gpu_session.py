#!/usr/bin/env python
"""Run a GPU session as named phases, each into its own results directory beside a manifest, resumably.

The 14 November 2026 pod deleted itself before its files were copied back, and only figures read over SSH during
the run survived (docs/benchmarks.md). This runner is the pod side of the fix: every phase writes into
<root>/<phase>/, and its manifest.json there says what ran (the command and a hash of it, start and end in UTC,
exit code, the engine versions installed when it started), so each phase is a unit that can be copied off the
pod the moment it ends and read on its own later.

    python benchmarks/gpu_session.py run --plan plan.json --root /workspace/session
    python benchmarks/gpu_session.py run --plan plan.json --root /workspace/session --dry-run
    python benchmarks/gpu_session.py status --root /workspace/session

A plan is JSON: {"session": "<name>", "env": {...}, "home": ..., "cleanup": ..., "pythons": {...},
"phases": [{"name", "cmd", "needs", "requires", "timeout_s", "why"}, ...]}. `cmd` is an argument list, or a
string run with `bash -o pipefail -c`; `{out}` (the phase's directory), `{root}` and `{python}` (this
interpreter) are replaced in it. Each phase starts in its own directory, which is also in $RIGTUNE_PHASE_DIR.
`env` is added to every phase's environment and recorded in its manifest, so it must not hold a secret: a token
the commands need (HF_TOKEN) is exported in the shell that starts the runner, and is neither recorded nor printed.

- One runner per root: <root>/runner.lock holds the running runner's pid; a second runner refuses to start
  while that process is alive, and takes the lock over once it is not.
- Phases run in plan order. A finished phase (status `done`) is skipped on a rerun, unless its command has
  changed since: then it is reported `stale` and kept, and `--redo-stale` runs it again. Any other phase is run
  again, after its earlier directory is moved, whole, to <root>/_attempts/<phase>.<n>, so an interrupted or
  failed attempt's files are kept as evidence rather than overwritten. A phase left `running` by a runner that
  died has its recorded process group and session killed (and the plan's `cleanup` run) before it is moved.
- A phase that fails never stops the session. A phase whose `needs` names one that is not `done` is recorded
  as `blocked` and not run; every other phase still runs.
- `requires`: a command that must exit 0 for the phase to run (an option a parallel series adds, say). When it
  fails the phase is recorded as `skipped`, with the reason, and is checked again on the next rerun.
- `timeout_s` ends a phase and records `timeout`. On Linux a phase leads its own session and process group, and
  whatever is left in either when the phase ends, by timeout or not, is killed (the session catches what a
  command such as `timeout` moved to a group of its own); then the plan's `cleanup` command
  runs (killing GPU processes an engine started in a session of its own), and only then is the phase's state
  written, so nothing is still writing into a phase recorded as finished.
- With "home" in the plan, $RIGTUNE_HOME is <root>/<home> for every phase, so a comparison reuses the
  calibration an earlier phase cached; the files a phase added or changed there (profiles, the trial ledger,
  engine logs) are copied into <phase>/rigtune-home/, so the phase's directory holds everything it produced.
  Temporary files (*.tmp, a profile half-saved when a phase was killed) are not copied.
- "pythons" names other interpreters ({"sglang": "$SGLANG_PYTHON", "llmopt": "/path/bin/python"}) whose package
  versions every manifest records beside this interpreter's.
- `touch <root>/STOP` ends the session at the next phase boundary; the runner renames it to
  STOP.seen-<time> when it stops, so the next run goes on.
- `--dry-run` prints the plan, each phase's state and the command it would run, and runs nothing.

Copy-back. When a directory reaches a final state (a phase done, failed, timed out, skipped or blocked, or an
attempt set aside as interrupted) the runner writes <dir>/SHA256SUMS (`sha256sum -c` reads it) over every file
in it, and lists the directory with its state in <root>/index.json. benchmarks/gpu_pull.py, run on the laptop,
reads that index over SSH and copies each finished directory, verified against its list. A finished directory
is never written to again, so its list stays true; one recorded as finished but without a list (the runner
died between the two) gets its list when the next run starts. The index also names the root's own logs
(session.log, copyback.log, the runner's output), which the puller copies as they are. `--after-phase CMD` also
runs CMD (with `bash -c`) after each finished directory, with {unit} (its path), {rel} (its path under the root)
and {root} replaced: a push to storage the user configures, with credentials from the environment only. A
failing CMD is reported on the console and in <root>/copyback.log, and never fails the phase.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHELL_SAFE = re.compile(r"^[A-Za-z0-9._/\\:-]+$")
DONE, FAILED, TIMEOUT, SKIPPED, BLOCKED, RUNNING, INTERRUPTED, STALE = (
    "done", "failed", "timeout", "skipped", "blocked", "running", "interrupted", "stale")
MANIFEST = "manifest.json"
ATTEMPTS = "_attempts"
LOG = "output.log"
LOCK = "runner.lock"
SUMS = "SHA256SUMS"
INDEX = "index.json"
FINAL = (DONE, FAILED, TIMEOUT, SKIPPED, BLOCKED, INTERRUPTED)
NOT_LOGS = (INDEX, LOCK, ".sums-staging")  # root-level files the index does not offer as logs
# Packages whose versions decide what a phase measured: read from metadata, so nothing is imported.
PACKAGES = ("rigtune", "vllm", "sglang", "torch", "transformers", "flashinfer-python", "arctic-inference",
            "nvidia-ml-py", "llm-optimizer", "aiconfigurator")
VERSIONS_CODE = ("import importlib.metadata as m, json\nout = {}\n"
                 f"for n in {PACKAGES!r}:\n"
                 "    try: out[n] = m.version(n)\n    except m.PackageNotFoundError: out[n] = None\n"
                 "print(json.dumps(out))")


class PlanError(ValueError):
    pass


class SessionLocked(RuntimeError):
    pass


@dataclass
class Phase:
    name: str
    cmd: Union[str, List[str]]
    needs: List[str] = field(default_factory=list)
    requires: Optional[Union[str, List[str]]] = None
    timeout_s: Optional[float] = None
    why: str = ""

    def digest(self) -> str:
        """A hash of what the phase runs, so a plan edited after a phase finished is noticed."""
        return hashlib.sha256(json.dumps([self.cmd, self.requires], sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Plan:
    session: str
    phases: List[Phase]
    env: Dict[str, str] = field(default_factory=dict)
    home: Optional[str] = None  # a directory under the root that becomes $RIGTUNE_HOME
    home_copy_max_mb: float = 512.0  # files a phase leaves in it above this are listed, not copied
    cleanup: Optional[Union[str, List[str]]] = None  # run after every phase's process has ended
    pythons: Dict[str, str] = field(default_factory=dict)  # other interpreters whose versions are recorded


def read_plan(data: dict) -> Plan:
    """A Plan from its JSON, refusing what would make a session ambiguous: a repeated or unsafe name, a need on
    a later or unknown phase, an unknown key (a typo must not drop a setting silently)."""
    if not isinstance(data, dict) or not isinstance(data.get("phases"), list) or not data["phases"]:
        raise PlanError('expected {"session": ..., "phases": [...]} with at least one phase')
    unknown = sorted(set(data) - {"session", "phases", "env", "home", "home_copy_max_mb", "notes", "cleanup",
                                  "pythons"})
    if unknown:
        raise PlanError(f"unknown plan key(s): {', '.join(unknown)}")
    phases: List[Phase] = []
    seen: List[str] = []
    for i, p in enumerate(data["phases"], 1):
        if not isinstance(p, dict):
            raise PlanError(f"phase {i}: expected an object")
        bad = sorted(set(p) - {"name", "cmd", "needs", "requires", "timeout_s", "why"})
        if bad:
            raise PlanError(f"phase {i}: unknown key(s) {', '.join(bad)}")
        name = p.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            raise PlanError(f"phase {i}: name {name!r} must be lower-case letters, digits, '.', '_' or '-'")
        if name in seen:
            raise PlanError(f"phase {i}: {name!r} is already a phase")
        cmd = p.get("cmd")
        if not (isinstance(cmd, str) and cmd.strip()) and not (
                isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
            raise PlanError(f"phase {name}: cmd must be a non-empty string or list of strings")
        needs = list(p.get("needs") or [])
        for n in needs:
            if n not in seen:
                raise PlanError(f"phase {name}: needs {n!r}, which is not an earlier phase")
        phases.append(Phase(name=name, cmd=cmd, needs=needs, requires=p.get("requires"),
                            timeout_s=p.get("timeout_s"), why=str(p.get("why") or "")))
        seen.append(name)
    env = data.get("env") or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise PlanError("env must map names to strings")
    pythons = data.get("pythons") or {}
    if not isinstance(pythons, dict) or not all(isinstance(v, str) for v in pythons.values()):
        raise PlanError("pythons must map labels to interpreter paths")
    home = data.get("home")
    if home is not None and (not isinstance(home, str) or not NAME.match(home) or home in seen):
        raise PlanError(f"home {home!r} must be a plain directory name that is not a phase's")
    return Plan(session=str(data.get("session") or "session"), phases=phases, env=env, home=home,
                home_copy_max_mb=float(data.get("home_copy_max_mb", 512.0)), cleanup=data.get("cleanup"),
                pythons=pythons)


def load_plan(path: Path) -> Plan:
    try:
        return read_plan(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        if isinstance(exc, PlanError):
            raise
        raise PlanError(f"{path}: {exc}") from None


# --------------------------------------------------------------------------- the pieces a test replaces


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def installed_versions(pythons: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """Each package in PACKAGES this interpreter has; the same from each interpreter in `pythons` (and from
    $SGLANG_PYTHON, where rigtune runs SGLang), keyed `<label>:<package>`; and the GPU driver from nvidia-smi.
    A missing one is None, never an error."""
    from importlib import metadata

    out: Dict[str, Optional[str]] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    others = dict(pythons or {})
    if os.environ.get("SGLANG_PYTHON") and "sglang" not in others:
        others["sglang"] = "$SGLANG_PYTHON"
    for label, interpreter in sorted(others.items()):
        exe = os.path.expandvars(interpreter)
        try:
            got = subprocess.run([exe, "-c", VERSIONS_CODE], capture_output=True, text=True, timeout=60,
                                 check=True, stdin=subprocess.DEVNULL)
            out.update({f"{label}:{k}": v for k, v in json.loads(got.stdout).items() if v is not None})
        except (OSError, subprocess.SubprocessError, ValueError):
            out[f"{label}:"] = None  # that interpreter could not be asked
    try:
        got = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True)
        out["gpus"] = "; ".join(line.strip() for line in got.stdout.splitlines() if line.strip()) or None
    except (OSError, subprocess.SubprocessError):
        out["gpus"] = None
    return out


def pid_alive(pid: int) -> bool:
    """Whether a process exists. On Windows os.kill would terminate it, so the process is opened instead."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def session_members(sid: int, proc: Path = Path("/proc")) -> List[int]:
    """The live processes in session `sid` (Linux: the session field of /proc/<pid>/stat), this one excluded;
    empty where there is no /proc. A zombie is left out: it is already dead and only waits for its parent.

    A phase starts a session of its own, and a command inside it can leave the phase's process group while
    staying in its session: `timeout` (without --foreground) puts itself and its command in a new group, so on the
    A40 in November 2026 ending the phase's group left llm-optimizer and auto_tune running under it."""
    members: List[int] = []
    try:
        entries = list(proc.iterdir())
    except OSError:
        return members
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            fields = stat[stat.rindex(")") + 2:].split()  # the command name may hold spaces and parentheses
            if fields[0] != "Z" and int(fields[3]) == sid:  # state, ppid, pgrp, session
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue  # it ended while being read, or is not a process directory
    return sorted(members)


def _signal_phase(pid: int, sig: int) -> bool:
    """Send `sig` to the phase's process group and to every process left in its session; whether any existed."""
    hit = False
    try:
        os.killpg(pid, sig)
        hit = True
    except ProcessLookupError:
        pass
    except PermissionError:
        hit = True
    for member in session_members(pid):
        try:
            os.kill(member, sig)
            hit = True
        except (ProcessLookupError, PermissionError):
            pass
    return hit


def group_alive(pid: int) -> bool:
    """Whether anything a phase started is still running: its process group or, on Linux, its session (the
    group's leader may be gone while its children are not). On Windows, the process itself."""
    if os.name == "nt":
        return pid_alive(pid)
    return _signal_phase(pid, 0)


def kill_group(pid: int, grace_s: float = 10.0) -> None:
    """End a phase: on Linux its process group and every process left in its session (the phase leads both), on
    Windows its process tree."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    if not _signal_phase(pid, signal.SIGTERM):
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _signal_phase(pid, 0):
            return
        time.sleep(0.2)
    _signal_phase(pid, signal.SIGKILL)


@dataclass
class Outcome:
    exit_code: Optional[int]
    timed_out: bool = False


def run_process(argv: List[str], cwd: Path, env: Dict[str, str], log: Path, timeout_s: Optional[float],
                on_start: Optional[Callable[[int], None]] = None) -> Outcome:
    """Run one phase's command with its output in `log`. On Linux it gets its own process group, and when the
    command ends (or times out) everything still in that group is ended too, an engine's child included."""
    posix = os.name == "posix"
    with log.open("ab") as fh:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=posix)
        if on_start:
            on_start(proc.pid)
        try:
            code = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            kill_group(proc.pid) if posix else proc.kill()
            code, timed_out = proc.wait(), True
        if posix:
            kill_group(proc.pid, grace_s=5.0)  # leftovers that outlived the command
        return Outcome(code, timed_out=timed_out)


# --------------------------------------------------------------------------- the session


def _argv(cmd: Union[str, List[str]], subs: Dict[str, str]) -> List[str]:
    def sub(s: str) -> str:
        for k, v in subs.items():
            s = s.replace("{" + k + "}", v)
        return s

    if isinstance(cmd, str):
        # pipefail: `a | tee log` must fail when `a` does, not take tee's exit code.
        return ["bash", "-o", "pipefail", "-c", sub(cmd)]
    return [sub(c) for c in cmd]


def read_manifest(phase_dir: Path) -> Optional[dict]:
    try:
        return json.loads((phase_dir / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    """Written to a temporary name and renamed, so a reader (or a copy) never sees half a manifest."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def is_temporary(name: str) -> bool:
    """A file a writer had not finished: rigtune's staging files (.<random>.tmp) and this runner's own."""
    return name.endswith(".tmp")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sums(unit: Path, staging: Path) -> Path:
    """<unit>/SHA256SUMS: one `sha256  relative/path` line per file, every file but the list itself, sorted by
    path, in the format sha256sum writes and `sha256sum -c` checks. It is staged outside the unit (`staging`, on
    the same disk) and renamed in, so the unit never holds a file the list does not name."""
    lines = [f"{_sha256(p)}  {p.relative_to(unit).as_posix()}"
             for p in sorted(unit.rglob("*"), key=lambda q: q.relative_to(unit).as_posix())
             if p.is_file() and p.relative_to(unit).as_posix() != SUMS]
    staging.write_bytes(("\n".join(lines) + "\n").encode("utf-8") if lines else b"")
    os.replace(staging, unit / SUMS)
    return unit / SUMS


def _scan(home: Path) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    if home.is_dir():
        for p in home.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[p.relative_to(home).as_posix()] = (st.st_size, st.st_mtime_ns)
    return out


class Session:
    def __init__(self, plan: Plan, root: Path, *, python: str = sys.executable,
                 versions: Optional[Callable[[], Dict[str, Optional[str]]]] = None,
                 runner: Optional[Callable[..., Outcome]] = None, now: Callable[[], str] = utc_now,
                 echo: Callable[[str], None] = print, redo_stale: bool = False,
                 after_phase: Optional[Union[str, List[str]]] = None):
        self.plan, self.root = plan, Path(root).resolve()
        self.python, self.now, self.echo, self.redo_stale = python, now, echo, redo_stale
        self.versions = versions or (lambda: installed_versions(self.plan.pythons))
        self.runner = runner or self._run_process
        self._current: Optional[Tuple[Path, dict]] = None
        self._locked = False
        self.after_phase = after_phase

    def phase_dir(self, name: str) -> Path:
        return self.root / name

    def status(self, name: str) -> Optional[str]:
        m = read_manifest(self.phase_dir(name))
        return m.get("status") if m else None

    def subs(self, phase: Phase) -> Dict[str, str]:
        return {"out": str(self.phase_dir(phase.name)), "root": str(self.root), "python": self.python}

    def env(self, phase: Phase) -> Dict[str, str]:
        env = {**os.environ, **self.plan.env, "RIGTUNE_PHASE_DIR": str(self.phase_dir(phase.name))}
        if self.plan.home:
            env["RIGTUNE_HOME"] = str(self.root / self.plan.home)
        return env

    def _log(self, line: str) -> None:
        self.echo(line)
        with (self.root / "session.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{self.now()} {line}\n")

    # ---- one runner per root

    def _lock(self) -> None:
        path = self.root / LOCK
        me = {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": self.now()}
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    held = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    held = {}
                pid, host = int(held.get("pid") or 0), held.get("host")
                if host not in (None, me["host"]):
                    raise SessionLocked(f"{path} is held by pid {pid} on {host}, which this machine cannot "
                                        "check; delete the file if that runner is gone")
                if pid_alive(pid):
                    raise SessionLocked(f"another runner (pid {pid}, since {held.get('started_utc')}) is using "
                                        f"{self.root}; wait for it, or stop it at a phase boundary with "
                                        f"`touch {self.root / 'STOP'}`")
                self._log(f"LOCK     {path} was left by pid {pid}, which is gone: taking it over")
                path.unlink()
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(me, fh)
            self._locked = True
            return
        raise SessionLocked(f"could not take {path}")

    def _unlock(self) -> None:
        if self._locked:
            try:
                (self.root / LOCK).unlink()
            except OSError:
                pass
            self._locked = False

    # ---- processes

    def _run_process(self, argv, cwd, env, log, timeout_s) -> Outcome:
        return run_process(argv, cwd, env, log, timeout_s, on_start=self._started)

    def _started(self, pid: int) -> None:
        """Record the phase's process (and on Linux its group), so a later runner can end what it left."""
        if self._current is None:
            return
        d, manifest = self._current
        manifest.update(pid=pid, pgid=pid if os.name == "posix" else None, host=socket.gethostname())
        write_json(d / MANIFEST, manifest)

    def _cleanup(self, why: str) -> Optional[int]:
        """The plan's cleanup command (kill leftover engines, wait for an empty GPU); its exit code."""
        if not self.plan.cleanup:
            return None
        argv = _argv(self.plan.cleanup, {"root": str(self.root), "python": self.python})
        try:
            got = subprocess.run(argv, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL,
                                 env={**os.environ, **self.plan.env})
            code, text = got.returncode, (got.stdout + got.stderr).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            code, text = None, str(exc)
        if text or code:
            self._log(f"CLEANUP  {why}: exit {code}" + (f": {text[-300:]}" if text else ""))
        return code

    # ---- directories

    def _set_aside(self, name: str) -> Optional[Path]:
        """Move a phase's earlier, unfinished directory to _attempts/<name>.<n>. A manifest left `running` (the
        runner itself was killed) is marked `interrupted` there, after what that phase started is ended."""
        d = self.phase_dir(name)
        if not d.exists():
            return None
        m = read_manifest(d)
        if m and m.get("status") in (BLOCKED, SKIPPED) and set(os.listdir(d)) <= self.bare_files():
            shutil.rmtree(d)  # nothing ran: a manifest alone, decided again now
            return None
        if m and m.get("status") == RUNNING:
            pid = int(m.get("pgid") or m.get("pid") or 0)
            if pid and m.get("host") in (None, socket.gethostname()) and group_alive(pid):
                self._log(f"KILL     {name}: ending process group {pid}, left running by an earlier runner")
                kill_group(pid)
            self._cleanup(f"before setting {name} aside")
        attempts = self.root / ATTEMPTS
        attempts.mkdir(exist_ok=True)
        n = 1
        while (attempts / f"{name}.{n}").exists():
            n += 1
        dest = attempts / f"{name}.{n}"
        os.replace(d, dest)
        if m and m.get("status") == RUNNING:
            m.update(status=INTERRUPTED, ended_utc=None,
                     note="the runner stopped while this phase ran (killed, or the pod ended); its files are "
                          "as they were left")
            write_json(dest / MANIFEST, m)
        self.after_unit(dest)
        return dest

    def bare_files(self) -> set:
        """The files a phase that never ran holds."""
        return {MANIFEST, SUMS}

    def after_unit(self, unit: Path) -> None:
        """A directory reached a final state: its checksum list, the index, then the user's push command."""
        write_sums(unit, self.root / ".sums-staging")
        self.write_index()
        if not self.after_phase:
            return
        rel = unit.relative_to(self.root).as_posix()
        argv = _argv(self.after_phase, {"unit": str(unit), "rel": rel, "root": str(self.root),
                                       "python": self.python})
        try:
            got = subprocess.run(argv, capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL,
                                 env={**os.environ, **self.plan.env})
            code, text = got.returncode, (got.stdout + got.stderr)
        except (OSError, subprocess.SubprocessError) as exc:
            code, text = None, str(exc)
        with (self.root / "copyback.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{self.now()} {rel}: exit {code}\n{text}")
        if code != 0:
            self.echo(f"COPY-BACK FAILED {rel}: exit {code} (see {self.root / 'copyback.log'}); the phase's "
                      "files are still on the pod")

    def units(self) -> List[Path]:
        """Every phase and attempt directory that has a manifest."""
        dirs = [p for p in self.root.iterdir() if p.is_dir() and p.name != ATTEMPTS]
        if (self.root / ATTEMPTS).is_dir():
            dirs += list((self.root / ATTEMPTS).iterdir())
        return sorted((d for d in dirs if read_manifest(d) is not None), key=lambda d: d.as_posix())

    def write_index(self, finished: bool = False) -> None:
        """<root>/index.json: every phase and attempt directory with its state and whether its checksum list is
        written, the root's own logs, and whether the runner got to the end of a full run. The puller copies
        only directories listed in a final state with their list."""
        units = []
        for d in self.units():
            status = read_manifest(d).get("status")
            units.append({"path": d.relative_to(self.root).as_posix(), "status": status,
                          "sums": (d / SUMS).is_file() and status in FINAL})
        logs = sorted(p.name for p in self.root.iterdir()
                      if p.is_file() and p.name not in NOT_LOGS and not p.name.startswith("STOP")
                      and not is_temporary(p.name))
        write_json(self.root / INDEX, {"session": self.plan.session, "updated_utc": self.now(),
                                       "finished": finished, "units": units, "logs": logs})

    def repair(self) -> None:
        """A directory recorded as finished without its checksum list (the runner died between the two) gets
        its list now: a finished state counts only once the list is written."""
        for d in self.units():
            if read_manifest(d).get("status") in FINAL and not (d / SUMS).is_file():
                self._log(f"REPAIR   {d.relative_to(self.root).as_posix()}: recorded as finished without "
                          f"{SUMS}; writing it now")
                write_sums(d, self.root / ".sums-staging")
        self.write_index()

    def _record(self, phase: Phase, status: str, reason: str) -> None:
        d = self.phase_dir(phase.name)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / MANIFEST, {"phase": phase.name, "why": phase.why, "status": status, "reason": reason,
                                  "command": _argv(phase.cmd, self.subs(phase)), "cmd_sha": phase.digest(),
                                  "recorded_utc": self.now(), "session": self.plan.session})
        self._log(f"{status.upper():8s} {phase.name}: {reason}")
        self.after_unit(d)

    def _requires(self, phase: Phase) -> Optional[str]:
        if not phase.requires:
            return None
        argv = _argv(phase.requires, self.subs(phase))
        try:
            got = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=self.env(phase),
                                 stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"its requirement {' '.join(argv)} could not run: {exc}"
        if got.returncode != 0:
            tail = (got.stdout + got.stderr).strip().splitlines()[-3:]
            return (f"its requirement `{' '.join(argv)}` exited {got.returncode}"
                    + (f": {' / '.join(tail)}" if tail else ""))
        return None

    def finished_before(self, phase: Phase) -> Optional[str]:
        """DONE or STALE when an earlier run finished this phase, else None."""
        m = read_manifest(self.phase_dir(phase.name))
        if not m or m.get("status") != DONE:
            return None
        return DONE if m.get("cmd_sha") == phase.digest() else STALE

    def run_phase(self, phase: Phase) -> str:
        before_state = self.finished_before(phase)
        if before_state == DONE:
            self._log(f"SKIP     {phase.name}: done in an earlier run")
            return DONE
        if before_state == STALE and not self.redo_stale:
            self._log(f"STALE    {phase.name}: done in an earlier run with a command the plan no longer has; "
                      "kept as it was (--redo-stale runs it again)")
            return STALE
        self._set_aside(phase.name)
        unmet = [n for n in phase.needs if self.status(n) != DONE]
        if unmet:
            self._record(phase, BLOCKED, f"needs {', '.join(unmet)}, which did not finish")
            return BLOCKED
        missing = self._requires(phase)
        if missing:
            self._record(phase, SKIPPED, missing)
            return SKIPPED
        d = self.phase_dir(phase.name)
        d.mkdir(parents=True)
        home = self.root / self.plan.home if self.plan.home else None
        before = _scan(home) if home else {}
        argv = _argv(phase.cmd, self.subs(phase))
        manifest = {"phase": phase.name, "why": phase.why, "session": self.plan.session, "status": RUNNING,
                    "command": argv, "cmd_sha": phase.digest(), "started_utc": self.now(), "ended_utc": None,
                    "exit_code": None, "timeout_s": phase.timeout_s, "needs": phase.needs,
                    "env": dict(self.plan.env), "versions": self.versions()}
        write_json(d / MANIFEST, manifest)
        self._current = (d, manifest)
        self.write_index()  # the index shows the phase as running from its start
        self._log(f"START    {phase.name}: {' '.join(argv)}")
        try:
            out = self.runner(argv, cwd=d, env=self.env(phase), log=d / LOG, timeout_s=phase.timeout_s)
        except OSError as exc:  # the command itself could not start (not installed, say)
            with (d / LOG).open("a", encoding="utf-8") as fh:
                fh.write(f"could not start: {exc}\n")
            out = Outcome(None)
        finally:
            self._current = None
        cleanup = self._cleanup(f"after {phase.name}")  # before the state: nothing may write after it
        status = TIMEOUT if out.timed_out else DONE if out.exit_code == 0 else FAILED
        manifest.update(status=status, ended_utc=self.now(), exit_code=out.exit_code)
        if cleanup is not None:
            manifest["cleanup_exit_code"] = cleanup
        if home:
            manifest.update(self._copy_home(home, before, d))
        write_json(d / MANIFEST, manifest)
        self._log(f"{status.upper():8s} {phase.name}: exit {out.exit_code}")
        self.after_unit(d)
        return status

    def _copy_home(self, home: Path, before: Dict[str, Tuple[int, int]], d: Path) -> dict:
        copied, left, temporary = [], [], []
        limit = self.plan.home_copy_max_mb * 1024 * 1024
        for rel, stat in sorted(_scan(home).items()):
            if before.get(rel) == stat:
                continue
            if is_temporary(rel):
                temporary.append(rel)
                continue
            if stat[0] > limit:
                left.append({"path": rel, "bytes": stat[0]})
                continue
            dest = d / "rigtune-home" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(home / rel, dest)
            copied.append(rel)
        return {"home_files_copied": len(copied), "home_files_not_copied": left,
                "home_temporary_files_skipped": temporary}

    def _stop_requested(self) -> bool:
        stop = self.root / "STOP"
        if not stop.exists():
            return False
        seen = self.root / f"STOP.seen-{self.now().replace(':', '')}"
        try:
            os.replace(stop, seen)  # consumed, so the next run goes on
        except OSError:
            pass
        return True

    def run(self, only: Optional[Sequence[str]] = None, redo: Sequence[str] = ()) -> Dict[str, str]:
        shell = any(isinstance(c, str) for p in self.plan.phases for c in (p.cmd, p.requires))
        shell = shell or isinstance(self.plan.cleanup, str)
        if shell and not SHELL_SAFE.match(str(self.root)):
            # {out} and {root} go into `bash -c` text unquoted, so a space would split them into two words.
            raise PlanError(f"the session root {self.root} must be letters, digits and . _ - / \\ : only, "
                            "because this plan's shell commands use it unquoted")
        names = {p.name for p in self.plan.phases}
        for name in list(redo) + list(only or []):
            if name not in names:
                raise PlanError(f"no phase named {name!r}")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock()
        try:
            self.repair()
            for name in redo:
                self.redo(name)
            results: Dict[str, str] = {}
            for phase in self.plan.phases:
                if only and phase.name not in only:
                    continue
                if self._stop_requested():
                    self._log(f"STOP     {self.root / 'STOP'} was found: stopping before {phase.name} (renamed "
                              "to STOP.seen-*, so the next run goes on)")
                    return results
                results[phase.name] = self.run_phase(phase)
            if not only:  # a partial run is not the session finishing
                self.finished()
            return results
        finally:
            self._unlock()

    def finished(self) -> None:
        """Every phase of a full run has had its turn (not when stopped, not under --only): the index says so."""
        self.write_index(finished=True)

    def redo(self, name: str) -> None:
        """Run a finished phase again on the next run: its directory is moved to _attempts first."""
        if name not in {p.name for p in self.plan.phases}:
            raise PlanError(f"no phase named {name!r}")
        self._set_aside(name)

    def describe(self) -> List[str]:
        lines = [f"session {self.plan.session}: {len(self.plan.phases)} phases under {self.root}"]
        if self.plan.home:
            lines.append(f"RIGTUNE_HOME={self.root / self.plan.home}")
        for k in sorted(self.plan.env):
            lines.append(f"env {k}={self.plan.env[k]}")
        if self.plan.cleanup:
            lines.append(f"cleanup after each phase: {' '.join(_argv(self.plan.cleanup, {}))}")
        for i, p in enumerate(self.plan.phases, 1):
            state = self.finished_before(p) or self.status(p.name) or "pending"
            lines.append(f"{i:2d}. {p.name} [{state}]" + (f" needs {', '.join(p.needs)}" if p.needs else "")
                         + (f", timeout {p.timeout_s:g}s" if p.timeout_s else ""))
            if p.why:
                lines.append(f"    why: {p.why}")
            if p.requires:
                lines.append(f"    requires: {' '.join(_argv(p.requires, self.subs(p)))}")
            lines.append(f"    run: {' '.join(_argv(p.cmd, self.subs(p)))}")
        return lines


def status_lines(root: Path) -> List[str]:
    """Every phase and attempt directory under a session root with its manifest's state."""
    root = Path(root)
    lines = []
    dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name != ATTEMPTS], key=lambda p: p.name)
    if (root / ATTEMPTS).is_dir():
        dirs += sorted((root / ATTEMPTS).iterdir(), key=lambda p: p.name)
    for d in dirs:
        m = read_manifest(d)
        if m is None:
            continue
        rel = d.relative_to(root).as_posix()
        lines.append(f"{rel:40s} {m.get('status', '?'):11s} exit {m.get('exit_code')}  "
                     f"{m.get('started_utc') or m.get('recorded_utc') or ''} {m.get('reason') or ''}".rstrip())
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="run every phase not yet done")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="the session's results directory")
    p.add_argument("--dry-run", action="store_true", help="print the plan and each phase's state; run nothing")
    p.add_argument("--only", nargs="*", default=None, help="run only these phases")
    p.add_argument("--redo", nargs="*", default=[], help="run these finished phases again")
    p.add_argument("--redo-stale", action="store_true",
                   help="run again every finished phase whose command the plan has changed since")
    p.add_argument("--after-phase", default=None, metavar="CMD",
                   help="run CMD (bash -c) after each phase, {unit} {rel} {root} replaced: a push to storage")
    p = sub.add_parser("status", help="each phase's state, from its manifest")
    p.add_argument("--root", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        for line in status_lines(a.root):
            print(line)
        return 0
    try:
        plan = load_plan(a.plan)
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    session = Session(plan, a.root, redo_stale=a.redo_stale, after_phase=a.after_phase)
    if a.dry_run:
        for line in session.describe():
            print(line)
        return 0
    try:
        results = session.run(only=a.only, redo=a.redo)
    except (PlanError, SessionLocked) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bad = sorted(n for n, s in results.items() if s not in (DONE, SKIPPED))
    print(f"session: {sum(s == DONE for s in results.values())} done, "
          f"{sum(s == SKIPPED for s in results.values())} skipped, {len(bad)} not done"
          + (f" ({', '.join(f'{n}: {results[n]}' for n in bad)})" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
