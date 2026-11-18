"""benchmarks/gpu_session.py: phases in order, each into its own directory with a manifest; resumable after a
failure, a timeout or the runner itself being killed mid-phase; a failed phase never stops an independent one.

Every phase here is a real subprocess (this interpreter running a line of Python), so the runner's own process
handling is what is tested, not a stand-in for it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu_session.py"
_spec = importlib.util.spec_from_file_location("gpu_session", SCRIPT)
gs = importlib.util.module_from_spec(_spec)
sys.modules["gpu_session"] = gs  # dataclasses look their module up while the class is made
_spec.loader.exec_module(gs)


def py(code: str) -> list:
    """A phase command: this interpreter running `code` (the runner replaces {python})."""
    return ["{python}", "-c", code]


# A phase that counts its runs in <root>/count-<name>, writes a file into its own directory and exits `rc`.
def counting(name: str, rc: int = 0) -> list:
    return py("import os, pathlib, sys\n"
              "out = pathlib.Path(os.environ['RIGTUNE_PHASE_DIR'])\n"
              f"c = out.parent / 'count-{name}'\n"
              "c.write_text(str(int(c.read_text()) + 1) if c.exists() else '1')\n"
              f"(out / 'result.txt').write_text('{name}')\n"
              f"sys.exit({rc})")


def runs(root: Path, name: str) -> int:
    c = root / f"count-{name}"
    return int(c.read_text()) if c.exists() else 0


def session(tmp_path: Path, phases: list, **plan) -> "gs.Session":
    p = gs.read_plan({"session": "t", "phases": phases, **plan})
    return gs.Session(p, tmp_path / "s", versions=lambda: {"vllm": "0.0-test"}, echo=lambda s: None)


def manifest(root: Path, rel: str) -> dict:
    return json.loads((root / rel / "manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- plans


@pytest.mark.parametrize("phases, needle", [
    ([{"name": "a", "cmd": "true"}, {"name": "a", "cmd": "true"}], "already a phase"),
    ([{"name": "a", "cmd": "true", "needs": ["b"]}, {"name": "b", "cmd": "true"}], "not an earlier phase"),
    ([{"name": "a", "cmd": "true", "timeout": 5}], "unknown key"),
    ([{"name": "A b", "cmd": "true"}], "lower-case"),
    ([{"name": "a", "cmd": []}], "non-empty"),
])
def test_a_plan_that_would_be_ambiguous_is_refused(phases, needle):
    with pytest.raises(gs.PlanError, match=needle):
        gs.read_plan({"phases": phases})


def test_home_must_not_be_a_phase_name():
    with pytest.raises(gs.PlanError, match="home"):
        gs.read_plan({"phases": [{"name": "a", "cmd": "true"}], "home": "a"})


# --------------------------------------------------------------------------- running


def test_each_phase_writes_its_own_directory_and_manifest(tmp_path):
    s = session(tmp_path, [{"name": "env", "cmd": counting("env"), "why": "record the machine"},
                           {"name": "smoke", "cmd": counting("smoke")}])
    assert s.run() == {"env": "done", "smoke": "done"}
    m = manifest(s.root, "env")
    assert m["status"] == "done" and m["exit_code"] == 0 and m["why"] == "record the machine"
    assert m["started_utc"] and m["ended_utc"] and m["versions"] == {"vllm": "0.0-test"}
    assert m["command"][0] == sys.executable  # {python} was replaced
    assert (s.root / "env" / "result.txt").read_text() == "env"
    assert (s.root / "smoke" / "result.txt").read_text() == "smoke"


def test_a_failed_phase_does_not_stop_independent_ones_and_blocks_its_dependents(tmp_path):
    s = session(tmp_path, [{"name": "cal", "cmd": counting("cal", rc=3)},
                           {"name": "cmp", "cmd": counting("cmp"), "needs": ["cal"]},
                           {"name": "other", "cmd": counting("other")}])
    assert s.run() == {"cal": "failed", "cmp": "blocked", "other": "done"}
    assert manifest(s.root, "cal")["exit_code"] == 3
    assert "cal" in manifest(s.root, "cmp")["reason"] and runs(s.root, "cmp") == 0
    assert runs(s.root, "other") == 1


def test_a_rerun_skips_done_phases_and_keeps_the_failed_attempt(tmp_path):
    phases = [{"name": "a", "cmd": counting("a")}, {"name": "b", "cmd": counting("b", rc=1)},
              {"name": "c", "cmd": counting("c"), "needs": ["b"]}]
    s = session(tmp_path, phases)
    assert s.run() == {"a": "done", "b": "failed", "c": "blocked"}
    phases[1]["cmd"] = counting("b")  # fixed between runs
    s = session(tmp_path, phases)
    assert s.run() == {"a": "done", "b": "done", "c": "done"}
    assert (runs(s.root, "a"), runs(s.root, "b"), runs(s.root, "c")) == (1, 2, 1)
    kept = s.root / "_attempts" / "b.1"
    assert manifest(s.root, "_attempts/b.1")["status"] == "failed" and (kept / "result.txt").exists()
    assert not (s.root / "_attempts" / "c.1").exists()  # a blocked phase ran nothing, so there is nothing to keep


def test_a_phase_whose_runner_was_killed_midway_is_run_again_and_its_files_kept(tmp_path):
    """The runner process itself dies while a phase runs (the pod's case): its manifest says `running`. A rerun
    moves that directory aside as `interrupted`, runs the phase again, and does not repeat finished phases."""
    root = tmp_path / "s"
    flag = tmp_path / "second-time"
    killer = py("import os, pathlib, signal, sys, time\n"
                "out = pathlib.Path(os.environ['RIGTUNE_PHASE_DIR'])\n"
                f"flag = pathlib.Path({str(flag)!r})\n"
                "if flag.exists():\n    (out / 'full.txt').write_text('all of it'); sys.exit(0)\n"
                "(out / 'partial.txt').write_text('half')\n"
                # Let go of the phase directory first (Windows cannot move a directory a process is in, or
                # has a file open in), then kill the runner and outlive it, as an engine would.
                f"os.chdir({str(tmp_path)!r}); os.close(1); os.close(2)\n"
                "os.kill(os.getppid(), signal.SIGTERM)\n"
                "time.sleep(2)\n")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"session": "t", "phases": [
        {"name": "first", "cmd": counting("first")}, {"name": "cal", "cmd": killer},
        {"name": "after", "cmd": counting("after")}]}), encoding="utf-8")
    cmd = [sys.executable, str(SCRIPT), "run", "--plan", str(plan), "--root", str(root)]
    got = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert got.returncode != 0
    assert manifest(root, "cal")["status"] == "running" and not (root / "after").exists()

    flag.write_text("1")
    got = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stdout + got.stderr
    assert runs(root, "first") == 1 and runs(root, "after") == 1
    old = manifest(root, "_attempts/cal.1")
    assert old["status"] == "interrupted" and old["exit_code"] is None
    assert (root / "_attempts" / "cal.1" / "partial.txt").read_text() == "half"
    assert manifest(root, "cal")["status"] == "done" and (root / "cal" / "full.txt").exists()


def test_a_timeout_ends_the_phase_and_the_session_goes_on(tmp_path):
    s = session(tmp_path, [{"name": "slow", "cmd": py("import time; time.sleep(60)"), "timeout_s": 1},
                           {"name": "next", "cmd": counting("next")}])
    assert s.run() == {"slow": "timeout", "next": "done"}
    assert manifest(s.root, "slow")["status"] == "timeout"


def test_a_command_that_cannot_start_is_a_failed_phase(tmp_path):
    s = session(tmp_path, [{"name": "x", "cmd": ["no-such-binary-for-rigtune-tests"]},
                           {"name": "y", "cmd": counting("y")}])
    assert s.run() == {"x": "failed", "y": "done"}
    assert "could not start" in (s.root / "x" / "output.log").read_text(encoding="utf-8")


def test_a_phase_whose_requirement_is_missing_is_skipped_with_why_and_checked_again(tmp_path):
    flag = tmp_path / "installed"
    req = py(f"import pathlib, sys; print('no such command: conformance'); "
             f"sys.exit(0 if pathlib.Path({str(flag)!r}).exists() else 2)")
    phases = [{"name": "conformance", "cmd": counting("conformance"), "requires": req},
              {"name": "after", "cmd": counting("after")}]
    s = session(tmp_path, phases)
    assert s.run() == {"conformance": "skipped", "after": "done"}
    reason = manifest(s.root, "conformance")["reason"]
    assert "exited 2" in reason and "no such command: conformance" in reason
    assert runs(s.root, "conformance") == 0
    flag.write_text("1")
    assert session(tmp_path, phases).run() == {"conformance": "done", "after": "done"}
    assert runs(s.root, "conformance") == 1 and runs(s.root, "after") == 1


def test_a_dry_run_runs_nothing(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"session": "t", "home": "rigtune-home", "phases": [
        {"name": "a", "cmd": counting("a"), "why": "because"},
        {"name": "b", "cmd": "echo {out}", "needs": ["a"], "requires": "true"}]}), encoding="utf-8")
    root = tmp_path / "s"
    assert gs.main(["run", "--plan", str(plan), "--root", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert " 1. a [pending]" in out and "why: because" in out and "needs a" in out
    assert f"echo {root.resolve() / 'b'}" in out and "requires: bash -o pipefail -c true" in out
    assert not root.exists()


def test_stop_file_ends_the_session_at_a_phase_boundary(tmp_path):
    stop = tmp_path / "s" / "STOP"
    s = session(tmp_path, [{"name": "a", "cmd": py(f"import pathlib; pathlib.Path({str(stop)!r}).touch()")},
                           {"name": "b", "cmd": counting("b")}])
    assert s.run() == {"a": "done"}
    assert runs(s.root, "b") == 0
    # The STOP file is consumed when it is honoured, so the next run (a resume) goes on.
    assert not stop.exists() and len(list(s.root.glob("STOP.seen-*"))) == 1
    again = session(tmp_path, [{"name": "a", "cmd": py(f"import pathlib; pathlib.Path({str(stop)!r}).touch()")},
                               {"name": "b", "cmd": counting("b")}])
    assert again.run() == {"a": "done", "b": "done"} and runs(s.root, "b") == 1


def test_redo_runs_a_finished_phase_again_and_keeps_the_old_one(tmp_path):
    phases = [{"name": "a", "cmd": counting("a")}]
    s = session(tmp_path, phases)
    s.run()
    s = session(tmp_path, phases)
    s.redo("a")
    s.run()
    assert runs(s.root, "a") == 2 and manifest(s.root, "_attempts/a.1")["status"] == "done"


def test_files_a_phase_leaves_in_rigtune_home_are_copied_into_its_directory(tmp_path):
    write = py("import os, pathlib\n"
               "h = pathlib.Path(os.environ['RIGTUNE_HOME'])\n"
               "(h / 'profiles').mkdir(parents=True, exist_ok=True)\n"
               "(h / 'profiles' / 'p.json').write_text('{}')\n"
               "(h / 'profiles' / '.k3j2.tmp').write_text('{half a profi')\n"
               "(h / 'ledger').mkdir(); (h / 'ledger' / 'm.jsonl').write_text('{}')\n"
               "(h / 'big.gguf').write_bytes(b'x' * 4096)\n")
    other = py("import os, pathlib\n"
               "h = pathlib.Path(os.environ['RIGTUNE_HOME'])\n"
               "(h / 'logs').mkdir(exist_ok=True); (h / 'logs' / 'engine.log').write_text('started')\n")
    s = session(tmp_path, [{"name": "cal", "cmd": write}, {"name": "cmp", "cmd": other}],
                home="rigtune-home", home_copy_max_mb=0.001)
    s.run()
    assert (s.root / "cal" / "rigtune-home" / "profiles" / "p.json").exists()
    m = manifest(s.root, "cal")
    assert (s.root / "cal" / "rigtune-home" / "ledger" / "m.jsonl").exists()  # the trial ledger too
    assert m["home_files_copied"] == 2 and m["home_files_not_copied"] == [{"path": "big.gguf", "bytes": 4096}]
    # A file a writer had not finished is not copied: its phase could not be verified against its list.
    assert m["home_temporary_files_skipped"] == ["profiles/.k3j2.tmp"]
    assert not list((s.root / "cal").rglob("*.tmp"))
    # The second phase gets only what it changed, not the first phase's files again.
    assert sorted(p.relative_to(s.root / "cmp" / "rigtune-home").as_posix()
                  for p in (s.root / "cmp" / "rigtune-home").rglob("*") if p.is_file()) == ["logs/engine.log"]


def test_the_shell_environment_reaches_a_phase_but_not_its_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_not_a_real_token_123")
    s = session(tmp_path, [{"name": "a", "cmd": py(
        "import os, pathlib; pathlib.Path(os.environ['RIGTUNE_PHASE_DIR'], 'seen').write_text("
        "str(len(os.environ.get('HF_TOKEN', ''))))")}], env={"HF_HUB_OFFLINE": "0"})
    s.run()
    assert (s.root / "a" / "seen").read_text() == str(len("hf_not_a_real_token_123"))
    text = (s.root / "a" / "manifest.json").read_text(encoding="utf-8")
    assert "hf_not_a_real_token_123" not in text and json.loads(text)["env"] == {"HF_HUB_OFFLINE": "0"}


def test_status_lists_phases_and_attempts(tmp_path, capsys):
    phases = [{"name": "a", "cmd": counting("a", rc=1)}]
    session(tmp_path, phases).run()
    session(tmp_path, phases).run()
    assert gs.main(["status", "--root", str(tmp_path / "s")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split()[:2] for line in out] == [["a", "failed"], ["_attempts/a.1", "failed"]]


# --------------------------------------------------------------------------- the review's cases


def test_a_second_runner_on_the_same_root_refuses_while_the_first_lives(tmp_path):
    """Two runners on one root would move each other's running phase aside (review demo 4)."""
    started, release = threading.Event(), threading.Event()

    def slow(argv, cwd, env, log, timeout_s):
        started.set()
        release.wait(30)
        (Path(cwd) / "result.json").write_text("{}")
        return gs.Outcome(0)

    first = gs.Session(gs.read_plan({"phases": [{"name": "cal", "cmd": ["x"]}]}), tmp_path / "s",
                       versions=lambda: {}, echo=lambda s: None, runner=slow)
    t = threading.Thread(target=first.run)
    t.start()
    try:
        assert started.wait(30)
        second = session(tmp_path, [{"name": "cal", "cmd": counting("cal")}])
        with pytest.raises(gs.SessionLocked, match="another runner"):
            second.run()
    finally:
        release.set()
        t.join()
    assert manifest(first.root, "cal")["status"] == "done" and (first.root / "cal" / "result.json").exists()
    assert not (first.root / "_attempts").exists() and not (first.root / "runner.lock").exists()


def test_a_lock_left_by_a_runner_that_died_is_taken_over(tmp_path):
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                          text=True).stdout.strip()
    root = tmp_path / "s"
    root.mkdir()
    (root / "runner.lock").write_text(json.dumps({"pid": int(dead), "host": gs.socket.gethostname()}))
    assert session(tmp_path, [{"name": "a", "cmd": counting("a")}]).run() == {"a": "done"}


def test_a_phase_left_running_has_its_process_group_ended_before_it_is_set_aside(tmp_path):
    root = tmp_path / "s"
    (root / "cal").mkdir(parents=True)
    engine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                              start_new_session=os.name == "posix")
    try:
        (root / "cal" / "manifest.json").write_text(json.dumps(
            {"phase": "cal", "status": "running", "pid": engine.pid,
             "pgid": engine.pid if os.name == "posix" else None, "host": gs.socket.gethostname()}))
        assert session(tmp_path, [{"name": "cal", "cmd": counting("cal")}]).run() == {"cal": "done"}
        assert engine.wait(timeout=30) is not None  # ended, not left writing into the attempt
        assert manifest(root, "_attempts/cal.1")["status"] == "interrupted"
    finally:
        if engine.poll() is None:
            engine.kill()


def test_a_phase_left_running_is_ended_before_any_phase_starts_even_one_this_run_leaves_out(tmp_path):
    """On the A40 a killed runner's comparison ran on for 45 minutes beside the next runner's phases, because a
    phase left running was ended only when the plan reached it."""
    root = tmp_path / "s"
    (root / "cal").mkdir(parents=True)
    engine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                              start_new_session=os.name == "posix")
    try:
        (root / "cal" / "manifest.json").write_text(json.dumps(
            {"phase": "cal", "status": "running", "pid": engine.pid,
             "pgid": engine.pid if os.name == "posix" else None, "host": gs.socket.gethostname()}))
        s = session(tmp_path, [{"name": "first", "cmd": counting("first")}, {"name": "cal", "cmd": counting("cal")}])
        assert s.run(only=["first"]) == {"first": "done"}
        assert engine.wait(timeout=30) is not None
        log = (root / "session.log").read_text(encoding="utf-8")
        assert log.index("LEFT     cal") < log.index("KILL     cal") < log.index("START    first")
        assert manifest(root, "_attempts/cal.1")["status"] == "interrupted" and not (root / "cal").exists()
    finally:
        if engine.poll() is None:
            engine.kill()


def test_the_processes_of_a_session_are_read_from_proc(tmp_path):
    """The session field is the sixth of /proc/<pid>/stat, counted after the command name, which may hold spaces
    and parentheses; a zombie is already dead and is left out."""
    stats = {"101": "101 (bash) S 1 101 500 0 -1", "102": "102 (python (x) y) S 101 102 500 0 -1",
             "103": "103 (gone) Z 101 103 500 0 -1", "104": "104 (sshd) S 1 104 104 0 -1"}
    for pid, stat in stats.items():
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "stat").write_text(stat)
    (tmp_path / "self").mkdir()
    (tmp_path / "999").mkdir()  # ended while being read: no stat
    assert gs.session_members(500, proc=tmp_path) == [101, 102]
    assert gs.session_members(500, proc=tmp_path / "missing") == []


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="sessions are read from Linux's /proc")
def test_a_child_that_leaves_the_phases_process_group_is_still_ended(tmp_path):
    """`timeout` puts its command in a process group of its own, so on the A40 ending the phase's group left
    llm-optimizer running; the phase's session still holds it."""
    pidfile = tmp_path / "child.pid"
    escape = py("import os, subprocess, sys\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
                "preexec_fn=os.setpgrp)\n"  # a group of its own, in the phase's session, as `timeout` makes
                f"open({str(pidfile)!r}, 'w').write(str(child.pid))\n")
    s = session(tmp_path, [{"name": "llmopt", "cmd": escape}])
    assert s.run() == {"llmopt": "done"}
    pid = int(pidfile.read_text())
    assert not gs.pid_alive(pid) or Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] == "Z"


def test_cleanup_runs_after_each_phase_before_its_state_is_written(tmp_path):
    seen = tmp_path / "seen.txt"
    cleanup = py("import glob, json, sys\n"
                 f"open({str(seen)!r}, 'a').write(' '.join(json.load(open(p))['status'] for p in "
                 f"sorted(glob.glob(sys.argv[1] + '/*/manifest.json'))) + chr(10))") + ["{root}"]
    s = session(tmp_path, [{"name": "a", "cmd": counting("a")}, {"name": "b", "cmd": counting("b", rc=1)}],
                cleanup=cleanup)
    assert s.run() == {"a": "done", "b": "failed"}
    # When cleanup ran, the phase that had just ended was still recorded as running.
    assert seen.read_text().splitlines() == ["running", "done running"]
    assert manifest(s.root, "b")["cleanup_exit_code"] == 0


def test_only_does_not_mark_the_session_finished(tmp_path):
    calls = []

    class Recorded(gs.Session):
        def finished(self):
            calls.append(1)

    plan = gs.read_plan({"phases": [{"name": "a", "cmd": counting("a")}, {"name": "b", "cmd": counting("b")}]})
    Recorded(plan, tmp_path / "s", versions=lambda: {}, echo=lambda s: None).run(only=["a"])
    assert calls == []
    Recorded(plan, tmp_path / "s", versions=lambda: {}, echo=lambda s: None).run()
    assert calls == [1]


def test_a_phase_whose_command_changed_is_reported_stale_not_skipped_silently(tmp_path):
    lines = []
    s = gs.Session(gs.read_plan({"phases": [{"name": "a", "cmd": counting("a")}]}), tmp_path / "s",
                   versions=lambda: {}, echo=lines.append)
    s.run()
    edited = gs.read_plan({"phases": [{"name": "a", "cmd": counting("a") + ["--edited"]}]})
    assert gs.Session(edited, tmp_path / "s", versions=lambda: {}, echo=lines.append).run() == {"a": "stale"}
    assert runs(s.root, "a") == 1 and any(line.startswith("STALE") for line in lines)
    again = gs.Session(edited, tmp_path / "s", versions=lambda: {}, echo=lines.append, redo_stale=True)
    assert again.run() == {"a": "done"} and runs(s.root, "a") == 2
    assert manifest(s.root, "_attempts/a.1")["status"] == "done"  # the old result is kept


def _bash_works() -> bool:
    import shutil
    if shutil.which("bash") is None:
        return False
    try:
        return subprocess.run(["bash", "-c", "true"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _bash_works(), reason="needs a working bash (CI's Linux has one)")
def test_a_shell_pipeline_fails_when_its_first_command_fails(tmp_path):
    s = session(tmp_path, [{"name": "a", "cmd": "false | tee {out}/log.txt"}])
    assert s.run() == {"a": "failed"}


def test_versions_are_read_from_other_interpreters_too():
    got = gs.installed_versions({"tools": sys.executable})
    assert got["tools:rigtune"] == got["rigtune"] is not None
