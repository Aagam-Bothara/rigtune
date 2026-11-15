"""Copy-back: benchmarks/gpu_session.py writes a checksum list and an index entry as each phase ends, and
benchmarks/gpu_pull.py copies finished phases to the laptop, verified, idempotently, loudly incomplete.

The pod is a session root built by the real runner; the transport is either a fake that tars that root in
process (and can cut a copy short), or the real SshRemote driving a fake `ssh` client, so the subprocess and
streaming path is exercised too.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gs = _load("gpu_session")
gp = _load("gpu_pull")


def py(code: str) -> list:
    return ["{python}", "-c", code]


def writes(files: dict, rc: int = 0) -> list:
    return py("import os, pathlib, sys\nout = pathlib.Path(os.environ['RIGTUNE_PHASE_DIR'])\n"
              + "".join(f"(out / {k!r}).parent.mkdir(parents=True, exist_ok=True)\n"
                        f"(out / {k!r}).write_text({v!r})\n" for k, v in files.items())
              + f"sys.exit({rc})")


def make_pod(tmp_path: Path, phases=None, name: str = "pod", **kw) -> "gs.Session":
    phases = phases or [{"name": "env", "cmd": writes({"probe.json": "{}"})},
                        {"name": "cmp-qwen", "cmd": writes({"results/r.json": "[1, 2]", "verdict.md": "win"})},
                        {"name": "broken", "cmd": writes({"partial.txt": "x"}, rc=1)}]
    s = gs.Session(gs.read_plan({"session": "t", "phases": phases}), tmp_path / name,
                   versions=lambda: {}, echo=lambda s: None, **kw)
    s.run()
    return s


class FakeRemote:
    """The pod's session root, read in process. `cut` maps a directory to the number of bytes of its tar stream
    that arrive; `error` says whether the transport then reports the failure (ssh exiting 255) or not."""

    def __init__(self, root: Path, cut=None, error=True):
        self.root, self.cut, self.error, self.tars = Path(root), dict(cut or {}), error, []

    def read(self, rel):
        p = self.root / rel
        return p.read_bytes() if p.is_file() else None

    @contextmanager
    def tar(self, rel):
        self.tars.append(rel)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            tf.add(self.root / rel, arcname=rel)
        data = buf.getvalue()
        if rel in self.cut:
            data = data[:self.cut[rel]]
        yield io.BytesIO(data)
        if rel in self.cut and self.error:
            raise gp.PullError("Connection to 203.0.113.7 closed by remote host.")


def files(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- the pod side


def test_each_finished_phase_gets_a_checksum_list_and_an_index_entry(tmp_path):
    s = make_pod(tmp_path)
    index = json.loads((s.root / "index.json").read_text(encoding="utf-8"))
    assert index["finished"] is True
    assert [(u["path"], u["status"]) for u in index["units"]] == [
        ("broken", "failed"), ("cmp-qwen", "done"), ("env", "done")]
    sums = gp.parse_sums((s.root / "cmp-qwen" / "SHA256SUMS").read_bytes())
    assert sorted(sums) == ["manifest.json", "output.log", "results/r.json", "verdict.md"]
    assert gp.problems(s.root / "cmp-qwen", (s.root / "cmp-qwen" / "SHA256SUMS").read_bytes()) == []


def test_the_index_lists_a_running_phase_without_a_checksum_list(tmp_path):
    seen = {}

    def peek(argv, cwd, env, log, timeout_s):  # what the index says while the phase runs
        seen.update(json.loads((Path(cwd).parent / "index.json").read_text(encoding="utf-8")))
        return gs.Outcome(0)

    gs.Session(gs.read_plan({"phases": [{"name": "a", "cmd": "x"}, {"name": "b", "cmd": "x"}]}),
               tmp_path / "pod", versions=lambda: {}, echo=lambda s: None, runner=peek).run()
    assert seen["finished"] is False
    assert {u["path"]: u["status"] for u in seen["units"]} == {"a": "done", "b": "running"}


def test_after_phase_runs_the_users_push_command_for_each_finished_phase(tmp_path):
    log = tmp_path / "pushed.txt"
    push = py(f"import sys; open({str(log)!r}, 'a').write(sys.argv[1] + '\\n')") + ["{rel}"]
    make_pod(tmp_path, after_phase=push)
    assert sorted(log.read_text().split()) == ["broken", "cmp-qwen", "env"]


def test_a_failing_push_command_is_loud_but_does_not_fail_the_phase(tmp_path):
    lines = []
    s = gs.Session(gs.read_plan({"phases": [{"name": "a", "cmd": writes({"f": "1"})}]}), tmp_path / "pod",
                   versions=lambda: {}, echo=lines.append, after_phase=py("import sys; sys.exit(7)"))
    assert s.run() == {"a": "done"}
    assert any(line.startswith("COPY-BACK FAILED") and "exit 7" in line for line in lines)


# --------------------------------------------------------------------------- pulling


def test_a_pull_copies_every_finished_phase_verified_and_a_rerun_copies_nothing(tmp_path):
    s = make_pod(tmp_path)
    dest = tmp_path / "laptop"
    remote = FakeRemote(s.root)
    rep = gp.pull(remote, dest)
    assert rep.ok and rep.finished and sorted(rep.copied) == ["broken", "cmp-qwen", "env"]
    for unit in ("env", "cmp-qwen", "broken"):
        assert files(dest / unit) == files(s.root / unit)
    assert gp.verify(dest) == {}

    remote.tars.clear()
    rep = gp.pull(remote, dest)
    assert rep.ok and rep.copied == [] and sorted(rep.current) == ["broken", "cmp-qwen", "env"]
    assert remote.tars == []  # idempotent: nothing crossed the wire again


@pytest.mark.parametrize("error", [True, False], ids=["transport-reports-it", "silent-truncation"])
def test_an_interrupted_copy_is_incomplete_loudly_and_a_rerun_completes_it(tmp_path, error):
    s = make_pod(tmp_path)
    dest = tmp_path / "laptop"
    rep = gp.pull(FakeRemote(s.root, cut={"cmp-qwen": 1536}, error=error), dest)
    assert not rep.ok and list(rep.incomplete) == ["cmp-qwen"]
    assert not (dest / "cmp-qwen").exists()  # nothing half-copied is put in place
    assert "INCOMPLETE cmp-qwen" in "\n".join(gp.report_lines(rep))
    assert sorted(rep.copied) == ["broken", "env"]  # the other phases were not held up

    rep = gp.pull(FakeRemote(s.root), dest)
    assert rep.ok and rep.copied == ["cmp-qwen"]
    assert files(dest / "cmp-qwen") == files(s.root / "cmp-qwen")


def test_an_interrupted_copy_leaves_the_earlier_good_copy_untouched(tmp_path):
    s = make_pod(tmp_path)
    dest = tmp_path / "laptop"
    gp.pull(FakeRemote(s.root), dest)
    before = files(dest / "broken")
    # The failed phase is run again on the pod (now it passes), so its copy is out of date...
    s2 = gs.Session(gs.read_plan({"phases": [{"name": "broken", "cmd": writes({"full.txt": "y"})}]}), s.root,
                    versions=lambda: {}, echo=lambda s: None)
    s2.run()
    # ...and the copy of the new one is cut short.
    rep = gp.pull(FakeRemote(s.root, cut={"broken": 700}), dest)
    assert "broken" in rep.incomplete and files(dest / "broken") == before
    assert "broken" not in gp.verify(dest)  # still the complete earlier attempt


def test_a_file_changed_after_its_checksum_list_is_reported_not_copied(tmp_path):
    s = make_pod(tmp_path)
    (s.root / "env" / "probe.json").write_text('{"edited": true}')
    rep = gp.pull(FakeRemote(s.root), tmp_path / "laptop")
    assert rep.incomplete == {"env": ["checksum differs: probe.json"]}
    assert not (tmp_path / "laptop" / "env").exists()


def test_a_finished_phase_without_a_checksum_list_is_incomplete(tmp_path):
    s = make_pod(tmp_path)
    (s.root / "env" / "SHA256SUMS").unlink()
    rep = gp.pull(FakeRemote(s.root), tmp_path / "laptop")
    assert "no env/SHA256SUMS" in rep.incomplete["env"][0]


def test_a_member_outside_its_directory_is_refused(tmp_path):
    s = make_pod(tmp_path)

    class Evil(FakeRemote):
        @contextmanager
        def tar(self, rel):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                info = tarfile.TarInfo("../escaped.txt")
                info.size = 1
                tf.addfile(info, io.BytesIO(b"x"))
            yield io.BytesIO(buf.getvalue())

    rep = gp.pull(Evil(s.root), tmp_path / "laptop")
    assert set(rep.incomplete) == {"broken", "cmp-qwen", "env"}
    assert not list(tmp_path.rglob("escaped.txt"))


def test_a_running_phase_is_never_copied_as_complete(tmp_path):
    s = make_pod(tmp_path)
    m = json.loads((s.root / "env" / "manifest.json").read_text(encoding="utf-8"))
    index = json.loads((s.root / "index.json").read_text(encoding="utf-8"))
    index["units"].append({"path": "cal-qwen", "status": "running"})
    index["finished"] = False
    (s.root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (s.root / "cal-qwen").mkdir()
    (s.root / "cal-qwen" / "manifest.json").write_text(json.dumps({**m, "status": "running"}))
    dest = tmp_path / "laptop"
    rep = gp.pull(FakeRemote(s.root), dest, running=True)
    assert rep.running == ["cal-qwen"] and not (dest / "cal-qwen").exists()
    assert (dest / "cal-qwen.running-snapshot" / "manifest.json").exists()
    assert gp.verify(dest) == {"cal-qwen": ["still running on the pod when last pulled"],
                               "(session)": ["the runner had not finished a full run when last pulled"]}


def test_verify_reports_a_damaged_local_copy(tmp_path, capsys):
    s = make_pod(tmp_path)
    dest = tmp_path / "laptop"
    gp.pull(FakeRemote(s.root), dest)
    (dest / "cmp-qwen" / "verdict.md").write_text("loss")
    assert gp.main(["verify", "--dest", str(dest)]) == 3
    assert "INCOMPLETE cmp-qwen: checksum differs: verdict.md" in capsys.readouterr().out


def test_watch_retries_an_unreachable_pod_and_stops_when_all_is_copied(tmp_path):
    reps = iter([gp.PullError("ssh: connect to host 203.0.113.7 port 40022: Connection timed out"),
                 gp.Report(copied=["env"], running=["cal"]),
                 gp.Report(copied=["cal"], finished=True)])

    def once():
        r = next(reps)
        if isinstance(r, Exception):
            raise r
        return r

    slept, said = [], []
    assert gp.watch(once, 300, sleep=slept.append, echo=said.append) == 0
    assert slept == [300, 300] and said[0].startswith("PULL FAILED")


# --------------------------------------------------------------------------- over a (fake) ssh client


FAKE_SSH = r'''
import io, os, re, shlex, sys, tarfile
cmd = sys.argv[-1]
m = re.fullmatch(r"if \[ -f (.+?) \]; then cat .+; else exit 44; fi", cmd)
if m:
    path = shlex.split(m.group(1))[0]
    if not os.path.isfile(path):
        sys.exit(44)
    sys.stdout.buffer.write(open(path, "rb").read())
    sys.exit(0)
words = shlex.split(cmd)
if words[:2] == ["tar", "-C"] and words[3:5] == ["-cf", "-"]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(os.path.join(words[2], words[5]), arcname=words[5])
    sys.stdout.buffer.write(buf.getvalue())
    sys.exit(0)
sys.stderr.write("fake ssh: unexpected command " + cmd)
sys.exit(2)
'''


def test_a_pull_over_the_ssh_client_copies_and_verifies(tmp_path):
    s = make_pod(tmp_path, name="my pod")  # a space: every remote path is quoted
    fake = tmp_path / "fake_ssh.py"
    fake.write_text(FAKE_SSH, encoding="utf-8")
    remote = gp.SshRemote(host="203.0.113.7", port=40022, identity="id_test", root=str(s.root),
                          ssh=[sys.executable, str(fake)])
    argv = remote.argv("true")
    assert argv[argv.index("-i") + 1] == "id_test" and "root@203.0.113.7" in argv
    dest = tmp_path / "laptop"
    rep = gp.pull(remote, dest)
    assert rep.ok and sorted(rep.copied) == ["broken", "cmp-qwen", "env"]
    assert files(dest / "cmp-qwen") == files(s.root / "cmp-qwen")
    assert remote.read("no/such/file") is None



# --------------------------------------------------------------------------- the review's cases


def test_a_phases_own_tmp_file_is_listed_and_copied(tmp_path):
    """The runner lists every file but the list itself, and the puller requires exactly that set (demo 1)."""
    s = make_pod(tmp_path, [{"name": "cal", "cmd": writes({"result.json": "{}", "engine.lock.tmp": "x"})}])
    rep = gp.pull(FakeRemote(s.root), tmp_path / "laptop")
    assert rep.ok and rep.copied == ["cal"] and (tmp_path / "laptop" / "cal" / "engine.lock.tmp").exists()


def test_verify_fails_when_a_finished_phase_is_missing_and_on_an_empty_destination(tmp_path, capsys):
    s = make_pod(tmp_path)
    dest = tmp_path / "laptop"
    gp.pull(FakeRemote(s.root), dest)
    assert gp.main(["verify", "--dest", str(dest)]) == 0
    shutil.rmtree(dest / "cmp-qwen")
    assert gp.main(["verify", "--dest", str(dest)]) == 3
    assert "INCOMPLETE cmp-qwen: not copied" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert gp.main(["verify", "--dest", str(empty)]) == 3


def test_a_phase_recorded_done_without_its_list_gets_it_on_the_next_run(tmp_path, monkeypatch):
    """The runner dies after the manifest says done and before SHA256SUMS is written (demo 3)."""
    phases = [{"name": "a", "cmd": writes({"r.txt": "r"})}]

    def dies(unit, staging):
        raise KeyboardInterrupt("runner killed while hashing")

    monkeypatch.setattr(gs, "write_sums", dies)
    with pytest.raises(KeyboardInterrupt):
        make_pod(tmp_path, phases)
    monkeypatch.undo()
    pod = tmp_path / "pod"
    assert json.loads((pod / "a" / "manifest.json").read_text())["status"] == "done"
    assert not (pod / "a" / "SHA256SUMS").exists()
    s = gs.Session(gs.read_plan({"phases": phases}), pod, versions=lambda: {}, echo=lambda s: None)
    assert s.run() == {"a": "done"} and (pod / "a" / "SHA256SUMS").exists()
    rep = gp.pull(FakeRemote(pod), tmp_path / "laptop")
    assert rep.ok and rep.copied == ["a"]


def test_a_verified_copy_is_kept_when_a_rerun_replaces_it(tmp_path):
    """A redone phase that fails replaces the local copy, and the good one is kept, not deleted (demo 8)."""
    s = make_pod(tmp_path, [{"name": "cal", "cmd": writes({"profile.json": "THE GOOD RESULT"})}])
    dest = tmp_path / "laptop"
    gp.pull(FakeRemote(s.root), dest)
    s2 = gs.Session(gs.read_plan({"phases": [{"name": "cal", "cmd": writes({"log": "x"}, rc=1)}]}), s.root,
                    versions=lambda: {}, echo=lambda s: None)
    s2.run(redo=["cal"])
    rep = gp.pull(FakeRemote(s.root, cut={"_attempts/cal.1": 600}), dest)
    assert "cal" in rep.copied and rep.superseded == ["_superseded/cal.1"]
    assert (dest / "_superseded" / "cal.1" / "profile.json").read_text() == "THE GOOD RESULT"
    assert json.loads((dest / "cal" / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("ssh", 120), PermissionError(13, "in use")],
                         ids=["ssh-timeout", "file-locked"])
def test_watch_survives_any_failed_pass(error):
    calls, said = [], []

    def once():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return gp.Report(finished=True)

    assert gp.watch(once, 5, sleep=lambda s: None, echo=said.append) == 0
    assert len(calls) == 2 and said[0].startswith(f"PULL FAILED ({type(error).__name__})")


def test_watch_stops_on_a_configuration_error():
    def once():
        raise gp.ConfigError("--remote is a Windows path")

    with pytest.raises(gp.ConfigError):
        gp.watch(once, 5, sleep=lambda s: None, echo=lambda s: None)


@pytest.mark.parametrize("remote", ["C:/Program Files/Git/workspace/session", "C:\\workspace\\session",
                                    "workspace/session"])
def test_a_remote_root_git_bash_rewrote_is_refused(remote, capsys):
    assert gp.main(["pull", "--host", "203.0.113.7", "--port", "40022", "--remote", remote,
                    "--dest", "unused"]) == 2
    err = capsys.readouterr().err
    assert "MSYS_NO_PATHCONV=1" in err or "absolute path" in err
    assert gp.check_remote("//workspace/session") == "/workspace/session"


def test_a_phase_deeper_than_260_characters_is_copied(tmp_path):
    deep = "/".join(["a-rather-long-directory-name-%02d" % i for i in range(4)]) + "/result.json"
    s = make_pod(tmp_path, [{"name": "cal", "cmd": writes({deep: "{}"})}])
    dest = tmp_path / ("laptop-" + "d" * 120)  # the pod's side stays under 260; the copy goes past it
    assert len(str((dest / "cal" / deep).resolve())) > 260
    rep = gp.pull(FakeRemote(s.root), dest)
    assert rep.ok and rep.copied == ["cal"]
    assert gp.verify(dest) == {}


def test_the_roots_own_logs_are_copied(tmp_path):
    s = make_pod(tmp_path)
    (s.root / "runner.out").write_text("session: 2 done")
    s.write_index(finished=True)
    dest = tmp_path / "laptop"
    gp.pull(FakeRemote(s.root), dest)
    assert (dest / "_root" / "runner.out").read_text() == "session: 2 done"
    assert "START" in (dest / "_root" / "session.log").read_text()
    assert not (dest / "_root" / "runner.lock").exists()
