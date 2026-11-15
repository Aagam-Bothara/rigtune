#!/usr/bin/env python
"""Copy a GPU session's finished phases from the pod to this machine, each verified against its checksum list.

Runs on the laptop (Windows Git Bash, macOS or Linux; the Python standard library and an `ssh` client only):

    MSYS_NO_PATHCONV=1 python benchmarks/gpu_pull.py pull --host 203.0.113.7 --port 40022 \
        --identity ~/.ssh/id_ed25519 --remote /workspace/session --dest gpu-results/session --watch 300
    python benchmarks/gpu_pull.py verify --dest gpu-results/session

benchmarks/gpu_session.py writes <phase>/SHA256SUMS when a phase reaches a final state (done, failed, timeout,
skipped, blocked, interrupted) and lists every such directory in <root>/index.json. A pull reads the index,
and for each listed directory whose local copy is missing or differs, streams it over SSH as one tar
(`tar -cf -` on the pod, unpacked here by Python's tarfile, so no tar or rsync is needed locally), unpacks it
into a short staging folder beside the destination, and checks every file against the pod's SHA256SUMS: each
listed file present with that hash, and no file the list does not name. Only then does it put the copy in
place; a local copy it replaces is moved to _superseded/<phase>.<n>, never deleted. A copy cut short, a file
changed after its list was written, or a list that is missing leaves the local copy as it was, prints
INCOMPLETE with what is wrong, and exits 3. Running it again is always safe: a directory whose local copy
already verifies is not copied again. The root's own logs (session.log, copyback.log, the runner's output) are
copied as they are into _root/, unverified, on every pull.

`verify` checks the local copy against the pod's index as last pulled: every directory the pod recorded as
finished present and matching its list, and the runner finished. It exits 3 otherwise, and on an empty
destination.

RunPod: this needs SSH over the pod's exposed TCP port (the "SSH over exposed TCP" line in the pod's Connect
tab: a public IP and a port mapped to 22). The proxied `ssh <pod>@ssh.runpod.io` login does not support scp or
sftp, and a non-interactive command such as this one's is not documented to work through it either. The key
is read by ssh from --identity; this script never reads, stores or prints it. In Git Bash, MSYS rewrites an
argument such as /workspace/session into a Windows path before Python sees it; this script refuses such a
--remote, and MSYS_NO_PATHCONV=1 stops the rewriting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Callable, Dict, Iterator, List, Optional, Protocol

SUMS = "SHA256SUMS"
INDEX = "index.json"
FINAL = ("done", "failed", "timeout", "skipped", "blocked", "interrupted")
STAGING = ".p"
SUPERSEDED = "_superseded"
ROOT_LOGS = "_root"
WINDOWS_PATH = re.compile(r"^[A-Za-z]:|\\|Program Files", re.I)


class PullError(RuntimeError):
    """The pod could not be read (connection, missing index): nothing was changed locally."""


class ConfigError(ValueError):
    """An argument that can never work (a Windows path as the pod's root): retrying will not help."""


def check_remote(root: str) -> str:
    if WINDOWS_PATH.search(root):
        raise ConfigError(f"--remote {root!r} looks like a Windows path, so it cannot be the pod's session root. "
                          "Git Bash rewrites /workspace/... into a Windows path before Python sees it: run the "
                          "command with MSYS_NO_PATHCONV=1 in front, or write the path as //workspace/session")
    if not root.startswith("/"):
        raise ConfigError(f"--remote {root!r} must be the absolute path of the session root on the pod")
    return root.replace("//", "/") if root.startswith("//") else root


def _lp(p: Path) -> Path:
    """On Windows, the extended-length form of a path (\\\\?\\C:\\...), so a deep phase directory under a long
    destination is not cut off at 260 characters. Unchanged elsewhere."""
    if os.name != "nt":
        return Path(p)
    s = str(p)
    if s.startswith("\\\\?\\"):
        return Path(s)
    return Path("\\\\?\\" + os.path.abspath(s))


class Remote(Protocol):
    def read(self, rel: str) -> Optional[bytes]:
        """A file under the session root, or None when it does not exist."""

    def tar(self, rel: str):
        """A context manager yielding a binary stream: `tar -cf -` of one directory under the root. It raises
        PullError on leaving if the stream did not end cleanly."""


@dataclass
class SshRemote:
    """The pod over SSH. `ssh` is the client's argument list before the destination (a test substitutes a fake
    client here); every remote path is quoted for the pod's shell."""

    host: str
    root: str
    port: int = 22
    user: str = "root"
    identity: Optional[str] = None
    ssh: List[str] = field(default_factory=lambda: ["ssh"])
    options: List[str] = field(default_factory=list)

    def argv(self, command: str) -> List[str]:
        out = [*self.ssh, "-T", "-p", str(self.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
        if self.identity:
            out += ["-i", self.identity]
        for o in self.options:
            out += ["-o", o]
        return [*out, f"{self.user}@{self.host}", command]

    def _path(self, rel: str) -> str:
        return str(PurePosixPath(self.root) / rel)

    def read(self, rel: str) -> Optional[bytes]:
        p = shlex.quote(self._path(rel))
        got = subprocess.run(self.argv(f"if [ -f {p} ]; then cat {p}; else exit 44; fi"), capture_output=True,
                             timeout=120, stdin=subprocess.DEVNULL)
        if got.returncode == 44:
            return None
        if got.returncode != 0:
            raise PullError(f"ssh to the pod failed (exit {got.returncode}): "
                            f"{got.stderr.decode(errors='replace').strip()[-300:]}")
        return got.stdout

    @contextmanager
    def tar(self, rel: str) -> Iterator[IO[bytes]]:
        cmd = f"tar -C {shlex.quote(self.root)} -cf - {shlex.quote(rel)}"
        proc = subprocess.Popen(self.argv(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL)
        assert proc.stdout is not None
        try:
            yield proc.stdout
        finally:
            proc.stdout.read()  # drain what is left so the process can exit
            _, err = proc.communicate(timeout=600)
            if proc.returncode != 0:
                raise PullError(f"tar of {rel} on the pod ended with exit {proc.returncode}: "
                                f"{err.decode(errors='replace').strip()[-300:]}")


# --------------------------------------------------------------------------- checking


def parse_sums(text: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if len(digest) != 64 or not name:
            raise ValueError(f"not a sha256sum line: {line!r}")
        out[name] = digest
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def problems(unit: Path, sums: bytes) -> List[str]:
    """What is wrong with a local copy of one directory against its SHA256SUMS: nothing when it is complete.
    Every file but the list itself must be named in it, exactly as the runner writes it."""
    unit = _lp(unit)
    if not unit.is_dir():
        return ["not copied"]
    try:
        want = parse_sums(sums)
    except ValueError as exc:
        return [str(exc)]
    have = {p.relative_to(unit).as_posix() for p in unit.rglob("*") if p.is_file()} - {SUMS}
    out = [f"missing {n}" for n in sorted(set(want) - have)]
    out += [f"not in its checksum list: {n}" for n in sorted(have - set(want))]
    out += [f"checksum differs: {n}" for n in sorted(set(want) & have) if sha256(unit / n) != want[n]]
    if (unit / SUMS).is_file() and (unit / SUMS).read_bytes() != sums:
        out.append(f"{SUMS} differs from the pod's")
    return out


def _unpack(stream: IO[bytes], rel: str, into: Path) -> None:
    """Unpack a tar stream of `rel` into `into`, which becomes the directory itself (members are written without
    the `rel/` prefix, so the staging path is no longer than the final one). Any member outside `rel`, and
    anything but plain files and directories (a link, a device), is refused: the stream comes from another
    machine."""
    top = PurePosixPath(rel)
    into = _lp(into)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            name = PurePosixPath(m.name)
            if name.is_absolute() or ".." in name.parts or (name != top and top not in name.parents):
                raise ValueError(f"unexpected member {m.name!r} in the stream for {rel}")
            inner = name.relative_to(top)
            dest = into.joinpath(*inner.parts) if inner.parts else into
            if m.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                assert src is not None
                with dest.open("wb") as fh:
                    shutil.copyfileobj(src, fh)
            else:
                raise ValueError(f"member {m.name!r} is not a plain file or directory")


# --------------------------------------------------------------------------- pulling


@dataclass
class Report:
    copied: List[str] = field(default_factory=list)
    current: List[str] = field(default_factory=list)
    incomplete: Dict[str, List[str]] = field(default_factory=dict)
    running: List[str] = field(default_factory=list)
    superseded: List[str] = field(default_factory=list)
    finished: bool = False  # the runner finished every phase of a full run, per the index

    @property
    def ok(self) -> bool:
        return not self.incomplete


def _local(dest: Path, rel: str) -> Path:
    return dest.joinpath(*PurePosixPath(rel).parts)


def _staging(dest: Path, rel: str) -> Path:
    return dest / STAGING / hashlib.sha256(rel.encode()).hexdigest()[:10]


def _supersede(dest: Path, rel: str) -> str:
    """Move the local copy of `rel` to _superseded/<rel with / as __>.<n>; its new name."""
    base = rel.replace("/", "__")
    n = 1
    while _lp(dest / SUPERSEDED / f"{base}.{n}").exists():
        n += 1
    target = dest / SUPERSEDED / f"{base}.{n}"
    _lp(target.parent).mkdir(parents=True, exist_ok=True)
    os.replace(_lp(_local(dest, rel)), _lp(target))
    return target.relative_to(dest).as_posix()


def pull_unit(remote: Remote, dest: Path, rel: str, rep: Optional[Report] = None) -> Optional[List[str]]:
    """Copy one finished directory if its local copy does not verify. None when it was already current, [] when
    it was copied and verified, else what is wrong (the local copy is then left untouched). A local copy that a
    verified one replaces is kept under _superseded/."""
    sums = remote.read(f"{rel}/{SUMS}")
    if sums is None:
        return [f"the pod has no {rel}/{SUMS}, though its index lists the phase as finished"]
    local = _local(dest, rel)
    if not problems(local, sums):
        return None
    staging = _staging(dest, rel)
    if _lp(staging).exists():
        shutil.rmtree(_lp(staging))
    _lp(staging).mkdir(parents=True)
    try:
        with remote.tar(rel) as stream:
            _unpack(stream, rel, staging)
    except (PullError, tarfile.TarError, ValueError, OSError, EOFError) as exc:
        return [f"copy did not complete: {exc}"]
    wrong = problems(staging, sums)
    if wrong:
        return wrong
    if _lp(local).exists():
        moved = _supersede(dest, rel)
        if rep is not None:
            rep.superseded.append(moved)
    _lp(local.parent).mkdir(parents=True, exist_ok=True)
    os.replace(_lp(staging), _lp(local))
    return []


def pull(remote: Remote, dest: Path, running: bool = False) -> Report:
    raw = remote.read(INDEX)
    if raw is None:
        raise PullError(f"the pod has no {INDEX} under the session root: is --remote the root given to "
                        "gpu_session.py, and has its first phase ended?")
    index = json.loads(raw.decode("utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    rep = Report(finished=bool(index.get("finished")))
    for unit in sorted(index.get("units", []), key=lambda u: u["path"]):
        rel, status = unit["path"], unit.get("status")
        if status not in FINAL:
            rep.running.append(rel)
            if running:  # a look at a phase still being written: copied as it is, never called complete
                snap = _local(dest, rel + ".running-snapshot")
                if _lp(snap).exists():
                    shutil.rmtree(_lp(snap))
                _lp(snap).mkdir(parents=True)
                try:
                    with remote.tar(rel) as stream:
                        _unpack(stream, rel, snap)
                except (PullError, tarfile.TarError, ValueError, OSError, EOFError):
                    pass
            continue
        got = pull_unit(remote, dest, rel, rep)
        if got is None:
            rep.current.append(rel)
        elif got:
            rep.incomplete[rel] = got
        else:
            rep.copied.append(rel)
    for name in index.get("logs", []):  # live files: copied as they are, never verified
        if "/" in name or "\\" in name or name.startswith("."):
            continue
        data = remote.read(name)
        if data is not None:
            (dest / ROOT_LOGS).mkdir(exist_ok=True)
            (dest / ROOT_LOGS / name).write_bytes(data)
    (dest / INDEX).write_bytes(raw)
    return rep


def verify(dest: Path) -> Dict[str, List[str]]:
    """What is missing or wrong in a local copy, against the pod's index as last pulled into `dest`: every
    directory the pod recorded as finished must be here and match its list, and the runner must have finished.
    Empty when the copy is complete."""
    try:
        index = json.loads((dest / INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {INDEX: [f"no readable {INDEX} in {dest}: nothing has been pulled here"]}
    out: Dict[str, List[str]] = {}
    for unit in sorted(index.get("units", []), key=lambda u: u["path"]):
        rel = unit["path"]
        if unit.get("status") not in FINAL:
            out[rel] = [f"still {unit.get('status')} on the pod when last pulled"]
            continue
        local = _local(dest, rel)
        sums = _lp(local / SUMS)
        wrong = problems(local, sums.read_bytes()) if sums.is_file() else ["not copied"]
        if wrong:
            out[rel] = wrong
    if not index.get("finished"):
        out["(session)"] = ["the runner had not finished a full run when last pulled"]
    return out


def report_lines(rep: Report) -> List[str]:
    lines = [f"copied and verified: {', '.join(rep.copied) or '-'}",
             f"already current: {len(rep.current)}"]
    if rep.superseded:
        lines.append(f"earlier local copies kept: {', '.join(rep.superseded)}")
    if rep.running:
        lines.append(f"still running on the pod (not copied as complete): {', '.join(rep.running)}")
    for rel, wrong in sorted(rep.incomplete.items()):
        lines.append(f"INCOMPLETE {rel}: " + "; ".join(wrong[:8]) + (" ..." if len(wrong) > 8 else ""))
    lines.append("the session runner has finished" if rep.finished else "the session runner has not finished")
    return lines


def watch(do_pull: Callable[[], Report], interval: float, sleep: Callable[[float], None] = time.sleep,
          echo: Callable[[str], None] = print, max_passes: Optional[int] = None) -> int:
    """Pull every `interval` seconds until the runner has finished and everything is copied and verified. A
    pass that fails for any reason (the pod unreachable, ssh timing out, a file locked on this machine) is
    reported and retried at the next interval; only a ConfigError or Ctrl-C ends the loop."""
    passes = 0
    while True:
        passes += 1
        try:
            rep: Optional[Report] = do_pull()
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001 - anything else is retried: a lost pull is worse than a noisy one
            echo(f"PULL FAILED ({type(exc).__name__}): {exc}; retrying in {interval:g}s")
            rep = None
        if rep is not None:
            for line in report_lines(rep):
                echo(line)
            if rep.finished and rep.ok and not rep.running:
                return 0
        if max_passes is not None and passes >= max_passes:
            return 3 if rep is None or not rep.ok else 1
        sleep(interval)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull", help="copy every finished phase not yet copied, and verify it")
    p.add_argument("--host", required=True, help="the pod's public IP (SSH over exposed TCP)")
    p.add_argument("--port", type=int, required=True, help="the public port mapped to the pod's port 22")
    p.add_argument("--user", default="root")
    p.add_argument("--identity", default=None, help="the private key file ssh should use")
    p.add_argument("--ssh-option", action="append", default=[], help="passed to ssh as -o (repeatable)")
    p.add_argument("--remote", required=True, help="the session root on the pod")
    p.add_argument("--dest", type=Path, required=True, help="where the copy goes on this machine")
    p.add_argument("--running", action="store_true",
                   help="also copy each running phase as it is, to <phase>.running-snapshot (unverified)")
    p.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                   help="pull again every SECONDS until the session has finished and all of it is copied")
    p = sub.add_parser("verify", help="check a local copy against the pod's index and each checksum list")
    p.add_argument("--dest", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        bad = verify(a.dest)
        for rel, wrong in sorted(bad.items()):
            print(f"INCOMPLETE {rel}: " + "; ".join(wrong[:8]))
        print("complete: every finished phase is here and verified" if not bad
              else f"{len(bad)} problem(s): the copy is not complete")
        return 3 if bad else 0
    try:
        remote = SshRemote(host=a.host, port=a.port, user=a.user, identity=a.identity,
                           root=check_remote(a.remote), options=a.ssh_option)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def once() -> Report:
        return pull(remote, a.dest, running=a.running)

    if a.watch:
        return watch(once, a.watch)
    try:
        rep = once()
    except (PullError, OSError, subprocess.SubprocessError) as exc:
        print(f"PULL FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    for line in report_lines(rep):
        print(line)
    return 0 if rep.ok else 3


if __name__ == "__main__":
    sys.exit(main())
