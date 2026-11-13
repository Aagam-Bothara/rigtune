"""Where a disk KV-cache tier may live (--kv-tier), and keeping it from outliving the process that made it.

A disk tier's files are the KV cache of the prompts it served, so they are derived from user prompts: they are
created only under rigtune's own cache dir, never on a network filesystem such as RunPod's /workspace volume
(storage that outlives the pod), and removed after every trial (SECURITY.md).
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import signal
import threading
from pathlib import Path
from typing import Any, Optional, Set

from rigtune import paths

logger = logging.getLogger(__name__)

TIER_DIR = "kv-tier"
# Filesystem types a disk tier is never written to: network and FUSE-mounted storage (RunPod's persistent
# volume among them), read from /proc/mounts. The path list is the fallback where /proc/mounts cannot be read.
NETWORK_FS = ("nfs", "nfs4", "cifs", "smb3", "ceph", "glusterfs", "9p")
NETWORK_VOLUMES = ("/workspace",)
MOUNTS = "/proc/mounts"


def tier_root() -> Path:
    """The directory every disk tier lives under: <rigtune home>/kv-tier."""
    return paths.home() / TIER_DIR


def _mount_type(where: str, mounts: str) -> Optional[str]:
    """The filesystem type of the mount `where` (an absolute, resolved POSIX path) is on, from `mounts`."""
    try:
        with open(mounts, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    best, kind = "", None
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[1])  # \040 is a space
        inside = where == point or where.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, kind = point, fields[2]
    return kind


def refusal(root: Optional[Path] = None, resolve: bool = True, mounts: str = MOUNTS) -> Optional[str]:
    """Why no disk tier may be written under `root` (default tier_root()), or None when it may: the mount it
    is on is a network or FUSE filesystem (NETWORK_FS, fuse.*), or, where the mount table cannot be read, its
    path is a known network volume (NETWORK_VOLUMES). Symbolic links are followed on POSIX, so a home linked
    into one is caught too."""
    root = Path(root) if root is not None else tier_root()
    where = os.path.realpath(root) if resolve and os.name == "posix" else str(root)
    where = where.replace("\\", "/")
    hint = ("a disk KV tier holds the served prompts' KV cache and is never written there (point "
            "RIGTUNE_HOME at a local disk)")
    kind = _mount_type(where, mounts)
    if kind is not None and (kind in NETWORK_FS or kind.startswith("fuse.")):
        return f"rigtune's cache dir {where} is on a {kind} filesystem; {hint}"
    for volume in NETWORK_VOLUMES:
        if where == volume or where.startswith(volume.rstrip("/") + "/"):
            return f"rigtune's cache dir {where} is on the network volume {volume}; {hint}"
    return None


# --------------------------------------------------------------------------- one directory per engine

DIR_PLACEHOLDER = "<rigtune kv-tier dir>"  # in a profile's recorded launch: each server makes its own
_owned: Set[Path] = set()  # the tier directories this process made and has not removed yet
_hooks_installed = False
_DIR_NAME = re.compile(r"pid(\d+)-port(\d+)")


def trial_dir(port: int) -> Path:
    """The disk tier of the engine this process runs on `port`: <tier root>/pid<rigtune pid>-port<port>. One
    directory per engine, so two engines never share files: SGLang 0.5.19's file backend keeps what it wrote
    across restarts and finds it again by name (mem_cache/hicache_storage.HiCacheFile.exists,
    storage/file/lru_file_evictor._scan_existing_files), and a trial finding an earlier trial's files would
    measure hits it never earned. The pid lets a later rigtune sweep a directory whose owner died (sweep)."""
    return tier_root() / f"pid{os.getpid()}-port{int(port)}"


def prepare(port: int) -> Path:
    """An empty directory for the engine on `port`, readable by this user only (0700), removed again on exit
    and on SIGTERM if it is still there. Raises OSError (a full disk, say) for the trial to record."""
    path = trial_dir(port)
    clear(port)
    tier_root().mkdir(parents=True, exist_ok=True)
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)  # mkdir's mode is masked by the umask
    _owned.add(path)
    _install_hooks()
    return path


def clear(port: int) -> None:
    """Remove the disk tier of the engine on `port`, before it starts and after it stops. Only a directory
    under tier_root() is ever removed."""
    _remove(trial_dir(port))


def _remove(path: Path) -> None:
    _owned.discard(path)
    if path.parent != tier_root() or not path.is_dir():
        return
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        logger.warning("could not remove the disk KV tier at %s; delete it by hand (it holds served "
                       "prompts' KV cache)", path)


def sweep() -> int:
    """Remove every tier directory whose owning rigtune process is gone: what a SIGKILL or an out-of-memory
    kill left behind. Run when a calibration or a server starts. Returns how many were removed."""
    import psutil

    root, removed = tier_root(), 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        m = _DIR_NAME.fullmatch(entry.name)
        if m and entry.is_dir() and int(m.group(1)) != os.getpid() and not psutil.pid_exists(int(m.group(1))):
            _remove(entry)
            removed += 1
    if removed:
        logger.info("removed %d disk KV tier directories left by rigtune processes that are gone", removed)
    return removed


def cleanup() -> None:
    """Remove every tier directory this process still owns (on exit, and on SIGTERM)."""
    for path in list(_owned):
        _remove(path)


def _install_hooks() -> None:
    """On interpreter exit and on SIGTERM, as rigtune.power restores the GPU. SIGKILL and an out-of-memory
    kill run neither; the next sweep() removes what they leave."""
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    atexit.register(cleanup)
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        previous = signal.getsignal(signal.SIGTERM)

        def _on_term(signum: int, frame: Any) -> None:
            cleanup()
            if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                previous(signum, frame)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError, AttributeError):
        pass
