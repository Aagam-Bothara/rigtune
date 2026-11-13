"""Where a disk KV-cache tier may live (--kv-tier).

A disk tier's files are the KV cache of the prompts it served, so they are derived from user prompts: they are
created only under rigtune's own cache dir, never on a network filesystem such as RunPod's /workspace volume
(storage that outlives the pod), and removed after every trial (SECURITY.md).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from rigtune import paths

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
