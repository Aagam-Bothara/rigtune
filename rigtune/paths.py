"""Where rigtune keeps its state, and the environment variables that move it.

rigtune was called PolyServe until September 2026. A calibration takes half an hour to an hour of GPU time, so an
upgrade keeps finding the ones made under the old name: the old environment variables are still read, and
~/.polyserve is used for as long as ~/.rigtune does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

FORMER = "polyserve"


def env(name: str) -> Optional[str]:
    """RIGTUNE_<name>, or the POLYSERVE_<name> it replaced; the new one wins when both are set."""
    return os.environ.get(f"RIGTUNE_{name}") or os.environ.get(f"POLYSERVE_{name}")


def home() -> Path:
    """Profiles, downloaded GGUFs and the performance and memory models fitted to this machine."""
    given = env("HOME")
    if given:
        return Path(given)
    current, former = Path.home() / ".rigtune", Path.home() / f".{FORMER}"
    return former if former.is_dir() and not current.exists() else current


def pairs_dir() -> Path:
    """Checked second-model evidence; independently relocatable for offline sites."""
    return Path(env("PAIRS")) if env("PAIRS") else home() / "pairs"


_OFF = ("off", "0", "false", "no", "none", "disabled")  # as RIGTUNE_LEDGER reads them, in any case


def kernel_cache_dir() -> Optional[Path]:
    """Where engines keep compiled kernels and their autotune results from one launch to the next, or None when
    RIGTUNE_KERNEL_CACHE turns it off. A value that turns it off is never read as a directory of that name."""
    given = (env("KERNEL_CACHE") or "").strip()
    if given.lower() in _OFF:
        return None
    return Path(given) if given else home() / "kernel-cache"
