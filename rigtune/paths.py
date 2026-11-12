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
