"""The engine releases rigtune was checked with, and a warning for any other.

rigtune's engine rules are read from particular releases' source (memory planning, flags, speculative methods)
and measured on particular releases (fp8 on Ampere, which proposers start). A release it was not checked with may
take a flag differently or plan memory another way, and nothing would say so: SGLang 0.5.23 changed a 24 GB
card's default prefill chunk behind an unchanged flag. So each release rigtune was run with is listed in
rigtune/data/conformance/<engine>-<version>.json, one record per machine and session, and any other release gets
a note: logged once, added to the profile's notes, and never a refusal. Versions are compared exactly, so a patch
release is unchecked until it is listed.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATA = Path(__file__).parent / "data" / "conformance"
CHECKED_ENGINES = ("vllm", "sglang")  # the engines with a table; others are never warned about


class Tested(BaseModel):
    """One machine and session an engine release ran on with rigtune."""

    engine: str
    version: str
    card: str
    date: str
    torch: Optional[str] = None  # None where the session did not record it
    transformers: Optional[str] = None
    checks: Dict[str, str] = Field(default_factory=dict)  # what was run -> "pass" | "fail" | "skip"
    source: str = ""  # where the evidence is


@functools.lru_cache(maxsize=1)
def tested() -> Dict[str, List[Tested]]:
    """Every record under rigtune/data/conformance, by engine, ordered by version and date."""
    out: Dict[str, List[Tested]] = {}
    for path in sorted(DATA.glob("*.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            t = Tested.model_validate(rec)
            out.setdefault(t.engine, []).append(t)
    for recs in out.values():
        recs.sort(key=lambda t: (_order(t.version), t.date, t.card))
    return out


def _order(version: str):
    return tuple(int(p) if p.isdigit() else -1 for p in version.replace("-", ".").split("."))


def checked_versions(engine: str) -> List[str]:
    return list(dict.fromkeys(t.version for t in tested().get(engine, [])))


def support_note(engine: str, version: Optional[str]) -> Optional[str]:
    """Why `engine` `version` deserves a warning, or None: a listed release, an engine with no table, or no
    version known (not installed)."""
    if engine not in CHECKED_ENGINES or not version:
        return None
    checked = checked_versions(engine)
    if version in checked:
        return None
    return (f"{engine} {version} has not been checked with rigtune (checked: {', '.join(checked) or 'none'}); "
            f"its flags, defaults and memory rules may differ from what rigtune plans for")


_warned: set = set()


def warn_once(note: str) -> None:
    if note not in _warned:
        _warned.add(note)
        logger.warning(note)
