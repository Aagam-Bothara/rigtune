"""Profile cache: ~/.rigtune/profiles/<hardware_hash>/<model_id>/<objective>.json"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from rigtune import paths
from rigtune.hardware import hardware_hash
from rigtune.hfconfig import snapshot_revision
from rigtune.models import HardwareDescriptor, ModelSpec, Profile

logger = logging.getLogger(__name__)

# How much of each engine library's version must match for a cached profile to be reused: torch, transformers and
# tokenizers change kernels and model code in minor releases, and a patch release only reuses with a note. FlashInfer
# ships the attention kernels themselves, so any release of it can change what was measured. A library not listed
# here is compared exactly.
LIBRARY_MATCH: Dict[str, Optional[int]] = {"torch": 2, "transformers": 2, "tokenizers": 2, "flashinfer": None}


def rigtune_home() -> Path:
    return paths.home()


def profiles_dir() -> Path:
    return rigtune_home() / "profiles"


def logs_dir() -> Path:
    return rigtune_home() / "logs"


def profile_path(hw_hash: str, model: ModelSpec, objective: str, workload: str = "default",
                 power: str = "off", phases: str = "unified", options: Optional[Dict[str, str]] = None) -> Path:
    name = objective if workload == "default" else f"{objective}-{workload}"
    if power != "off":
        name += f"-power-{power}"
    if phases != "unified":
        name += f"-phases-{phases}"
    for k in sorted(options or {}):
        name += f"-{k}-{str(options[k]).replace('/', '_')}"
    if model.revision is not None:  # an explicitly requested revision (ModelSpec.revision) is its own profile
        name += "-rev-" + hashlib.sha256(model.revision.encode()).hexdigest()[:16]
    return profiles_dir() / hw_hash / model.safe_id / f"{name}.json"


def save(profile: Profile, requested: Optional[ModelSpec] = None) -> Path:
    """Write the profile where `load` looks for it: under the revision the user asked for.

    `requested` is the model as the user named it. A verified-drafts calibration pins `prepared.spec.revision` to
    the commit it checked, and a profile filed under that pin would never be found by the next request, which
    names no commit; the pin is recorded in the profile (`model_revision`) and compared on reuse instead. Without
    `requested` the prepared model's revision is used, which is the requested one whenever nothing was pinned, so
    profiles calibrated without verified drafts keep the names they always had."""
    if requested is not None:
        revision = requested.revision
    else:
        revision = profile.prepared.spec.revision if profile.prepared is not None else None
    path = profile_path(profile.hardware_hash, ModelSpec(hf_id=profile.model_id, revision=revision),
                        profile.objective, profile.workload, profile.power_mode, profile.phases, profile.options)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A staging file of its own per writer, so two saves of one profile never replace each other's half-written
    # file. Its name is short and fixed in length, so a long profile name cannot push it past Windows' MAX_PATH.
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".tmp")
    tmp = Path(staged)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(profile.model_dump_json(indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info("saved profile %s", path)
    return path


def load(hw: HardwareDescriptor, model: ModelSpec, objective: str, workload: str = "default",
         power: str = "off", phases: str = "unified", options: Optional[Dict[str, str]] = None) -> Optional[Profile]:
    """Return a cached profile if present and still valid for this hardware + backend version."""
    path = profile_path(hardware_hash(hw), model, objective, workload, power, phases, options)
    if not path.exists():
        return None
    try:
        profile = Profile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("ignoring unreadable profile %s: %s", path, exc)
        return None
    reason = invalid_reason(profile, hw) or revision_reason(profile, model)
    if reason:
        logger.info("cached profile invalid (%s); recalibrating", reason)
        return None
    for note in reuse_notes(profile, hw, model):
        logger.info("reusing the cached profile: %s", note)
    return profile


def invalid_reason(profile: Profile, hw: HardwareDescriptor) -> Optional[str]:
    if profile.hardware_hash != hardware_hash(hw):
        return "hardware changed"
    current = hw.backend_version(profile.backend)
    if profile.backend_version != current:
        return f"{profile.backend} version changed ({profile.backend_version} -> {current})"
    if not hw.backend_available(profile.backend):
        return f"{profile.backend} no longer available"
    changed, _ = library_changes(profile, hw)
    if changed:
        return f"{profile.backend} libraries changed ({', '.join(changed)})"
    old = [(g.driver_version, g.cuda_version) for g in profile.hardware.gpus]
    new = [(g.driver_version, g.cuda_version) for g in hw.gpus]
    if old != new:
        return f"GPU driver or CUDA version changed ({_drivers(old)} -> {_drivers(new)})"
    return None


def _drivers(pairs: List[Tuple[Optional[str], Optional[str]]]) -> str:
    return ", ".join(f"driver {d or '?'} CUDA {c or '?'}" for d, c in dict.fromkeys(pairs)) or "no GPU"


def _release(version: str) -> Optional[Tuple[int, ...]]:
    m = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    return tuple(int(x) for x in m.groups() if x is not None) if m else None


def library_changes(profile: Profile, hw: HardwareDescriptor) -> Tuple[List[str], List[str]]:
    """The winning engine's libraries that differ from calibration: (changes that invalidate the profile, changes
    that only earn a note), per LIBRARY_MATCH. Libraries are compared only when both sides recorded some: profiles
    and probes from before they were read, or an engine environment that could not be asked, record none."""
    recorded = profile.hardware.backends.get(profile.backend)
    present = hw.backends.get(profile.backend)
    old = recorded.libs if recorded is not None else {}
    new = present.libs if present is not None else {}
    if not old or not new:
        return [], []
    changed: List[str] = []
    patch: List[str] = []
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name), new.get(name)
        if a == b:
            continue
        text = f"{name} {a or 'absent'} -> {b or 'absent'}"
        depth = LIBRARY_MATCH.get(name)
        ra, rb = (_release(a), _release(b)) if a and b else (None, None)
        if depth is not None and ra is not None and rb is not None and ra[:depth] == rb[:depth]:
            patch.append(text)
        else:
            changed.append(text)
    return changed, patch


def revision_reason(profile: Profile, model: ModelSpec) -> Optional[str]:
    """Set when the local snapshot of the model is no longer the commit the profile was calibrated on. Compared
    only when both are known; resolving the current commit never asks the Hub.

    A profile whose calibration pinned the model to a commit the request did not name (verified drafts) launches
    at that commit whatever the local `main` points at, so it is not compared: `rigtune recalibrate` moves it."""
    pinned = profile.prepared.spec.revision if profile.prepared is not None else None
    if pinned is not None and pinned != model.revision and re.fullmatch(r"[a-f0-9]{40}", pinned):
        return None
    current = snapshot_revision(model) if profile.model_revision else None
    if current is not None and current != profile.model_revision:
        return f"{model.hf_id} is now commit {current[:12]}, calibrated on {profile.model_revision[:12]}"
    return None


def reuse_notes(profile: Profile, hw: HardwareDescriptor, model: ModelSpec) -> List[str]:
    """What a reused profile cannot vouch for: patch-level library changes, and a model commit that cannot be
    resolved offline."""
    _, patch = library_changes(profile, hw)
    notes = [f"{profile.backend} libraries changed at patch level only ({', '.join(patch)})"] if patch else []
    if profile.model_revision and snapshot_revision(model) is None:
        notes.append(f"the current commit of {model.hf_id} cannot be resolved offline; calibrated on "
                     f"{profile.model_revision[:12]}")
    return notes


def delete(hw: HardwareDescriptor, model: ModelSpec, objective: Optional[str] = None,
           workload: str = "default") -> int:
    base = profiles_dir() / hardware_hash(hw) / model.safe_id
    n = 0
    if objective:
        p = profile_path(hardware_hash(hw), model, objective, workload)
        if p.exists():
            p.unlink()
            n = 1
    elif base.exists():
        for p in base.glob("*.json"):
            p.unlink()
            n += 1
    return n


def iter_profiles() -> Iterator[Tuple[Path, Profile]]:
    root = profiles_dir()
    if not root.exists():
        return
    for path in sorted(root.glob("*/*/*.json")):
        try:
            yield path, Profile.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skipping %s: %s", path, exc)


def list_profiles() -> List[Tuple[Path, Profile]]:
    return list(iter_profiles())
