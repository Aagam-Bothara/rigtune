"""Carry a calibrated pick, and the caches that start it warm, to another machine with the same GPU.

A bundle is a tar file of three parts: `profile.json`, the profile as calibrated; `kernel-cache/`, rigtune's kernel
cache (paths.kernel_cache_dir: Triton's kernels, vLLM's compile and AOT caches and startup plans, FlashInfer's
kernels); and `rigtune-bundle.json`, a manifest of what the bundle is for and the sha256 of every file. On a fresh
A100, these caches started Qwen3.5-9B's pick in 140 s against 280 s cold (benchmarks/sessions/2026-11/
modal-warm-serve).

Importing checks the profile against the machine as `rigtune serve --profile` does (cache.portable_problems),
extracts only regular files and directories inside the kernel cache, never overwrites a file already there, and
checks each file it writes against the manifest. Compiled caches can hold the exporting machine's paths, so a bundle
is meant for machines its owner controls.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from rigtune import __version__, paths
from rigtune.models import HardwareDescriptor, Profile

MANIFEST = "rigtune-bundle.json"
PROFILE = "profile.json"
CACHE = "kernel-cache"
SCHEMA = 1


class BundleError(RuntimeError):
    """A bundle that cannot be written or read."""


@dataclass
class ImportResult:
    profile: Profile
    written: int = 0  # cache files extracted
    kept: int = 0  # cache files already present, left as they were
    refusals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_files(root: Path) -> Dict[str, Path]:
    """Every regular file under `root`, by its POSIX path relative to it. Links are left out: a cache holds none,
    and one could point anywhere on the importing machine."""
    out: Dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out[p.relative_to(root).as_posix()] = p
    return out


def manifest_for(profile: Profile, files: Dict[str, str], cache_bytes: int) -> dict:
    """What the bundle was made from: enough to say, before extracting, whether it can start warm here."""
    recorded = profile.hardware.backends.get(profile.backend)
    return {
        "schema": SCHEMA,
        "rigtune": __version__,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"id": profile.model_id, "revision": profile.model_revision},
        "engine": {"name": profile.backend, "version": profile.backend_version,
                   "libs": dict(recorded.libs) if recorded is not None else {}},
        "gpus": [{"name": g.name, "compute_capability": g.compute_capability,
                  "vram_gib": round(g.vram_total_bytes / 2**30)} for g in profile.hardware.gpus],
        "config": profile.config.key(),
        "cache_bytes": cache_bytes,
        "files": files,
    }


def export(profile: Profile, out: Path, cache_dir: Optional[Path] = None) -> dict:
    """Write `profile` and the kernel cache to the tar file `out` (gzipped when its name ends in .gz). Returns the
    manifest."""
    cache_dir = cache_dir if cache_dir is not None else paths.kernel_cache_dir()
    if cache_dir is None:
        raise BundleError("RIGTUNE_KERNEL_CACHE is off, so there is no kernel cache to bundle")
    files = _cache_files(cache_dir) if cache_dir.is_dir() else {}
    out = out.resolve()
    files = {rel: p for rel, p in files.items() if p.resolve() != out}  # a bundle written into the cache itself
    digests = {rel: _sha256(p) for rel, p in files.items()}
    manifest = manifest_for(profile, digests, sum(p.stat().st_size for p in files.values()))
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz" if out.name.endswith(".gz") else "w") as tar:
        for name, data in ((MANIFEST, json.dumps(manifest, indent=1).encode()),
                           (PROFILE, profile.model_dump_json(indent=1).encode())):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), int(time.time()), 0o644
            tar.addfile(info, io.BytesIO(data))
        for rel, p in files.items():
            tar.add(p, arcname=f"{CACHE}/{rel}", recursive=False)
    return manifest


def _member_path(member: tarfile.TarInfo) -> PurePosixPath:
    """A member's path, refused unless it is a plain relative path: no absolute path, drive or `..`."""
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or ":" in member.name or "\\" in member.name:
        raise BundleError(f"refused: the bundle holds an unsafe path, {member.name!r}")
    return path


def read(bundle: Path) -> tuple:
    """(manifest, profile) from a bundle, without extracting its cache."""
    try:
        with tarfile.open(bundle) as tar:
            manifest = json.loads(tar.extractfile(MANIFEST).read())  # type: ignore[union-attr]
            profile = Profile.model_validate_json(tar.extractfile(PROFILE).read())  # type: ignore[union-attr]
    except (OSError, KeyError, ValueError, tarfile.TarError, AttributeError) as exc:
        raise BundleError(f"{bundle} is not a rigtune bundle: {exc}") from None
    if manifest.get("schema") != SCHEMA:
        raise BundleError(f"{bundle} is a bundle of schema {manifest.get('schema')}; this rigtune reads {SCHEMA}")
    return manifest, profile


def import_(bundle: Path, hw: HardwareDescriptor, cache_dir: Optional[Path] = None) -> ImportResult:
    """Check `bundle` against this machine and, unless it is refused, extract its cache into the kernel cache.
    Nothing is extracted when it is refused."""
    from rigtune.cache import portable_problems

    manifest, profile = read(bundle)
    refusals, warnings = portable_problems(profile, hw)
    result = ImportResult(profile=profile, refusals=refusals, warnings=warnings)
    if refusals:
        return result
    cache_dir = cache_dir if cache_dir is not None else paths.kernel_cache_dir()
    if cache_dir is None:
        raise BundleError("RIGTUNE_KERNEL_CACHE is off, so there is no kernel cache to import into")
    digests: Dict[str, str] = manifest.get("files", {})
    root = cache_dir.resolve()
    with tarfile.open(bundle) as tar:
        planned = []  # every member is checked before any is written, so a refusal leaves nothing behind
        for member in tar.getmembers():
            path = _member_path(member)
            if path.parts[:1] != (CACHE,) or len(path.parts) < 2:
                if member.name in (MANIFEST, PROFILE):
                    continue
                raise BundleError(f"refused: the bundle holds {member.name!r}, outside {CACHE}/")
            if member.isdir():
                continue
            if not member.isfile():
                raise BundleError(f"refused: {member.name!r} in the bundle is not a regular file")
            rel = PurePosixPath(*path.parts[1:]).as_posix()
            if rel not in digests:
                raise BundleError(f"refused: {member.name!r} is not in the bundle's manifest")
            dest = (root / rel).resolve()
            if root not in dest.parents:
                raise BundleError(f"refused: {member.name!r} would be written outside the kernel cache")
            planned.append((member, rel, dest))
        for member, rel, dest in planned:
            if dest.exists():
                result.kept += 1
                continue
            src = tar.extractfile(member)
            if src is None:
                raise BundleError(f"refused: {member.name!r} could not be read")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".rigtune-import")
            h = hashlib.sha256()
            with open(tmp, "wb") as fh:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    h.update(chunk)
                    fh.write(chunk)
            if h.hexdigest() != digests[rel]:
                tmp.unlink()
                raise BundleError(f"refused: {member.name!r} does not match the bundle's manifest")
            tmp.replace(dest)
            result.written += 1
    return result
