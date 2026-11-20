"""rigtune bundle: a profile and the kernel cache that starts it warm, carried to another machine with the same GPU.

Import checks the profile as `serve --profile` does, writes only regular files inside the kernel cache, never over a
file already there, and only files the manifest lists with a matching sha256.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rigtune import bundle as bd
from rigtune.cli import app
from tests.conftest import make_hw
from tests.test_cache_and_pipeline import _profile

FILES = {"triton/abc/kernel.cubin": b"\x7fELF cubin", "vllm/torch_compile_cache/h1/model": b"aot artifact",
         "flashinfer/.cache/flashinfer/0.7.2/80/cached_ops/x.so": b"jit op"}


@pytest.fixture
def profile(spec, prepared_vllm):
    return _profile(make_hw("a100"), spec, prepared_vllm)


def _cache(root: Path, files=FILES) -> Path:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def _read_back(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["winner.tar", "winner.tar.gz"])
def test_a_bundle_carries_the_cache_and_the_profile(profile, tmp_path, name):
    src, dst = _cache(tmp_path / "src"), tmp_path / "dst"
    manifest = bd.export(profile, tmp_path / name, cache_dir=src)
    assert set(manifest["files"]) == set(FILES) and manifest["cache_bytes"] == sum(map(len, FILES.values()))
    assert manifest["engine"] == {"name": "vllm", "version": "1.0", "libs": {}}
    assert manifest["gpus"][0]["name"] == "NVIDIA A100-SXM4-80GB" and manifest["config"] == profile.config.key()
    res = bd.import_(tmp_path / name, make_hw("a100"), cache_dir=dst)
    assert (res.refusals, res.written, res.kept) == ([], 3, 0)
    assert _read_back(dst) == FILES
    assert res.profile.model_dump() == profile.model_dump()


def test_a_second_import_keeps_what_is_there(profile, tmp_path):
    bd.export(profile, tmp_path / "b.tar", cache_dir=_cache(tmp_path / "src"))
    dst = _cache(tmp_path / "dst", {"triton/abc/kernel.cubin": b"mine"})
    res = bd.import_(tmp_path / "b.tar", make_hw("a100"), cache_dir=dst)
    assert (res.written, res.kept) == (2, 1)
    assert (dst / "triton/abc/kernel.cubin").read_bytes() == b"mine"  # never overwritten


def test_a_bundle_for_another_gpu_is_refused_and_nothing_is_written(profile, tmp_path):
    bd.export(profile, tmp_path / "b.tar", cache_dir=_cache(tmp_path / "src"))
    res = bd.import_(tmp_path / "b.tar", make_hw("rtx4090"), cache_dir=tmp_path / "dst")
    assert res.refusals and "NVIDIA GeForce RTX 4090" in res.refusals[0]
    assert not (tmp_path / "dst").exists()


def _forged(tmp_path: Path, profile, members: dict, files: dict) -> Path:
    """A bundle written by hand: `members` are extra tar entries, `files` the manifest's digests."""
    manifest = bd.manifest_for(profile, files, 0)
    path = tmp_path / "forged.tar"
    with tarfile.open(path, "w") as tar:
        for name, data in {bd.MANIFEST: json.dumps(manifest).encode(),
                           bd.PROFILE: profile.model_dump_json().encode(), **members}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize("name", ["kernel-cache/../../evil.txt", "/etc/evil", "kernel-cache\\..\\evil",
                                  "elsewhere/evil.txt"])
def test_an_unsafe_path_is_refused_before_anything_is_written(profile, tmp_path, name):
    import hashlib

    ok = {"kernel-cache/triton/fine.bin": b"fine"}
    digests = {"triton/fine.bin": hashlib.sha256(b"fine").hexdigest()}
    path = _forged(tmp_path, profile, {**ok, name: b"evil"}, digests)
    with pytest.raises(bd.BundleError, match="refused"):
        bd.import_(path, make_hw("a100"), cache_dir=tmp_path / "dst")
    assert not (tmp_path / "dst").exists() and not (tmp_path / "evil.txt").exists()


def test_a_file_the_manifest_does_not_list_is_refused(profile, tmp_path):
    path = _forged(tmp_path, profile, {"kernel-cache/triton/extra.bin": b"x"}, {})
    with pytest.raises(bd.BundleError, match="not in the bundle's manifest"):
        bd.import_(path, make_hw("a100"), cache_dir=tmp_path / "dst")


def test_a_tampered_file_is_refused_and_not_left_behind(profile, tmp_path):
    path = _forged(tmp_path, profile, {"kernel-cache/triton/k.bin": b"changed"}, {"triton/k.bin": "0" * 64})
    with pytest.raises(bd.BundleError, match="does not match"):
        bd.import_(path, make_hw("a100"), cache_dir=tmp_path / "dst")
    assert not [p for p in (tmp_path / "dst").rglob("*") if p.is_file()]


def test_links_in_the_cache_are_left_out(profile, tmp_path):
    src = _cache(tmp_path / "src")
    try:
        os.symlink(tmp_path / "outside.txt", src / "triton" / "link")
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot make symlinks")
    assert "triton/link" not in bd.export(profile, tmp_path / "b.tar", cache_dir=src)["files"]


def test_no_kernel_cache_means_nothing_to_bundle(profile, tmp_path, monkeypatch):
    monkeypatch.setenv("RIGTUNE_KERNEL_CACHE", "off")
    with pytest.raises(bd.BundleError, match="RIGTUNE_KERNEL_CACHE is off"):
        bd.export(profile, tmp_path / "b.tar")


def test_the_cli_exports_imports_and_says_how_to_serve_it(profile, tmp_path, monkeypatch):
    profile_file = tmp_path / "p.json"
    profile_file.write_text(profile.model_dump_json(), encoding="utf-8")
    _cache(tmp_path / "kernel-cache")  # conftest points RIGTUNE_KERNEL_CACHE here
    runner = CliRunner()
    out = runner.invoke(app, ["bundle", "export", str(profile_file), "--out", str(tmp_path / "w.tar.gz")],
                        env={"COLUMNS": "300"})
    assert out.exit_code == 0, out.output
    assert "3 cache files" in out.output
    monkeypatch.setenv("RIGTUNE_KERNEL_CACHE", str(tmp_path / "other-machine"))
    monkeypatch.setattr("rigtune.hardware.probe", lambda: make_hw("a100"))
    res = runner.invoke(app, ["bundle", "import", str(tmp_path / "w.tar.gz")], env={"COLUMNS": "300"})
    assert res.exit_code == 0, res.output
    assert "3 files added, 0 already there" in res.output
    assert f"rigtune serve {profile.model_id} --profile {tmp_path / 'w.profile.json'}" in res.output
    assert _read_back(tmp_path / "other-machine") == FILES
    monkeypatch.setattr("rigtune.hardware.probe", lambda: make_hw("rtx4090"))
    refused = runner.invoke(app, ["bundle", "import", str(tmp_path / "w.tar.gz")], env={"COLUMNS": "300"})
    assert refused.exit_code == 2 and "not imported: it was calibrated on NVIDIA A100" in refused.output
