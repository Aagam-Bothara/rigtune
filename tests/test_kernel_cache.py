"""vLLM launches with its caches in rigtune's kernel cache (backends/vllm.py kernel_cache_env): Triton's, vLLM's own
compile and AOT caches (VLLM_CACHE_ROOT) and FlashInfer's (FLASHINFER_WORKSPACE_BASE).

The directories are set at launch only (LaunchSpec.runtime_env): never in the recorded launch environment, so no
local path reaches a profile or a verdict, and never over a variable the caller set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rigtune import paths
from rigtune.backends import get_backend
from rigtune.backends.base import LaunchSpec, Process
from rigtune.models import Config

CFG = Config(backend="vllm", quant="bf16", ctx=4096, batch=64)
NAMES = ("TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "FLASHINFER_WORKSPACE_BASE")


def _want(tmp_path: Path) -> dict:
    base = tmp_path / "kernel-cache"  # conftest points RIGTUNE_KERNEL_CACHE here
    return {"TRITON_CACHE_DIR": str(base / "triton"), "VLLM_CACHE_ROOT": str(base / "vllm"),
            "FLASHINFER_WORKSPACE_BASE": str(base / "flashinfer")}


def _child_env(spec: LaunchSpec, tmp_path: Path, name: str = "TRITON_CACHE_DIR") -> str:
    """What a process started from `spec` sees in `name` ("" when unset)."""
    out = tmp_path / "seen.txt"
    code = f"import os, sys; open(sys.argv[1], 'w').write(os.environ.get({name!r}, ''))"
    probe = LaunchSpec(args=[sys.executable, "-c", code, str(out)], env=spec.env, runtime_env=spec.runtime_env)
    proc = Process(probe, port=1, health_url="http://127.0.0.1:1/health").start()
    proc._proc.wait(timeout=60)
    return out.read_text()


def test_vllm_launch_sets_its_caches_at_launch_only(prepared_vllm, tmp_path):
    spec = get_backend("vllm").launch_spec(CFG, prepared_vllm, 1)
    assert spec.runtime_env == _want(tmp_path)
    assert not set(NAMES) & set(spec.env)
    assert not any("kernel-cache" in a for a in spec.args)


@pytest.mark.parametrize("name", NAMES)
def test_the_launch_creates_each_directory_and_the_engine_sees_it(prepared_vllm, tmp_path, monkeypatch, name):
    for n in NAMES:
        monkeypatch.delenv(n, raising=False)
    spec = get_backend("vllm").launch_spec(CFG, prepared_vllm, 1)
    assert _child_env(spec, tmp_path, name) == spec.runtime_env[name]
    assert Path(spec.runtime_env[name]).is_dir()


def test_a_vllm_cache_root_the_caller_set_wins_and_the_others_stay(prepared_vllm, tmp_path, monkeypatch):
    for n in NAMES:
        monkeypatch.delenv(n, raising=False)
    mine = str(tmp_path / "my-own-vllm-cache")
    monkeypatch.setenv("VLLM_CACHE_ROOT", mine)
    spec = get_backend("vllm").launch_spec(CFG, prepared_vllm, 1)
    assert _child_env(spec, tmp_path, "VLLM_CACHE_ROOT") == mine
    assert not Path(spec.runtime_env["VLLM_CACHE_ROOT"]).exists()  # rigtune's was not even created
    assert _child_env(spec, tmp_path, "TRITON_CACHE_DIR") == spec.runtime_env["TRITON_CACHE_DIR"]


def test_a_triton_cache_dir_the_caller_set_wins(prepared_vllm, tmp_path, monkeypatch):
    mine = str(tmp_path / "my-own-triton")
    monkeypatch.setenv("TRITON_CACHE_DIR", mine)
    spec = get_backend("vllm").launch_spec(CFG, prepared_vllm, 1)
    assert _child_env(spec, tmp_path) == mine
    assert not Path(spec.runtime_env["TRITON_CACHE_DIR"]).exists()  # rigtune's was not even created


def test_a_directory_that_cannot_be_created_is_left_to_the_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory")
    spec = LaunchSpec(args=[], runtime_env={"TRITON_CACHE_DIR": str(blocker / "triton")})
    assert _child_env(spec, tmp_path) == ""


@pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "no", "none", "disabled"])
def test_rigtune_kernel_cache_off_sets_nothing(prepared_vllm, monkeypatch, value):
    monkeypatch.setenv("RIGTUNE_KERNEL_CACHE", value)
    assert paths.kernel_cache_dir() is None
    assert get_backend("vllm").launch_spec(CFG, prepared_vllm, 1).runtime_env == {}


def test_the_default_is_under_rigtune_home(tmp_path, monkeypatch):
    monkeypatch.delenv("RIGTUNE_KERNEL_CACHE", raising=False)
    monkeypatch.delenv("POLYSERVE_KERNEL_CACHE", raising=False)
    monkeypatch.setenv("RIGTUNE_HOME", str(tmp_path / "home"))
    assert paths.kernel_cache_dir() == tmp_path / "home" / "kernel-cache"


def test_replica_and_phase_launches_keep_them(prepared_vllm, tmp_path):
    be = get_backend("vllm")
    assert be.replica_launch_spec(CFG, prepared_vllm, 1, 0).runtime_env == _want(tmp_path)
    assert be.disagg_launch_spec(CFG, prepared_vllm, 1, "prefill", {}, 0, 5600).runtime_env == _want(tmp_path)


def test_sglang_launches_are_unchanged(no_network, prepared_vllm):
    from rigtune.models import PreparedModel

    model = PreparedModel(spec=prepared_vllm.spec, backend="sglang", arch=prepared_vllm.arch,
                          hf_path=prepared_vllm.hf_path, weights_bytes=prepared_vllm.weights_bytes)
    spec = get_backend("sglang").launch_spec(Config(backend="sglang", quant="bf16"), model, 1)
    assert spec.runtime_env == {} and not set(NAMES) & set(spec.env)
