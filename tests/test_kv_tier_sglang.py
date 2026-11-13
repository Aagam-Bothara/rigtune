"""SGLang's hierarchical cache as a KV-cache tier: exact arguments and environment, and feature checks read
from the source of the SGLang that would run."""

from __future__ import annotations

import os

import pytest

import rigtune.backends.sglang as sg
from rigtune import kvtier
from rigtune.backends import get_backend
from rigtune.memory import HOST_TIER_RESERVE
from rigtune.models import Config, GiB

ALL = frozenset({"hicache", "file", "file_cap"})
ARGS = "enable_hierarchical_cache: bool = False\nhicache_size: int = 0\nhicache_storage_backend: str = None\n"
ENVIRON = ("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)\n"
           "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)\n")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGTUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    return tmp_path / "home"


def test_a_host_tier_is_a_hicache_size_in_decimal_gb(prepared_vllm, home, monkeypatch):
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL)
    cfg = Config(backend="sglang", quant="bf16", batch=64, gpu_memory_utilization=0.88, kv_tier="host:40")
    spec = get_backend("sglang").launch_spec(cfg, prepared_vllm, 8100)
    i = spec.args.index("--enable-hierarchical-cache")
    assert spec.args[i:] == ["--enable-hierarchical-cache", "--hicache-size", "42"]  # 40 GiB = 42.9 GB
    assert spec.env == {}


def test_a_disk_tier_is_the_file_backend_in_this_engines_own_directory(prepared_vllm, home, monkeypatch):
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL)
    cfg = Config(backend="sglang", quant="bf16", batch=64, kv_tier="host:40+disk:200")
    spec = get_backend("sglang").launch_spec(cfg, prepared_vllm, 8100)
    assert spec.args[-5:] == ["--enable-hierarchical-cache", "--hicache-size", "42",
                              "--hicache-storage-backend", "file"]
    own = home / "kv-tier" / f"pid{os.getpid()}-port8100"
    assert spec.env == {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(own),
                        "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": str(200 * GiB)}  # always capped
    # A profile records its launch at port 0, with a placeholder each server replaces with its own directory.
    recorded = get_backend("sglang").launch_spec(cfg, prepared_vllm, 0)
    assert recorded.env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] == kvtier.DIR_PLACEHOLDER


def test_a_disk_tier_is_offered_only_where_sglang_can_cap_it(monkeypatch):
    be = get_backend("sglang")
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL)
    assert be.tier_disk_refusal() is None
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL - {"file_cap"})
    assert "cannot cap" in be.tier_disk_refusal()
    monkeypatch.setattr(sg, "_hicache_features", lambda: frozenset({"hicache"}))
    assert "no file backend" in be.tier_disk_refusal()


def test_a_disk_tier_on_a_network_filesystem_is_refused_not_launched_host_only(prepared_vllm, monkeypatch):
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL)
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    monkeypatch.setattr(kvtier, "refusal", lambda *a, **k: "rigtune's cache dir is on a nfs4 filesystem")
    cfg = Config(backend="sglang", quant="bf16", batch=64, kv_tier="host:40+disk:200")
    with pytest.raises(ValueError, match="nfs4"):
        get_backend("sglang").launch_spec(cfg, prepared_vllm, 8100)
    host_cfg = cfg.model_copy(update={"kv_tier": "host:40"})
    host_only = get_backend("sglang").launch_spec(host_cfg, prepared_vllm, 8100)
    assert "--hicache-storage-backend" not in host_only.args  # a host tier needs no directory at all


def test_without_a_tier_the_launch_is_unchanged(prepared_vllm, home, monkeypatch):
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL)
    cfg = Config(backend="sglang", quant="bf16", batch=64, gpu_memory_utilization=0.88)
    plain = get_backend("sglang").launch_spec(cfg, prepared_vllm, 8100)
    assert not any("hicache" in a or "hierarchical" in a for a in plain.args) and plain.env == {}
    tiered = get_backend("sglang").launch_spec(cfg.model_copy(update={"kv_tier": "host:8"}), prepared_vllm,
                                               8100)
    assert tiered.args[:len(plain.args)] == plain.args


def test_features_are_read_from_source_text():
    assert sg.hicache_features_in(ARGS, ENVIRON) == ALL
    assert sg.hicache_features_in("", ENVIRON) == frozenset()
    assert sg.hicache_features_in(ARGS, "") == {"hicache"}  # no storage-dir variable: files would go to /tmp
    assert sg.hicache_features_in(ARGS, "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR\n") == {"hicache", "file"}


def test_features_of_the_sglang_that_would_run(tmp_path, monkeypatch):
    pkg = tmp_path / "sglang"
    (pkg / "srt").mkdir(parents=True)
    (pkg / "srt" / "server_args.py").write_text(ARGS, encoding="utf-8")
    (pkg / "srt" / "environ.py").write_text(ENVIRON, encoding="utf-8")
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    monkeypatch.setattr(sg, "_sglang_package_dir", lambda python: pkg)
    sg._hicache_features.cache_clear()
    try:
        assert sg._hicache_features() == ALL
        monkeypatch.setattr(sg, "_sglang_package_dir", lambda python: None)  # not found: nothing, no crash
        sg._hicache_features.cache_clear()
        assert sg._hicache_features() == frozenset()
    finally:
        sg._hicache_features.cache_clear()


def test_what_refuses_a_tier(prepared_vllm, monkeypatch):
    be = get_backend("sglang")
    cfg = Config(backend="sglang", quant="bf16", batch=64)
    monkeypatch.setattr(sg, "_hicache_features", lambda: ALL)
    assert be.tier_refusal(cfg, prepared_vllm) is None
    no_radix = cfg.model_copy(update={"prefix_cache": False})
    assert "disable-radix-cache" in be.tier_refusal(no_radix, prepared_vllm)
    assert "tensor parallelism" in be.tier_refusal(cfg.model_copy(update={"tp": 2}), prepared_vllm)
    monkeypatch.setattr(sg, "_hicache_features", lambda: frozenset({"hicache"}))
    assert be.tier_refusal(cfg, prepared_vllm) is None
    monkeypatch.setattr(sg, "_hicache_features", lambda: frozenset())
    assert "hierarchical-cache" in be.tier_refusal(cfg, prepared_vllm)


def test_host_ram_sglang_needs_beyond_the_pool():
    be = get_backend("sglang")
    assert be.tier_host_reserve() == max(HOST_TIER_RESERVE, 10 * GiB)  # SGLang itself keeps 10 GiB free
    cfg = Config(backend="sglang", quant="bf16", batch=64)
    assert be.tier_host_need(cfg, 40 * GiB) == 40 * GiB
    assert be.tier_host_need(cfg.model_copy(update={"spec_decode": "ngram:12"}), 40 * GiB) == 40 * GiB
    eagle = cfg.model_copy(update={"spec_decode": "eagle3:lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B:3"})
    assert be.tier_host_need(eagle, 40 * GiB) == 80 * GiB  # the head's own host pool, at the target's size
