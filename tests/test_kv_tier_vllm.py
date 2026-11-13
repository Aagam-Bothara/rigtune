"""vLLM's native CPU KV tier: exact launch arguments and feature checks read from source; its unbounded disk
tier is never launched. And the disk-tier directories every engine gets, which never outlive their trial."""

from __future__ import annotations

import errno
import json
import os
import signal
import sys
from typing import Dict

import pytest

import rigtune.backends.vllm as vl
import rigtune.calibrate.search as searchmod
from rigtune import kvtier
from rigtune.backends import get_backend
from rigtune.backends.base import BaseBackend, LlmtraceHooks, free_port
from rigtune.calibrate.search import SubprocessTrialRunner
from rigtune.calibrate.workload import Workload
from rigtune.memory import MemoryModel
from rigtune.models import ArchInfo, Config, GiB, ModelSpec, PreparedModel

ALL = frozenset({"cpu", "hma", "shm"})


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGTUNE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _flag(args, name):
    return json.loads(args[args.index(name) + 1])


def test_a_host_tier_is_one_kv_transfer_config(prepared_vllm, home):
    cfg = Config(backend="vllm", quant="bf16", batch=64, gpu_memory_utilization=0.9, kv_tier="host:40")
    args = get_backend("vllm").launch_spec(cfg, prepared_vllm, 8100).args
    assert args.count("--kv-transfer-config") == 1
    assert _flag(args, "--kv-transfer-config") == {
        "kv_connector": "OffloadingConnector", "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": 40 * GiB, "offload_prompt_only": False}}
    assert not (home / "kv-tier").exists()  # the launch spec creates nothing


def test_the_unbounded_disk_tier_is_refused_never_launched_host_only(prepared_vllm, home):
    be = get_backend("vllm")
    cfg = Config(backend="vllm", quant="bf16", batch=64, kv_tier="host:40+disk:200")
    with pytest.raises(ValueError, match="no size limit"):
        be.launch_spec(cfg, prepared_vllm, 8100)
    assert be.tier_disk_refusal() == "vLLM's disk tier has no size limit in 0.29; not offered"


def test_without_a_tier_the_launch_is_unchanged(prepared_vllm):
    cfg = Config(backend="vllm", quant="bf16", batch=64, gpu_memory_utilization=0.9)
    args = get_backend("vllm").launch_spec(cfg, prepared_vllm, 8100).args
    assert "--kv-transfer-config" not in args
    tiered = get_backend("vllm").launch_spec(cfg.model_copy(update={"kv_tier": "host:8"}), prepared_vllm,
                                             8100).args
    assert tiered[:len(args)] == args  # the tier only appends


def test_disaggregated_and_replica_engines_never_carry_a_tier(prepared_vllm):
    be = get_backend("vllm")
    cfg = Config(backend="vllm", quant="bf16", batch=64, kv_tier="host:40")
    nixl = {"kv_connector": "NixlConnector"}
    pd = be.disagg_launch_spec(cfg, prepared_vllm, 8100, "prefill", nixl, 0, 5600).args
    assert pd.count("--kv-transfer-config") == 1 and _flag(pd, "--kv-transfer-config") == nixl
    assert "--kv-transfer-config" not in be.replica_launch_spec(cfg, prepared_vllm, 8100, 1).args


def _sources(**drop: bool) -> Dict[str, str]:
    files = {
        "distributed/kv_transfer/kv_connector/v1/offloading_connector.py":
            "class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):\n",
        "v1/kv_offload/base.py": 'self.extra_config.get("offload_prompt_only", True)\n',
        "v1/kv_offload/cpu/shared_offload_region.py": 'path = f"/dev/shm/vllm_offload_{engine_id}.mmap"\n',
    }
    for name, gone in drop.items():
        if gone:
            files.pop(next(k for k in files if name in k.replace("/", "_")))
    return files


def test_features_are_read_from_source_text():
    assert vl.offload_features_in(_sources()) == ALL
    assert vl.offload_features_in(_sources(offloading_connector=True)) == frozenset()
    assert vl.offload_features_in(_sources(kv_offload_base=True)) == frozenset()  # no offload_prompt_only
    no_hma = _sources()
    no_hma[vl.OFFLOAD_SOURCES[0]] = "class OffloadingConnector(KVConnectorBase_V1):\n"
    assert vl.offload_features_in(no_hma) == ALL - {"hma"}
    assert vl.offload_features_in(_sources(shared_offload_region=True)) == ALL - {"shm"}


def test_features_of_the_vllm_that_will_run(tmp_path, monkeypatch):
    root = tmp_path / "vllm"
    for rel, text in _sources().items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    monkeypatch.setattr(vl, "_vllm_package", lambda: root)
    vl._offload_features.cache_clear()
    try:
        assert vl._offload_features() == ALL
        monkeypatch.setattr(vl, "_vllm_package", lambda: None)  # no vLLM found: nothing offered, no crash
        vl._offload_features.cache_clear()
        assert vl._offload_features() == frozenset()
    finally:
        vl._offload_features.cache_clear()


def _hybrid(linear: bool) -> PreparedModel:
    arch = ArchInfo(architecture="Qwen3_5ForCausalLM" if linear else "Gemma3ForCausalLM", num_layers=32,
                    hidden_size=4096, num_attention_heads=16, num_kv_heads=4, head_dim=256,
                    full_attention_layers=8,
                    sliding_layers=0 if linear else 24, sliding_window=None if linear else 1024,
                    linear_layers=24 if linear else 0, linear_state_bytes=50 * 2**20 if linear else 0)
    return PreparedModel(spec=ModelSpec(hf_id="x/y"), backend="vllm", arch=arch,
                         weights_bytes={"bf16": 18 * GiB})


def test_what_refuses_a_tier(prepared_vllm, monkeypatch):
    be = get_backend("vllm")
    cfg = Config(backend="vllm", quant="bf16", batch=64)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(vl, "_offload_features", lambda: ALL)
    assert be.tier_refusal(cfg, prepared_vllm) is None
    assert "tensor parallelism" in be.tier_refusal(cfg.model_copy(update={"tp": 2}), prepared_vllm)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert "expandable_segments" in be.tier_refusal(cfg, prepared_vllm)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    assert be.tier_refusal(cfg, _hybrid(linear=True)) is None
    no_prefix = cfg.model_copy(update={"prefix_cache": False})
    assert "prefix caching" in be.tier_refusal(no_prefix, _hybrid(linear=True))
    assert be.tier_refusal(no_prefix, _hybrid(linear=False)) is None
    monkeypatch.setattr(vl, "_offload_features", lambda: ALL - {"hma"})
    assert "hybrid KV cache manager" in be.tier_refusal(cfg, _hybrid(linear=False))
    assert be.tier_refusal(cfg, prepared_vllm) is None  # a dense model does not need it
    monkeypatch.setattr(vl, "_offload_features", lambda: frozenset())
    assert "OffloadingConnector" in be.tier_refusal(cfg, prepared_vllm)
    assert be.tier_host_cap() is None
    assert get_backend("vllm-cpu").tier_refusal(cfg, prepared_vllm)


def test_dev_shm_caps_the_cpu_tier(monkeypatch):
    be = get_backend("vllm")
    monkeypatch.setattr(vl, "_offload_features", lambda: ALL)
    monkeypatch.setattr(vl, "_shm_free", lambda: 64 * 2**20)  # Docker's default /dev/shm
    assert be.tier_host_cap() == 64 * 2**20
    monkeypatch.setattr(vl, "_offload_features", lambda: ALL - {"shm"})
    assert be.tier_host_cap() is None


# --------------------------------------------------------------------------- the disk tier's directory


class _TierEngine(BaseBackend):
    """Stands in for an engine that writes its disk tier, then dies before it is ready."""

    name = "fake"
    seen: list = []

    def memory_model(self, hw):
        return MemoryModel(runtime_workspace=0, kv_tokens_fn=lambda c: 0, device="cpu")

    def workload_hooks(self, hw, model) -> LlmtraceHooks:
        return LlmtraceHooks(model_name="fake", gpu_ids=[])

    def launch(self, cfg, model, port, log_path=None):
        d = kvtier.trial_dir(port)
        mode = (d.stat().st_mode & 0o777) if d.exists() else None
        self.seen.append((sorted(p.name for p in d.iterdir()) if d.exists() else None, mode))
        d.mkdir(parents=True, exist_ok=True)
        (d / "block.bin").write_bytes(b"kv of a user prompt")
        raise RuntimeError("engine died")


def _runner(be):
    return SubprocessTrialRunner(backends={"fake": be}, models={"fake": None}, hw=None, workload=Workload())


def test_the_runner_gives_a_disk_tier_an_empty_private_directory_and_removes_it(home, monkeypatch):
    port = free_port()
    monkeypatch.setattr(searchmod, "free_port", lambda: port)
    stale = kvtier.trial_dir(port)
    assert stale.name == f"pid{os.getpid()}-port{port}"
    stale.mkdir(parents=True)
    (stale / "old.bin").write_bytes(b"an earlier trial's cache")
    be = _TierEngine()
    be.seen = []
    res = _runner(be).run(Config(backend="fake", quant="none", kv_tier="host:4+disk:8"), "tier")
    assert not res.ok and "engine died" in res.error
    names, mode = be.seen[0]
    assert names == []  # the earlier trial's files were gone before the engine started
    if sys.platform != "win32":
        assert mode == 0o700
    assert not stale.exists()  # and what the engine wrote is gone after it stopped
    # A host-only tier gets no directory; a trial without a tier touches nothing.
    be.seen = []
    _runner(be).run(Config(backend="fake", quant="none", kv_tier="host:4"), "tier")
    assert be.seen[0] == (None, None)
    stale.mkdir(parents=True)
    _runner(be).run(Config(backend="fake", quant="none"), "quant")
    assert stale.exists()


def test_a_full_disk_fails_the_trial_and_launches_nothing(home, monkeypatch):
    def full(port):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(searchmod.kvtier, "prepare", full)
    be = _TierEngine()
    be.seen = []
    res = _runner(be).run(Config(backend="fake", quant="none", kv_tier="host:4+disk:8"), "tier")
    assert not res.ok and not res.launched and "No space left on device" in res.error
    assert be.seen == []  # the engine was never started
    assert not (home / "kv-tier").exists() or not any((home / "kv-tier").iterdir())


def test_clear_only_removes_tier_directories(home):
    kvtier.clear(1234)  # nothing there: nothing happens
    other = home / "profiles"
    other.mkdir(parents=True)
    kvtier.trial_dir(1234).mkdir(parents=True)
    kvtier.clear(1234)
    assert other.exists() and not kvtier.trial_dir(1234).exists()


def test_a_sweep_removes_what_dead_processes_left(home, monkeypatch):
    import psutil

    root = kvtier.tier_root()
    for name in ("pid111-port8000", "pid222-port8001", f"pid{os.getpid()}-port8002", "notes"):
        (root / name).mkdir(parents=True)
        (root / name / "block.bin").write_bytes(b"kv")
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid == 222)  # 111 was SIGKILLed; 222 still runs
    assert kvtier.sweep() == 1
    assert sorted(p.name for p in root.iterdir()) == sorted(["pid222-port8001", f"pid{os.getpid()}-port8002",
                                                             "notes"])
    assert kvtier.sweep() == 0


def test_exit_and_sigterm_remove_what_this_process_owns(home, monkeypatch):
    registered, handlers = [], {}
    monkeypatch.setattr(kvtier, "_hooks_installed", False)
    monkeypatch.setattr(kvtier.atexit, "register", registered.append)
    monkeypatch.setattr(kvtier.signal, "getsignal", lambda s: signal.SIG_DFL)
    monkeypatch.setattr(kvtier.signal, "signal", lambda s, h: handlers.setdefault(s, h))
    a, b = kvtier.prepare(9001), kvtier.prepare(9002)
    assert registered == [kvtier.cleanup] and signal.SIGTERM in handlers  # installed once
    registered[0]()  # interpreter exit
    assert not a.exists() and not b.exists()
    c = kvtier.prepare(9003)
    with pytest.raises(SystemExit):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert not c.exists()
