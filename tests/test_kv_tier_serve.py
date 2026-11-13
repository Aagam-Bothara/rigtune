"""Serving a pick with a KV-cache tier: checked again before launch by calibration's own rules, served
without the tier when it no longer fits, and a disk tier never outlives its server or lands on a network
filesystem."""

from __future__ import annotations

import os

import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
import rigtune.hardware as hwmod
import rigtune.memory as memmod
from rigtune import kvtier
from rigtune.backends import get_backend
from rigtune.models import Config, CPUInfo, GiB
from rigtune.serve.supervisor import Supervisor


class _Proc:
    def __init__(self, spec):
        self.spec, self.pid, self.stopped = spec, 4242, False

    def wait_ready(self, timeout=0):
        return True

    def alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGTUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(vl, "_offload_features", lambda: frozenset({"cpu", "hma", "shm"}))
    monkeypatch.setattr(vl, "_shm_free", lambda: 500 * GiB)
    monkeypatch.setattr(sg, "_hicache_features", lambda: frozenset({"hicache", "file", "file_cap"}))
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    monkeypatch.setattr(kvtier, "refusal", lambda *a, **k: None)  # a local disk, whatever this machine has
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(memmod, "disk_tier_budget", lambda root: 1000 * GiB)
    return tmp_path / "home"


def _engine(monkeypatch, name):
    """The backend, launched as a stand-in process that records its config, launch spec and disk directory."""
    be = get_backend(name)
    launched = []

    def launch(cfg, model, port, log_path=None):
        d = kvtier.trial_dir(port)
        files = sorted(os.listdir(d)) if d.exists() else None
        launched.append((cfg, be.launch_spec(cfg, model, port), files))
        return _Proc(launched[-1][1])

    monkeypatch.setattr(be, "launch", launch)
    return be, launched


def _ram(monkeypatch, free_gib):
    cpu = CPUInfo(ram_total_bytes=256 * GiB, ram_free_bytes=free_gib * GiB)
    monkeypatch.setattr(hwmod, "probe_cpu", lambda: cpu)


def _cfg(backend="vllm", tier=None) -> Config:
    return Config(backend=backend, quant="bf16", batch=64, kv_tier=tier)


def test_a_tier_that_still_fits_serves_quietly(home, prepared_vllm, monkeypatch):
    be, launched = _engine(monkeypatch, "vllm")
    _ram(monkeypatch, 128)
    sup = Supervisor(be, _cfg(tier="host:40"), prepared_vllm, port=8123)
    sup.start()
    assert sup.tier_warning is None and sup.status()["kv_tier"] == {"tier": "host:40", "warning": None}
    assert "--kv-transfer-config" in launched[0][1].args
    sup.stop()


def test_a_tier_that_no_longer_fits_is_served_without_it(home, prepared_vllm, monkeypatch, caplog):
    be, launched = _engine(monkeypatch, "vllm")
    _ram(monkeypatch, 24)  # calibrated with 128 GiB free; 40 GiB no longer fits
    sup = Supervisor(be, _cfg(tier="host:40"), prepared_vllm, port=8124)
    sup.start()
    assert "serving without the host:40 KV-cache tier" in sup.tier_warning and "host RAM" in caplog.text
    cfg, spec, _ = launched[0]
    assert cfg.kv_tier is None and "--kv-transfer-config" not in spec.args  # the GPU-only twin
    assert sup.status()["kv_tier"] == {"tier": None, "warning": sup.tier_warning}
    sup.stop()


def test_an_old_vllm_disk_tier_is_never_launched(home, prepared_vllm, monkeypatch):
    be, launched = _engine(monkeypatch, "vllm")
    _ram(monkeypatch, 128)
    sup = Supervisor(be, _cfg(tier="host:8+disk:60"), prepared_vllm, port=8125)
    sup.start()
    assert "no size limit" in sup.tier_warning
    assert launched[0][0].kv_tier is None and "--kv-transfer-config" not in launched[0][1].args
    sup.stop()


def test_sglang_serves_its_disk_tier_from_an_empty_private_directory_and_removes_it(home, prepared_vllm,
                                                                                    monkeypatch):
    be, launched = _engine(monkeypatch, "sglang")
    _ram(monkeypatch, 128)
    port = 8126
    stale = kvtier.trial_dir(port)
    stale.mkdir(parents=True)
    (stale / "old.bin").write_bytes(b"another server's cache")
    dead = kvtier.tier_root() / "pid999999-port8000"
    dead.mkdir()
    monkeypatch.setattr("psutil.pid_exists", lambda pid: pid == os.getpid())
    sup = Supervisor(be, _cfg("sglang", "host:8+disk:60"), prepared_vllm, port=port)
    sup.start()
    cfg, spec, files = launched[0]
    assert files == [] and not dead.exists()  # a fresh directory, and a dead server's swept away
    assert spec.env == {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(stale),
                        "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": str(60 * GiB)}
    (stale / "block.bin").write_bytes(b"a served prompt's KV cache")
    sup.stop()
    assert not stale.exists()


def test_sglang_on_a_shrunken_host_is_served_without_its_tier(home, prepared_vllm, monkeypatch):
    be, launched = _engine(monkeypatch, "sglang")
    _ram(monkeypatch, 25)  # 0.95 x 25 GiB, less the margin and the 10 GiB SGLang keeps: 12.5 GiB, under 16
    sup = Supervisor(be, _cfg("sglang", "host:16+disk:60"), prepared_vllm, port=8127)
    sup.start()
    cfg, spec, files = launched[0]
    assert cfg.kv_tier is None and "--enable-hierarchical-cache" not in spec.args and files is None
    assert "serving without the host:16+disk:60 KV-cache tier" in sup.tier_warning
    sup.stop()


def test_a_disk_tier_on_a_network_filesystem_is_served_without_the_tier(home, prepared_vllm, monkeypatch):
    be, launched = _engine(monkeypatch, "sglang")
    _ram(monkeypatch, 128)
    monkeypatch.setattr(kvtier, "refusal", lambda *a, **k: "rigtune's cache dir /workspace is on a fuse.mfs "
                                                          "filesystem")
    sup = Supervisor(be, _cfg("sglang", "host:8+disk:60"), prepared_vllm, port=8128)
    sup.start()
    cfg, spec, _ = launched[0]
    assert cfg.kv_tier is None and "fuse.mfs" in sup.tier_warning  # not quietly host-only, either
    assert "--hicache-storage-backend" not in spec.args
    sup.stop()


def test_servers_without_a_tier_are_untouched(home, prepared_vllm, monkeypatch):
    be, launched = _engine(monkeypatch, "vllm")
    monkeypatch.setattr(hwmod, "probe_cpu", lambda: pytest.fail("no tier: the host is not probed"))
    sup = Supervisor(be, _cfg(), prepared_vllm, port=8129)
    sup.start()
    assert "kv_tier" not in sup.status() and sup.tier_warning is None
    sup.stop()
