"""`rigtune serve --profile`: serve a pick calibrated on another machine with the same GPU (cache.portable_problems).

What the GPU work depends on must match: the GPUs, the engine and its libraries, and a model the machine can load.
A different CPU, RAM, driver or CUDA is a warning, unless the pick leans on the host.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rigtune import cache
from rigtune.cli import app
from rigtune.models import DisaggSpec, GPUInfo
from tests.conftest import GiB, make_hw
from tests.test_cache_and_pipeline import _profile


def _here(**cpu):
    hw = make_hw("a100")
    return hw.model_copy(update={"cpu": hw.cpu.model_copy(update=cpu)}) if cpu else hw


def _with_gpu(hw, **gpu):
    return hw.model_copy(update={"gpus": [hw.gpus[0].model_copy(update=gpu)]})


@pytest.fixture
def profile(spec, prepared_vllm):
    return _profile(make_hw("a100"), spec, prepared_vllm)


def test_the_same_machine_serves_it_without_a_word(profile):
    assert cache.portable_problems(profile, _here()) == ([], [])


def test_another_cpu_and_ram_are_only_warnings(profile):
    refuse, warn = cache.portable_problems(profile, _here(model_name="Other CPU", physical_cores=4,
                                                          ram_total_bytes=16 * GiB))
    assert refuse == []
    assert warn == ["calibrated with a Test CPU (8 cores), served on a Other CPU (4 cores)",
                    "calibrated with 64 GiB of RAM, served with 16 GiB"]


def test_another_driver_is_a_warning(profile):
    refuse, warn = cache.portable_problems(profile, _with_gpu(_here(), driver_version="595.1", cuda_version="13.2"))
    assert refuse == [] and warn == ["GPU driver or CUDA differs (driver ? CUDA ? -> driver 595.1 CUDA 13.2)"]


def test_another_gpu_is_refused(profile):
    refuse, _ = cache.portable_problems(profile, make_hw("rtx4090"))
    assert refuse == ["it was calibrated on NVIDIA A100-SXM4-80GB (80 GiB) and this machine has "
                      "NVIDIA GeForce RTX 4090 (24 GiB)"]


def test_another_engine_release_is_refused(profile):
    hw = _here()
    vllm = hw.backends["vllm"].model_copy(update={"version": "2.0"})
    refuse, _ = cache.portable_problems(profile, hw.model_copy(update={"backends": {**hw.backends, "vllm": vllm}}))
    assert refuse == ["it was calibrated on vllm 1.0 and this machine has 2.0"]


def test_too_few_gpus_for_its_replicas_are_refused(profile):
    two = make_hw("a100")
    two = two.model_copy(update={"gpus": [two.gpus[0], two.gpus[0].model_copy(update={"index": 1})]})
    replicated = profile.model_copy(update={"replicas": 2, "hardware": two})
    refuse, _ = cache.portable_problems(replicated, _here())
    assert refuse == ["it needs 2 GPU(s) and this machine has 1"]


def test_a_disaggregated_pick_is_refused(profile):
    pd = profile.model_copy(update={"disagg": DisaggSpec(prefill=profile.config, decode=profile.config,
                                                         prefill_gpu=0, decode_gpu=1)})
    refuse, _ = cache.portable_problems(pd, _here())
    assert any("disaggregated prefill and decode" in r for r in refuse)


def test_a_kv_tier_sized_for_other_ram_is_refused_and_the_same_ram_serves_it(profile):
    tiered = profile.model_copy(update={"config": profile.config.model_copy(update={"kv_tier": "host"})})
    refuse, warn = cache.portable_problems(tiered, _here(ram_total_bytes=16 * GiB))
    assert refuse == ["its host KV-cache tier was sized for that machine's host memory, and this machine's CPU or "
                      "RAM differs"] and warn == []
    assert cache.portable_problems(tiered, _here()) == ([], [])


def test_a_local_model_folder_this_machine_lacks_is_refused(profile, tmp_path):
    missing = str(tmp_path / "gone" / "model")
    local = profile.model_copy(update={"prepared": profile.prepared.model_copy(update={"hf_path": missing})})
    refuse, _ = cache.portable_problems(local, _here())
    assert refuse == [f"its model files are at {missing}, which this machine does not have"]


def _serve(profile_path, model, hw, monkeypatch):
    monkeypatch.setattr("rigtune.hardware.probe", lambda: hw)
    return CliRunner().invoke(app, ["serve", model, "--profile", str(profile_path)], env={"COLUMNS": "300"})


def test_serve_refuses_a_profile_of_another_model(profile, tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    res = _serve(path, "someone/else", _here(), monkeypatch)
    assert res.exit_code != 0 and "is a profile of meta-llama/Llama-3.2-3B-Instruct, not someone/else" in res.output


def test_serve_says_why_it_will_not_serve_it_here(profile, tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    res = _serve(path, profile.model_id, make_hw("rtx4090"), monkeypatch)
    assert res.exit_code == 2 and "not served here: it was calibrated on NVIDIA A100-SXM4-80GB" in res.output


def test_serve_launches_the_files_pick_without_calibrating(profile, tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    path.write_text(profile.model_dump_json(), encoding="utf-8")
    launched = {}

    def no_calibration(*a, **k):
        raise AssertionError("serve --profile must not calibrate or look up the cache")

    class StopHere(Exception):
        pass

    def fake_supervisor(backend, cfg, prepared, **kw):
        launched["config"] = cfg
        raise StopHere

    monkeypatch.setattr("rigtune.pipeline.resolve_profile", no_calibration)
    monkeypatch.setattr("rigtune.serve.Supervisor", fake_supervisor)
    res = _serve(path, profile.model_id, _with_gpu(_here(), driver_version="595.1"), monkeypatch)
    assert isinstance(res.exception, StopHere), res.output
    assert launched["config"].key() == profile.config.key()
    assert "--profile: GPU driver or CUDA differs" in res.output


def test_gpu_shape_matches_what_the_hardware_hash_keys():
    a = GPUInfo(vendor="nvidia", name="X", compute_capability=(8, 0), vram_total_bytes=80 * GiB)
    b = a.model_copy(update={"vram_total_bytes": 80 * GiB + 2**20, "vram_free_bytes": 1})  # within one 256 MiB step
    assert cache._gpu_shape(a) == cache._gpu_shape(b)
