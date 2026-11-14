"""torch, transformers, flashinfer and tokenizers versions, read in each engine's own environment and recorded
beside the engine's version: in the probe, in every profile, never in the hardware hash or the cache check."""

from __future__ import annotations

import json
import subprocess
import sys

import rigtune.hardware as H
from rigtune.cache import invalid_reason
from rigtune.models import BackendAvailability, Config, ModelSpec, PreparedModel, Profile
from rigtune.selector import explain

# vLLM 0.29.0's environment on the 14 November 2026 pod: vLLM pins torch==2.13.0 and flashinfer-python==0.6.18
# (requirements/cuda.txt) and only transformers>=5.10.4, which resolved to 5.23.0.
VLLM_ENV = {"vllm": "0.29.0", "torch": "2.13.0", "transformers": "5.23.0", "flashinfer-python": "0.6.18"}


def test_a_backend_recorded_before_libraries_loads_with_none():
    old = BackendAvailability.model_validate({"name": "vllm", "available": True, "version": "0.29.0"})
    assert old.libs == {}


def test_vllm_libraries_come_from_its_environment_and_only_where_vllm_is(monkeypatch, hw_a100):
    monkeypatch.setattr(H, "_pkg_version", lambda name: VLLM_ENV.get(name))
    monkeypatch.setattr(H, "_torch_is_cuda_build", lambda: True)
    found = H.probe_backends(hw_a100.gpus)
    assert found["vllm"].libs == {"torch": "2.13.0", "transformers": "5.23.0", "flashinfer": "0.6.18"}
    assert found["vllm"].version == "0.29.0"  # unchanged
    monkeypatch.setattr(H, "_pkg_version", lambda name: None if name == "vllm" else VLLM_ENV.get(name))
    assert H.probe_backends(hw_a100.gpus)["vllm"].libs == {}  # torch alone is not an engine's


def test_sglang_in_its_own_environment_is_asked_once_for_its_version_and_libraries(monkeypatch, tmp_path,
                                                                                    hw_a100):
    py = tmp_path / "sgl" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    monkeypatch.setenv("SGLANG_PYTHON", str(py))
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        # SGLang 0.5.23's pins (python/pyproject.toml): torch 2.14.1, transformers 5.19.0, tokenizers 0.23.2.
        env = {"sglang": "0.5.23", "torch": "2.14.1", "transformers": "5.19.0", "flashinfer-python": "0.7.2",
               "tokenizers": "0.23.2"}
        return subprocess.CompletedProcess(cmd, 0, json.dumps({n: env.get(n) for n in cmd[3:]}) + "\n", "")

    monkeypatch.setattr(H.subprocess, "run", run)
    sgl = H.probe_backends(hw_a100.gpus)["sglang"]
    assert sgl.version == "0.5.23" and len(calls) == 1 and calls[0][0] == str(py)
    assert sgl.libs == {"torch": "2.14.1", "transformers": "5.19.0", "flashinfer": "0.7.2", "tokenizers": "0.23.2"}


def test_an_environment_that_cannot_answer_has_no_version_and_no_libraries(monkeypatch, tmp_path, hw_a100):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setenv("SGLANG_PYTHON", str(py))
    monkeypatch.setattr(H.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    sgl = H.probe_backends(hw_a100.gpus)["sglang"]
    assert sgl.version is None and sgl.libs == {} and not sgl.available


def test_the_question_runs_in_a_real_interpreter():
    """The code another environment runs, run here: one JSON line, null for what is not installed."""
    out = subprocess.run([sys.executable, "-c", H._VERSIONS_CODE, "pip", "no-such-distribution-here"],
                         capture_output=True, text=True, timeout=60, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["no-such-distribution-here"] is None and set(got) == {"pip", "no-such-distribution-here"}


def test_libraries_change_neither_the_hardware_hash_nor_whether_a_profile_is_valid(hw_a100, prepared_vllm):
    before = H.hardware_hash(hw_a100)
    newer = hw_a100.model_copy(deep=True)
    newer.backends["vllm"] = newer.backends["vllm"].model_copy(update={"libs": {"transformers": "5.23.0"}})
    assert H.hardware_hash(newer) == before
    profile = Profile(rigtune_version="x", hardware_hash=before, hardware=hw_a100, model_id="m",
                      objective="balanced", workload="chat", backend="vllm",
                      backend_version=hw_a100.backend_version("vllm"),
                      config=Config(backend="vllm", quant="bf16"),
                      prepared=PreparedModel(spec=ModelSpec(hf_id="m"), backend="vllm", arch=prepared_vllm.arch),
                      launch_args=[])
    assert invalid_reason(profile, newer) is None


def test_the_probe_lists_them_beside_the_engine(hw_a100):
    hw = hw_a100.model_copy(deep=True)
    hw.backends["vllm"] = hw.backends["vllm"].model_copy(update={"version": "0.29.0",
                                                                 "libs": {"transformers": "5.23.0", "torch": "2.13.0"}})
    lines = explain(hw)
    assert "vllm v0.29.0: available (torch 2.13.0, transformers 5.23.0)" in lines
    assert "sglang v1.0: available" in lines  # no libraries recorded, nothing added
