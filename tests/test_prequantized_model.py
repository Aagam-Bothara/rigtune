"""A checkpoint its author quantized is served as stored: no online fp8 on top, no Hub copies quantized from it.

Fixtures are the Hub's own files: Qwen/Qwen3-8B-FP8's quantization_config and nvidia/Llama-3.1-8B-Instruct-FP4's
hf_quant_config.json (21 Nov 2026).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import huggingface_hub
import pytest

import rigtune.backends.sglang as sg
import rigtune.backends.vllm as vl
import rigtune.quantized as qz
from rigtune.hfconfig import HF_QUANT, arch_from_config, load_arch
from rigtune.models import ModelSpec
from rigtune.pipeline import prepare_and_plan, select
from rigtune.quantized import as_stored
from tests.conftest import LLAMA_3B_CONFIG

QWEN3_FP8 = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128]}
NVFP4_HF_QUANT = {"producer": {"name": "modelopt", "version": "0.37.0.dev5+g76fb12d47.d20250905"},
                  "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16,
                                   "exclude_modules": ["lm_head"]}}
FP8_DESCRIBED = "fp8 (e4m3), 8-bit weights"


def test_the_config_says_how_the_checkpoint_is_stored():
    assert arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": QWEN3_FP8}).checkpoint_quant == FP8_DESCRIBED
    assert arch_from_config(LLAMA_3B_CONFIG).checkpoint_quant is None
    multimodal = {"architectures": ["X"], "text_config": {**LLAMA_3B_CONFIG, "quantization_config": QWEN3_FP8}}
    assert arch_from_config(multimodal).checkpoint_quant == FP8_DESCRIBED


def test_it_is_never_saved_in_a_profile():
    arch = arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": QWEN3_FP8})
    assert "checkpoint_quant" not in arch.model_dump()


def test_modelopts_own_file_is_read_when_the_config_declares_nothing(monkeypatch, tmp_path):
    path = tmp_path / HF_QUANT
    path.write_text(json.dumps(NVFP4_HF_QUANT))
    info = SimpleNamespace(sha="0" * 40, safetensors=None,
                           siblings=[SimpleNamespace(rfilename=HF_QUANT, size=300)])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: SimpleNamespace(model_info=lambda *a, **k: info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    monkeypatch.setattr("rigtune.hfconfig.fetch_config", lambda spec, token=None: LLAMA_3B_CONFIG)
    assert load_arch(ModelSpec(hf_id="nvidia/x")).checkpoint_quant == ("modelopt (NVFP4), 4-bit weights, "
                                                                       "4-bit activations, KV cache FP8")
    (tmp_path / "config.json").write_text(json.dumps(LLAMA_3B_CONFIG))  # the same layout on disk
    monkeypatch.undo()
    assert load_arch(ModelSpec(hf_id=str(tmp_path))).checkpoint_quant.startswith("modelopt (NVFP4)")


def test_an_unquantized_checkpoint_keeps_every_precision():
    assert as_stored(["bf16", "fp8", "awq"], None, "m") == ["bf16", "fp8", "awq"]


def test_a_quantized_checkpoint_keeps_only_its_stored_precision():
    assert as_stored(["bf16", "fp8", "awq", "gptq", "w8a8"], FP8_DESCRIBED, "m") == ["bf16"]
    assert as_stored(["fp16", "awq"], FP8_DESCRIBED, "m") == ["fp16"]  # Turing


def test_a_quant_filter_that_leaves_no_way_to_serve_it_says_so():
    with pytest.raises(RuntimeError, match=r"already quantized \(fp8 \(e4m3\), 8-bit weights\).*--quant fp8"):
        as_stored(["fp8"], FP8_DESCRIBED, "Qwen/Qwen3-8B-FP8")


@pytest.mark.parametrize("module,backend", [(vl, vl.VllmBackend), (sg, sg.SglangBackend)])
def test_both_engines_prepare_only_the_stored_precision(module, backend, hw_rtx4090, monkeypatch):
    arch = arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": QWEN3_FP8})
    arch.num_params, arch.checkpoint_bytes = 8190735360, 9436628784  # the Hub's count and the files' size
    monkeypatch.setattr(module, "load_arch", lambda spec, token=None: arch)

    def no_search(*a, **k):
        raise AssertionError("no Hub copy of an already quantized checkpoint is looked for")
    monkeypatch.setattr(qz, "find_prequantized_repos", no_search)
    prepared = backend().prepare(ModelSpec(hf_id="Qwen/Qwen3-8B-FP8"), hw_rtx4090,
                                 quants=["bf16", "fp8", "awq", "gptq"])
    assert prepared.weights_bytes == {"bf16": 9436628784}


def test_the_profile_says_why(hw_rtx4090, spec, no_network, monkeypatch):
    arch = arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": QWEN3_FP8})
    monkeypatch.setattr(vl, "load_arch", lambda s, token=None: arch)
    candidates, reg = select(hw_rtx4090, spec, force="vllm")
    plan = prepare_and_plan(hw_rtx4090, spec, candidates, reg)
    assert {c.quant for c, _ in plan.feasible["vllm"]} == {"bf16"}
    assert any(f"is stored quantized ({FP8_DESCRIBED}), so it is served as stored" in n for n in plan.notes)
