"""A checkpoint its author quantized is served as stored, under its own label ("ckpt-fp8"): no online fp8 on top,
no Hub copies quantized from it, and the label everywhere a precision is chosen, launched or compared.

Fixtures are the Hub's own files (21 Nov 2026): each checkpoint's quantization_config, trimmed to the fields read,
and nvidia/Llama-3.1-8B-Instruct-FP4's hf_quant_config.json.
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
from rigtune.quantized import as_stored, is_stored, stored_quant
from tests.conftest import LLAMA_3B_CONFIG

QWEN3_FP8 = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8", "weight_block_size": [128, 128]}
NVFP4_HF_QUANT = {"producer": {"name": "modelopt", "version": "0.37.0.dev5+g76fb12d47.d20250905"},
                  "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16,
                                   "exclude_modules": ["lm_head"]}}
FP8_DESCRIBED = "fp8 (e4m3), 8-bit weights"
# Each checkpoint's quantization_config, by the label it is served as stored under.
CONFIGS = {
    "ckpt-fp8": QWEN3_FP8,  # Qwen/Qwen3-8B-FP8
    "ckpt-mxfp4": {"quant_method": "mxfp4"},  # openai/gpt-oss-20b
    "ckpt-nvfp4": {"config_groups": {"group_0": {  # nvidia/Llama-3.1-8B-Instruct-FP4
        "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
        "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}}},
        "quant_algo": "NVFP4", "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "quant_method": "modelopt"},
    "ckpt-awq": {"bits": 4, "group_size": 128, "quant_method": "awq", "version": "gemm", "zero_point": True},
    "ckpt-w8a8": {"config_groups": {"group_0": {  # RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8
        "input_activations": {"num_bits": 8, "type": "int", "strategy": "token", "dynamic": True},
        "weights": {"num_bits": 8, "type": "int", "strategy": "channel", "dynamic": False}}},
        "format": "int-quantized", "quant_method": "compressed-tensors"},
}
NVFP4_CT = {"config_groups": {"group_0": {  # RedHatAI/Llama-3.1-8B-Instruct-NVFP4
    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
    "weights": {"num_bits": 4, "type": "float", "group_size": 16}}},
    "format": "nvfp4-pack-quantized", "quant_method": "compressed-tensors"}


def _quantized(config=QWEN3_FP8):
    arch = arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": config})
    arch.num_params, arch.checkpoint_bytes = 8190735360, 9436628784  # Qwen3-8B-FP8: the Hub's count, its files
    return arch


def test_the_config_says_how_the_checkpoint_is_stored():
    assert arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": QWEN3_FP8}).checkpoint_quant == FP8_DESCRIBED
    assert arch_from_config(LLAMA_3B_CONFIG).checkpoint_quant is None
    assert arch_from_config(LLAMA_3B_CONFIG).stored_as is None
    multimodal = {"architectures": ["X"], "text_config": {**LLAMA_3B_CONFIG, "quantization_config": QWEN3_FP8}}
    assert arch_from_config(multimodal).checkpoint_quant == FP8_DESCRIBED


@pytest.mark.parametrize("label", sorted(CONFIGS))
def test_each_checkpoint_gets_a_label_that_says_what_it_is(label):
    assert arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": CONFIGS[label]}).stored_as == label


def test_modelopt_and_compressed_tensors_nvfp4_get_the_same_label():
    assert arch_from_config({**LLAMA_3B_CONFIG, "quantization_config": NVFP4_CT}).stored_as == "ckpt-nvfp4"


def test_neither_is_saved_in_a_profile():
    dumped = _quantized().model_dump()
    assert "checkpoint_quant" not in dumped and "stored_as" not in dumped


def test_modelopts_own_file_is_read_when_the_config_declares_nothing(monkeypatch, tmp_path):
    path = tmp_path / HF_QUANT
    path.write_text(json.dumps(NVFP4_HF_QUANT))
    info = SimpleNamespace(sha="0" * 40, safetensors=None,
                           siblings=[SimpleNamespace(rfilename=HF_QUANT, size=300)])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: SimpleNamespace(model_info=lambda *a, **k: info))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    monkeypatch.setattr("rigtune.hfconfig.fetch_config", lambda spec, token=None: LLAMA_3B_CONFIG)
    arch = load_arch(ModelSpec(hf_id="nvidia/x"))
    assert arch.checkpoint_quant == "modelopt (NVFP4), 4-bit weights, 4-bit activations, KV cache FP8"
    assert arch.stored_as == "ckpt-nvfp4"
    (tmp_path / "config.json").write_text(json.dumps(LLAMA_3B_CONFIG))  # the same layout on disk
    monkeypatch.undo()
    assert load_arch(ModelSpec(hf_id=str(tmp_path))).stored_as == "ckpt-nvfp4"


def test_an_unquantized_checkpoint_keeps_every_precision():
    assert as_stored(["bf16", "fp8", "awq"], arch_from_config(LLAMA_3B_CONFIG), "m") == ["bf16", "fp8", "awq"]


def test_a_quantized_checkpoint_gets_its_stored_label_in_place_of_16_bit():
    assert as_stored(["bf16", "fp8", "awq", "gptq", "w8a8"], _quantized(), "m") == ["ckpt-fp8"]
    assert as_stored(["fp16", "awq"], _quantized(), "m") == ["ckpt-fp8"]  # Turing


def test_a_quant_filter_that_leaves_no_way_to_serve_it_says_so():
    with pytest.raises(RuntimeError, match=r"already quantized \(fp8 \(e4m3\), 8-bit weights\).*--quant fp8"):
        as_stored(["fp8"], _quantized(), "Qwen/Qwen3-8B-FP8")


@pytest.mark.parametrize("module,backend", [(vl, vl.VllmBackend), (sg, sg.SglangBackend)])
def test_both_engines_prepare_plan_and_launch_it_as_stored(module, backend, hw_rtx4090, monkeypatch):
    arch = _quantized()
    monkeypatch.setattr(module, "load_arch", lambda spec, token=None: arch)

    def no_search(*a, **k):
        raise AssertionError("no Hub copy of an already quantized checkpoint is looked for")
    monkeypatch.setattr(qz, "find_prequantized_repos", no_search)
    engine = backend()
    prepared = engine.prepare(ModelSpec(hf_id="Qwen/Qwen3-8B-FP8"), hw_rtx4090, quants=["bf16", "fp8", "awq", "gptq"])
    assert prepared.weights_bytes == {"ckpt-fp8": 9436628784}  # the files' size, not two bytes a parameter
    grid = engine.candidate_configs(hw_rtx4090, prepared)
    assert grid and {c.quant for c in grid} == {"ckpt-fp8"}
    assert engine.default_config(hw_rtx4090, prepared).quant == "ckpt-fp8"
    args = engine.launch_spec(grid[0], prepared, 8000).args
    assert "--quantization" not in args and "--dtype" not in args  # the checkpoint's own method and dtype


def test_vllm_passes_the_checkpoints_revision(hw_rtx4090, monkeypatch):
    """A quantized checkpoint given as the model is the model itself, not a Hub copy: its revision is its own."""
    arch = _quantized()
    monkeypatch.setattr(vl, "load_arch", lambda spec, token=None: arch)
    engine = vl.VllmBackend()
    prepared = engine.prepare(ModelSpec(hf_id="Qwen/Qwen3-8B-FP8", revision="a" * 40), hw_rtx4090)
    args = engine.launch_spec(engine.default_config(hw_rtx4090, prepared), prepared, 8000).args
    assert args[args.index("--revision") + 1] == "a" * 40


def test_the_stored_label_is_the_models_own_weights_to_the_answer_check():
    from rigtune.accuracy_gate import faithful
    from rigtune.quality import precision_rank

    assert faithful("ckpt-nvfp4") and not faithful("awq")
    assert precision_rank("ckpt-nvfp4") == precision_rank("bf16") > precision_rank("fp8")


def test_quant_filters_take_it_in_place_of_16_bit():
    from rigtune.pipeline import allowed_quants

    assert allowed_quants(["ckpt-fp8"], None) == ["ckpt-fp8"]  # auto
    assert allowed_quants(["ckpt-fp8"], ["bf16"]) == ["ckpt-fp8"]
    assert allowed_quants(["ckpt-fp8"], ["fp8"]) == []


def test_helpers():
    assert is_stored("ckpt-fp8") and not is_stored("fp8") and not is_stored("awq")
    assert stored_quant({"ckpt-awq": 1}) == "ckpt-awq" and stored_quant({"bf16": 1, "fp8": 1}) is None


def test_the_stock_rows_serve_it_as_stored(hw_rtx4090, monkeypatch):
    from rigtune.bench.references import reference_configs

    arch = _quantized()
    monkeypatch.setattr(vl, "load_arch", lambda spec, token=None: arch)
    prepared = vl.VllmBackend().prepare(ModelSpec(hf_id="Qwen/Qwen3-8B-FP8"), hw_rtx4090)
    refs = reference_configs(hw_rtx4090, {"vllm": prepared}, {"vllm": vl.VllmBackend()})
    assert refs["vllm-default"].quant == "ckpt-fp8" and "vllm-fp8-default" not in refs


def test_a_comparison_measures_the_stock_and_other_tools_rows_as_stored(hw_a100, spec, no_network, monkeypatch):
    """An auto_tune row asking for the model's own 16-bit weights gets them: the checkpoint as stored."""
    from rigtune.bench import compare
    from rigtune.bench.rows import ExtraRow
    from rigtune.models import Config
    from rigtune.pipeline import calibrate
    from tests.test_compare import CompareRunner
    from tests.test_extra_rows import AUTOTUNE
    from tests.test_objectives_and_search import FakeRunner

    arch = _quantized()
    monkeypatch.setattr(vl, "load_arch", lambda s, token=None: arch)
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    speeds = {**FakeRunner.QUANT_SPEED, "ckpt-fp8": FakeRunner.QUANT_SPEED["fp8"]}  # the fakes' table, plus it

    class Runner(FakeRunner):
        QUANT_SPEED = speeds

    class Comparing(CompareRunner):
        QUANT_SPEED = speeds

    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=Runner())
    result = compare(hw_a100, spec, profile, planned.prepared, reg, runner=Comparing(),
                     extra={"autotune-vllm": ExtraRow(config=Config.model_validate(AUTOTUNE), source="auto_tune.sh")})
    quants = {r.label: r.config.quant for r in result.rows}
    assert quants == {"rigtune": "ckpt-fp8", "vllm-default": "ckpt-fp8", "autotune-vllm": "ckpt-fp8"}


def test_the_profile_says_why(hw_rtx4090, spec, no_network, monkeypatch):
    arch = _quantized()
    monkeypatch.setattr(vl, "load_arch", lambda s, token=None: arch)
    candidates, reg = select(hw_rtx4090, spec, force="vllm")
    plan = prepare_and_plan(hw_rtx4090, spec, candidates, reg)
    assert {c.quant for c, _ in plan.feasible["vllm"]} == {"ckpt-fp8"}
    assert any(f"is stored quantized ({FP8_DESCRIBED}), so it is served as stored (ckpt-fp8)" in n
               for n in plan.notes)
