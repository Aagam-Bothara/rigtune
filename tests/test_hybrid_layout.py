"""Hybrid attention layouts read from model configs (trimmed Hub configs in tests/fixtures; no network)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rigtune.hfconfig import arch_from_config, gdn_state_bytes
from rigtune.models import ArchInfo

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


QWEN35 = "qwen3_5_9b_config.json"
GEMMA4 = "gemma4_12b_it_config.json"
QWEN25 = "qwen2_5_7b_instruct_config.json"

# Qwen3.5-9B, one Gated DeltaNet layer, one sequence: conv state (2*16*128 + 32*128) x (4 - 1) in bf16
# plus a temporal state of 32 x 128 x 128 in float32 (the config's mamba_ssm_dtype).
QWEN35_STATE_PER_LAYER = (2 * 16 * 128 + 32 * 128) * 3 * 2 + 32 * 128 * 128 * 4  # 2,146,304 B (~2.05 MiB)


def test_dense_control_has_no_hybrid_fields():
    arch = arch_from_config(load(QWEN25))
    assert not arch.is_hybrid and arch.layout_note is None
    assert (arch.sliding_layers, arch.linear_layers, arch.linear_state_bytes) == (0, 0, 0)
    # Its config carries sliding_window 131072 with use_sliding_window false: still every layer full.
    assert arch.sliding_window is None
    assert arch.kv_bytes_per_token(2) == 2 * 28 * 4 * 128 * 2


def test_dense_arch_is_unchanged_field_for_field():
    """A dense config yields exactly the ArchInfo it did before hybrid fields existed."""
    arch = arch_from_config(load(QWEN25))
    before = {"architecture", "num_layers", "hidden_size", "num_attention_heads", "num_kv_heads", "head_dim",
              "vocab_size", "max_position_embeddings", "num_params", "torch_dtype"}
    assert ArchInfo(**{k: getattr(arch, k) for k in before}) == arch


def test_qwen3_5_layout():
    arch = arch_from_config(load(QWEN35))
    assert arch.architecture == "Qwen3_5ForConditionalGeneration"
    assert arch.is_hybrid and arch.layout_note is None
    assert (arch.num_layers, arch.full_attention_layers, arch.linear_layers, arch.sliding_layers) == (32, 8, 24, 0)
    assert arch.linear_state_bytes == 24 * QWEN35_STATE_PER_LAYER == 51_511_296  # ~49.1 MiB per sequence
    # Each speculative token adds a conv column of (2*16*128 + 32*128) x 2 B per layer: 2,211,840 B a layer at
    # k = 4 and 2,539,520 B at k = 24 (vLLM gated_delta_net_state_shape, conv_kernel - 1 + num_spec).
    assert arch.linear_conv_step_bytes == 24 * 8192 * 2
    assert QWEN35_STATE_PER_LAYER + 4 * 8192 * 2 == 2_211_840 and QWEN35_STATE_PER_LAYER + 24 * 8192 * 2 == 2_539_520
    # Full layers: 2 x 8 x 4 heads x 256 x 2 B = 32 KiB per token, a quarter of the every-layer-full figure.
    assert arch.full_kv_bytes_per_token(2) == 32_768
    assert arch.kv_bytes_per_token(2) == 4 * 32_768


def test_qwen3_5_interval_fallback_matches_layer_types():
    cfg = load(QWEN35)
    del cfg["text_config"]["layer_types"]
    assert arch_from_config(cfg) == arch_from_config(load(QWEN35))


def test_gemma4_layout_uses_global_heads_on_full_layers():
    arch = arch_from_config(load(GEMMA4))
    assert arch.architecture == "Gemma4UnifiedForConditionalGeneration"
    assert (arch.num_layers, arch.full_attention_layers, arch.sliding_layers) == (48, 8, 40)
    assert arch.sliding_window == 1024 and arch.linear_layers == 0
    # attention_k_eq_v: full layers keep 1 KV head of global_head_dim 512; sliding layers 8 heads of 256.
    assert (arch.full_kv_heads, arch.full_head_dim) == (1, 512)
    assert arch.full_kv_bytes_per_token(2) == 2 * 8 * 1 * 512 * 2  # 16 KiB per token
    assert arch.sliding_kv_bytes_per_token(2) == 2 * 40 * 8 * 256 * 2  # 320 KiB per token


def test_gemma3_sliding_window_pattern():
    # The fields google/gemma-3-12b-it carries (no layer_types): 48 layers, pattern 6, window 1024.
    cfg = {"architectures": ["Gemma3ForConditionalGeneration"],
           "text_config": {"num_hidden_layers": 48, "num_key_value_heads": 8, "head_dim": 256, "hidden_size": 3840,
                           "num_attention_heads": 16, "sliding_window": 1024, "sliding_window_pattern": 6}}
    arch = arch_from_config(cfg)
    assert (arch.full_attention_layers, arch.sliding_layers, arch.sliding_window) == (8, 40, 1024)


def test_all_full_layer_types_is_dense():
    cfg = load(QWEN25)
    cfg["layer_types"] = ["full_attention"] * 28
    assert arch_from_config(cfg) == arch_from_config(load(QWEN25))


def test_unknown_layer_type_falls_back_with_a_warning(caplog):
    cfg = load(QWEN35)
    cfg["text_config"]["layer_types"] = ["mamba"] * 24 + ["full_attention"] * 8
    cfg["architectures"] = ["SomeHybridForCausalLM"]
    with caplog.at_level(logging.WARNING, logger="rigtune.hfconfig"):
        arch = arch_from_config(cfg)
    assert not arch.is_hybrid and arch.linear_state_bytes == 0
    assert arch.layout_note == "unrecognised layer types mamba"
    assert "every layer kept a full cache" in caplog.text


def test_missing_window_or_state_fields_fall_back():
    cfg = load(GEMMA4)
    del cfg["text_config"]["sliding_window"]
    assert arch_from_config(cfg).layout_note == "sliding layers without a sliding_window"
    cfg = load(QWEN35)
    del cfg["text_config"]["linear_key_head_dim"]
    assert not arch_from_config(cfg).is_hybrid


def test_gdn_state_defaults_to_float32_without_mamba_ssm_dtype():
    text = dict(load(QWEN35)["text_config"])
    assert gdn_state_bytes(text, "bfloat16") == QWEN35_STATE_PER_LAYER
    del text["mamba_ssm_dtype"]
    assert gdn_state_bytes(text, "bfloat16") == QWEN35_STATE_PER_LAYER


def test_profiles_cached_before_hybrid_fields_still_load():
    old = {"architecture": "LlamaForCausalLM", "num_layers": 28, "hidden_size": 3072, "num_attention_heads": 24,
           "num_kv_heads": 8, "head_dim": 128, "vocab_size": 128256, "max_position_embeddings": 131072,
           "num_params": 3_212_749_824, "torch_dtype": "bfloat16"}
    arch = ArchInfo.model_validate(old)
    assert not arch.is_hybrid and arch.kv_bytes_per_token(2) == 114_688


def test_qwen3_6_and_qwen3_8_share_the_layout():
    """Qwen3.6-35B-A3B (MoE) and Qwen3.8-27B: the same 3 linear : 1 full interleave, parsed only."""
    moe = arch_from_config(load("qwen3_6_35b_a3b_config.json"))
    assert moe.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert (moe.num_layers, moe.full_attention_layers, moe.linear_layers) == (40, 10, 30)
    assert moe.linear_state_bytes == 30 * QWEN35_STATE_PER_LAYER  # same GDN shape as Qwen3.5-9B
    assert moe.full_kv_bytes_per_token(2) == 2 * 10 * 2 * 256 * 2
    dense = arch_from_config(load("qwen3_8_27b_config.json"))
    assert dense.architecture == "Qwen3_5ForConditionalGeneration"
    assert (dense.num_layers, dense.full_attention_layers, dense.linear_layers) == (64, 16, 48)
    # 48 value heads: conv (2*16*128 + 48*128) x 3 x 2 B + temporal 48 x 128 x 128 x 4 B per layer.
    assert dense.linear_state_bytes == 48 * ((2 * 16 * 128 + 48 * 128) * 3 * 2 + 48 * 128 * 128 * 4)


HYBRID_ARCHS = ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLM",
                "Qwen3_5MoeForCausalLM", "Gemma4UnifiedForConditionalGeneration", "Gemma4ForConditionalGeneration",
                "Gemma4ForCausalLM")


def test_hybrid_architectures_are_known_without_the_engine_registry(hw_rtx4090, monkeypatch):
    """SGLang usually runs from $SGLANG_PYTHON, so rigtune cannot import its registry and falls back to KNOWN_ARCHS;
    before these entries both engines refused the 2026 hybrid models whenever that happened."""
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    from rigtune.models import ModelSpec

    base = arch_from_config(load(QWEN35))
    for name in HYBRID_ARCHS:
        arch = base.model_copy(update={"architecture": name})
        for mod in (vl, sg):
            monkeypatch.setattr(mod, "load_arch", lambda spec, token=None, a=arch: a)
        # Both registries are stubbed to "cannot be imported" (conftest).
        assert vl.VllmBackend().supports(hw_rtx4090, ModelSpec(hf_id="x")), name
        assert sg.SglangBackend().supports(hw_rtx4090, ModelSpec(hf_id="x")), name


def test_a_readable_registry_still_decides(hw_rtx4090, monkeypatch):
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    from rigtune.models import ModelSpec

    arch = arch_from_config(load(QWEN35))
    for mod in (vl, sg):
        monkeypatch.setattr(mod, "load_arch", lambda spec, token=None: arch)
    monkeypatch.setattr(vl, "vllm_registry_archs", lambda: {"LlamaForCausalLM"})
    monkeypatch.setattr(sg, "sglang_registry_archs", lambda: {"LlamaForCausalLM"})
    assert not vl.VllmBackend().supports(hw_rtx4090, ModelSpec(hf_id="x"))
    assert not sg.SglangBackend().supports(hw_rtx4090, ModelSpec(hf_id="x"))
    monkeypatch.setattr(vl, "vllm_registry_archs", lambda: {arch.architecture})
    monkeypatch.setattr(sg, "sglang_registry_archs", lambda: {arch.architecture})
    assert vl.VllmBackend().supports(hw_rtx4090, ModelSpec(hf_id="x"))
    assert sg.SglangBackend().supports(hw_rtx4090, ModelSpec(hf_id="x"))
