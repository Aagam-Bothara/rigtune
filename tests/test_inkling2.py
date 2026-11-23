"""thinkingmachines/Inkling-2-Mini read from its config (tests/fixtures/inkling_2_mini_config.json, the Hub's config.json
of 20 Nov 2026 with the per-tensor quantization lists dropped; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigtune.hfconfig import arch_from_config, vllm_context_length
from rigtune.models import Config

INKLING2 = json.loads((Path(__file__).parent / "fixtures" / "inkling_2_mini_config.json").read_text("utf-8"))


def test_its_context_is_its_model_max_length():
    # text_config sets only model_max_length, which vLLM's config class copies to max_position_embeddings.
    assert arch_from_config(INKLING2).max_position_embeddings == 1_048_576


def test_its_layout_counts_each_loop_pass_s_caches():
    # 20 local layers (window 1,024) and 4 full (5, 11, 17, 23); layers 6-17 run 3 passes, each with its own caches:
    # full 5, 11 x 3, 17 x 3, 23 = 8 and 40 sliding, as vLLM and SGLang lay them out.
    a = arch_from_config(INKLING2)
    assert (a.num_layers, a.loop_extra_layers, a.full_attention_layers, a.sliding_layers, a.sliding_window) == \
        (24, 24, 8, 40, 1024)
    assert a.full_kv_bytes_per_token(2) == 8 * 2 * 8 * 128 * 2  # 32 KiB a token in bf16
    assert a.kv_bytes_per_token(2) == 48 * 2 * 8 * 128 * 2  # the every-layer-full fallback counts 48 caches too


def _text(**fields):
    return {"architectures": ["X"], "num_hidden_layers": 6, "hidden_size": 256, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 64, **fields}


INKLING2_ARCH = ["Inkling2LoopForConditionalGeneration"]


def test_local_layer_ids_alone_make_a_sliding_layout():
    a = arch_from_config(_text(architectures=INKLING2_ARCH, local_layer_ids=[0, 1, 3, 4], sliding_window_size=512))
    assert (a.full_attention_layers, a.sliding_layers, a.sliding_window, a.loop_extra_layers) == (2, 4, 512, 0)
    # Without a window the ids say nothing that can be planned: every layer full, as before.
    assert not arch_from_config(_text(architectures=INKLING2_ARCH, local_layer_ids=[0, 1])).is_hybrid
    # Only Inkling 2's are read: another family's (the first Inkling's) were not checked.
    assert not arch_from_config(_text(local_layer_ids=[0, 1, 3, 4], sliding_window_size=512)).is_hybrid


def test_sliding_layers_sized_apart_from_the_full_ones_are_not_planned_as_if_alike():
    for field in ({"swa_head_dim": 32}, {"swa_num_key_value_heads": 1}):
        a = arch_from_config(_text(architectures=INKLING2_ARCH, local_layer_ids=[0, 1], sliding_window_size=512,
                                   **field))
        assert not a.is_hybrid and "are not modelled" in a.layout_note
    # Equal to the full layers' (Inkling-2-Mini's own: 128 and 8), as planned.
    assert arch_from_config(INKLING2).is_hybrid


def test_both_engines_plan_it_as_every_layer_full_and_say_why(hw_a100):
    from rigtune.backends.sglang import SglangBackend
    from rigtune.backends.vllm import VllmBackend

    arch = arch_from_config(INKLING2)
    for be, says in ((VllmBackend(), "conv state for every layer and loop pass"), (SglangBackend(), "not modelled")):
        mm = be.memory_model(hw_a100)
        reason = mm.full_kv_reason(arch)
        assert says in reason and "every layer full" in reason
        assert mm.hybrid_for(Config(backend=be.name, quant="bf16", ctx=4096, batch=16), arch) is None
    # A family whose rules apply is unchanged.
    gemma_like = arch.model_copy(update={"architecture": "Gemma4ForConditionalGeneration"})
    assert VllmBackend().memory_model(hw_a100).full_kv_reason(gemma_like) is None


def test_a_loop_s_passes_are_counted_whatever_the_layout():
    loop = {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8, "is_varloop": False}
    dense = arch_from_config(_text(loop_config=loop))
    assert (dense.loop_extra_layers, dense.is_hybrid) == (4, False)
    assert dense.kv_bytes_per_token(2) == (6 + 4) * 2 * 2 * 64 * 2
    # A layout that falls back to every layer full still keeps the passes.
    odd = arch_from_config(_text(loop_config=loop, layer_types=["full_attention", "sliding_attention"]))
    assert odd.layout_note and odd.loop_extra_layers == 4
    # A loop whose fields do not make one is ignored, as is one pass.
    for bad in ({**loop, "core_end": 9}, {**loop, "num_passes": "3"}, {**loop, "num_passes": 1}):
        assert arch_from_config(_text(loop_config=bad)).loop_extra_layers == 0


@pytest.mark.parametrize("cfg, expected", [
    ({"max_position_embeddings": 32768, "seq_length": 8192}, 8192),  # the smallest key wins (Qwen/Qwen-7B-Chat)
    ({"n_positions": 1024}, 1024),  # GPT-2
    ({"max_position_embeddings": 8192, "model_max_length": 131072}, 131072),  # model_max_length replaces it (Cohere)
    ({"max_seq_len": 2048, "max_target_positions": 448}, 448),
    ({"n_ctx": 2048}, 2048),  # none of vLLM's keys: as before
    ({}, 4096),
])
def test_the_context_is_derived_as_vllm_derives_it(cfg, expected):
    assert vllm_context_length(cfg) == expected
