"""thinkingmachines/Inkling-2-Mini read from its config (tests/fixtures/inkling_2_mini_config.json, the Hub's config.json
of 20 Nov 2026 with the per-tensor quantization lists dropped; no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigtune.hfconfig import arch_from_config, vllm_context_length

INKLING2 = json.loads((Path(__file__).parent / "fixtures" / "inkling_2_mini_config.json").read_text("utf-8"))


def test_its_context_is_its_model_max_length():
    # text_config sets only model_max_length, which vLLM's config class copies to max_position_embeddings.
    assert arch_from_config(INKLING2).max_position_embeddings == 1_048_576


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
