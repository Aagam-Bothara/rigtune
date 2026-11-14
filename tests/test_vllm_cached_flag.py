"""vLLM 0.33 moved its server arguments to entrypoints/common/cli_args.py, where rigtune did not look, so a
session-file calibration on 0.33 launched without --enable-prompt-tokens-details and got no per-reply cached
tokens. Each check is run against a source tree shaped like each release (tests/engine_trees.py)."""

from __future__ import annotations

from unittest import mock

import pytest

import rigtune.backends.vllm as vl
from rigtune.backends import get_backend
from rigtune.models import Config
from tests.engine_trees import installed

RELEASES = ("0.29.0", "0.33.0")
FLAG = "--enable-prompt-tokens-details"


def _answers() -> dict:
    return {"attention_backend": vl._attention_backend_flag(),
            "prompt_tokens_details": vl._prompt_tokens_details_flag(),
            "language_model_only": vl._language_model_only_flag(),
            "offload": sorted(vl._offload_features())}


@pytest.mark.parametrize("version", RELEASES)
def test_every_feature_check_answers_the_same_on_both_releases(version, tmp_path):
    with installed("vllm", version, tmp_path):
        assert _answers() == {"attention_backend": True, "prompt_tokens_details": True,
                              "language_model_only": True, "offload": ["cpu", "hma", "shm"]}


@pytest.mark.parametrize("where, found", [
    ("entrypoints/common/cli_args.py", True),  # 0.33
    ("entrypoints/launchers/cli_args.py", True),  # 0.29
    ("entrypoints/openai/cli_args.py", True),  # 0.11
    ("entrypoints/cli_args.py", False),  # nowhere a release keeps it
])
def test_the_flag_is_looked_for_where_each_release_defines_its_server_arguments(where, found, tmp_path):
    pkg = tmp_path / "vllm"
    (pkg / where).parent.mkdir(parents=True)
    (pkg / where).write_text("    enable_prompt_tokens_details: bool = False\n", encoding="utf-8")
    with installed("vllm", "0.33.0", tmp_path):
        assert vl._prompt_tokens_details_flag() is found


CONFIGS = [
    Config(backend="vllm", quant="bf16", ctx=4096, batch=256, gpu_memory_utilization=0.90),
    Config(backend="vllm", quant="fp8", ctx=8192, batch=16, prefix_cache=False, prefill_budget=8192),
    Config(backend="vllm", quant="bf16", spec_decode="ngram_gpu:4", kv_dtype="fp8"),
    Config(backend="vllm", quant="bf16", kv_tier="host:16"),
]


def _launches(version, tmp_path, prepared) -> dict:
    with installed("vllm", version, tmp_path / version):
        return {(c.key(), reuse): get_backend("vllm", report_reuse=reuse).launch_spec(c, prepared, 8000)
                for c in CONFIGS for reuse in (False, True)}


def test_a_session_launch_on_033_asks_for_cached_tokens_and_nothing_else_moves(tmp_path, prepared_vllm):
    old, new = _launches("0.29.0", tmp_path, prepared_vllm), _launches("0.33.0", tmp_path, prepared_vllm)
    # 0.33 now launches exactly as 0.29 always did, with and without per-request reuse reports.
    assert {k: (v.args, v.env) for k, v in new.items()} == {k: (v.args, v.env) for k, v in old.items()}
    for (key, reuse), spec in new.items():
        assert spec.args.count(FLAG) == (1 if reuse else 0)
        if reuse:  # the flag is the only difference a session file makes
            assert [a for a in spec.args if a != FLAG] == new[(key, False)].args
    # Before this change the 0.33 session launch was this list without its last argument.
    assert new[(CONFIGS[0].key(), True)].args == [
        "vllm", "serve", "meta-llama/Llama-3.2-3B-Instruct", "--served-model-name",
        "meta-llama/Llama-3.2-3B-Instruct", "--host", "127.0.0.1", "--port", "8000", "--max-model-len", "4096",
        "--max-num-seqs", "256", "--gpu-memory-utilization", "0.90", "--dtype", "bfloat16", FLAG]


def test_without_a_vllm_source_tree_nothing_is_asked_for(tmp_path, prepared_vllm):
    with installed("vllm", "0.33.0", tmp_path), mock.patch.object(vl, "_vllm_package", lambda: None):
        vl._prompt_tokens_details_flag.cache_clear()
        assert FLAG not in get_backend("vllm", report_reuse=True).launch_spec(CONFIGS[0], prepared_vllm, 8000).args
