"""rigtune.heads: which drafters each engine loads, and whether one fits a target, on configs read from the Hub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rigtune.heads import check_head, head_kind, target_layer_ids
from rigtune.models import ArchInfo

FIXTURES = Path(__file__).parent / "fixtures" / "heads"


def _head(repo: str):
    """(config, files) of a drafter as read on the Hub on 21 Nov 2026."""
    data = json.loads((FIXTURES / (repo.replace("/", "__") + ".json")).read_text(encoding="utf-8"))
    return data["config"], data["files"]


def _target(layers: int, hidden: int, vocab: int, arch: str = "unknown") -> ArchInfo:
    return ArchInfo(architecture=arch, num_layers=layers, hidden_size=hidden, num_attention_heads=32, num_kv_heads=8,
                    head_dim=128, vocab_size=vocab)


# Each target's config.json: layers, hidden size, vocabulary.
LLAMA31_8B = _target(32, 4096, 128256)
QWEN3_8B = _target(36, 4096, 151936)
QWEN35_9B = _target(32, 4096, 248320)
QWEN3_30B_A3B = _target(48, 2048, 151936)
GEMMA4_12B = _target(48, 3840, 262144)
GPT_OSS_20B = _target(24, 2880, 201088, "GptOssForCausalLM")
GPT_OSS_120B = _target(36, 2880, 201088, "GptOssForCausalLM")
GEMMA4_31B = _target(60, 5376, 262144)
QWEN38_27B = _target(64, 5120, 248320)
KIMI_K3 = _target(93, 7168, 163840)
MUSE_GLIMMER_30B = _target(52, 6656, 202048)

OK, NOT_HERE = "ok", "not_a_head_here"

CASES = [
    # repo, target, aliases, vLLM verdict, SGLang verdict (status, or the first reason refused)
    ("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B", LLAMA31_8B, (), OK, OK),
    ("AngelSlim/Qwen3-8B_eagle3", QWEN3_8B, (), OK, OK),
    # Its layers [1, 11, 21] sit in eagle_config only: vLLM 0.33's default runner never reads them there and
    # captures its defaults (2, 12, 21), so the head would run on other layers with no error.
    ("zhuyksir/EAGLE3-gpt-oss-20b-bf16", GPT_OSS_20B, (), "layer_ids_unread_by_engine", OK),
    # Plain LlamaForCausalLM with one layer and a draft vocabulary: vLLM renames it into Eagle3LlamaForCausalLM.
    ("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", LLAMA31_8B, (), OK, NOT_HERE),
    ("nm-testing/Speculator-Qwen3-8B-Eagle3", QWEN3_8B, (), OK, NOT_HERE),
    # The speculators format: vLLM converts it while reading it; SGLang has no converter.
    ("RedHatAI/Qwen3-8B-speculator.eagle3", QWEN3_8B, ("Qwen/Qwen3-8B",), OK, NOT_HERE),
    ("RedHatAI/Qwen3-30B-A3B-speculator.dflash", QWEN3_30B_A3B, ("Qwen/Qwen3-30B-A3B",), OK, NOT_HERE),
    ("z-lab/Qwen3-8B-DFlash-b16", QWEN3_8B, (), OK, OK),
    ("z-lab/Qwen3.5-9B-DFlash", QWEN35_9B, (), OK, OK),
    ("z-lab/gemma4-12B-it-DFlash", GEMMA4_12B, (), OK, OK),
    ("deepseek-ai/dspark_qwen3_8b_block7", QWEN3_8B, (), OK, OK),
    ("deepseek-ai/dspark_gemma4_12b_block7", GEMMA4_12B, (), OK, NOT_HERE),  # Gemma4DSparkModel: vLLM only
    # Traps found on the Hub: a head for a 2048-wide model under a Qwen3-8B tag; a 36-layer "head" with its own
    # vocabulary; a head that ships a ModelOpt quantization.
    ("Tengyunw/qwen3_30b_moe_eagle3", QWEN3_8B, (), "hidden_differs", "hidden_differs"),
    ("AvitoTech/avibe-eagle", QWEN3_8B, (), "vocab_differs", "vocab_differs"),
    ("nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3", QWEN3_30B_A3B, (), "quantized_head", "quantized_head"),
    # Its last id is the target's depth: vLLM captures it (the last layer's output); SGLang keeps such ids as they
    # are for gpt-oss alone.
    ("nvidia/gpt-oss-120b-Eagle3-v3", GPT_OSS_120B, (), OK, OK),
    ("RedHatAI/gemma-4-31B-it-speculator.eagle3", GEMMA4_31B, ("google/gemma-4-31B-it",), OK, NOT_HERE),
    # DSparkDraftModel of model_type qwen3: vLLM renames it to Qwen3DSparkModel; SGLang's generic DSpark.
    ("RadixArk/Qwen3.8-27B-DSpark", QWEN38_27B, (), OK, OK),
    # num_target_layers 5: the layers it reads, as vLLM's MLA DSpark requires, not Kimi-K3's 93.
    ("Inferact/Kimi-K3-DSpark", KIMI_K3, (), OK, NOT_HERE),
]


@pytest.mark.parametrize("repo, target, aliases, vllm, sglang", CASES, ids=[c[0] for c in CASES])
def test_each_engine_s_verdict_on_real_drafters(repo, target, aliases, vllm, sglang):
    cfg, files = _head(repo)
    for engine, expected in (("vllm", vllm), ("sglang", sglang)):
        verdict = check_head(target, cfg, engine, files=files, aliases=aliases)
        got = verdict.status if verdict.status == "ok" else verdict.reasons[0]
        assert got == expected, (engine, verdict)


def test_the_kind_each_drafter_is_loaded_as():
    expected = {"lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B": "eagle3", "RedHatAI/Qwen3-8B-speculator.eagle3": "eagle3",
                "RedHatAI/Qwen3-30B-A3B-speculator.dflash": "dflash", "z-lab/Qwen3.5-9B-DFlash": "dflash",
                "deepseek-ai/dspark_qwen3_8b_block7": "dspark", "deepseek-ai/dspark_gemma4_12b_block7": "dspark"}
    for repo, kind in expected.items():
        assert head_kind(_head(repo)[0], "vllm") == kind


def test_a_drafter_for_another_size_of_the_family_is_refused():
    """z-lab's Qwen3-8B DFlash on Qwen3-30B-A3B: same vocabulary, another width and depth."""
    verdict = check_head(QWEN3_30B_A3B, _head("z-lab/Qwen3-8B-DFlash-b16")[0], "vllm")
    assert verdict.status == "refused" and {"hidden_differs", "target_layers_differ"} <= set(verdict.reasons)


def test_target_layers_are_read_in_each_format():
    assert target_layer_ids(_head("z-lab/Qwen3.5-9B-DFlash")[0], "dflash") == [1, 5, 9, 13, 17, 21, 25, 29]
    assert target_layer_ids(_head("deepseek-ai/dspark_qwen3_8b_block7")[0], "dspark") == [1, 9, 17, 25, 33]
    # speculators DFlash stores them one higher (vLLM's algos.py: aux_id - 1)
    assert target_layer_ids(_head("RedHatAI/Qwen3-30B-A3B-speculator.dflash")[0], "dflash") == [0, 11, 22, 33, 44]
    assert target_layer_ids(_head("zhuyksir/EAGLE3-gpt-oss-20b-bf16")[0], "eagle3") == [1, 11, 21]
    assert target_layer_ids(_head("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")[0], "eagle3") is None


def test_layers_the_target_does_not_have_are_refused():
    cfg = {**_head("z-lab/Qwen3.5-9B-DFlash")[0], "num_target_layers": None,
           "dflash_config": {"mask_token_id": 248077, "target_layer_ids": [1, 31, 32]}}
    assert check_head(QWEN35_9B, cfg, "sglang").reasons == ["layer_ids_outside_target"]


def test_a_speculators_verifier_naming_another_model_is_refused():
    cfg, _ = _head("RedHatAI/Qwen3-8B-speculator.eagle3")
    assert check_head(QWEN3_8B, cfg, "vllm", aliases=("qwen/qwen3-8b",)).status == "ok"  # case-insensitive
    assert check_head(QWEN3_8B, cfg, "vllm", aliases=("RedHatAI/Qwen3-8B-FP8-dynamic",)).reasons == [
        "verifier_differs"]
    assert check_head(QWEN3_8B, cfg, "vllm").status == "ok"  # no aliases given: not checked


def test_a_full_model_is_not_a_head_on_either_engine():
    """A plain Llama 3.2 1B would be renamed into vLLM's Eagle3 model too; only a head's marks let one through."""
    full = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 16,
            "vocab_size": 128256, "hidden_size": 2048}
    for engine in ("vllm", "sglang"):
        assert check_head(LLAMA31_8B, full, engine).reasons == ["not_a_head_here"]
    assert check_head(LLAMA31_8B, {}, "vllm").reasons == ["not_a_head_here"]


def test_missing_facts_are_unknown_never_a_fit():
    cfg = {"architectures": ["DFlashDraftModel"], "num_hidden_layers": 5}
    verdict = check_head(QWEN3_8B, cfg, "vllm")
    assert verdict.status == "unknown"
    assert set(verdict.reasons) == {"vocab_unread", "hidden_unread", "layer_ids_unread", "mask_token_unread"}


def test_an_unknown_engine_loads_nothing():
    assert head_kind(_head("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")[0], "llamacpp-cuda") is None


def test_a_speculators_eagle3_head_s_layers_are_checked_as_stored():
    """vLLM takes a speculators EAGLE3 head's eagle_aux_hidden_state_layer_ids as stored (only DFlash and DSpark ids
    are one higher), so a target without those layers is refused."""
    cfg, _ = _head("RedHatAI/gemma-4-31B-it-speculator.eagle3")
    assert target_layer_ids(cfg, "eagle3") == [2, 30, 57]
    shallow = _target(20, 5376, 262144)
    assert check_head(shallow, cfg, "vllm").reasons == ["layer_ids_outside_target"]


def test_eagle3_layer_bounds_follow_each_engine_s_numbering():
    cfg = {"architectures": ["LlamaForCausalLMEagle3"], "num_hidden_layers": 1, "vocab_size": 128256,
           "hidden_size": 4096, "draft_vocab_size": 32000}
    edge = "layer_ids_edge_unchecked"
    for ids, vllm, sglang in (([1, 15, 29], OK, OK),  # Llama: SGLang captures id + 1, at most layer 31
                              ([1, 15, 30], OK, OK),
                              ([1, 15, 31], OK, edge),  # captured by some SGLang targets' final norm, not Llama's
                              ([1, 15, 32], OK, "layer_ids_outside_target"),  # vLLM: the last layer's output
                              ([1, 15, 33], "layer_ids_outside_target", "layer_ids_outside_target")):
        # Named where both engines read them: vLLM at the top, SGLang in eagle_config.
        head = {**cfg, "eagle_aux_hidden_state_layer_ids": ids,
                "eagle_config": {"eagle_aux_hidden_state_layer_ids": ids}}
        for engine, expected in (("vllm", vllm), ("sglang", sglang)):
            verdict = check_head(LLAMA31_8B, head, engine)
            assert (verdict.status if verdict.status == "ok" else verdict.reasons[0]) == expected, (ids, engine)
    gpt_oss = {**cfg, "vocab_size": 201088, "hidden_size": 2880, "eagle_config": {"eagle_aux_hidden_state_layer_ids":
                                                                                   [1, 11, 23]}}
    assert check_head(GPT_OSS_20B, gpt_oss, "sglang").status == "ok"  # gpt-oss also captures its final output


def test_vllm_falls_back_to_its_default_layers_when_they_sit_in_eagle_config_only():
    cfg = {"architectures": ["LlamaForCausalLMEagle3"], "num_hidden_layers": 1, "vocab_size": 128256,
           "hidden_size": 4096, "draft_vocab_size": 32000}
    trained = {**cfg, "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 15, 28]}}
    assert check_head(LLAMA31_8B, trained, "vllm").reasons == ["layer_ids_unread_by_engine"]
    assert check_head(LLAMA31_8B, trained, "sglang").status == "ok"
    # Trained on the defaults (2, n // 2, n - 3): falling back to them changes nothing.
    defaults = {**cfg, "eagle_config": {"eagle_aux_hidden_state_layer_ids": [2, 16, 29]}}
    assert check_head(LLAMA31_8B, defaults, "vllm").status == "ok"
    # V2 does read eagle_config's "layer_ids".
    named = {**cfg, "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 15, 28], "layer_ids": [1, 15, 28]}}
    assert check_head(LLAMA31_8B, named, "vllm").status == "ok"


def test_sglang_reads_an_eagle3_head_s_layers_from_eagle_config_only():
    cfg = {"architectures": ["LlamaForCausalLMEagle3"], "num_hidden_layers": 1, "vocab_size": 128256,
           "hidden_size": 4096, "draft_vocab_size": 32000, "eagle_aux_hidden_state_layer_ids": [1, 15, 28]}
    assert check_head(LLAMA31_8B, cfg, "vllm").status == "ok"
    assert check_head(LLAMA31_8B, cfg, "sglang").reasons == ["layer_ids_unread_by_engine"]


def test_muse_glimmer_s_assistant_is_a_dflash_drafter_whose_vocabulary_is_the_target_s():
    cfg, files = _head("meta-models/Muse-Glimmer-30B-assistant")
    assert head_kind(cfg, "vllm") == head_kind(cfg, "sglang") == "dflash"
    verdict = check_head(MUSE_GLIMMER_30B, cfg, "vllm", files=files)
    assert verdict.status == "unknown" and verdict.reasons == ["vocab_unread"]  # it declares none: borrows the target's


def test_vllm_renames_into_dflash_nothing_a_full_model_could_declare():
    full = {"architectures": ["LagunaForCausalLM"], "num_hidden_layers": 1, "vocab_size": 151936,
            "hidden_size": 4096, "draft_vocab_size": 32000}
    assert head_kind(full, "vllm") is None


def test_vllm_finds_a_listed_head_too():
    from rigtune import speculative
    from rigtune.models import ModelSpec

    cfg, _ = _head("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B")
    found = speculative.find_eagle3_head(ModelSpec(hf_id="unsloth/Meta-Llama-3.1-8B-Instruct"), LLAMA31_8B,
                                         engine="vllm", fetch_config=lambda r: cfg, sizes=lambda r: 850_000_000)
    assert found == ("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B", 850_000_000)


def test_the_order_a_head_lists_its_layers_in_does_not_matter():
    cfg = {"architectures": ["LlamaForCausalLMEagle3"], "num_hidden_layers": 1, "vocab_size": 128256,
           "hidden_size": 4096, "draft_vocab_size": 32000,
           "eagle_config": {"eagle_aux_hidden_state_layer_ids": [29, 16, 2]}}  # the defaults, out of order
    assert check_head(LLAMA31_8B, cfg, "vllm").status == "ok"
