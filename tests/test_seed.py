"""A seed command (a user's own, or a model's published recipe) read into rigtune's settings (rigtune.seed)."""

from __future__ import annotations

import pytest

from rigtune.models import Config
from rigtune.seed import parse_command

# vLLM's recipe for Inkling-2-Mini on an H100 (recipes/models/thinkingmachines/Inkling-2-Mini.yaml, read 23 Nov 2026),
# as the recipe prints it, with its line continuations.
INKLING_RECIPE = """vllm serve thinkingmachines/Inkling-2-Mini \\
  --tensor-parallel-size 1 \\
  --reasoning-parser inkling \\
  --max-model-len 262144 \\
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'"""


def test_the_inkling_recipe_is_read_into_the_settings_rigtune_tunes():
    seed = parse_command(INKLING_RECIPE)
    assert (seed.engine, seed.model) == ("vllm", "thinkingmachines/Inkling-2-Mini")
    assert seed.fields == {"tp": 1, "ctx": 262144, "spec_decode": "dspark:7"}  # the checkpoint's own drafter
    assert seed.extra == {"reasoning_parser": "inkling"}  # changes how replies are split, not speed: passed through
    assert any("draft_sample_method" in n for n in seed.notes)  # rigtune sets the drafter's sampling itself
    # No quant: the checkpoint's own (NVFP4) is the model's to say. The rest builds a Config.
    Config(backend=seed.engine, quant="ckpt-nvfp4", **seed.fields, extra=seed.extra)


@pytest.mark.parametrize("command", [
    "vllm serve m --max-num-seqs=64 --gpu-memory-utilization=0.9",
    "vllm serve m --max-num-seqs 64 --gpu-memory-utilization 0.9",
    "python -m vllm.entrypoints.openai.api_server --model m --max-num-seqs 64 --gpu-memory-utilization 0.9",
])
def test_flags_are_read_with_or_without_an_equals_sign_and_either_entry_point(command):
    seed = parse_command(command)
    assert (seed.model, seed.fields) == ("m", {"batch": 64, "gpu_memory_utilization": 0.9})


def test_vllm_s_dotted_form_reads_as_the_json_it_stands_for():
    dotted = parse_command("vllm serve m --speculative-config.method mtp --speculative-config.num_speculative_tokens 3")
    assert dotted.fields["spec_decode"] == "mtp:3"


@pytest.mark.parametrize("conf, spec", [
    ('{"method":"ngram","num_speculative_tokens":4}', "ngram:4"),
    ('{"method":"ngram_gpu","num_speculative_tokens":5}', "ngram_gpu:5"),
    ('{"method":"suffix","num_speculative_tokens":24}', "suffix:24"),
    ('{"method":"mtp","num_speculative_tokens":2}', "mtp:2"),
    ('{"method":"eagle3","model":"AngelSlim/Qwen3-8B_eagle3","num_speculative_tokens":6}',
     "eagle3:AngelSlim/Qwen3-8B_eagle3:6"),
    ('{"method":"dspark","model":"deepseek-ai/dspark_qwen3_8b_block7","num_speculative_tokens":7}',
     "dspark:deepseek-ai/dspark_qwen3_8b_block7:7"),
    ('{"method":"draft_model","model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}', "draft:Qwen/Qwen3-0.6B:5"),
])
def test_each_drafter_vllm_names_becomes_rigtune_s_spec(conf, spec):
    assert parse_command(f"vllm serve m --speculative-config '{conf}'").fields["spec_decode"] == spec


def test_settings_rigtune_launches_its_own_way_are_noted_not_silently_dropped():
    seed = parse_command("vllm serve m --host 0.0.0.0 --port 9000 --api-key s3cr3t-key --enforce-eager "
                         "--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,"
                         "\"prompt_lookup_min\":3,\"prompt_lookup_max\":5}'")
    assert seed.extra == {"enforce_eager": True}
    assert "s3cr3t" not in " ".join(seed.notes)  # the key is never echoed
    for word in ("--host", "--port", "--api-key", "lookup (3-5)"):
        assert any(word in n for n in seed.notes), word


@pytest.mark.parametrize("command, why", [
    ("llama-server -m m.gguf", "not a vLLM or SGLang"),
    ("vllm serve", "names no model"),
    ("vllm serve m --max-num-seqs lots", "whole number"),
    ("vllm serve m --speculative-config '{\"method\":\"mtp\"}'", "num_speculative_tokens"),
    ("vllm serve m --speculative-config '{\"method\":\"dflash\",\"num_speculative_tokens\":7}'", "names its model"),
    ("vllm serve m --speculative-config not-json", "not JSON"),
    ("vllm serve m stray", "unexpected argument"),
])
def test_a_command_rigtune_cannot_read_says_why(command, why):
    with pytest.raises(ValueError, match=why):
        parse_command(command)


def test_vllm_s_short_flags_and_dtype():
    seed = parse_command("vllm serve m -tp 2 --dtype bfloat16 --kv-cache-dtype fp8 --no-enable-prefix-caching")
    assert seed.fields == {"tp": 2, "quant": "bf16", "kv_dtype": "fp8", "prefix_cache": False}
    # --quantization wins over --dtype, whichever comes first.
    assert parse_command("vllm serve m --quantization fp8 --dtype bfloat16").fields["quant"] == "fp8"


def test_sglang_s_command_is_read_with_its_own_names():
    seed = parse_command("python -m sglang.launch_server --model-path Qwen/Qwen3-8B --context-length 32768 "
                         "--max-running-requests 128 --mem-fraction-static 0.88 --chunked-prefill-size 4096 "
                         "--tp-size 1 --disable-radix-cache --speculative-algorithm EAGLE3 "
                         "--speculative-draft-model-path AngelSlim/Qwen3-8B_eagle3 --speculative-num-steps 3 "
                         "--speculative-eagle-topk 4 --speculative-num-draft-tokens 8 --enable-metrics")
    assert (seed.engine, seed.model) == ("sglang", "Qwen/Qwen3-8B")
    assert seed.fields == {"ctx": 32768, "batch": 128, "gpu_memory_utilization": 0.88, "prefill_budget": 4096,
                           "tp": 1, "prefix_cache": False, "spec_decode": "eagle3:AngelSlim/Qwen3-8B_eagle3:3"}
    assert seed.extra == {"enable_metrics": True}
    assert any("top-k 4" in n for n in seed.notes)  # rigtune drafts a chain


@pytest.mark.parametrize("flags, spec", [
    ("--speculative-algorithm NGRAM --speculative-num-draft-tokens 12", "ngram:12"),
    ("--speculative-algorithm NEXTN --speculative-num-steps 3", "mtp:3"),
    ("--speculative-algorithm EAGLE --speculative-draft-model-path Qwen/Qwen3.5-9B --speculative-num-steps 2", "mtp:2"),
    ("--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash "
     "--speculative-num-draft-tokens 16", "dflash:z-lab/Qwen3-8B-DFlash:15"),
    ("--speculative-algorithm DSPARK --speculative-draft-model-path deepseek-ai/dspark_qwen3_8b_block7 "
     "--speculative-dspark-block-size 7", "dspark:deepseek-ai/dspark_qwen3_8b_block7:7"),
    ("--speculative-algorithm STANDALONE --speculative-draft-model-path Qwen/Qwen3-0.6B --speculative-num-steps 4",
     "draft:Qwen/Qwen3-0.6B:4"),
])
def test_each_sglang_drafter_becomes_rigtune_s_spec(flags, spec):
    seed = parse_command(f"python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B {flags}")
    assert seed.fields["spec_decode"] == spec


# ---------------------------------------------------------------- what the review of S1 found


def test_underscored_names_are_the_same_flags_and_cannot_override_what_rigtune_reads():
    # vLLM's parser takes --max_model_len for --max-model-len; passed through, it would override rigtune's own.
    seed = parse_command("vllm serve m --max_model_len 8192 --speculative_config '{\"method\":\"mtp\","
                         "\"num_speculative_tokens\":2}'")
    assert seed.fields == {"ctx": 8192, "spec_decode": "mtp:2"} and seed.extra == {}


def test_a_flag_missing_its_value_or_giving_one_to_a_bare_flag_is_refused():
    for command, why in [("vllm serve m -tp --max-num-seqs 8", "--tensor-parallel-size needs a value"),
                         ("vllm serve m --gpu-memory-utilization --enforce-eager", "needs a value"),
                         ("vllm serve m --enable-prefix-caching=false", "takes no value"),
                         ("vllm serve m --gpu-memory-utilization 1.5", "at most 1"),
                         ("vllm serve m --max-num-seqs 0", "1 or more")]:
        with pytest.raises(ValueError, match=why):
            parse_command(command)
    # The engine's automatic length is noted, not read as -1.
    auto = parse_command("vllm serve m --max-model-len -1")
    assert "ctx" not in auto.fields and any("automatic length" in n for n in auto.notes)


def test_other_dotted_flags_pass_through_as_json_nested_and_merged():
    import json

    seed = parse_command("vllm serve m --override-generation-config.temperature 0.6 "
                         "--hf-overrides.text_config.rope_theta 1000000")
    assert json.loads(seed.extra["override_generation_config"]) == {"temperature": 0.6}
    assert json.loads(seed.extra["hf_overrides"]) == {"text_config": {"rope_theta": 1000000}}
    # A JSON value and dotted keys for the same flag merge, the dotted keys winning (as vLLM does).
    merged = parse_command("vllm serve m --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":2}' "
                           "--speculative-config.num_speculative_tokens 6")
    assert merged.fields["spec_decode"] == "ngram:6"


def test_a_method_vllm_would_guess_from_the_drafter_s_name_must_be_named():
    with pytest.raises(ValueError, match="names no method"):
        parse_command("vllm serve m --speculative-config '{\"model\":\"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B\","
                      "\"num_speculative_tokens\":4}'")
    with pytest.raises(ValueError, match="whole number"):
        parse_command("vllm serve m --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":true}'")


def test_n_gram_lookup_left_to_vllm_is_noted_since_rigtune_launches_its_own():
    seed = parse_command("vllm serve m --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4}'")
    assert any("vLLM's own default" in n for n in seed.notes)
    kept = parse_command("vllm serve m --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,"
                         "\"prompt_lookup_min\":2,\"prompt_lookup_max\":4}'")
    assert not kept.notes


def test_settings_the_plan_cannot_see_are_refused_and_a_stage_s_own_are_not_kept():
    for flag in ("-pp 2", "--data-parallel-size 2", "--cpu-offload-gb 8", "--kv-cache-memory-bytes 1000"):
        with pytest.raises(ValueError, match="does not model"):
            parse_command(f"vllm serve m {flag}")
    with pytest.raises(ValueError, match="does not model"):
        parse_command("python -m sglang.launch_server --model-path m --max-total-tokens 100000")
    seed = parse_command("vllm serve m --compilation-config.mode 3 --attention-backend FLASHINFER "
                         "--language-model-only --revision abc123")
    assert seed.extra == {} and seed.fields == {"text_only": True} and seed.revision == "abc123"
    sg = parse_command("python -m sglang.launch_server --model-path m --enable-torch-compile")
    assert sg.extra == {} and any("kernel stage" in n for n in sg.notes)


def test_secrets_are_never_kept_or_echoed():
    command = ("VLLM_API_KEY=sk-SECRET HF_TOKEN=hf_SECRET vllm serve m --hf-token hf_SECRET2 "
               "--api-key=sk-SECRET3 --max-num-seqs 8")
    seed = parse_command(command)
    text = repr(seed)
    assert "SECRET" not in text and seed.fields == {"batch": 8} and seed.extra == {}
    assert any("VLLM_API_KEY, HF_TOKEN" in n for n in seed.notes)  # names only
    for bad in ("VLLM_API_KEY=sk-SECRET llama-server -m m", "vllm serve m sk-SECRET-positional"):
        with pytest.raises(ValueError) as err:
            parse_command(bad)
        assert "SECRET" not in str(err.value)


def test_quantization_is_read_only_where_rigtune_launches_it():
    assert parse_command("vllm serve m --quantization fp8 --dtype bfloat16").fields == {"quant": "fp8"}
    awq = parse_command("vllm serve m --quantization awq")
    assert "quant" not in awq.fields and any("checkpoint's own" in n for n in awq.notes)
    for bad in ("--quantization bitsandbytes", "--dtype float32"):
        with pytest.raises(ValueError, match="not one rigtune launches"):
            parse_command(f"vllm serve m {bad}")


def test_sglang_counts_it_would_choose_itself_must_be_given():
    with pytest.raises(ValueError, match="speculative-num-draft-tokens"):
        parse_command("python -m sglang.launch_server --model-path m --speculative-algorithm DFLASH "
                      "--speculative-draft-model-path d")
    with pytest.raises(ValueError, match="speculative-num-steps"):
        parse_command("python -m sglang.launch_server --model-path m --speculative-algorithm EAGLE3 "
                      "--speculative-draft-model-path h")
    # Drafter flags with no algorithm, or with one that does not read them, are noted.
    loose = parse_command("sglang serve --model-path m --speculative-num-steps 3")
    assert "spec_decode" not in loose.fields and any("--speculative-num-steps" in n for n in loose.notes)


def test_mtp_with_a_second_model_is_an_assistant():
    seed = parse_command("vllm serve google/gemma-4-12B-it --speculative-config "
                         "'{\"method\":\"mtp\",\"model\":\"google/gemma-4-12B-it-assistant\",\"num_speculative_tokens\":1}'")
    assert seed.fields["spec_decode"] == "assist:google/gemma-4-12B-it-assistant:1"
