"""What a checkpoint declares about itself (rigtune/introspect.py), read through injected fetchers: no test here
touches the network. Configs are the fixtures in tests/fixtures; quantization configs are copied from the
repositories named beside them; the MTP, embedding and output-head tensors are those the safetensors headers of
Qwen/Qwen3.5-9B (revision c202236) and Qwen/Qwen3.8-27B (37561816) list, read on 14 November 2026."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from rigtune import introspect
from rigtune.models import ModelSpec

FIXTURES = Path(__file__).parent / "fixtures"
QWEN35 = json.loads((FIXTURES / "qwen3_5_9b_config.json").read_text(encoding="utf-8"))
QWEN36_MOE = json.loads((FIXTURES / "qwen3_6_35b_a3b_config.json").read_text(encoding="utf-8"))
QWEN25 = json.loads((FIXTURES / "qwen2_5_7b_instruct_config.json").read_text(encoding="utf-8"))
GEMMA4 = json.loads((FIXTURES / "gemma4_12b_it_config.json").read_text(encoding="utf-8"))

def _mtp_layer(hidden: int, mlp: int, q: int, o_in: int, fc_in: int) -> dict:
    """Shapes of a Qwen3.5 MTP head's 15 tensors (all BF16), as both checkpoints list them."""
    return {"mtp.fc.weight": [hidden, fc_in], "mtp.norm.weight": [hidden],
            "mtp.pre_fc_norm_embedding.weight": [hidden], "mtp.pre_fc_norm_hidden.weight": [hidden],
            "mtp.layers.0.input_layernorm.weight": [hidden], "mtp.layers.0.post_attention_layernorm.weight": [hidden],
            "mtp.layers.0.mlp.down_proj.weight": [hidden, mlp], "mtp.layers.0.mlp.gate_proj.weight": [mlp, hidden],
            "mtp.layers.0.mlp.up_proj.weight": [mlp, hidden], "mtp.layers.0.self_attn.q_proj.weight": [q, hidden],
            "mtp.layers.0.self_attn.k_proj.weight": [1024, hidden], "mtp.layers.0.self_attn.v_proj.weight": [1024, hidden],
            "mtp.layers.0.self_attn.o_proj.weight": [hidden, o_in], "mtp.layers.0.self_attn.q_norm.weight": [256],
            "mtp.layers.0.self_attn.k_norm.weight": [256]}


def _bf16(shapes: dict) -> dict:
    out = {}
    for name, shape in shapes.items():
        n = 2
        for d in shape:
            n *= d
        out[name] = n
    return out


QWEN35_HEAD = _bf16(_mtp_layer(hidden=4096, mlp=12288, q=8192, o_in=4096, fc_in=8192))
QWEN35_HEAD_BYTES = 486_581_248  # the 15 tensors' data offsets, summed
QWEN35_SHARED_BYTES = 2 * 2_034_237_440  # embed_tokens and lm_head, each 248,320 x 4,096 x 2 B (not tied)
QWEN35_TENSORS = {
    "model.language_model.embed_tokens.weight": 248_320 * 4096 * 2,
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": 4096 * 8192 * 2,  # made up, as below
    "model.visual.blocks.0.attn.qkv.weight": 1152 * 3456 * 2,
    "lm_head.weight": 248_320 * 4096 * 2,
    **QWEN35_HEAD,
}
SPEC = ModelSpec(hf_id="Qwen/Qwen3.5-9B")
QWEN38 = json.loads((FIXTURES / "qwen3_8_27b_config.json").read_text(encoding="utf-8"))
QWEN38_TENSORS = {"model.language_model.embed_tokens.weight": 2_542_796_800, "lm_head.weight": 2_542_796_800,
                  **_bf16(_mtp_layer(hidden=5120, mlp=17408, q=12288, o_in=6144, fc_in=10240))}

# Lines copied from vLLM v0.33.0 rust/src/chat/tests/templates/qwen3.jinja (84-87) and qwen35.jinja (147-153).
QWEN3_TEMPLATE = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}"""
QWEN35_TEMPLATE = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\\n' }}
    {%- else %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""
# Lines 147-154 of Qwen/Qwen3.5-9B's own chat_template.jinja at revision c202236235762e1c871ad0ccb60c8ee5ba337b9a,
# the checkpoint the A40 run of November 2026 served: unlike vLLM's qwen35.jinja above, it thinks by default.
QWEN35_9B_TEMPLATE = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}
{%- endif %}"""

# quantization_config, verbatim, of hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 and
# RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8.
AWQ = {"bits": 4, "group_size": 128, "modules_to_not_convert": None, "quant_method": "awq", "version": "gemm",
       "zero_point": True}
W8A8 = {"config_groups": {"group_0": {
    "input_activations": {"block_structure": None, "dynamic": True, "group_size": None, "num_bits": 8,
                          "observer": "memoryless", "observer_kwargs": {}, "strategy": "token", "symmetric": True,
                          "type": "int"},
    "output_activations": None, "targets": ["Linear"],
    "weights": {"block_structure": None, "dynamic": False, "group_size": None, "num_bits": 8, "observer": "minmax",
                "observer_kwargs": {}, "strategy": "channel", "symmetric": True, "type": "int"}}},
    "format": "int-quantized", "global_compression_ratio": 1.239290831149584, "ignore": ["lm_head"],
    "kv_cache_scheme": None, "quant_method": "compressed-tensors", "quantization_status": "compressed",
    "sparsity_config": {}}


def fetchers(config=None, texts=None, files=(), tensors=None, calls=None):
    """Fetchers over fixed data; each records its calls in `calls`, and a value that is an exception is raised."""
    calls = calls if calls is not None else []

    def give(what, value):
        calls.append(what)
        if isinstance(value, Exception):
            raise value
        return value

    return dict(fetch_config=lambda s: give("config", config),
                fetch_text=lambda s, f: give(f, (texts or {}).get(f)),
                list_files=lambda s: give("files", files if isinstance(files, Exception) else list(files)),
                list_tensors=lambda s: give("tensors", tensors))


# ---------------------------------------------------------------- MTP heads


def test_a_qwen3_5_checkpoint_with_its_head():
    calls = []
    facts = introspect.inspect_model(SPEC, features=("mtp",), **fetchers(QWEN35, tensors=QWEN35_TENSORS, calls=calls))
    assert sum(QWEN35_HEAD.values()) == QWEN35_HEAD_BYTES
    assert facts.mtp == introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15,
                                           bytes=QWEN35_HEAD_BYTES, shared_bytes=QWEN35_SHARED_BYTES)
    assert facts.notes == [] and calls == ["config", "tensors"]  # nothing else is fetched for the head
    assert facts.describe()["mtp"] == "1 layer (mtp_num_hidden_layers), 15 tensors, 0.49 GB"


def test_qwen3_8_27b_declares_and_ships_its_head():
    head, notes = introspect.mtp_head(QWEN38, QWEN38_TENSORS, "Qwen/Qwen3.8-27B")
    # 15 tensors summing to 849,398,784 B; embed_tokens and lm_head 248,320 x 5,120 x 2 B each.
    assert head == introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=15, bytes=849_398_784,
                                      shared_bytes=5_085_593_600) and notes == []


def test_tied_embeddings_count_once():
    tied = {k: v for k, v in QWEN35_TENSORS.items() if k != "lm_head.weight"}
    assert introspect.mtp_head(QWEN35, tied)[0].shared_bytes == 2_034_237_440


def test_the_moe_sibling_declares_its_head_the_same_way():
    assert introspect.mtp_layers(QWEN36_MOE) == (1, "mtp_num_hidden_layers")


def test_a_dense_model_has_no_head_and_its_weights_are_never_listed():
    calls = []
    facts = introspect.inspect_model(ModelSpec(hf_id="Qwen/Qwen2.5-7B-Instruct"), features=("mtp",),
                                     **fetchers(QWEN25, tensors=pytest.fail, calls=calls))
    assert facts.mtp is None and facts.notes == [] and calls == ["config"]
    assert facts.describe()["mtp"] == "none"


def test_a_declared_head_without_its_tensors_is_a_note_not_a_head():
    no_head = {k: v for k, v in QWEN35_TENSORS.items() if not k.startswith("mtp.")}
    facts = introspect.inspect_model(SPEC, features=("mtp",), **fetchers(QWEN35, tensors=no_head))
    assert facts.mtp is None
    assert facts.notes == ["Qwen/Qwen3.5-9B declares 1 MTP layer(s) (mtp_num_hidden_layers) but its weights hold no "
                           "MTP tensors (none named mtp or nextn, and no decoder layer past the last); no MTP head is "
                           "reported"]


def test_a_head_shipped_without_a_declared_count_is_counted_from_its_tensors():
    """Qwen3-Next, LongCat-Flash and MiMo-V2-Flash ship mtp.* tensors and declare no count."""
    head, notes = introspect.mtp_head(QWEN25, QWEN35_TENSORS, "x")
    assert (head.layers, head.config_key, head.tensors) == (1, "the tensors", 15)
    assert notes == ["x declares no MTP layers but ships 15 MTP-named tensors; counted 1 layer(s) from them"]
    mimo_v2 = {f"model.mtp.layers.{i}.mlp.weight": 1 for i in range(3)}  # MiMo-V2-Flash: model.mtp.layers.0-2
    assert introspect.mtp_head({"num_hidden_layers": 48}, mimo_v2)[0].layers == 3


# Other families' layouts, as their configs and model.safetensors.index.json name them on the Hub (read on
# 20 November 2026); sizes made up.
def _decoder(prefix: str, layers: int, *, norm: bool = True) -> dict:
    """A decoder's layers, with the final norm that sits beside them (prefix + "norm.weight") unless `norm` is off."""
    out = {f"{prefix}layers.{i}.self_attn.q_proj.weight": 1000 for i in range(layers)}
    return {**out, f"{prefix}norm.weight": 1} if norm else out


INKLING = {"architectures": ["InklingForConditionalGeneration"], "model_type": "inkling_mm_model",
           "text_config": {"num_hidden_layers": 42},
           "mtp_config": {"num_nextn_predict_layers": 8, "chain_hidden_post_norm": False}}
INKLING_TENSORS = {**_decoder("model.llm.", 42), "model.llm.embed.weight": 50, "model.llm.unembed.weight": 50,
                   **{f"model.mtp.layers.{i}.mlp.down_proj.weight": 7 for i in range(8)}}
DEEPSEEK = {"architectures": ["DeepseekV3ForCausalLM"], "num_hidden_layers": 61, "num_nextn_predict_layers": 1}
DEEPSEEK_TENSORS = {**_decoder("model.", 61), "model.embed_tokens.weight": 50, "lm_head.weight": 50,
                    "model.layers.61.self_attn.q_a_proj.weight": 7, "model.layers.61.eh_proj.weight": 7}
GLM_AIR = {"architectures": ["Glm4MoeForCausalLM"], "num_hidden_layers": 46, "num_nextn_predict_layers": 1}
GLM_AIR_TENSORS = {**_decoder("model.", 46), "model.embed_tokens.weight": 50, "lm_head.weight": 50,
                   "model.layers.46.embed_tokens.weight": 9, "model.layers.46.shared_head.head.weight": 9}
MIMO = {"architectures": ["MiMoForCausalLM"], "num_hidden_layers": 36, "num_nextn_predict_layers": 1}
MIMO_TENSORS = {**_decoder("model.", 36), "model.mtp_layers.0.input_layernorm.weight": 3,
                "model.mtp_layers.0.eh_proj.weight": 4}
STEP = {"architectures": ["Step3p5ForCausalLM"], "num_hidden_layers": 45, "num_nextn_predict_layers": 3}
STEP_TENSORS = {**_decoder("model.", 45), "model.embed_tokens.weight": 50,
                **{f"model.layers.{i}.transformer.shared_head.output.weight": 9 for i in (45, 46, 47)}}


@pytest.mark.parametrize("config, tensors, layers, key, count, size", [
    (INKLING, INKLING_TENSORS, 8, "mtp_config.num_nextn_predict_layers", 8, 56),  # a sub-config, model.mtp.*
    (DEEPSEEK, DEEPSEEK_TENSORS, 1, "num_nextn_predict_layers", 2, 14),  # a decoder layer past the 61st
    (GLM_AIR, GLM_AIR_TENSORS, 1, "num_nextn_predict_layers", 2, 18),
    (MIMO, MIMO_TENSORS, 1, "num_nextn_predict_layers", 2, 7),  # model.mtp_layers.*
    (STEP, STEP_TENSORS, 3, "num_nextn_predict_layers", 3, 27),  # three layers past the last
])
def test_every_family_s_layout_is_found(config, tensors, layers, key, count, size):
    head, notes = introspect.mtp_head(config, tensors, "x")
    assert (head.layers, head.config_key, head.tensors, head.bytes) == (layers, key, count, size) and notes == []


GLM_46 = {"architectures": ["Glm4MoeForCausalLM"], "num_hidden_layers": 92, "num_nextn_predict_layers": 1}
GLM_46_TENSORS = {**_decoder("model.", 92), "model.embed_tokens.weight": 50, "lm_head.weight": 50,
                  "model.layers.92.eh_proj.weight": 9, "model.layers.92.shared_head.norm.weight": 1}


@pytest.mark.parametrize("config, tensors, shared", [
    (GLM_AIR, GLM_AIR_TENSORS, 0),  # it brings its own embed_tokens and shared_head.head: none of the target's
    (GLM_46, GLM_46_TENSORS, 100),  # only a shared_head.norm: both of the target's are copied
    (DEEPSEEK, {**DEEPSEEK_TENSORS, "model.layers.61.embed_tokens.weight": 9}, 50),  # its own embedding only
    (STEP, STEP_TENSORS, 50),  # its own output head (shared_head.output) only: the target's embedding is copied
    ({**MIMO, "num_hidden_layers": 2}, {**_decoder("model.", 2), "model.embed_tokens.weight": 50, "lm_head.weight": 50,
                                        "model.mtp.embed_tokens.weight": 9, "model.mtp.layers.0.w": 3}, 50),  # LongCat
    (INKLING, INKLING_TENSORS, 100),  # model.llm.embed and model.llm.unembed
    (QWEN35, QWEN35_TENSORS, QWEN35_SHARED_BYTES),
    ({"num_hidden_layers": 2, "num_nextn_predict_layers": 1},  # Nemotron-H's backbone.embeddings
     {"backbone.embeddings.weight": 40, "lm_head.weight": 40, "mtp.layers.0.mixer.weight": 3}, 80),
])
def test_the_target_s_embedding_and_output_head_as_each_family_names_them(config, tensors, shared):
    assert introspect.mtp_head(config, tensors)[0].shared_bytes == shared


def test_a_head_that_brings_its_own_embedding_is_sized_by_its_own_bytes():
    from rigtune import speculative
    from rigtune.hfconfig import arch_from_config
    from rigtune.models import PreparedModel

    arch = arch_from_config({k: v for k, v in GLM_AIR.items() if k != "num_nextn_predict_layers"})  # planner: 0
    pm = PreparedModel(spec=ModelSpec(hf_id="zai-org/GLM-4.5-Air"), backend="sglang", arch=arch,
                       hf_path="zai-org/GLM-4.5-Air")
    head = introspect.mtp_head(GLM_AIR, GLM_AIR_TENSORS)[0]
    notes = speculative.record_mtp_head(pm, "sglang", inspect=lambda spec: introspect.ModelFacts(mtp=head))
    assert notes == [] and speculative.draft_bytes("mtp:1", pm) == 18  # its 18 bytes, nothing added
    assert pm.arch.mtp_layers == 1  # the planner gets the head's count
    glm46 = introspect.mtp_head(GLM_46, GLM_46_TENSORS)[0]
    speculative.record_mtp_head(pm, "sglang", inspect=lambda spec: introspect.ModelFacts(mtp=glm46))
    assert speculative.draft_bytes("mtp:1", pm) == 10 + 100  # its own 10, and both of the target's copied


@pytest.mark.parametrize("tower_layers", [29, 32])  # one past the text model's 28, and well past it
def test_a_vision_tower_s_layers_are_never_taken_for_a_head(tower_layers):
    config = {"text_config": {"num_hidden_layers": 28}, "num_nextn_predict_layers": 1}
    tensors = {**_decoder("model.language_model.", 28), **_decoder("model.visual.encoder.", tower_layers)}
    head, notes = introspect.mtp_head(config, tensors, "x")
    assert head is None and "hold no MTP tensors" in notes[0]


def test_a_layer_stack_with_no_embedding_or_norm_beside_it_is_not_the_language_model():
    config = {"num_hidden_layers": 28, "num_nextn_predict_layers": 1}
    tensors = {**_decoder("model.", 28), **_decoder("encoder.", 29, norm=False)}
    assert introspect.mtp_head(config, tensors)[0] is None


def test_a_named_head_ends_the_search():
    # Inkling's 8 named MTP layers beside a 47-layer audio encoder whose layers 42-46 run past the text model's.
    tensors = {**INKLING_TENSORS, **_decoder("model.audio.encoder.", 47)}
    head, _ = introspect.mtp_head(INKLING, tensors)
    assert head.tensors == 8


@pytest.mark.parametrize("extra", [
    {"model.layers.62.mlp.weight": 1},  # past the declared head: not a head layer
    {"model.layers.61.mlp.weight": 1, "model.layers.63.mlp.weight": 1},
])
def test_layers_past_the_declared_count_are_not_a_head(extra):
    head, _ = introspect.mtp_head(DEEPSEEK, {**_decoder("model.", 61), **extra})
    assert head is None


def test_a_decoder_with_a_gap_is_not_read_as_one():
    tensors = {k: v for k, v in DEEPSEEK_TENSORS.items() if not k.startswith("model.layers.5.")}
    assert introspect.mtp_head(DEEPSEEK, tensors)[0] is None


def test_the_planner_counts_a_head_declared_in_a_sub_config():
    from rigtune.hfconfig import arch_from_config

    assert arch_from_config({**INKLING, "text_config": {**INKLING["text_config"], "hidden_size": 4096}}).mtp_layers == 8


def test_unreadable_weights_or_config_are_notes_not_exceptions():
    facts = introspect.inspect_model(SPEC, features=("mtp",), **fetchers(QWEN35, tensors=OSError("offline")))
    assert facts.mtp is None
    assert facts.notes == ["Qwen/Qwen3.5-9B: could not read the safetensors headers (offline)",
                           "Qwen/Qwen3.5-9B declares 1 MTP layer(s) (mtp_num_hidden_layers) but its weights could not "
                           "be listed; no MTP head is reported"]
    facts = introspect.inspect_model(SPEC, **fetchers(RuntimeError("gated"), tensors=QWEN35_TENSORS))
    assert facts.describe() == {"mtp": "none", "quantization": "none", "thinking": "none", "modalities": "text"}
    assert facts.notes[0] == "Qwen/Qwen3.5-9B: could not read config.json (gated)"


def test_a_listing_without_sizes_still_finds_the_head():
    sizes = dict(QWEN35_TENSORS, **{"mtp.norm.weight": None})
    head, _ = introspect.mtp_head(QWEN35, sizes)
    assert (head.tensors, head.bytes) == (15, None) and head.describe() == "1 layer (mtp_num_hidden_layers), 15 tensors"


@pytest.mark.parametrize("config, expected", [
    ({"num_nextn_predict_layers": 1}, (1, "num_nextn_predict_layers")),  # the DeepSeek-style key, top level
    ({"text_config": {"mtp_num_hidden_layers": 2}}, (2, "mtp_num_hidden_layers")),
    ({"mtp_config": {"num_nextn_predict_layers": 8}}, (8, "mtp_config.num_nextn_predict_layers")),  # Inkling
    ({"mtp_config": {"num_nextn_predict_layers": 0}, "num_nextn_predict_layers": 1}, (1, "num_nextn_predict_layers")),
    ({"vision_config": {"num_nextn_predict_layers": 3}}, None),  # only a sub-config named for MTP
    ({"mtp_num_hidden_layers": 0}, None),
    ({"mtp_num_hidden_layers": True}, None),
    ({"mtp_num_hidden_layers": "1"}, None),
    ({}, None),
    (None, None),
])
def test_what_counts_as_a_declared_head(config, expected):
    assert introspect.mtp_layers(config) == expected


def _write_safetensors(path: Path, tensors: dict) -> None:
    """A safetensors file: an 8-byte little-endian header length, the JSON header, then the data."""
    header, offset = {}, 0
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def test_a_local_checkpoint_is_read_from_its_files(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(QWEN35), encoding="utf-8")
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {
        "model.language_model.norm.weight": ("BF16", [4096], 8192)})
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", {
        "mtp.fc.weight": ("BF16", [64, 128], 16_384), "mtp.norm.weight": ("BF16", [64], 128)})
    (tmp_path / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "chat_template.jinja").write_text(QWEN35_TEMPLATE, encoding="utf-8")
    facts = introspect.inspect_model(ModelSpec(hf_id=str(tmp_path)))
    assert facts.mtp == introspect.MtpHead(layers=1, config_key="mtp_num_hidden_layers", tensors=2, bytes=16_512,
                                           shared_bytes=None)  # this listing has no embedding or output head
    assert facts.thinking == introspect.ThinkingSwitch("enable_thinking", "off")
    assert facts.modalities == ("text", "video") and facts.quantization is None and facts.notes == []


def test_the_hub_listing_is_read_from_safetensors_headers(monkeypatch):
    """The default Hub fetchers, against a fake HfApi shaped like huggingface_hub's (SafetensorsRepoMetadata
    .files_metadata -> SafetensorsFileMetadata.tensors -> TensorInfo with dtype, shape and data_offsets)."""
    import types

    import huggingface_hub

    seen = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def get_safetensors_metadata(self, repo_id, revision=None):
            seen.append((repo_id, revision))
            info = types.SimpleNamespace
            return info(files_metadata={
                "model-00001-of-00002.safetensors": info(tensors={
                    "lm_head.weight": info(dtype="BF16", shape=[10, 4], data_offsets=(0, 80))}),
                "model-00002-of-00002.safetensors": info(tensors={
                    "mtp.fc.weight": info(dtype="BF16", shape=[8, 4], data_offsets=(80, 144))})})

        def list_repo_files(self, repo_id, revision=None):
            return ["config.json", "preprocessor_config.json", "video_preprocessor_config.json"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    spec = ModelSpec(hf_id="Qwen/Qwen3.5-9B", revision="abc123")
    assert introspect._default_tensors(spec) == {"lm_head.weight": 80, "mtp.fc.weight": 64}
    assert seen == [("Qwen/Qwen3.5-9B", "abc123")]
    assert introspect._default_files(spec)[-1] == "video_preprocessor_config.json"


def test_offline_the_cached_snapshot_is_read(monkeypatch, tmp_path):
    """The Hub cannot be asked, but the snapshot is in the local Hugging Face cache: its headers answer, and only
    when every file the index lists is there."""
    import huggingface_hub

    class Offline:
        def __init__(self, token=None):
            pass

        def get_safetensors_metadata(self, repo_id, revision=None):
            raise ConnectionError("offline")

    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "lm_head.weight": "model-00001-of-00002.safetensors",
        "mtp.fc.weight": "model-00002-of-00002.safetensors"}}), encoding="utf-8")
    _write_safetensors(snap / "model-00001-of-00002.safetensors", {"lm_head.weight": ("BF16", [10, 4], 80)})
    _write_safetensors(snap / "model-00002-of-00002.safetensors", {"mtp.fc.weight": ("BF16", [8, 4], 64)})
    asked = []

    def from_cache(repo_id, filename, revision=None):
        asked.append((repo_id, filename, revision))
        path = snap / filename
        return str(path) if path.exists() else None

    monkeypatch.setattr(huggingface_hub, "HfApi", Offline)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", from_cache)
    spec = ModelSpec(hf_id="Qwen/Qwen3.5-9B", revision="abc123")
    assert introspect._default_tensors(spec) == {"lm_head.weight": 80, "mtp.fc.weight": 64}
    assert asked[0] == ("Qwen/Qwen3.5-9B", "model.safetensors.index.json", "abc123")
    (snap / "model-00002-of-00002.safetensors").unlink()  # a partly cached snapshot does not answer
    with pytest.raises(ConnectionError):
        introspect._default_tensors(spec)
    (snap / "model.safetensors.index.json").unlink()  # nor does an empty cache (no single model.safetensors)
    with pytest.raises(ConnectionError):
        introspect._default_tensors(spec)


def test_a_file_the_hub_does_not_have_is_none_and_other_failures_raise(monkeypatch):
    import huggingface_hub

    class EntryNotFoundError(Exception):
        pass

    def download(repo, filename, revision=None, token=None):
        if filename == "chat_template.jinja":
            raise EntryNotFoundError("404")
        raise ConnectionError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert introspect._default_text(SPEC, "chat_template.jinja") is None
    with pytest.raises(ConnectionError):
        introspect._default_text(SPEC, "tokenizer_config.json")


def test_a_header_without_offsets_is_sized_from_dtype_and_shape():
    assert introspect._tensor_bytes("BF16", [4096, 2], None) == 16_384
    assert introspect._tensor_bytes("F8_E4M3", [10], None) == 10
    assert introspect._tensor_bytes("F4", [10], None) is None  # a dtype this does not know has no size


# ---------------------------------------------------------------- quantization


def test_an_unquantized_checkpoint_reports_none():
    assert introspect.quantization(QWEN35) is None and introspect.quantization(None) is None


def test_awq_and_w8a8_as_their_configs_declare_them():
    assert introspect.quantization({"quantization_config": AWQ}) == introspect.QuantFacts(
        method="awq", weight_bits=4)
    w8a8 = introspect.quantization({"quantization_config": W8A8})
    assert w8a8 == introspect.QuantFacts(method="compressed-tensors", fmt="int-quantized", weight_bits=8,
                                         activation_bits=8)
    assert w8a8.describe() == "compressed-tensors (int-quantized), 8-bit weights, 8-bit activations"


def test_fp8_a_kv_scheme_and_a_nested_config():
    fp8 = {"text_config": {"quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
                                                   "kv_cache_scheme": {"type": "float", "num_bits": 8}}}}
    assert introspect.quantization(fp8) == introspect.QuantFacts(method="fp8", fmt="e4m3", weight_bits=8,
                                                                 kv_scheme="float8")


def test_groups_that_disagree_give_no_bit_width():
    q = {"quant_method": "compressed-tensors", "config_groups": {
        "a": {"weights": {"num_bits": 4}}, "b": {"weights": {"num_bits": 8}}}}
    assert introspect.quantization({"quantization_config": q}).weight_bits is None


def test_modelopt_from_hf_quant_config():
    # vLLM v0.33.0 model_executor/layers/quantization/modelopt.py: {"quantization": {"quant_algo": ...,
    # "kv_cache_quant_algo": ...}} (lines 326-336).
    hq = {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}}
    facts = introspect.inspect_model(SPEC, features=("quantization",),
                                     **fetchers(QWEN35, texts={"hf_quant_config.json": json.dumps(hq)}))
    assert facts.quantization == introspect.QuantFacts(method="modelopt", fmt="FP8", weight_bits=8, activation_bits=8,
                                                       kv_scheme="FP8", source="hf_quant_config.json")
    broken = introspect.inspect_model(SPEC, features=("quantization",),
                                      **fetchers(QWEN35, texts={"hf_quant_config.json": "{nope"}))
    assert broken.quantization is None and broken.notes[0].startswith("Qwen/Qwen3.5-9B: hf_quant_config.json is not")


# ---------------------------------------------------------------- thinking switches


@pytest.mark.parametrize("template, expected", [
    (QWEN3_TEMPLATE, introspect.ThinkingSwitch("enable_thinking", "on")),
    (QWEN35_TEMPLATE, introspect.ThinkingSwitch("enable_thinking", "off")),
    (QWEN35_9B_TEMPLATE, introspect.ThinkingSwitch("enable_thinking", "on")),
    ("{{ enable_thinking }}", introspect.ThinkingSwitch("enable_thinking", "unknown")),
    (QWEN3_TEMPLATE + QWEN35_TEMPLATE, introspect.ThinkingSwitch("enable_thinking", "unknown")),
    ("{{ messages }}", None),
    ("", None),
    (None, None),
])
def test_the_thinking_switch_and_its_default(template, expected):
    assert introspect.thinking_switch(template) == expected


def test_the_template_comes_from_its_file_or_from_tokenizer_config():
    texts = {"tokenizer_config.json": json.dumps({"chat_template": QWEN3_TEMPLATE})}
    calls = []
    facts = introspect.inspect_model(SPEC, features=("thinking",), **fetchers(QWEN35, texts=texts, calls=calls))
    assert facts.thinking.default == "on" and calls == ["config", "chat_template.jinja", "tokenizer_config.json"]
    assert facts.describe()["thinking"] == "enable_thinking, on by default"
    named = {"chat_template": [{"name": "tool_use", "template": "x"}, {"name": "default", "template": QWEN35_TEMPLATE}]}
    assert introspect.chat_template(named, None) == QWEN35_TEMPLATE
    assert introspect.chat_template({"chat_template": QWEN3_TEMPLATE}, QWEN35_TEMPLATE) == QWEN35_TEMPLATE


def test_no_template_reports_none_with_a_note():
    facts = introspect.inspect_model(SPEC, features=("thinking",), **fetchers(QWEN35))
    assert facts.thinking is None
    assert facts.notes == ["Qwen/Qwen3.5-9B: no chat template found; no thinking switch is reported"]


# ---------------------------------------------------------------- modalities


def test_modalities_from_sub_configs_and_processor_files():
    assert introspect.modalities(QWEN25) == ("text",)
    assert introspect.modalities({"vision_config": {}, "audio_config": {}}) == ("text", "image", "audio")
    assert introspect.modalities(QWEN35, ["sub/video_preprocessor_config.json"]) == ("text", "video")
    assert introspect.modalities(GEMMA4, None) == ("text",)  # this fixture keeps no vision_config
    assert introspect.modalities(None) == ("text",)


def test_an_unreadable_listing_leaves_the_config_answer():
    facts = introspect.inspect_model(SPEC, features=("modalities",),
                                     **fetchers({"vision_config": {}}, files=OSError("offline")))
    assert facts.modalities == ("text", "image")
    assert facts.notes == ["Qwen/Qwen3.5-9B: could not read the file listing (offline)"]


def test_nothing_asked_nothing_fetched():
    calls = []
    assert introspect.inspect_model(SPEC, features=(), **fetchers(QWEN35, calls=calls)).describe()["mtp"] == "none"
    assert calls == []
