"""Fake engine source trees, shaped like each release rigtune is checked against.

rigtune tells what an installed engine offers by reading its source text (importing vLLM or SGLang costs seconds,
and SGLang often lives in another environment). These trees hold, at the paths each release has them, only the
lines those checks look for, copied from the release tag (file and line in each comment), so a test can point
rigtune at "vLLM 0.33.0" with no engine installed. Paths are relative to the vllm or sglang package.

    with installed("vllm", "0.33.0", tmp_path):
        ...  # rigtune's feature checks now read that tree
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from typing import Dict, Iterator
from unittest import mock

VERSIONS = ("vllm-0.29.0", "vllm-0.33.0", "sglang-0.5.19", "sglang-0.5.23")
SGLANG_PYTHON = "/opt/sglang/bin/python"  # what $SGLANG_PYTHON points at, in launch commands built from a tree

_VLLM_COMMON = {
    # engine/arg_utils.py: line 38 (both tags) imports AttentionConfig; EngineArgs.language_model_only is line 613
    # at v0.29.0 and 651 at v0.33.0.
    "engine/arg_utils.py": "    AttentionConfig,\n"
                           "    language_model_only: bool = MultiModalConfig.language_model_only\n",
    # distributed/kv_transfer/kv_connector/v1/offloading_connector.py: lines 49 / 53.
    "distributed/kv_transfer/kv_connector/v1/offloading_connector.py":
        "class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):\n",
    # v1/kv_offload/base.py: lines 598-599 / 729-730.
    "v1/kv_offload/base.py": "        self.offload_prompt_only: bool = bool(\n"
                             "            self.extra_config.get(\"offload_prompt_only\", True)\n",
    # v1/kv_offload/cpu/shared_offload_region.py: lines 98 / 128.
    "v1/kv_offload/cpu/shared_offload_region.py":
        "        self.mmap_path = f\"/dev/shm/vllm_offload_{engine_id}.mmap\"\n",
}

_SGLANG_ENVIRON = ("    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)\n"
                   "    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)\n")
_SGLANG_PROTOCOL = "    return_cached_tokens_details: bool = False\n"
# The MTP checks read these files whole, so each is a verbatim excerpt of the release's own text.
# v0.29.0 config/speculative.py, lines 882-899: SpeculativeConfig.hf_config_override's Qwen3.5 branch.
_VLLM_0_29_MTP_REWRITE = (
    "        if hf_config.model_type in (\n"
    "            \"qwen3_5\",\n"
    "            \"qwen3_5_moe\",\n"
    "            \"qwen3_5_text\",\n"
    "            \"qwen3_5_moe_text\",\n"
    "        ):\n"
    "            # Checkpoints that ship only the text config resolve to the\n"
    "            # `qwen3_5_text` / `qwen3_5_moe_text` model types and carry the\n"
    "            # same `mtp_num_hidden_layers` field as the multimodal ones.\n"
    "            is_moe = hf_config.model_type in (\"qwen3_5_moe\", \"qwen3_5_moe_text\")\n"
    "            hf_config.model_type = \"qwen3_5_mtp\"\n"
    "            n_predict = getattr(hf_config, \"mtp_num_hidden_layers\", None)\n"
    "            hf_config.update(\n"
    "                {\n"
    "                    \"n_predict\": n_predict,\n"
    "                    \"architectures\": [\"Qwen3_5MoeMTP\" if is_moe else \"Qwen3_5MTP\"],\n"
    "                }\n"
    "            )\n"
)
# v0.33.0 config/speculative.py, lines 987-1007.
_VLLM_0_33_MTP_REWRITE = (
    "        if hf_config.model_type in (\n"
    "            \"qwen3_5\",\n"
    "            \"qwen3_5_moe\",\n"
    "            \"qwen3_5_text\",\n"
    "            \"qwen3_5_moe_text\",\n"
    "        ):\n"
    "            # Checkpoints that ship only the text config resolve to the\n"
    "            # `qwen3_5_text` / `qwen3_5_moe_text` model types and carry the\n"
    "            # same `mtp_num_hidden_layers` field as the multimodal ones.\n"
    "            is_moe = hf_config.model_type in (\"qwen3_5_moe\", \"qwen3_5_moe_text\")\n"
    "            hf_config.model_type = \"qwen3_5_mtp\"\n"
    "            n_predict = getattr(hf_config, \"mtp_num_hidden_layers\", None)\n"
    "            if n_predict is None:\n"
    "                text_config = get_hf_text_config(hf_config)\n"
    "                n_predict = getattr(text_config, \"mtp_num_hidden_layers\", None)\n"
    "            hf_config.update(\n"
    "                {\n"
    "                    \"n_predict\": n_predict,\n"
    "                    \"architectures\": [\"Qwen3_5MoeMTP\" if is_moe else \"Qwen3_5MTP\"],\n"
    "                }\n"
    "            )\n"
)
# v0.29.0 model_executor/models/qwen3_5_mtp.py, lines 215-224 and 320-323.
_VLLM_0_29_MTP_MODEL = (
    "class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal):\n"
    "    packed_modules_mapping = {\n"
    "        \"qkv_proj\": [\n"
    "            \"q_proj\",\n"
    "            \"k_proj\",\n"
    "            \"v_proj\",\n"
    "        ],\n"
    "        \"gate_up_proj\": [\"gate_proj\", \"up_proj\"],\n"
    "    }\n"
    "\n"
    "class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):\n"
    "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
    "        super().__init__(vllm_config=vllm_config, prefix=prefix)\n"
    "        self.set_moe_parameters()\n"
)
# v0.33.0 model_executor/models/qwen3_5_mtp.py, lines 216-225 and 322-325.
_VLLM_0_33_MTP_MODEL = (
    "class Qwen3_5MTP(LocalArgmaxMixin, nn.Module, SupportsMultiModal, SupportsPP):\n"
    "    packed_modules_mapping = {\n"
    "        \"qkv_proj\": [\n"
    "            \"q_proj\",\n"
    "            \"k_proj\",\n"
    "            \"v_proj\",\n"
    "        ],\n"
    "        \"gate_up_proj\": [\"gate_proj\", \"up_proj\"],\n"
    "    }\n"
    "\n"
    "class Qwen3_5MoeMTP(Qwen3_5MTP, QwenNextMixtureOfExperts):\n"
    "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
    "        super().__init__(vllm_config=vllm_config, prefix=prefix)\n"
    "        self.set_moe_parameters()\n"
)
# v0.29.0 distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py, lines 220-231 (v0.33.0 has no every-group fallback).
_VLLM_0_29_OFFLOAD_SCHEDULER = (
    "        eagle_groups = {\n"
    "            idx\n"
    "            for idx, g in enumerate(kv_cache_config.kv_cache_groups)\n"
    "            if g.is_eagle_group\n"
    "        }\n"
    "\n"
    "        use_eagle = (\n"
    "            vllm_config.speculative_config is not None\n"
    "            and vllm_config.speculative_config.use_eagle()\n"
    "        )\n"
    "        if use_eagle and not eagle_groups:\n"
    "            eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))\n"
)
# v0.5.19 configs/model_config.py, lines 751-773: ModelConfig._config_draft_model's Qwen3.5 branch.
_SGLANG_0_5_19_MTP_REWRITE = (
    "        if is_draft_model and self.hf_config.architectures[0] in [\n"
    "            \"Qwen3_5ForConditionalGeneration\",\n"
    "            \"Qwen3_5MoeForConditionalGeneration\",\n"
    "            \"Qwen3_5ForCausalLM\",\n"
    "            \"Qwen3_5MoeForCausalLM\",\n"
    "            \"InternS2PreviewForConditionalGeneration\",\n"
    "            \"InternS2MobiusForConditionalGeneration\",\n"
    "        ]:\n"
    "            if (\n"
    "                self.hf_config.architectures[0]\n"
    "                == \"InternS2MobiusForConditionalGeneration\"\n"
    "            ):\n"
    "                # The target owns 2,560 experts through four shared physical\n"
    "                # banks, while its bundled MTP layer is an ordinary Qwen3.5\n"
    "                # MoE layer with the checkpoint-declared smaller expert set.\n"
    "                self.hf_text_config.model_type = \"qwen3_5_moe_text\"\n"
    "                self.hf_text_config.num_experts = self.hf_text_config.mtp_num_experts\n"
    "                self.hf_text_config.num_experts_per_tok = (\n"
    "                    self.hf_text_config.mtp_num_experts_per_tok\n"
    "                )\n"
    "            self.hf_config.architectures[0] = \"Qwen3_5ForCausalLMMTP\"\n"
    "            self.hf_config.num_nextn_predict_layers = 1\n"
    "            self.hf_text_config.num_nextn_predict_layers = 1\n"
)
# v0.5.23 configs/model_config.py, lines 1159-1181.
_SGLANG_0_5_23_MTP_REWRITE = (
    "        if is_draft_model and self.hf_config.architectures[0] in [\n"
    "            \"Qwen3_5ForConditionalGeneration\",\n"
    "            \"Qwen3_5MoeForConditionalGeneration\",\n"
    "            \"Qwen3_5ForCausalLM\",\n"
    "            \"Qwen3_5MoeForCausalLM\",\n"
    "            \"InternS2PreviewForConditionalGeneration\",\n"
    "            \"InternS2MobiusForConditionalGeneration\",\n"
    "        ]:\n"
    "            if (\n"
    "                self.hf_config.architectures[0]\n"
    "                == \"InternS2MobiusForConditionalGeneration\"\n"
    "            ):\n"
    "                # The target owns 2,560 experts through four shared physical\n"
    "                # banks, while its bundled MTP layer is an ordinary Qwen3.5\n"
    "                # MoE layer with the checkpoint-declared smaller expert set.\n"
    "                self.hf_text_config.model_type = \"qwen3_5_moe_text\"\n"
    "                self.hf_text_config.num_experts = self.hf_text_config.mtp_num_experts\n"
    "                self.hf_text_config.num_experts_per_tok = (\n"
    "                    self.hf_text_config.mtp_num_experts_per_tok\n"
    "                )\n"
    "            self.hf_config.architectures[0] = \"Qwen3_5ForCausalLMMTP\"\n"
    "            self.hf_config.num_nextn_predict_layers = 1\n"
    "            self.hf_text_config.num_nextn_predict_layers = 1\n"
)
# v0.5.19 models/qwen3_5_mtp.py, lines 84-90.
_SGLANG_0_5_19_MTP_MODEL = (
    "class Qwen3_5ForCausalLMMTP(nn.Module):\n"
    "\n"
    "    @staticmethod\n"
    "    def shared_experts_fusion_disable_reason(hf_config, quant_config):\n"
    "        return Qwen3_5ForCausalLM.shared_experts_fusion_disable_reason(\n"
    "            getattr(hf_config, \"text_config\", hf_config),\n"
    "            _mtp_quant_config(quant_config),\n"
)
# v0.5.23 models/qwen3_5_mtp.py, lines 94-99.
_SGLANG_0_5_23_MTP_MODEL = (
    "class Qwen3_5ForCausalLMMTP(nn.Module):\n"
    "    # The loader reads this off the model class and hands it to the quant\n"
    "    # config, which needs it to expand fused module names (qkv_proj ->\n"
    "    # q/k/v_proj) before matching them against an exclude list. Without it an\n"
    "    # excluded attention projection is not recognised as excluded.\n"
    "    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping\n"
)
# v0.5.23 arg_groups/speculative_hook.py, lines 109-115 (_resolve_speculative_algorithm_alias).
_SGLANG_0_5_23_NEXTN = (
    "    if speculative_algorithm == \"NEXTN\" or speculative_algorithm == \"EAGLE\":\n"
    "        if is_gemma4_draft:\n"
    "            logger.info(\n"
    "                \"Detected Gemma4AssistantForCausalLM draft; \"\n"
    "                f\"promoting --speculative-algorithm {speculative_algorithm} to FROZEN_KV_MTP.\"\n"
    "            )\n"
    "            return \"FROZEN_KV_MTP\"\n"
)

TREES: Dict[str, Dict[str, str]] = {
    "vllm-0.29.0": dict(_VLLM_COMMON, **{
        # The server's own arguments: entrypoints/launchers/cli_args.py, BaseFrontendArgs, line 132.
        "entrypoints/launchers/cli_args.py": "    enable_prompt_tokens_details: bool = False\n",
        "config/speculative.py": _VLLM_0_29_MTP_REWRITE,
        "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py": _VLLM_0_29_OFFLOAD_SCHEDULER,
        "model_executor/models/qwen3_5_mtp.py": _VLLM_0_29_MTP_MODEL,
    }),
    "vllm-0.33.0": dict(_VLLM_COMMON, **{
        # Moved to entrypoints/common/cli_args.py, BaseFrontendArgs, line 156; launchers/cli_args.py is gone.
        "entrypoints/common/cli_args.py": "    enable_prompt_tokens_details: bool = False\n",
        "config/speculative.py": _VLLM_0_33_MTP_REWRITE,
        "model_executor/models/qwen3_5_mtp.py": _VLLM_0_33_MTP_MODEL,
    }),
    "sglang-0.5.19": {
        # Every argument is defined in srt/server_args.py (lines 1353, 2072, 2324, 2719, 2735, 2770).
        "srt/server_args.py": (
            "    enable_cache_report: A[\n"
            "        \"Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK. Or any "
            "name registered via `SpeculativeAlgorithm.register`.\",\n"
            "    speculative_ngram_max_bfs_breadth: A[\n"
            "    enable_hierarchical_cache: A[bool, \"Enable hierarchical cache\", NS(\"memory\")] = (\n"
            "    hicache_size: A[\n"
            "    hicache_storage_backend: A[\n"),
        "srt/arg_groups/speculative_hook.py": "        cfg.speculative_algorithm == \"STANDALONE\"\n",  # line 645
        "srt/environ.py": _SGLANG_ENVIRON,  # lines 717, 719
        "srt/entrypoints/openai/protocol.py": _SGLANG_PROTOCOL,  # line 355
        "srt/configs/model_config.py": _SGLANG_0_5_19_MTP_REWRITE,
        "srt/models/qwen3_5_mtp.py": _SGLANG_0_5_19_MTP_MODEL,
    },
    "sglang-0.5.23": {
        # Arguments moved to srt/arg_groups/fields/*.py; server_args.py keeps add_cli_args (line 351). rigtune reads
        # server_args.py and arg_groups/*.py, not fields/, and still finds every name through
        # arg_groups/field_order.py's POSITIONAL_FIELD_ORDER (lines 146, 300, 359, 362, 366) and the hooks.
        "srt/server_args.py": "    def add_cli_args(parser: argparse.ArgumentParser):\n",
        "srt/arg_groups/field_order.py": (
            "    \"enable_cache_report\",\n"
            "    \"speculative_ngram_max_bfs_breadth\",\n"
            "    \"enable_hierarchical_cache\",\n"
            "    \"hicache_size\",\n"
            "    \"hicache_storage_backend\",\n"),
        # lines 109-115 and 1078
        "srt/arg_groups/speculative_hook.py":
            _SGLANG_0_5_23_NEXTN + "    if cfg.speculative_algorithm == \"STANDALONE\" and attn_dp_enabled_of(\n",
        # line 480
        "srt/arg_groups/attention_hook.py": "            if _algo not in (\"DSPARK\", \"DFLASH\") or verify not in (\n",
        "srt/arg_groups/fields/serving.py": "    enable_cache_report: A[\n",  # line 196, not read by rigtune
        "srt/environ.py": _SGLANG_ENVIRON,  # lines 828, 830
        "srt/entrypoints/openai/protocol.py": _SGLANG_PROTOCOL,  # line 392
        "srt/configs/model_config.py": _SGLANG_0_5_23_MTP_REWRITE,
        "srt/models/qwen3_5_mtp.py": _SGLANG_0_5_23_MTP_MODEL,
    },
}


def build(engine: str, version: str, root: Path) -> Path:
    """Write the tree for `engine` `version` under `root`; returns the package directory."""
    pkg = root / engine
    for rel, text in TREES[f"{engine}-{version}"].items():
        path = pkg.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return pkg


def _clear() -> None:
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl

    for fn in (vl._vllm_package, vl._serve_cli, vl._attention_backend_flag, vl._prompt_tokens_details_flag,
               vl._language_model_only_flag, vl._offload_features, vl._mtp_archs, vl._own_head, vl._vllm_python,
               vl._suffix_refusal, vl._arctic_report, vl._engine_facts, sg._engine_facts,
               vl._offload_counts_every_group_as_draft, sg.sglang_spec_methods,
               sg._cache_report_flag, sg._hicache_features, sg._tier_details_flag, sg.sglang_mtp_archs):
        if hasattr(fn, "cache_clear"):  # a test may have put a plain function in its place
            fn.cache_clear()


def _version_tuple(version: str):
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


@contextlib.contextmanager
def installed(engine: str, version: str, root: Path, flashinfer: bool = True,
              arctic_inference: bool = True) -> Iterator[Path]:
    """rigtune as it sees `engine` `version` installed from the tree under `root` (built if missing), with the
    optional packages rigtune looks for present or not. Yields the package directory."""
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    import rigtune.speculative as speculative

    pkg = root / engine
    if not pkg.exists():
        build(engine, version, root)
    present = {"flashinfer": flashinfer, "arctic_inference": arctic_inference}
    real_find_spec = importlib.util.find_spec

    class _Spec:
        submodule_search_locations = [str(pkg)]

    def find_spec(name, *args, **kwargs):
        if name == engine:
            return _Spec()
        if name in present:
            return object() if present[name] else None
        return real_find_spec(name, *args, **kwargs)

    _clear()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(importlib.util, "find_spec", find_spec))
            if engine == "vllm":
                stack.enter_context(mock.patch.object(vl, "_serve_cli", lambda: "vllm"))
                stack.enter_context(mock.patch.object(vl, "_vllm_package", lambda: pkg))
                stack.enter_context(mock.patch.object(vl, "_vllm_version", lambda: _version_tuple(version)))
                stack.enter_context(mock.patch.object(speculative, "_vllm_version", lambda: version))
                # Arctic Inference as vLLM's own Python would report it: its suffix tree loads, or it is absent.
                absent = "ModuleNotFoundError: No module named 'arctic_inference'"
                report = ({"arctic_inference": speculative.ARCTIC_ADVISED, "toplevel": "ok", "suffix": "ok",
                           "vllm": version} if arctic_inference
                          else {"arctic_inference": None, "toplevel": absent, "suffix": absent})
                stack.enter_context(mock.patch.object(vl, "_arctic_report",
                                                      lambda: ("/opt/vllm/bin/python", report, None)))
            else:
                stack.enter_context(mock.patch.object(sg, "sglang_python", lambda: SGLANG_PYTHON))
                stack.enter_context(mock.patch.object(sg, "_sglang_package_dir", lambda python: pkg))
            yield pkg
    finally:
        _clear()  # every answer cached while the tree stood in for the engine
