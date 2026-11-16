"""Hand-made engine-probe answers for the kernel-setting tests: the names read in vLLM v0.33.0 and SGLang v0.5.23
source, never a real probe's output (their provenance says so)."""

from __future__ import annotations

from typing import Dict, Optional

from rigtune.backends.facts import EngineFacts

VLLM_KERNEL = {
    "cudagraph_modes": ["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"],
    "compilation_modes": ["NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE"],
    "compilation_fields": ["cudagraph_capture_sizes", "cudagraph_mode", "max_cudagraph_capture_size", "mode",
                           "pass_config", "use_inductor_graph_partition"],
    "pass_fields": ["eliminate_noops", "fuse_act_quant", "fuse_norm_quant"],
    "attention_backends": ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION"],
    "opt_levels": ["O0", "O1", "O2", "O3"],
}
VLLM_FLAGS = ("--attention-backend", "--compilation-config", "--optimization-level", "--max-num-seqs")


# SGLang's kernel flags and their choices, as the two releases' parsers build them (v0.5.19 server_args.py /
# v0.5.23 arg_groups/fields/exec_.py and choices.py). 0.5.23 dropped tc_piecewise.
_CG_0519 = ["full", "breakable", "tc_piecewise", "disabled"]
_CG_0523 = ["full", "breakable", "disabled"]
_ATTN = ["fa3", "flashinfer", "triton", "torch_native", "flex_attention", "trtllm_mha"]


def sglang_facts(version: str = "0.5.23", drop=()) -> EngineFacts:
    cg = _CG_0519 if version == "0.5.19" else _CG_0523
    flags = {
        "--attention-backend": {"choices": _ATTN, "default": None},
        "--prefill-attention-backend": {"choices": _ATTN, "default": None},
        "--decode-attention-backend": {"choices": _ATTN, "default": None},
        "--cuda-graph-backend-decode": {"choices": cg, "default": None},
        "--cuda-graph-backend-prefill": {"choices": cg, "default": None},
        "--cuda-graph-max-bs-decode": {"choices": None, "default": None},
        "--cuda-graph-max-bs-prefill": {"choices": None, "default": None},
        "--enable-torch-compile": {"choices": None, "default": False},
        "--torch-compile-max-bs": {"choices": None, "default": 32},
        "--max-running-requests": {"choices": None, "default": None},
    }
    if version == "0.5.19":
        flags["--cuda-graph-max-bs"] = {"choices": None, "default": None}  # the deprecated alias
    return EngineFacts(engine="sglang", version=version, provenance="test: names read in SGLang source",
                       flags={k: v for k, v in flags.items() if k not in drop})


def vllm_facts(version: str = "0.33.0", kernel: Optional[Dict] = VLLM_KERNEL, flags=VLLM_FLAGS) -> EngineFacts:
    return EngineFacts(engine="vllm", version=version, kernel=kernel, provenance="test: names read in vLLM source",
                       flags=None if flags is None else {f: {"choices": None, "default": None} for f in flags})
