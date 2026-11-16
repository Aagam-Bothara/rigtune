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


def vllm_facts(version: str = "0.33.0", kernel: Optional[Dict] = VLLM_KERNEL, flags=VLLM_FLAGS) -> EngineFacts:
    return EngineFacts(engine="vllm", version=version, kernel=kernel, provenance="test: names read in vLLM source",
                       flags=None if flags is None else {f: {"choices": None, "default": None} for f in flags})
