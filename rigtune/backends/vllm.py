"""vLLM backend (NVIDIA, compute capability >= 7.5)."""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rigtune import speculative
from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid, render_extra
from rigtune.hfconfig import dtype_bytes, kv_element_bytes, load_arch
from rigtune.memory import HybridKV, MemoryModel, estimate
from rigtune.models import ArchInfo, Config, GiB, HardwareDescriptor, ModelSpec, PreparedModel
from rigtune.hardware import nvlink_between
from rigtune.quantized import INT4_METHODS, INT8_METHODS, PREQUANTIZED, hf_weight_options

# The fp8 KV-cache type offered on Ampere. It runs through FlashInfer: vLLM 0.11's default Triton
# attention builds its fp8 kernels with e4m3, which Ampere cannot compile, whatever the cache type.
AMPERE_FP8_KV = "fp8_e5m2"
# int8 with a scale per token and head, computed at run time (no calibration data). In vLLM 0.29 it is
# one of Triton attention's cache types; earlier releases were not checked. It is offered from Ada on:
# on an A40 it started but lost to the unquantized cache (398 against 444 tok/s at 8 users on `extract`,
# with a slower tail) and to fp8_e5m2 (538), so on Ampere it would only cost a trial. Ada is untested.
INT8_KV = "int8_per_token_head"
INT8_KV_FROM = (0, 29)
# Cache types that need a particular attention backend.
KV_ATTENTION = {AMPERE_FP8_KV: "FLASHINFER", INT8_KV: "TRITON_ATTN"}

logger = logging.getLogger(__name__)

# Architectures known to load in vLLM, consulted when neither the engine probe (rigtune.backends.facts: the registry
# of the engine that will run, listed in its own Python) nor an importable registry here lists the model's. Frozen: no
# name is ever added, since the probe lists a new architecture as soon as an installed engine has it.
FALLBACK_ARCHS = frozenset({
    "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM", "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration", "Phi3ForCausalLM", "PhiForCausalLM", "GPT2LMHeadModel", "GPTNeoXForCausalLM",
    "OPTForCausalLM", "FalconForCausalLM", "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "OlmoForCausalLM",
    "Olmo2ForCausalLM", "InternLM2ForCausalLM", "StableLmForCausalLM", "CohereForCausalLM", "GraniteForCausalLM",
    "Starcoder2ForCausalLM", "BloomForCausalLM", "MPTForCausalLM", "ExaoneForCausalLM", "SmolLM3ForCausalLM",
    # 2026 hybrid-attention models, registered by both engines. vLLM 0.29.0 model_executor/models/registry.py:
    # _TEXT_GENERATION_MODELS has Gemma4ForCausalLM (line 112), Qwen3_5ForCausalLM and Qwen3_5MoeForCausalLM
    # (202-203); _MULTIMODAL_MODELS has Gemma4ForConditionalGeneration (401),
    # Gemma4UnifiedForConditionalGeneration (402-404), Qwen3_5ForConditionalGeneration and
    # Qwen3_5MoeForConditionalGeneration (575-578). SGLang 0.5.19 EntryClass: srt/models/qwen3_5.py:2690,
    # qwen3_5_text.py:257, gemma4_unified.py:438, gemma4_mm.py:1135, gemma4_causal.py:1437.
    # SglangBackend.supports falls back to this list too, which is the usual case for SGLang: it often runs from
    # $SGLANG_PYTHON, and rigtune's own interpreter cannot import its registry. The list knows no versions, so
    # an older engine there may be offered these models and fail at launch.
    "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM", "Gemma4UnifiedForConditionalGeneration", "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
})

# Paged-KV runtimes do not need KV for every (ctx x batch) token up front; they preempt when
# occupancy is exceeded. Budget a fraction of the worst case so realistic batches survive planning.
PAGED_KV_FRACTION = 0.25


VLLM_BLOCK = 16  # config/cache.CacheConfig.DEFAULT_BLOCK_SIZE; FlashAttention and Triton take MultipleOf(16)
VLLM_CONCURRENT_BATCHES = 2  # config/vllm.VllmConfig.max_concurrent_batches with async scheduling (the default)


def vllm_default_batched_tokens(hw: HardwareDescriptor) -> int:
    """max_num_batched_tokens `vllm serve` picks when none is given (engine/arg_utils.EngineArgs.get_batch_defaults,
    UsageContext.OPENAI_API_SERVER): 16384 from 160 GiB, 8192 from 70 GiB except on an A100, else 2048."""
    total = hw.gpu.vram_total_bytes if hw.gpu else 0
    name = (hw.gpu.name if hw.gpu else "").lower()
    if total >= 160 * GiB:
        return 16384
    if total >= 70 * GiB and "a100" not in name:
        return 8192
    return 2048


def _cdiv(a: float, b: float) -> int:
    return int(-(-a // b))


def vllm_hybrid_kv(cfg: Config, arch: ArchInfo, default_batched_tokens: int = 2048) -> HybridKV:
    """vLLM 0.29's hybrid KV cache manager, read from source (not yet run on hardware).

    - One block pool for every layer type: v1/core/kv_cache_utils.get_kv_cache_config_from_groups sizes
      num_blocks = available // bytes_per_block, and every KV cache group draws from it, so the pool is not
      pre-split by type. The manager is on by default on CUDA (config/vllm.py VllmConfig.__post_init__ turns it off
      only for a platform without it, chunked local attention, or a KV connector without HMA support).
    - Blocks: 16 tokens (CacheConfig.DEFAULT_BLOCK_SIZE). With linear layers,
      platforms/interface.Platform._align_hybrid_block_size raises the attention block to
      16 x cdiv(state page, 16 x attention page per token) and pads the state page up to one attention page
      (528 tokens for Qwen3.5-9B with a bf16 cache, 1056 with fp8). Mixed page sizes without linear layers
      (Gemma 4: 8 heads of 256 on sliding layers, 1 of 512 on full ones) are unified by raising the smaller page's
      block (kv_cache_utils.unify_kv_cache_spec_page_size: 64 tokens on Gemma 4's full layers).
    - Layer groups: _get_kv_cache_groups_uniform_page_size makes every group as large as the smallest layer type
      (the largest, when that is under 1.5x the smallest) and pads the rest: gemma-3-27b's 52 sliding layers take
      60 layer pages beside its 10 full ones.
    - Sliding layers free out-of-window blocks (single_type_kv_cache_manager.SlidingWindowManager), but keep the
      blocks of tokens scheduled and not yet settled: kv_cache_interface.SlidingWindowSpec.max_admission_blocks_
      per_request reserves window - 1 + VllmConfig.max_in_flight_tokens (+1 block), where max_in_flight_tokens is
      max_concurrent_batches x max_num_batched_tokens. Counted once across the batch.
    - Linear (mamba) layers hold one state block per running request (kv_cache_interface.MambaSpec,
      MambaManager). With prefix caching, which vLLM enables for hybrid generative models
      (config/model.ModelConfig.is_prefix_caching_supported), mamba_cache_mode becomes "align"
      (model_executor/models/config.MambaModelConfig.verify_and_update_config): a request briefly holds a second
      block while crossing a block boundary (MambaManager.remove_skipped_blocks frees it), so the steady state,
      one per running sequence, is what the batch is planned for.
    - Speculative decoding: mamba/abstract.MambaBase.get_kv_cache_spec sets num_speculative_blocks =
      num_speculative_tokens, so a spec with k tokens holds 1 + k states per sequence, and the conv state grows
      by k columns (mamba_utils.gated_delta_net_state_shape, conv_kernel - 1 + num_spec).
    - Start-up: kv_cache_utils.get_kv_cache_configs refuses to start unless one request at max_model_len fits
      (_check_enough_kv_cache_memory, _max_memory_usage_bytes_from_groups: per group, full layers
      cdiv(max_model_len, block) blocks, sliding cdiv(min(window - 1 + in-flight, max_model_len), block) + 1, mamba
      2 + k in align mode and 1 + k without prefix caching), after holding back one null block. The plan is never
      below that.
    - The checkpoint's own MTP head ("mtp:<k>"): its layers are full-attention decoder layers
      (model_executor/models/qwen3_5_mtp.py Qwen3_5MultiTokenPredictor builds mtp_num_hidden_layers of them with
      layer_type="full_attention", and shares no KV with the target). GPUModelRunner.get_kv_cache_spec gives every
      attention layer of the static forward context, the drafter's too, a spec, and theirs equals the target's
      full-attention spec (v0.33.0's FullAttentionSpec.is_draft is compare=False), so they join the full layers'
      group: Qwen3.5-9B with one MTP layer has 9 full layers against 24 linear ones, the group size becomes 9 and
      the linear layers pad to 27. A second model's layers (draft, EAGLE3) are not counted: the planner does not
      know them.
    """
    elem = kv_element_bytes(cfg.kv_dtype if cfg.kv_dtype != "auto" else arch.torch_dtype)
    kind, _, k = speculative.parse(cfg.spec_decode) if cfg.spec_decode else (None, None, 0)
    n_full, n_sliding, n_linear = arch.full_attention_layers or 0, arch.sliding_layers, arch.linear_layers
    if kind in speculative.SELF_HEAD:
        n_full += arch.mtp_layers
    full_page1 = 2 * (arch.full_kv_heads or arch.num_kv_heads) * (arch.full_head_dim or arch.head_dim) * elem
    sliding_page1 = 2 * arch.num_kv_heads * arch.head_dim * elem
    full_block = sliding_block = VLLM_BLOCK
    state_page = 0.0
    if n_linear:
        raw_state = (arch.linear_state_bytes + k * arch.linear_conv_step_bytes) / n_linear
        full_block = sliding_block = VLLM_BLOCK * _cdiv(raw_state, VLLM_BLOCK * full_page1)
        state_page = full_block * full_page1
    elif n_sliding and n_full:
        big = max(VLLM_BLOCK * full_page1, VLLM_BLOCK * sliding_page1)
        if big % (VLLM_BLOCK * full_page1) == 0:
            full_block = int(big // full_page1)
        if big % (VLLM_BLOCK * sliding_page1) == 0:
            sliding_block = int(big // sliding_page1)
    counts = [n for n in (n_full, n_sliding, n_linear) if n]
    group = min(counts)
    if max(counts) < 1.5 * group:
        group = max(counts)

    def padded(n: int) -> int:
        return _cdiv(n, group) * group

    in_flight = VLLM_CONCURRENT_BATCHES * (cfg.prefill_budget or default_batched_tokens)
    ctx = cfg.ctx
    one_request = padded(n_full) * _cdiv(ctx, full_block) * full_block * full_page1
    page = max(full_block * full_page1, state_page)
    if n_sliding:
        window = arch.sliding_window or 0
        one_request += padded(n_sliding) * (_cdiv(min(window - 1 + in_flight, ctx), sliding_block) + 1) \
            * sliding_block * sliding_page1
        page = max(page, sliding_block * sliding_page1)
    if n_linear:
        align = cfg.prefix_cache is not False
        one_request += padded(n_linear) * ((2 if align else 1) + k) * state_page
    null_block = group * page
    return HybridKV(
        state_slots=cfg.batch * (1 + k),
        state_bytes=int(padded(n_linear) * state_page) if n_linear else None,
        full_block=full_block,
        sliding_block=sliding_block,
        full_layers=padded(n_full),
        sliding_layers=padded(n_sliding) if n_sliding else None,
        swa_extra_tokens=in_flight if n_sliding else 0,
        # SlidingWindowSpec.max_admission_blocks_per_request: min(window - 1 + in-flight, max_model_len) + 1 block.
        sliding_cap=(_cdiv(min((arch.sliding_window or 0) - 1 + in_flight, ctx), sliding_block) + 1) * sliding_block
        if n_sliding else None,
        min_bytes=int(one_request + null_block),
    )


def vllm_mtp_extra_weights(cfg: Config, model: PreparedModel) -> int:
    """Weight bytes an MTP setting adds to the plan's. None for a Hub checkpoint: load_arch takes its parameter count
    from the Hub's safetensors total, which already holds the mtp.* tensors (vLLM skips them without MTP). For a
    local checkpoint directory the count is estimated from the config and leaves the head out, so its bytes
    (spec_bytes["mtp"]) are added. Qwen3_5MTP.load_weights also loads its own copy of the embedding and output head
    before v1/spec_decode/llm_base_proposer.py load_model swaps in the target's (_maybe_share_embeddings, _maybe_share_lm_head): a
    transient of shared_bytes at load, which is not planned (docs/benchmarks.md lists it for the GPU run)."""
    if speculative.own_head(cfg.spec_decode) and os.path.isdir(model.spec.hf_id):
        return int(model.spec_bytes.get(speculative.MTP_KEY) or 0)
    from rigtune.pairing import extra_weights
    return extra_weights(cfg, model)


def vllm_registry_archs() -> Optional[set]:
    if importlib.util.find_spec("vllm") is None:
        return None
    try:
        from vllm.model_executor.models import ModelRegistry  # type: ignore

        return set(ModelRegistry.get_supported_archs())
    except Exception as exc:
        logger.debug("vllm registry unavailable: %s", exc)
        return None


# vLLM 0.29 cannot run fp8 weights on Ampere. With torch.compile the engine fails in Inductor
# ("auto_functionalized was not removed"); without it (-O0 or --enforce-eager) a CUTLASS sm80 kernel
# fails. Measured on an A40 with vLLM 0.29.0 and torch 2.13; vLLM 0.11 ran fp8 there through Marlin.
FP8_ON_AMPERE_BROKEN_FROM = (0, 29)


@functools.lru_cache(maxsize=1)
def _vllm_version() -> Tuple[int, int]:
    """Installed vLLM's (major, minor), or (0, 0) when it is not installed."""
    try:
        from importlib.metadata import version

        m = re.match(r"(\d+)\.(\d+)", version("vllm"))
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    except Exception:
        return (0, 0)


@functools.lru_cache(maxsize=1)
def _serve_cli() -> Optional[str]:
    """The `vllm` command line next to this interpreter, or on PATH."""
    sibling = Path(sys.executable).with_name("vllm.exe" if sys.platform == "win32" else "vllm")
    return str(sibling) if sibling.exists() else shutil.which("vllm")


def vllm_serve_head(model_path: str) -> List[str]:
    """How a vLLM server command line starts: `vllm serve <model>` where that CLI is installed.
    vLLM 0.29 deprecates the `vllm.entrypoints.openai.api_server` module used before, which remains
    the fallback without the CLI."""
    cli = _serve_cli()
    if cli:
        return [cli, "serve", model_path]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path]


@functools.lru_cache(maxsize=1)
def _attention_backend_flag() -> bool:
    """Whether the installed vLLM takes --attention-backend (its AttentionConfig). vLLM 0.11 had no
    such flag and read VLLM_ATTENTION_BACKEND instead. Read from source: importing vLLM costs seconds."""
    spec = importlib.util.find_spec("vllm")
    locs = list(spec.submodule_search_locations or []) if spec else []
    try:
        return bool(locs) and "AttentionConfig" in (Path(locs[0]) / "engine" / "arg_utils.py").read_text("utf-8")
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def _vllm_package() -> Optional[Path]:
    """The vllm package that the launch command will run, or None when it cannot be told.

    The `vllm` command can come from PATH, in another environment than this interpreter's, so a flag found in the
    importable source here would say nothing about it: look in that command's own environment instead."""
    cli = _serve_cli()
    sibling = Path(sys.executable).with_name("vllm.exe" if sys.platform == "win32" else "vllm")
    if cli is None or Path(cli) == sibling:  # the module next to this interpreter, or its own CLI
        spec = importlib.util.find_spec("vllm")
        locs = list(spec.submodule_search_locations or []) if spec else []
        return Path(locs[0]) if locs else None
    env = Path(cli).resolve().parent.parent
    found = sorted(env.glob("lib/python3*/site-packages/vllm")) + sorted(env.glob("Lib/site-packages/vllm"))
    return found[0] if found else None


def _is_python(path: str) -> bool:
    """Whether a path names a Python interpreter (python3.12, python, pypy3, python.exe), not a shell."""
    return re.match(r"(?:python|pypy)[\w.-]*$", Path(path).name, re.IGNORECASE) is not None


def script_python(script: Path) -> Optional[str]:
    """The Python a console script runs with, or None when it does not name one: its `#!` interpreter when that is a
    Python (through `/usr/bin/env`, its options and assignments skipped, looked up on PATH), or, for pip's launcher for
    long paths (`#!/bin/sh` followed by `'''exec' <python> "$0" "$@"`), the one it execs. A shell wrapper
    (`#!/bin/bash ... exec .../vllm`) names none: probing bash would refuse a working install."""
    try:
        with open(script, "rb") as fh:
            head = fh.read(1024).decode("utf-8", "replace").splitlines()[:2]
    except OSError:
        return None
    if not head or not head[0].startswith("#!"):
        return None
    words = head[0][2:].split()
    if words and Path(words[0]).name == "env":
        words = words[1:]
        while words and (words[0].startswith("-") or "=" in words[0]):  # env -S, -i, -u NAME is rare; VAR=value
            words = words[1:]
        if words:
            words[0] = shutil.which(words[0]) or words[0]
    if words and _is_python(words[0]):
        return words[0]
    if words and Path(words[0]).name == "sh" and len(head) > 1:
        m = re.match(r"""'''exec' (?:'([^']+)'|"([^"]+)"|(\S+)) "\$0" "\$@\"""", head[1])
        named = next((g for g in m.groups() if g), None) if m else None
        return named if named and _is_python(named) else None
    return None


@functools.lru_cache(maxsize=1)
def _vllm_python() -> Optional[str]:
    """The Python the launch command runs vLLM with: this interpreter when the command is its own `vllm` or there is
    none (vllm_serve_head then runs `python -m`); else the Python the `vllm` script names; else, for a launcher with no
    `#!` line, a Python beside it; else this interpreter, which rigtune already requires to import vLLM before it
    offers the backend at all (hardware.probe_backends)."""
    cli = _serve_cli()
    sibling = Path(sys.executable).with_name("vllm.exe" if sys.platform == "win32" else "vllm")
    if cli is None or Path(cli) == sibling:
        return sys.executable
    named = script_python(Path(cli))
    if named:
        return named
    try:
        with open(cli, "rb") as fh:
            has_shebang = fh.read(2) == b"#!"
    except OSError:
        has_shebang = True
    if not has_shebang:
        for name in ("python3", "python", "python.exe"):
            beside = Path(cli).with_name(name)
            if beside.exists():
                return str(beside)
    return sys.executable


@functools.lru_cache(maxsize=1)
def _engine_facts():
    """What the vLLM the launch command will run says it offers (rigtune.backends.facts), asked in its own Python once
    per install; None when probing is off or that vLLM cannot be found."""
    from rigtune.backends import facts

    return facts.cached_facts("vllm", _vllm_python(), _vllm_package())


@functools.lru_cache(maxsize=1)
def _arctic_report() ->Tuple[Optional[str], Optional[dict], Optional[str]]:
    """(the Python vLLM runs from, speculative.SUFFIX_PROBE's report there or None, why there is none), asked once per
    process, so once per calibration. Where that Python is this one and has no `arctic_inference` at all, nothing is
    spawned."""
    python = _vllm_python()
    if python is None:
        return None, None, "no interpreter found for the vllm command"
    if python == sys.executable and importlib.util.find_spec("arctic_inference") is None:
        absent = "ModuleNotFoundError: No module named 'arctic_inference'"
        return python, {"arctic_inference": None, "toplevel": absent, "suffix": absent}, None
    try:
        return python, speculative.run_suffix_probe(python), None
    except Exception as exc:  # TimeoutExpired, OSError, a crash or unreadable output: could not ask
        logger.warning("could not ask %s about arctic-inference: %s", python, exc)
        return python, None, type(exc).__name__


@functools.lru_cache(maxsize=1)
def _suffix_refusal() -> Optional[str]:
    """speculative.suffix_refusal for the Python vLLM runs from: a positive finding only, never a check that could not
    run (_suffix_caveat). The profile's note names no path; the full install line for that Python is logged."""
    python, report, _ = _arctic_report()
    refusal = speculative.suffix_refusal(report) if report is not None else None
    if refusal and python:
        logger.warning("suffix decoding not offered (asked %s): %s. Install line: %s", python, refusal,
                       speculative.arctic_install_line(python))
    return refusal


def _suffix_caveat() -> Optional[str]:
    """speculative.suffix_caveat when the check could not run; suffix decoding is then offered as before."""
    _, report, why = _arctic_report()
    return speculative.suffix_caveat(why or "no report") if report is None else None


def arctic_plugin_note() -> Optional[str]:
    """When ARCTIC_INFERENCE_ENABLED=1 is set for the engines rigtune launches (they inherit its environment) and
    arctic-inference's plugin in vLLM's Python would refuse that vLLM: every vLLM launch, of any method, would fail at
    start-up (speculative.plugin_refusal). Asks vLLM's Python only when the variable is set."""
    if os.environ.get("ARCTIC_INFERENCE_ENABLED") != "1" or os.environ.get("ARCTIC_INFERENCE_SKIP_VERSION_CHECK") == "1":
        return None
    _, report, _ = _arctic_report()
    refusal = speculative.plugin_refusal(report) if report is not None else None
    return f"vllm: {refusal}" if refusal else None


# Where a vLLM release defines its server's own arguments (BaseFrontendArgs, which holds
# enable_prompt_tokens_details): entrypoints/common/cli_args.py in v0.33.0 (line 156), entrypoints/launchers/
# cli_args.py in v0.29.0 (line 132), entrypoints/openai/cli_args.py before. Each tag has only its own file.
FRONTEND_ARGS = (("entrypoints", "common", "cli_args.py"), ("entrypoints", "launchers", "cli_args.py"),
                 ("entrypoints", "openai", "cli_args.py"))


@functools.lru_cache(maxsize=1)
def _prompt_tokens_details_flag() -> bool:
    """Whether the vLLM that will run takes --enable-prompt-tokens-details, which puts cached_tokens in each reply's
    usage. Looked for in each release's server-argument file (FRONTEND_ARGS); read from source, since importing
    vLLM costs seconds."""
    root = _vllm_package()
    if root is None:
        return False
    for parts in FRONTEND_ARGS:
        try:
            if "enable_prompt_tokens_details" in root.joinpath(*parts).read_text("utf-8"):
                return True
        except OSError:
            continue
    return False


@functools.lru_cache(maxsize=1)
def _language_model_only_flag() -> bool:
    """Whether the vLLM that will run takes --language-model-only, which serves a multimodal model's language model
    alone: no vision or audio encoder is loaded and no multimodal input is profiled at start-up. vLLM 0.29.0 has it
    in engine/arg_utils.py (EngineArgs.language_model_only, MultiModalConfig.language_model_only). Read from source,
    since importing vLLM costs seconds."""
    root = _vllm_package()
    try:
        return root is not None and "language_model_only" in (root / "engine" / "arg_utils.py").read_text("utf-8")
    except OSError:
        return False


# What the vLLM that will run offers for a KV-cache tier, read from its source (importing vLLM costs seconds).
# The rules below were read in v0.29.0 and have not run on hardware.
#   "cpu": distributed/kv_transfer/kv_connector/v1/offloading_connector.py has OffloadingConnector, and
#          v1/kv_offload/base.py reads offload_prompt_only (default True: decode tokens are not offloaded).
#   "hma": that connector is SupportsHMA, so config/vllm.py VllmConfig.__post_init__ keeps the hybrid KV
#          cache manager on; without it, hybrid layouts would be planned wrong (sliding layers promoted to
#          full by v1/core/kv_cache_utils._promote_local_kv_cache_specs) or fail to start (mamba layers).
#   "shm": v1/kv_offload/cpu/shared_offload_region.py maps the CPU tier from /dev/shm (CPUOffloadingSpec.
#          create_worker on CUDA, whatever the tensor-parallel size), so free /dev/shm caps the tier too.
# The filesystem tier behind it (TieringOffloadingSpec) is not offered: every block stored to the CPU tier is
# cascaded to every secondary tier (v1/kv_offload/tiering/manager.py), and the fs tier takes no size limit and
# never deletes (tiering/fs/manager.py), so one session trial could write the whole replay to disk.
OFFLOAD_FEATURES = ("cpu", "hma", "shm")
VLLM_NO_DISK = "vLLM's disk tier has no size limit in 0.29; not offered"


def offload_features_in(sources: dict) -> frozenset:
    """The OFFLOAD_FEATURES that vLLM source text (relative path -> text) shows."""
    connector = sources.get("distributed/kv_transfer/kv_connector/v1/offloading_connector.py", "")
    spec_base = sources.get("v1/kv_offload/base.py", "")
    out = set()
    if "class OffloadingConnector" in connector and "offload_prompt_only" in spec_base:
        out.add("cpu")
        if "SupportsHMA" in connector:
            out.add("hma")
        if "/dev/shm" in sources.get("v1/kv_offload/cpu/shared_offload_region.py", ""):
            out.add("shm")
    return frozenset(out)


OFFLOAD_SOURCES = ("distributed/kv_transfer/kv_connector/v1/offloading_connector.py", "v1/kv_offload/base.py",
                   "v1/kv_offload/cpu/shared_offload_region.py")


@functools.lru_cache(maxsize=1)
def _offload_features() -> frozenset:
    """OFFLOAD_FEATURES of the vLLM the launch command will run; empty when it cannot be found."""
    root = _vllm_package()
    if root is None:
        return frozenset()
    sources = {}
    for rel in OFFLOAD_SOURCES:
        try:
            sources[rel] = root.joinpath(*rel.split("/")).read_text("utf-8")
        except OSError:
            continue
    return offload_features_in(sources)


def offload_features_from(probed: Optional[dict]) -> Optional[frozenset]:
    """The OFFLOAD_FEATURES the engine probe's `features` answer shows (None when it gave none), by the rules
    offload_features_in applies to the source: "cpu" needs the connector and offload_prompt_only, the others need
    "cpu"."""
    if probed is None:
        return None
    if not (probed.get("offloading_connector") and probed.get("offload_prompt_only")):
        return frozenset()
    return frozenset({"cpu"} | ({"hma"} if probed.get("supports_hma") else set())
                     | ({"shm"} if probed.get("shm_region") else set()))


def probe_answer(read: Callable[[Any], Any]) -> Any:
    """`read` applied to what the vLLM that will run told the engine probe (rigtune.backends.facts), or None when it
    was not asked or did not answer that question."""
    try:
        facts = _engine_facts()
        return None if facts is None else read(facts)
    except Exception as exc:  # never a reason to fail: the source check answers instead
        logger.debug("no engine probe answer: %s", exc)
        return None


# Each feature check rigtune makes of vLLM: (the engine probe's answer, None when it has none; the check that reads
# vLLM's source, which answers otherwise). The probe asks the parser `vllm serve` builds, wherever the release defines
# its arguments: the source checks look in the files one release kept them in, and found no
# --enable-prompt-tokens-details on 0.33 until told where 0.33 moved it.
# The source checks are looked up when asked, so a test that replaces one replaces it here too.
FEATURE_CHECKS: Dict[str, Tuple[Callable[[Any], Any], Callable[[], Any]]] = {
    "attention_backend": (lambda f: f.has_flag("--attention-backend"), lambda: _attention_backend_flag()),
    "prompt_tokens_details": (lambda f: f.has_flag("--enable-prompt-tokens-details"),
                              lambda: _prompt_tokens_details_flag()),
    "language_model_only": (lambda f: f.has_flag("--language-model-only"), lambda: _language_model_only_flag()),
    "offload": (lambda f: offload_features_from(f.features), lambda: _offload_features()),
}


def feature_answer(name: str) -> Tuple[Any, str]:
    """(a feature check's answer, where it came from): the engine probe's (FEATURE_CHECKS), else the source's. The
    second part is for a refusal's text: "(read from its source)" as before the probe, or what the probe asked."""
    probed, from_source = FEATURE_CHECKS[name]
    answer = probe_answer(probed)
    if answer is None:
        return from_source(), "(read from its source)"
    return answer, "(asked in its own Python)"


def feature(name: str) -> Any:
    """A feature check's answer (FEATURE_CHECKS): the engine probe's, else the source's."""
    return feature_answer(name)[0]


def feature_report() -> Dict[str, Tuple[Any, Any]]:
    """Each feature check's (probe answer or None, source answer), for `rigtune probe --engines`: where both answered
    and they differ, one of the two is wrong."""
    return {name: (probe_answer(probed), from_source()) for name, (probed, from_source) in FEATURE_CHECKS.items()}


def _probe_lists(field: str, value: str) -> bool:
    """False only when the engine probe answered `field` (spec_methods, kv_dtypes) without `value`: a version gate that
    holds still needs the engine to list what it gates."""
    answer = probe_answer(lambda f: None if getattr(f, field) is None else value in getattr(f, field))
    return answer is not False


# Which architectures the vLLM that will run can serve with their own MTP head, read from its source (importing vLLM
# costs seconds). config/speculative.py SpeculativeConfig.hf_config_override rewrites a draft of each model type below
# to "qwen3_5_mtp" (v0.29.0 lines 882-899, v0.33.0 987-1007), and model_executor/models/qwen3_5_mtp.py defines the
# draft class it then loads (Qwen3_5MTP line 215 / 216, Qwen3_5MoeMTP line 320 / 322). Not yet run on hardware.
VLLM_MTP_REWRITE = 'hf_config.model_type = "qwen3_5_mtp"'
VLLM_MTP_TEST = "if hf_config.model_type in ("
VLLM_MTP_TARGETS = (("qwen3_5", "Qwen3_5ForConditionalGeneration", "class Qwen3_5MTP("),
                    ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration", "class Qwen3_5MoeMTP("),
                    ("qwen3_5_text", "Qwen3_5ForCausalLM", "class Qwen3_5MTP("),
                    ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM", "class Qwen3_5MoeMTP("))
MTP_SOURCES = ("config/speculative.py", "model_executor/models/qwen3_5_mtp.py")


def mtp_archs_in(sources: dict) -> frozenset:
    """The architectures whose own MTP head vLLM source text (relative path -> text) shows it can load."""
    spec, model = (sources.get(rel, "") for rel in MTP_SOURCES)
    at = spec.find(VLLM_MTP_REWRITE)
    # The model types in the `hf_config.model_type in (...)` test whose branch makes the rewrite.
    opens = spec.rfind(VLLM_MTP_TEST, 0, at) if at >= 0 else -1
    if opens < 0:
        return frozenset()
    listed = set(re.findall(r'"([a-z0-9_]+)"', spec[opens:spec.find(")", opens + len(VLLM_MTP_TEST))]))
    return frozenset(arch for model_type, arch, cls in VLLM_MTP_TARGETS if model_type in listed and cls in model)


@functools.lru_cache(maxsize=1)
def _mtp_archs() -> frozenset:
    """mtp_archs_in for the vLLM the launch command will run; empty when it cannot be found."""
    root = _vllm_package()
    if root is None:
        return frozenset()
    sources = {}
    for rel in MTP_SOURCES:
        try:
            sources[rel] = root.joinpath(*rel.split("/")).read_text("utf-8")
        except OSError:
            continue
    return mtp_archs_in(sources)


# With an EAGLE-family proposer (MTP included) and no KV cache group marked as the draft's, vLLM 0.29's offloading
# scheduler counts every group as a draft group (distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py,
# line 231). A Qwen3.5 MTP layer's spec marks nothing (only MLA drafts set non_causal_multi_token_decode,
# v1/core/kv_cache_utils.py _annotate_eagle_groups), and vLLM's own note on that case says a linear-attention group
# then needs a lookup window "align-mode checkpointing never produces, so reuse drops to zero with no error"
# (_warn_if_unannotated_eagle_mamba, line 1863). v0.33.0 treats unmarked groups as non-draft instead and has no
# such line. Read from source; not run.
OFFLOAD_SCHEDULER = "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
OFFLOAD_EVERY_GROUP_DRAFT = "eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))"


@functools.lru_cache(maxsize=1)
def _offload_counts_every_group_as_draft() -> bool:
    """Whether the vLLM that will run has 0.29's every-group-is-a-draft fallback in its offloading scheduler."""
    root = _vllm_package()
    if root is None:
        return False
    try:
        return OFFLOAD_EVERY_GROUP_DRAFT in root.joinpath(*OFFLOAD_SCHEDULER.split("/")).read_text("utf-8")
    except OSError:
        return False


def _shm_free() -> Optional[int]:
    """Free bytes in /dev/shm, or None where there is none (not Linux)."""
    try:
        return shutil.disk_usage("/dev/shm").free if os.path.isdir("/dev/shm") else None
    except OSError:
        return None


def vllm_tier_config(cfg: Config, port: int) -> dict:
    """--kv-transfer-config for cfg.kv_tier: vLLM's native CPU tier (OffloadingConnector). Equivalent to
    --kv-offloading-size with the native backend (config/vllm.py VllmConfig._post_init_kv_transfer_config sets
    the same connector, kv_role and cpu_bytes_to_use), plus offload_prompt_only false: an answer is the next
    turn's prompt; with the default (prompt tokens only, v1/kv_offload/base.py) it is stored only when that
    turn arrives, by when the GPU may have evicted it. cpu_bytes_to_use is the total over every worker
    (v1/kv_offload/cpu/spec.py CPUOffloadingSpec); tiers are offered at tensor-parallel size 1 only. A "+disk"
    tier is refused, never launched as host-only (VLLM_NO_DISK)."""
    host_gib, disk_gib = cfg.tier_gib
    if disk_gib:
        raise ValueError(f"{cfg.kv_tier}: {VLLM_NO_DISK}")
    extra: dict = {"cpu_bytes_to_use": host_gib * GiB, "offload_prompt_only": False}
    return {"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": extra}


class VllmBackend(BaseBackend):
    name = "vllm"
    runtime_workspace_bytes = int(1.5 * GiB)  # CUDA context + CUDA graphs + activation workspace

    # ---- capability

    def facts(self, refresh: bool = False):
        if refresh:
            from rigtune.backends import facts

            facts.forget("vllm", _vllm_python(), _vllm_package())
            _engine_facts.cache_clear()
        return _engine_facts()

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        if hw.gpu is None or hw.gpu.vendor != "nvidia" or hw.gpu.cc < (7, 5):
            return False
        try:
            arch = load_arch(model).architecture
        except Exception as exc:
            logger.warning("could not read config for %s: %s", model.hf_id, exc)
            return True
        # The registry of the vLLM that will run, listed in its own Python by the engine probe. It can only add an
        # architecture, never take one away: the probe imports no vLLM plugins, which can register out-of-tree models,
        # and SGLang's registry skips any model module that fails to import, which one needing a GPU may do with none
        # visible. A model refused that way would never be tried, while one offered wrongly costs a failed launch.
        if probe_answer(lambda f: None if f.archs is None else arch in f.archs):
            return True
        live = vllm_registry_archs()
        if live is not None:
            return arch in live
        return arch in FALLBACK_ARCHS or arch == "unknown"

    # ---- prepare

    def precisions(self, hw: HardwareDescriptor) -> List[str]:
        cc = hw.gpu.cc if hw.gpu else (0, 0)
        out = ["bf16" if cc >= (8, 0) else "fp16"]
        # fp8 online weight quantisation: native on Ada/Hopper, Marlin on Ampere up to vLLM 0.28.
        if cc >= (8, 9) or (cc >= (8, 0) and _vllm_version() < FP8_ON_AMPERE_BROKEN_FROM):
            out.append("fp8")
        if cc >= (7, 5):  # pre-quantized 4-bit checkpoints: Marlin kernels on Ampere+, plain AWQ/GPTQ on Turing
            out += list(INT4_METHODS)
            out += list(INT8_METHODS)  # W8A8: vLLM's int8 kernels need compute capability 7.5
        return out

    def supported_quants(self, hw: HardwareDescriptor) -> List[str]:
        return self.precisions(hw)

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants: Optional[List[str]] = None) -> PreparedModel:
        arch = load_arch(model)
        weights, paths = hf_weight_options(model, arch.num_params or 0, list(quants or self.precisions(hw)),
                                           lambda p: 1 if p == "fp8" else dtype_bytes(p))
        return PreparedModel(spec=model, backend=self.name, arch=arch, hf_path=model.hf_id, hf_paths=paths,
                             weights_bytes=weights)

    # ---- optional search dimensions

    supports_tp = True

    def kv_dtypes(self, hw: HardwareDescriptor) -> List[str]:
        cc = hw.gpu.cc if hw.gpu else (0, 0)
        out: List[str] = []
        if cc >= (8, 9):  # Ada, Hopper: e4m3
            out.append("fp8")
        elif cc >= (8, 0) and importlib.util.find_spec("flashinfer") is not None:  # Ampere, via FlashInfer
            out.append(AMPERE_FP8_KV)
        if cc >= (8, 9) and _vllm_version() >= INT8_KV_FROM and _probe_lists("kv_dtypes", INT8_KV):
            out.append(INT8_KV)
        return out

    def batch_ladder(self) -> Tuple[int, ...]:
        return (16, 64, 256, 512)

    def spec_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                      all_methods: bool = False, min_batch: Optional[int] = None) -> List[Config]:
        # n-gram lookup on the GPU where vLLM has it, since the CPU proposer turns async scheduling off.
        specs = [speculative.ngram_gpu() if speculative.vllm_supports_ngram_gpu() else speculative.ngram()]
        if speculative.vllm_supports_suffix():
            specs.append(speculative.suffix())
        if model.drafts_mode == "verified":
            specs += [v.spec_decode for v in self._pair_variants(cfg, model, hw, all_methods)[0]]
        else:
            draft = speculative.draft_for(model.spec.hf_id)
            if draft and speculative.vllm_supports_draft():
                specs.append(speculative.draft(draft, speculative.DRAFT_TOKENS_VLLM))
        head = model.spec_repos.get("eagle3")
        if all_methods and head and speculative.vllm_supports_eagle3():  # the head and tokens SGLang is offered
            specs.append(speculative.eagle3(head, speculative.EAGLE3_TOKENS))
        out = [cfg.model_copy(update={"spec_decode": s, "spec_head_quant": None}) for s in specs if s != cfg.spec_decode]
        out = [v for v in out if self.known_failure(v) is None]
        return out + self._mtp_variants(cfg, model, hw, all_methods, min_batch)[0]

    def spec_notes(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                   all_methods: bool = False, min_batch: Optional[int] = None) -> List[str]:
        """What the search should say about the MTP settings for `cfg` (see _mtp_variants); where this card's
        search has the fp8_e5m2 KV cache, that MTP is never paired with it (known_failure); and why suffix decoding
        is not tried on a release that has it (speculative.suffix_refusal: on 14 Nov 2026 every suffix trial on an
        A40 pod failed to start with arctic-inference 0.1.1 installed, and each cost a launch)."""
        notes = self._mtp_variants(cfg, model, hw, all_methods, min_batch)[1]
        if model.drafts_mode == "verified":
            notes += self._pair_variants(cfg, model, hw, all_methods)[1]
        if speculative.vllm_suffix_release():
            if not speculative.vllm_supports_suffix():
                notes.append(f"vllm: suffix decoding not tried: {speculative.vllm_suffix_refusal()}")
            elif _suffix_caveat():
                notes.append(f"vllm: {_suffix_caveat()}")
        if all_methods and hw is not None and self._offers_mtp(cfg, model) and AMPERE_FP8_KV in self.kv_dtypes(hw):
            notes.append(f"vllm: MTP is not paired with the {AMPERE_FP8_KV} KV cache through FlashInfer on this card, "
                         "by analogy with EAGLE3, which shares its proposer path and is excluded after vLLM's draft "
                         "model failed so on an A40; unproven for MTP")
        return notes

    def find_pairs(self, prepared: PreparedModel, ctx: int = 4096, *, draft: Optional[str] = None,
                   hub: str = "on", hw: Optional[HardwareDescriptor] = None) -> List[str]:
        from rigtune.pairing import discover
        return discover(prepared, self.name, ctx, draft=draft, hub=hub,
                        version=self.version(hw) if hw else None)

    def _pair_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor], all_methods=False):
        from rigtune.pairing import eligible_pairs
        if hw is None or cfg.quant in PREQUANTIZED:
            return [], []
        out, notes = [], []
        if cfg.kv_dtype == AMPERE_FP8_KV:
            return [], ["vllm: verified pairs excluded from the measured failing fp8_e5m2 FlashInfer combination"]
        pairs = (eligible_pairs(model, "draft", cfg.ctx)
                 if speculative.vllm_supports_draft() and _probe_lists("spec_methods", "draft_model") else [])
        if all_methods and _vllm_version() >= (0, 29) and _probe_lists("spec_methods", "mtp"):
            pairs += eligible_pairs(model, "assistant", cfg.ctx)
        for pair in pairs:
            spec = (speculative.assist(pair["repo"], 1) if pair["kind"] == "assistant"
                    else speculative.draft(pair["repo"], speculative.DRAFT_TOKENS_VLLM))
            v = cfg.model_copy(update={"spec_decode": spec, "spec_head_quant": None})
            if estimate(hw, model, v, self.memory_model(hw)).feasible:
                out.append(v)
                notes.append(f"vllm: {pair['repo']} weights planned at target dtype; draft KV is unplanned")
            else:
                notes.append(f"vllm: {pair['repo']} does not fit with its draft weights at ctx {cfg.ctx}, batch {cfg.batch}")
        return out, notes

    def _mtp_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor], all_methods: bool,
                      min_batch: Optional[int]) -> Tuple[List[Config], List[str]]:
        """The checkpoint's own MTP head at each of speculative.MTP_TOKENS, under --speculative all and with the
        machine known (its extra states and the head's KV layer are planned in vllm_hybrid_kv), with notes. A setting
        that does not fit at `cfg`'s batch is tried at the largest batch that fits and still covers the workload's
        busiest level (`min_batch`), or not at all; either is noted (BaseBackend.own_head_fit)."""
        if not (all_methods and hw is not None and self._offers_mtp(cfg, model)):
            return [], []
        variants = [cfg.model_copy(update={"spec_decode": speculative.mtp(k), "spec_head_quant": None})
                    for k in speculative.MTP_TOKENS]
        facts = _engine_facts()
        if (_vllm_version() >= (0, 33) and hw.gpu is not None and hw.gpu.cc >= (8, 9)
                and "fp8" in self.precisions(hw) and facts is not None and facts.answered("spec_fields")
                and "draft_lm_head_quantization" in facts.spec_fields):
            variants.append(cfg.model_copy(update={"spec_decode": "mtp:1", "spec_head_quant": "fp8"}))
        variants = [v for v in variants if v.key() != cfg.key() and self.known_failure(v) is None]
        return self.own_head_fit(variants, model, hw, min_batch)

    def _offers_mtp(self, cfg: Config, model: PreparedModel) -> bool:
        """The checkpoint's own MTP head: its weights were found (find_second_models), the vLLM that will run can
        load it for this architecture (_mtp_archs), and the launch reads the checkpoint the head was found in, not a
        pre-quantized repository's."""
        return (speculative.MTP_KEY in model.spec_repos and cfg.quant not in PREQUANTIZED
                and model.arch.architecture in _mtp_archs())

    def known_failure(self, cfg: Config) -> Optional[str]:
        """Why `cfg` is expected to fail, when it is. On an A40, vLLM 0.29's draft model on the fp8 KV cache through
        FlashInfer (the Ampere route) failed most requests with "CUDA error: an illegal memory access", each working
        alone; those trials were measured and dropped as failures. An EAGLE3 head takes the same proposer path, so
        that pair is not tried at all (the draft-model pair is still tried, as before). Nor is an MTP head, which
        runs through the same EAGLE proposer (config/speculative.py SpeculativeConfig.use_eagle covers "mtp"). That is
        by analogy and unproven: no MTP trial has run with this cache; spec_notes says so in the profile, and the GPU
        validation in docs/benchmarks.md runs the pair once by hand."""
        spec = cfg.spec_decode or ""
        if spec.startswith("eagle3:") and cfg.kv_dtype == AMPERE_FP8_KV:
            return "EAGLE3 with the fp8 KV cache through FlashInfer (the draft model failed so on an A40)"
        if speculative.own_head(cfg.spec_decode) and cfg.kv_dtype == AMPERE_FP8_KV:
            return "MTP with the fp8 KV cache through FlashInfer (EAGLE3's proposer path, which failed so on an A40)"
        return None

    def find_second_models(self, prepared: PreparedModel) -> List[str]:
        """Under --speculative all: the verified EAGLE3 head for this model, the one SGLang gets, so the two engines
        are compared on one proposer. vLLM plans no memory for it, so its size is recorded but not needed. And the
        checkpoint's own MTP head, where this vLLM can load one for the architecture: its weights are already in the
        plan's weights, which load_arch takes from the Hub's safetensors total (vLLM skips them without MTP). For a
        local directory the parameter count is estimated from the config and leaves them out."""
        notes: List[str] = []
        if speculative.vllm_supports_eagle3() and speculative.eagle3_heads_for(prepared.spec.hf_id):
            try:
                found = speculative.find_eagle3_head(prepared.spec, prepared.arch)
            except Exception as exc:
                found = None
                notes.append(f"vllm: no EAGLE3 head tried, the Hub could not be asked ({exc})")
            if found:
                prepared.spec_repos["eagle3"] = found[0]
                if found[1]:
                    prepared.spec_bytes[found[0]] = int(found[1])
        if prepared.arch.architecture in _mtp_archs():
            notes += speculative.record_mtp_head(prepared, self.name)
        for n in notes:
            logger.warning(n)
        return notes

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        return model  # vLLM pulls HF weights itself at launch

    # ---- memory

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return self.calibrated_memory(hw, lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION), "gpu",
                                      hybrid_fn=lambda cfg, arch: vllm_hybrid_kv(
                                          cfg, arch, vllm_default_batched_tokens(hw)),
                                      extra_weights_fn=vllm_mtp_extra_weights)

    # ---- configs

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> List[Config]:
        ctxs = ctx_grid(model.arch.max_position_embeddings, min_ctx)
        out: List[Config] = []
        for quant in [q for q in self.precisions(hw) if q in model.weights_bytes]:
            for gmu in (0.80, 0.90, 0.95):
                for ctx in ctxs:
                    for batch in (16, 64, 256):
                        out.append(
                            Config(backend=self.name, quant=quant, ctx=ctx, batch=batch, gpu_memory_utilization=gmu)
                        )
        return out

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> Config:
        return Config(
            backend=self.name,
            quant=self.precisions(hw)[0],
            ctx=min(max(4096, min_ctx), model.arch.max_position_embeddings),
            batch=256,
            gpu_memory_utilization=0.90,
        )

    # ---- launch

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        args = [
            *vllm_serve_head(model.hf_paths.get(cfg.quant) or model.hf_path or model.spec.hf_id),
            # A stable name even when a 4-bit checkpoint repository is what gets loaded.
            "--served-model-name", model.spec.hf_id,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--max-model-len", str(cfg.ctx),
            "--max-num-seqs", str(cfg.batch),
        ]
        if cfg.gpu_memory_utilization is not None:
            args += ["--gpu-memory-utilization", f"{cfg.gpu_memory_utilization:.2f}"]
        if cfg.prefill_budget is not None:
            args += ["--max-num-batched-tokens", str(cfg.prefill_budget)]
        if cfg.quant == "fp8":
            args += ["--quantization", "fp8"]
        elif cfg.quant in ("bf16", "fp16"):
            args += ["--dtype", "bfloat16" if cfg.quant == "bf16" else "float16"]
        # awq / gptq / w8a8: the checkpoint's quantization_config selects the kernel; no flag needed.
        if cfg.kv_dtype != "auto":
            args += ["--kv-cache-dtype", cfg.kv_dtype]
        if cfg.prefix_cache is not None:
            args += ["--enable-prefix-caching" if cfg.prefix_cache else "--no-enable-prefix-caching"]
        if cfg.spec_decode:
            revision = model.spec.revision if cfg.quant not in PREQUANTIZED else None  # read by an MTP head only
            kind, second, _ = speculative.parse(cfg.spec_decode)
            if second:
                revision = model.spec_revisions.get(second)
            payload = speculative.vllm_config(cfg.spec_decode, revision, cfg.spec_head_quant)
            if second in model.spec_paths:
                payload["model"] = model.spec_paths[second]
            args += ["--speculative-config", json.dumps(payload)]
        if cfg.tp > 1:
            args += ["--tensor-parallel-size", str(cfg.tp)]
        if model.spec.revision and cfg.quant not in PREQUANTIZED:
            args += ["--revision", model.spec.revision]
        args += render_extra(cfg.extra, skip=("attention_backend",))
        env = {}
        # Some cache types need a particular attention backend (KV_ATTENTION); extra["attention_backend"]
        # forces one, so an ablation can keep the backend while removing the quantized cache.
        attention = cfg.extra.get("attention_backend") or KV_ATTENTION.get(cfg.kv_dtype)
        if attention:
            env["VLLM_ATTENTION_BACKEND"] = str(attention)  # vLLM 0.11
            if feature("attention_backend"):
                args += ["--attention-backend", str(attention)]
        # PCIe-only GPUs: peer-to-peer can hang at start-up inside containers. Measured on a pair of
        # A40s: NCCL's P2P path hangs at init, and with only that disabled the engine's custom
        # all-reduce (CUDA IPC, also P2P) hangs next. With both off it starts in under a minute.
        if cfg.tp > 1 and nvlink_between(list(range(cfg.tp))) is False:
            env["NCCL_P2P_DISABLE"] = "1"
            args += ["--disable-custom-all-reduce"]
        if self.report_reuse and feature("prompt_tokens_details"):
            args += ["--enable-prompt-tokens-details"]
        if cfg.kv_tier is not None:
            args += ["--kv-transfer-config", json.dumps(vllm_tier_config(cfg, port))]
        if cfg.text_only:  # set only by the retry after multimodal profiling failed at start-up
            args += ["--language-model-only"]
        return LaunchSpec(args=args, env=env)

    def launch_note(self) -> Optional[str]:
        return arctic_plugin_note()

    def text_only_refusal(self) -> Optional[str]:
        has, origin = feature_answer("language_model_only")
        if not has:
            return f"the vLLM that would run has no --language-model-only {origin}"
        return None

    # ---- KV-cache tier

    def tier_refusal(self, cfg: Config, model: PreparedModel) -> Optional[str]:
        features, origin = feature_answer("offload")
        if "cpu" not in features:
            return f"the vLLM that would run has no OffloadingConnector with offload_prompt_only {origin}"
        if cfg.tp > 1:
            return "not offered with tensor parallelism (not read for it, and never run)"
        if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            # config/vllm.py VllmConfig._verify_kv_transfer_compat refuses any KV connector with it.
            return "PYTORCH_CUDA_ALLOC_CONF sets expandable_segments:True, which vLLM refuses with connectors"
        if model.arch.is_hybrid and "hma" not in features:
            return ("this vLLM's offloading connector does not support the hybrid KV cache manager, which "
                    f"{model.arch.architecture} needs")
        if model.arch.linear_layers and cfg.prefix_cache is False:
            # distributed/kv_transfer/kv_connector/v1/offloading/config.py build_offloading_config asserts
            # "Hybrid models (e.g. Mamba+Attention) need --enable-prefix-caching".
            return "a model with linear-attention layers needs prefix caching on to offload"
        if model.arch.linear_layers and speculative.own_head(cfg.spec_decode) \
                and _offload_counts_every_group_as_draft():
            return ("this vLLM's offloading tier counts every KV cache group as MTP's, which leaves a "
                    "linear-attention model no reuse from the tier (read from its source)")
        return None

    def tier_disk_refusal(self) -> Optional[str]:
        return VLLM_NO_DISK

    def tier_host_cap(self) -> Optional[int]:
        return _shm_free() if "shm" in feature("offload") else None

    # Chunked-prefill token budgets to try. Small budgets interleave prefill with decode and protect
    # per-token latency; large ones finish long prompts in fewer steps and cut time to first token.
    PREFILL_BUDGETS = (2048, 8192, 16384)

    def prefill_variants(self, cfg: Config) -> List[Config]:
        # vLLM requires max_num_batched_tokens >= max_num_seqs.
        return [cfg.model_copy(update={"prefill_budget": b}) for b in self.PREFILL_BUDGETS
                if b >= cfg.batch and b != cfg.prefill_budget]

    def disagg_launch_spec(self, cfg: Config, model: PreparedModel, port: int, role: str,
                           kv_transfer_config: dict, gpu_index: int, side_channel_port: int) -> LaunchSpec:
        # One --kv-transfer-config per engine: the phases' connector, never a KV-cache tier besides it.
        spec = self.launch_spec(cfg.model_copy(update={"kv_tier": None}), model, port)
        spec.args += ["--kv-transfer-config", json.dumps(kv_transfer_config)]
        spec.env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "VLLM_NIXL_SIDE_CHANNEL_PORT": str(side_channel_port),
        })
        return spec

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(
            health_path="/health",
            model_name=model.spec.hf_id,  # --served-model-name
            tokenizer_id=model.spec.hf_id,
            gpu_ids=[hw.gpu.index] if hw.gpu else [],
            metrics_path="/metrics",  # vllm:prefix_cache_queries and vllm:prefix_cache_hits, on by default
        )
