"""SGLang backend (NVIDIA, compute capability >= 7.5)."""

from __future__ import annotations

import functools
import importlib.util
import logging
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from rigtune import kernels, kvtier, speculative
from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid, render_extra
from rigtune.backends.vllm import FALLBACK_ARCHS, PAGED_KV_FRACTION
from rigtune.hfconfig import dtype_bytes, load_arch
from rigtune.memory import HOST_TIER_RESERVE, MIN_SAFETY_MARGIN, HybridKV, MemoryModel, estimate
from rigtune.models import ArchInfo, Config, GiB, HardwareDescriptor, ModelSpec, PreparedModel
from rigtune.hardware import nvlink_between, sglang_python
from rigtune.quantized import INT4_METHODS, as_stored, hf_weight_options, stored_quant

logger = logging.getLogger(__name__)

# SGLang 0.5.19 defaults behind its hybrid-model pools (all read from source; not yet run on hardware).
MAMBA_FULL_MEMORY_RATIO = 0.9  # server_args.ServerArgs.mamba_full_memory_ratio
MAMBA_SLOTS_PER_REQUEST = 3  # mem_cache/kv_cache_configurator.MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
MAMBA_EXTRA_BUFFER_SLOTS = 2  # kv_cache_configurator.MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
MAMBA_EXTRA_BUFFER_SLOTS_LAZY = 1  # MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY (and _NO_OVERLAP)
SWA_FULL_TOKENS_RATIO = 0.8  # server_args.ServerArgs.swa_full_tokens_ratio
SWA_EVICTION_INTERVAL = 128  # environ.SGLANG_SWA_EVICTION_INTERVAL (line 637 at v0.5.19, 730 at v0.5.23)
# pool_configurator.SWAChunkCapPoolConfigurator (v0.5.19; compute_swa_request_cap at v0.5.23), under the default
# overlap scheduler.
SWA_CHUNKS_IN_FLIGHT = 2
# speculative_ngram_max_bfs_breadth's default, which arg_groups/speculative_hook.py::_handle_ngram makes the n-gram
# top-k (server_args.py:2324-2328 at v0.5.19, arg_groups/fields/spec.py:256-259 at v0.5.23).
NGRAM_BFS_BREADTH = 10
# Architectures whose sliding layers get SGLang's hybrid SWA pool (configs/model_config.is_hybrid_swa_model, the
# Gemma 4 entries and SWA_SINK_ARCHS). Gemma 2 and 3 are not among them, and arg_groups/model_overrides/
# gemma2_gemma3.py sets disable_hybrid_swa_memory for them: every layer keeps a full KV cache.
SGLANG_HYBRID_SWA_ARCHS = frozenset({
    "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration",
    "GptOssForCausalLM", "GraniteSWAForCausalLM", "GraniteMoeSWAForCausalLM",
})
# Architectures for which arg_groups/overrides._mamba_radix_cache_resolution can pick the "extra_buffer" strategy
# (overrides._MAMBA_EXTRA_BUFFER_ARCHS, with the default triton linear-attention backend): the Qwen3.5 family.
SGLANG_EXTRA_BUFFER_ARCHS = frozenset({
    "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM", "Qwen3NextForCausalLM",
})
# Layers of a Qwen3.5 MTP draft: configs/model_config.py ModelConfig._config_draft_model sets
# num_nextn_predict_layers = 1 (line 772 at v0.5.19, 1180 at v0.5.23) and models/qwen3_5_mtp.py
# Qwen3_5ForCausalLMMTP.__init__ builds its model with num_hidden_layers = 1 (both tags).
SGLANG_MTP_DRAFT_LAYERS = 1
# A draft runs at the target's context length (speculative/eagle_worker_v2.py line 182 at v0.5.19, 229 at v0.5.23),
# and configs/model_config.py ModelConfig._derive_context_length (lines 863-888 / 1291-1316) refuses a draft whose
# config derives a shorter one ("Target model's context_length (8192) is greater than the derived context_length
# (2048) ... set the env var SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1"), taking 2048 when the config names none
# (get_context_length, utils/hf_transformers/common.py at v0.5.23). lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B declares
# max_position_embeddings 2048 (read on the Hub, 21 Nov 2026), so above 2048 tokens it never started. With the
# variable set SGLang raises the draft's max_position_embeddings to the target's, as vLLM does for an EAGLE head
# unasked (config/speculative.py lines 1503-1511 at v0.33.0). Set only for these algorithms: a head reads the
# target's positions, while a draft model's own context limit is pairing's to check (context_too_short).
LONGER_CONTEXT_VAR = "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"  # environ.py 308 / 320, an EnvBool: "1" is true
# SGLang checks every draft's context against the target's (configs/model_config.py 1296-1319): a head's, and a DFlash
# or DSpark drafter's, which vLLM raises itself.
HEAD_ALGORITHMS = frozenset({"EAGLE3", "DFLASH", "DSPARK"})


# SGLang 0.5.23 raised the default prefill chunk of cards with 20 to 35 GiB (an RTX 4090, an L4) from 2048 to
# 4096 tokens: arg_groups/memory_hook.py::handle_gpu_memory_settings, line 111 at v0.5.23, line 81 at v0.5.19;
# every other tier is the same at both tags. Only those two releases were read, so 0.5.20 to 0.5.22 keep
# 0.5.19's rule, and so does a version not written as major.minor.patch (or none at all).
SGLANG_CHUNK_4096_FROM = (0, 5, 23)


def sglang_release(version: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """(major, minor, patch) of an SGLang version string such as "0.5.23" or "0.5.10.post1"; None otherwise."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def sglang_default_chunk(vram_total_bytes: int, version: Optional[str] = None) -> int:
    """chunked_prefill_size SGLang `version` picks from GPU memory (memory_hook.handle_gpu_memory_settings)."""
    mib = vram_total_bytes / 2**20
    if not mib:
        return 4096
    release = sglang_release(version)
    mid = 4096 if release is not None and release >= SGLANG_CHUNK_4096_FROM else 2048
    for limit, chunk in ((20 * 1024, 2048), (35 * 1024, mid), (60 * 1024, 4096), (160 * 1024, 8192)):
        if mib < limit:
            return chunk
    return 16384


def sglang_full_kv_reason(arch: ArchInfo) -> Optional[str]:
    """Why SGLang keeps a full KV cache on every layer of this hybrid layout, or None when it does not."""
    if arch.sliding_layers and arch.architecture not in SGLANG_HYBRID_SWA_ARCHS:
        return (f"SGLang keeps a full KV cache on every layer of {arch.architecture}: it is not one of the "
                "architectures it gives a sliding-window pool (and it disables that pool for Gemma 2 and 3)")
    return None


def sglang_mamba_slots_per_request(cfg: Config, arch: ArchInfo) -> int:
    """State slots SGLang holds per running request with the radix cache on (_calculate_mamba_ratio).

    Three, plus the extra buffer: _mamba_radix_cache_resolution resolves the "auto" strategy to "extra_buffer"
    when the overlap scheduler is on (the default) and the architecture supports it, which adds two slots
    (one for "extra_buffer_lazy", or without overlap); otherwise "no_buffer" and three. A strategy or
    disable_overlap_schedule passed through cfg.extra is honoured."""
    strategy = str(cfg.extra.get("mamba_radix_cache_strategy", "auto"))
    overlap = not cfg.extra.get("disable_overlap_schedule", False)
    if strategy == "auto":
        strategy = "extra_buffer" if overlap and arch.architecture in SGLANG_EXTRA_BUFFER_ARCHS else "no_buffer"
    if strategy == "extra_buffer":
        return MAMBA_SLOTS_PER_REQUEST + (MAMBA_EXTRA_BUFFER_SLOTS if overlap else MAMBA_EXTRA_BUFFER_SLOTS_LAZY)
    if strategy == "extra_buffer_lazy":
        return MAMBA_SLOTS_PER_REQUEST + MAMBA_EXTRA_BUFFER_SLOTS_LAZY
    return MAMBA_SLOTS_PER_REQUEST


def sglang_decode_alloc(spec: Optional[str]) -> int:
    """KV tokens a request may hold for its next decode step beyond the window, under the overlap scheduler.

    Without speculation, one page (1 token on CUDA). With it, the overlap allocator keeps two decode steps
    outstanding, 2 x mem_cache/allocation_sizing.py::get_alloc_len_per_decode = 2 x max(steps x top-k, draft
    tokens) (v0.5.19 line 17, v0.5.23 line 28). For an EAGLE3 head or a draft model, a chain of k steps at top-k 1
    with k + 1 draft tokens, that is 2(k + 1); for n-gram, top-k is NGRAM_BFS_BREADTH and steps draft // top-k
    (at least 1; speculative_hook.py::_handle_ngram), so 24 for its 12 draft tokens."""
    draft = sglang_draft_tokens(spec)
    if not draft:
        return 1
    kind, _, k = speculative.parse(spec or "")
    if kind == "ngram":
        steps_x_topk = max(1, draft // NGRAM_BFS_BREADTH) * NGRAM_BFS_BREADTH
    else:
        steps_x_topk = k  # top-k 1
    return 2 * max(steps_x_topk, draft)


def sglang_hybrid_kv(cfg: Config, arch: ArchInfo, vram_total_bytes: int, version: Optional[str] = None) -> HybridKV:
    """SGLang 0.5.19's and 0.5.23's pools for hybrid models, read from source (not yet run on hardware). Sliding
    layers only count for SGLANG_HYBRID_SWA_ARCHS; for others MemoryModel.full_kv_fn keeps every layer full.

    Radix cache on (the default; `--disable-radix-cache` when cfg.prefix_cache is False):
    - Linear layers: mem_cache/kv_cache_configurator._handle_max_mamba_cache gives the mamba pool
      ratio / (1 + ratio) of the cache budget (mamba_full_memory_ratio 0.9) and resolve_max_num_reqs caps
      running requests at slots // sglang_mamba_slots_per_request (5 for Qwen3.5 under the defaults). So
      cfg.batch needs that many slots per request (+1 padding slot) inside that share; with fewer, SGLang
      lowers max_running_requests ("max_running_requests is capped to N by the mamba state cache").
    - Sliding layers: model_executor/pool_configurator.HybridSWAPoolConfigurator sizes the SWA pool at
      swa_full_tokens_ratio (0.8) x the full pool's tokens, whatever the context.
    Radix cache off, with --max-running-requests (always passed):
    - Linear layers: max_mamba_cache_size = max_running_requests, plus the padding slot.
    - Sliding layers: SWAChunkCapPoolConfigurator.is_applicable holds (chunked_prefill_size is always
      resolved), so the SWA pool is (window + eviction interval x D + 1 page + the next decode allocation)
      x requests + 2 prefill chunks + 1 page (pool_configurator.py: SWAChunkCapPoolConfigurator.__init__ at
      v0.5.19, compute_swa_request_cap at v0.5.23, where the class is SWARequestCapPoolConfigurator; same formula).
      D is the draft tokens, 1 without speculation; the decode allocation is sglang_decode_alloc. So without
      speculation, window + eviction interval + 2 pages. The default chunk depends on the card and the release
      (sglang_default_chunk). With the overlap scheduler off (cfg.extra) SGLang holds one chunk and one decode
      step, fewer than planned.
    Speculative decoding (D = --speculative-num-draft-tokens, as speculative.sglang_config passes it):
    _handle_max_mamba_cache also reserves an intermediate state per draft token for every request slot,
    per_req x (requests + 1) x D, inside the mamba share with the radix cache (the joint solve for K) and beside it
    without. ReplaySSM, which would replace it, is off by default (server_args enable_linear_replayssm_spec).
    SGLang's GDN conv state does not grow with D (configs/mamba_utils.Mamba2StateShape.create: conv_kernel - 1).
    A second model's own KV cache is not counted, as for dense models (see spec_fraction). The checkpoint's own MTP
    head is: model_executor/pool_configurator.py DefaultPoolConfigurator, which sizes these models' KV pool, scales
    its cell by 1 + draft layers / full-attention layers for EAGLE (NEXTN resolves to it), with the draft's layers
    read as num_nextn_predict_layers (v0.5.19 lines 183-222, v0.5.23 lines 259-314; only with a draft path, which
    sglang_config passes). _config_draft_model sets that to 1 for a Qwen3.5 head, and Qwen3_5ForCausalLMMTP builds
    one layer whatever the checkpoint declares (mtp_config.num_hidden_layers = 1), so one full layer is added. For
    another family the count is the one SGLang said its draft reads (SglangBackend._own_head_notes puts it in
    arch.mtp_layers), else the checkpoint's.
    """
    draft = sglang_draft_tokens(cfg.spec_decode)
    full_layers = None
    if speculative.own_head(cfg.spec_decode) and arch.full_attention_layers:
        full_layers = arch.full_attention_layers + (arch.mtp_layers or SGLANG_MTP_DRAFT_LAYERS)
    if cfg.prefix_cache is not False:
        slots = sglang_mamba_slots_per_request(cfg, arch) * cfg.batch + 1
        return HybridKV(state_slots=slots + draft * (cfg.batch + 1),
                        swa_full_tokens_ratio=SWA_FULL_TOKENS_RATIO,
                        state_share=MAMBA_FULL_MEMORY_RATIO / (1 + MAMBA_FULL_MEMORY_RATIO),
                        full_layers=full_layers)
    swa_pool = None
    if arch.sliding_layers and arch.full_attention_layers:
        page = 1  # CUDA default (arg_groups/overrides.py page_size resolution)
        chunk = cfg.prefill_budget or sglang_default_chunk(vram_total_bytes, version)
        per_request = (arch.sliding_window or 0) + SWA_EVICTION_INTERVAL * max(1, draft) + page \
            + (sglang_decode_alloc(cfg.spec_decode) if draft else page)
        swa_pool = per_request * cfg.batch + SWA_CHUNKS_IN_FLIGHT * chunk + page
    return HybridKV(state_slots=(cfg.batch + 1) * (1 + draft), swa_pool_tokens=swa_pool, full_layers=full_layers)


def sglang_draft_tokens(spec: Optional[str]) -> int:
    """--speculative-num-draft-tokens for a spec: k for n-gram, k + 1 for a second model or the checkpoint's own MTP
    head (sglang_config)."""
    if not spec:
        return 0
    kind, _, k = speculative.parse(spec)
    return k if kind == "ngram" else k + 1


def sglang_registry_archs() -> Optional[set]:
    if importlib.util.find_spec("sglang") is None:
        return None
    try:
        from sglang.srt.models.registry import ModelRegistry  # type: ignore

        return set(ModelRegistry.get_supported_archs())
    except Exception as exc:
        logger.debug("sglang registry unavailable: %s", exc)
        return None


# Which speculative methods an SGLang release has, told by its argument parser's source rather than by importing
# it (seconds, and often a separate environment: $SGLANG_PYTHON). From the release sources:
#   0.5.2:  --speculative-algorithm choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE"]; no n-gram.
#   0.5.3:  adds "NGRAM" and the speculative_ngram_* arguments.
#   0.5.19: no choices list; the help reads "Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK",
#           and the n-gram settings are applied in srt/arg_groups/speculative_hook.py.
# What rigtune relies on for a second model (it always runs the V2 overlap worker, draft weights load before the
# KV pools are sized, the draft's KV is counted inside the fraction) was read in 0.5.19 only, so EAGLE3 and
# STANDALONE are offered where the source has 0.5.19's DSPARK; which release first had it was not checked.
# Older releases with n-gram get n-gram only, and releases without STANDALONE (before 0.5.2) nothing.
SGLANG_SPEC_FROM = "STANDALONE"
SGLANG_NGRAM_MARKER = "speculative_ngram_max_bfs_breadth"
SGLANG_SECOND_MODEL_MARKER = "DSPARK"


def spec_methods_in(source: str) -> FrozenSet[str]:
    """rigtune's spec kinds ("ngram", "eagle3", "draft") that SGLang argument-parser source text shows."""
    if SGLANG_SPEC_FROM not in source:
        return frozenset()
    out = {"ngram"} if SGLANG_NGRAM_MARKER in source else set()
    if SGLANG_SECOND_MODEL_MARKER in source:
        out |= {"eagle3", "draft"}
    return frozenset(out)


def _sglang_package_dir(python: str) -> Optional[Path]:
    """Where `python` would import sglang from, found without importing it."""
    if python == sys.executable:
        spec = importlib.util.find_spec("sglang")
        locs = list(spec.submodule_search_locations or []) if spec else []
        return Path(locs[0]) if locs else None
    code = ("import importlib.util as u; s = u.find_spec('sglang'); "
            "print(list(s.submodule_search_locations or [''])[0] if s else '')")
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60)
    except Exception as exc:
        logger.debug("%s could not locate sglang: %s", python, exc)
        return None
    found = out.stdout.strip() if out.returncode == 0 else ""
    return Path(found) if found else None


@functools.lru_cache(maxsize=1)
def _engine_facts():
    """What the SGLang that would be launched says it offers (rigtune.backends.facts), asked in its own Python
    ($SGLANG_PYTHON, or this one) once per install; None when probing is off or SGLang cannot be found."""
    from rigtune.backends import facts

    python = sglang_python()
    return facts.cached_facts("sglang", python, _sglang_package_dir(python) if python else None)


def _server_args_source(pkg: Path) -> str:
    """srt/server_args.py, plus srt/arg_groups/**/*.py where a release has split its arguments out (0.5.23 declares
    them in arg_groups/fields/ and arg_groups/field_order.py)."""
    files = [pkg / "srt" / "server_args.py"] + sorted((pkg / "srt" / "arg_groups").rglob("*.py"))
    texts = []
    for f in files:
        try:
            texts.append(f.read_text("utf-8"))
        except OSError:
            continue
    return "\n".join(texts)


# The flag each block drafter is launched with (speculative.sglang_config), as SGLang 0.5.23 declares it
# (arg_groups/fields/spec.py lines 70, 78). The flags came with the drafters themselves (DFlash in April, DSpark in July
# 2026), so 0.5.19 has them too; what rigtune plans by was read at 0.5.23 only: the draft pool budgeted from its own
# attention geometry (mem_cache/draft_kv_budget.py, added 24 Oct 2026, four days before 0.5.23), its weights inside
# the fraction and its logits buffer outside it. So the drafters are offered where that file is as well.
SGLANG_BLOCK_FLAGS = {"dflash": "speculative_dflash_block_size", "dspark": "speculative_dspark_block_size"}
SGLANG_DRAFT_POOL_MARKER = ("srt", "mem_cache", "draft_kv_budget.py")


def block_drafters_in(source: str, draft_pool_read: bool = True) -> FrozenSet[str]:
    """The block drafter kinds whose launch flag SGLang argument-parser source text declares, in a release whose draft
    memory was read (`draft_pool_read`: SGLANG_DRAFT_POOL_MARKER present); none otherwise."""
    if not draft_pool_read:
        return frozenset()
    return frozenset(kind for kind, flag in SGLANG_BLOCK_FLAGS.items() if flag in source)


@functools.lru_cache(maxsize=1)
def sglang_block_drafters() -> FrozenSet[str]:
    """The DFlash and DSpark kinds the SGLang that would be launched takes rigtune's flags for, in a release whose
    draft memory was read, from its source without importing it; empty when SGLang is not found."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    if not pkg:
        return frozenset()
    return block_drafters_in(_server_args_source(pkg), pkg.joinpath(*SGLANG_DRAFT_POOL_MARKER).is_file())


def sglang_drafter_weights(cfg: Config, model: PreparedModel) -> int:
    """Bytes of a DFlash or DSpark drafter's weights, inside --mem-fraction-static: SGLang 0.5.23 builds the draft
    worker before it sizes the KV pools from what is left of the fraction (managers/scheduler.py lines 1130-1137;
    kv_cache_configurator.py 2990-3060, "draft weights are now counted"). Both take the target's embedding and output
    head (models/dflash.py 708-716; models/dspark.py 550-555), so their own bytes are all. 0 for any other setting."""
    from rigtune.backends.vllm import block_drafter

    found = block_drafter(cfg, model)
    return int(model.spec_bytes.get(found[0]) or 0) if found is not None else 0


def sglang_drafter_layers(cfg: Config, model: PreparedModel) -> int:
    """A DFlash or DSpark drafter's own KV pool, which SGLang 0.5.23 gives a slot per target token at its exact size,
    its KV heads x (K + V head dims) x its layers (pool_configurator.py lines 316-338; dflash_utils.py 121-175),
    planned as that many of the target's full-attention layers, as vLLM's is (vllm_drafter_layers)."""
    from rigtune.backends.vllm import vllm_drafter_layers

    return vllm_drafter_layers(cfg, model)


def sglang_logits_bytes(spec: Optional[str], model: PreparedModel, batch: Optional[int]) -> int:
    """The fp32 logits buffer a DFlash or DSpark launch allocates for its CUDA graphs after the KV pools are sized,
    outside the fraction: batch x draft tokens x vocabulary x 4 B (graph_shared_output.py lines 61-70; model_runner.py
    905-930; the capture batch is at most the request pool's, base_cuda_graph_runner.py 77-97). 467 MB for
    Qwen3-8B's DFlash drafter at SGLang's default 48 requests. 0 for any other setting."""
    if not spec or not batch:
        return 0
    kind, _, k = speculative.parse(spec)
    if kind not in speculative.BLOCK_DRAFTERS:
        return 0
    tokens = k + 1  # DFlash: the last accepted token and k drafts; DSpark: gamma + 1
    return batch * tokens * (model.arch.vocab_size or 0) * 4


# DFlash's sampling rehearsal (0.5.23 speculative/dflash_worker_v2.py: prewarm_sampling, line 799), run after CUDA-graph
# capture, so outside the fraction: a fresh fp32 (batch x block, vocabulary) matrix (line 825) and the dense top-p
# path's two more of that size (dflash_utils.py: logits over the temperature, then the softmax, line 1094). The
# batch is min(max_decode_logits_rows // block, max_running_requests) (line 771): rigtune's batch. Measured on an A40
# and an A6000 (23 Nov 2026, Qwen3-8B, 64 requests): its logged need is 1.79 GiB, the input matrix and a peak of two
# more (1,302,737,408 B). SGLang PR #52548 (not merged) rehearses on the shared logits buffer, one matrix less, but
# not under a deferred TP gather or DP attention, which this cannot see, so a release with it is still counted at 3:
# one matrix too many, never too few. DSpark's worker inherits BaseSpecWorker.prewarm_sampling (base_spec_worker.py
# line 240), which allocates nothing (model_runner.py line 1165), and 0.5.19 has no rehearsal.
SGLANG_DFLASH_WORKER = ("srt", "speculative", "dflash_worker_v2.py")


def rehearsal_matrices_in(source: Optional[str]) -> int:
    """How many full-vocabulary fp32 matrices DFlash's start-up rehearsal holds at once in SGLang's
    dflash_worker_v2.py source text: 0 without a rehearsal, 3 with one."""
    return 3 if source and "def prewarm_sampling" in source else 0


@functools.lru_cache(maxsize=1)
def sglang_rehearsal_matrices() -> int:
    """rehearsal_matrices_in for the SGLang that would be launched; 0 when it is not found."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    if not pkg:
        return 0
    try:
        return rehearsal_matrices_in(pkg.joinpath(*SGLANG_DFLASH_WORKER).read_text("utf-8"))
    except OSError:
        return 0


def sglang_rehearsal_bytes(spec: Optional[str], model: PreparedModel, batch: Optional[int]) -> int:
    """What DFlash's sampling rehearsal takes outside the fraction at start-up, on top of the logits buffer: its
    matrices x batch x block x vocabulary x 4 B (1.74 GiB for Qwen3-8B's DFlash drafter at 64 requests on 0.5.23).
    0 for any other setting."""
    if not spec or not batch or speculative.parse(spec)[0] != "dflash":
        return 0
    return sglang_rehearsal_matrices() * sglang_logits_bytes(spec, model, batch)


@functools.lru_cache(maxsize=1)
def sglang_spec_methods() -> FrozenSet[str]:
    """rigtune's spec kinds ("ngram", "eagle3", "draft") the SGLang that would be launched offers. Empty when
    SGLang is not found or predates 0.5.2."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    return spec_methods_in(_server_args_source(pkg)) if pkg else frozenset()


# Which architectures the SGLang that would be launched can serve with their own MTP head, read from its source
# like the methods above. NEXTN must be an algorithm its arguments know (0.5.19's server_args.py help text, line
# 2072; 0.5.23's arg_groups/speculative_hook.py, line 109), and, as for an EAGLE3 head, the release must be one
# whose second-model behaviour was read (SGLANG_SECOND_MODEL_MARKER). configs/model_config.py must rewrite a draft of
# the architecture to Qwen3_5ForCausalLMMTP (ModelConfig._config_draft_model, line 771 at v0.5.19, 1179 at v0.5.23,
# in the branch of an `architectures[0] in [...]` test that lists the architecture), and models/qwen3_5_mtp.py must
# define it (line 84 / 94).
SGLANG_MTP_ALGORITHM = "NEXTN"
SGLANG_MTP_REWRITE = 'architectures[0] = "Qwen3_5ForCausalLMMTP"'
SGLANG_MTP_CLASS = "class Qwen3_5ForCausalLMMTP("
SGLANG_MTP_TEST = "architectures[0] in ["


def mtp_archs_in(args_source: str, model_config_source: str, mtp_model_source: str) -> FrozenSet[str]:
    """The architectures whose own MTP head SGLang source text shows it can load: its argument parser's, its
    srt/configs/model_config.py's and its srt/models/qwen3_5_mtp.py's."""
    if SGLANG_MTP_ALGORITHM not in args_source or SGLANG_SECOND_MODEL_MARKER not in args_source:
        return frozenset()
    at = model_config_source.find(SGLANG_MTP_REWRITE)
    if at < 0 or SGLANG_MTP_CLASS not in mtp_model_source:
        return frozenset()
    # The names in the `architectures[0] in [...]` test whose branch makes the rewrite, not the names anywhere.
    opens = model_config_source.rfind(SGLANG_MTP_TEST, 0, at)
    if opens < 0:
        return frozenset()
    closes = model_config_source.find("]", opens + len(SGLANG_MTP_TEST))
    listed = set(re.findall(r'"([A-Za-z0-9_]+)"', model_config_source[opens:closes]))
    return frozenset(a for a in speculative.MTP_ARCHS if a in listed)


# MTP drafts SGLang runs on its multi-layer EAGLE worker, which builds one model runner per step, each with its own
# embedding and output head and its own head layer (managers/tp_worker.py _init_multi_layer_eagle_model_runners,
# lines 502-519 at v0.5.23). The worker is turned on for MiMo-V2 (arg_groups/model_overrides/mimo_v2.py, lines 23-25 at
# v0.5.23, 26 at v0.5.19), GigaChat-3.5 (arg_groups/model_overrides/gigachat35.py 18-22, v0.5.23 only) and Step-3.5
# and 3.7 (arg_groups/overrides.py 346-350 at v0.5.23, 469 at v0.5.19), and Inkling's banded heads refuse to run
# without it (models/inkling.py 1792-1797; own_head.HeadAnswer.banded; every Inkling checkpoint on the Hub is banded).
# rigtune plans one runner, so these are not offered on SGLang until their layout is planned. The targets are listed
# too, by configs/model_config.py's own names (MIMO_V2_MODEL_ARCHS, line 74 at v0.5.23), for an answer that names no
# draft class (SGLang could not load the config).
SGLANG_MULTI_LAYER_DRAFTS = frozenset({"MiMoV2MTP", "GigaChat35ForCausalLMNextN", "Step3p5MTP"})
SGLANG_MULTI_LAYER_TARGETS = frozenset({"MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "GigaChat35ForCausalLM",
                                        "Step3p5ForCausalLM", "Step3p7ForConditionalGeneration",
                                        "InklingForConditionalGeneration"})


@functools.lru_cache(maxsize=32)
def _own_head(model_id: str, revision: Optional[str]):
    """Whether the SGLang that would be launched has an MTP draft for this model's architecture: asked of it
    (rigtune.backends.own_head, which builds the draft's ModelConfig as a launch does and reads what
    _config_draft_model renamed it to), with its model registry from the install probe. Until 20 Nov 2026 only the
    Qwen3.5 family was offered, from sglang_mtp_archs; that is still what is read when SGLang is not asked."""
    from rigtune.backends import own_head

    python = sglang_python()
    facts = _engine_facts()
    registered = facts.archs if facts is not None and facts.answered("archs") else None
    return own_head.ask("sglang", python, _sglang_package_dir(python) if python else None, model_id, revision,
                        registered)[0]


@functools.lru_cache(maxsize=1)
def sglang_mtp_archs() -> FrozenSet[str]:
    """mtp_archs_in for the SGLang that would be launched; empty when it cannot be found. What is read when SGLang is
    not asked (_own_head: probing off, or its Python not found), as before it was asked."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    if not pkg:
        return frozenset()
    return mtp_archs_in(_server_args_source(pkg), _package_text(pkg, "srt", "configs", "model_config.py"),
                        _package_text(pkg, "srt", "models", "qwen3_5_mtp.py"))


@functools.lru_cache(maxsize=1)
def _cache_report_flag() -> bool:
    """Whether the SGLang that would be launched takes --enable-cache-report, which puts cached_tokens in each
    reply's usage (0.5.19 has it in srt/server_args.py); read from that interpreter's source, as above."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    return bool(pkg) and "enable_cache_report" in _server_args_source(pkg)


# SGLang's hierarchical cache (HiCache), read in 0.5.19 and not yet run on hardware:
#   "hicache":  --enable-hierarchical-cache and --hicache-size (srt/server_args.py): a host pool of that many
#               decimal GB per rank (mem_cache/pool_host/base.py HostKVCache: host_size * 1e9 //
#               size_per_token), pinned, split between a hybrid model's pools by their bytes
#               (hybrid_cache/hybrid_pool_assembler.py _split_hicache_size). A pool smaller than the device
#               pool only warns there.
#   "file":     --hicache-storage-backend file, whose directory comes from
#               SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR (srt/environ.py; mem_cache/hicache_storage.HiCacheFile
#               writes /tmp/hicache without it, so without the variable no disk tier is offered).
#   "file_cap": SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE bounds that directory
#               (mem_cache/storage/file/lru_file_evictor.LRUFileEvictor, which evicts least recently used
#               files past it; unbounded by default). A disk tier is offered only with it, always set.
# HiCache refuses --disable-radix-cache (arg_groups/kv_cache_hook.py), and checks host RAM itself: it raises
# unless psutil's available RAM less HICACHE_HOST_MEMORY_RESERVE_BYTES covers the pool (pool_host/base.py).
SGLANG_HOST_RESERVE = 10 * GiB  # mem_cache/pool_host/base.py HICACHE_HOST_MEMORY_RESERVE_BYTES
FILE_DIR_VAR = "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"
FILE_CAP_VAR = "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE"


def hicache_features_in(args_source: str, environ_source: str) -> FrozenSet[str]:
    """The HiCache features ("hicache", "file", "file_cap") SGLang source text shows: its argument parser's,
    and srt/environ.py's."""
    out = set()
    if "enable_hierarchical_cache" in args_source and "hicache_size" in args_source:
        out.add("hicache")
        if "hicache_storage_backend" in args_source and FILE_DIR_VAR in environ_source:
            out.add("file")
            if FILE_CAP_VAR in environ_source:
                out.add("file_cap")
    return frozenset(out)


def _package_text(pkg: Path, *parts: str) -> str:
    try:
        return pkg.joinpath(*parts).read_text("utf-8")
    except OSError:
        return ""


@functools.lru_cache(maxsize=1)
def _hicache_features() -> FrozenSet[str]:
    """hicache_features_in for the SGLang that would be launched; empty when it cannot be found."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    if not pkg:
        return frozenset()
    return hicache_features_in(_server_args_source(pkg), _package_text(pkg, "srt", "environ.py"))


TIER_DETAILS_FIELD = "return_cached_tokens_details"


@functools.lru_cache(maxsize=1)
def _tier_details_flag() -> bool:
    """Whether the SGLang that would be launched takes return_cached_tokens_details on a chat request:
    0.5.19's srt/entrypoints/openai/protocol.py ChatCompletionRequest has it, and serving_chat.py then streams
    a chunk with no choices whose sglext.cached_tokens_details holds {device, host, storage} (storage only
    with a storage backend). It needs no server flag. Read from that interpreter's source, as above."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    protocol = _package_text(pkg, "srt", "entrypoints", "openai", "protocol.py") if pkg else ""
    return TIER_DETAILS_FIELD in protocol


# rigtune's spec kinds and the SpeculativeAlgorithm names each needs (speculative/spec_info.py; members at v0.5.19 and
# v0.5.23: DFLASH, DSPARK, EAGLE, EAGLE3, FROZEN_KV_MTP, STANDALONE, NGRAM, NONE, and UNO at v0.5.23). The same gates as
# spec_methods_in: nothing before STANDALONE, and a second model only where DSPARK marks a release whose second-model
# behaviour was read.
SGLANG_ALGORITHMS = {"ngram": (SGLANG_SPEC_FROM, "NGRAM"),
                     "eagle3": (SGLANG_SPEC_FROM, "EAGLE3", SGLANG_SECOND_MODEL_MARKER),
                     "draft": (SGLANG_SPEC_FROM, SGLANG_SECOND_MODEL_MARKER)}


def spec_methods_from(algorithms: Optional[List[str]]) -> Optional[FrozenSet[str]]:
    """rigtune's spec kinds the engine probe's list of SGLang's algorithms shows; None when it gave none."""
    if algorithms is None:
        return None
    have = set(algorithms)
    return frozenset(kind for kind, needs in SGLANG_ALGORITHMS.items() if have.issuperset(needs))


def hicache_features_from(flags: Optional[dict], env_vars: Optional[List[str]]) -> Optional[FrozenSet[str]]:
    """hicache_features_in's answer from the engine probe's flags and environment variables; None unless it gave
    both."""
    if flags is None or env_vars is None:
        return None
    out = set()
    if "--enable-hierarchical-cache" in flags and "--hicache-size" in flags:
        out.add("hicache")
        if "--hicache-storage-backend" in flags and FILE_DIR_VAR in env_vars:
            out.add("file")
            if FILE_CAP_VAR in env_vars:
                out.add("file_cap")
    return frozenset(out)


def probe_answer(read: Callable[[Any], Any]) -> Any:
    """`read` applied to what the SGLang that would be launched told the engine probe (rigtune.backends.facts), or
    None when it was not asked or did not answer that question."""
    try:
        facts = _engine_facts()
        return None if facts is None else read(facts)
    except Exception as exc:  # never a reason to fail: the source check answers instead
        logger.debug("no engine probe answer: %s", exc)
        return None


# Each feature check rigtune makes of SGLang: (the engine probe's answer, None when it has none; the check that reads
# SGLang's source, which answers otherwise). SGLang 0.5.23 moved its argument definitions to srt/arg_groups/fields/,
# which the source checks do not read: they still answer only because arg_groups/field_order.py lists the names. The
# probe asks the parser `sglang serve` builds.
# The source checks are looked up when asked, so a test that replaces one replaces it here too.
FEATURE_CHECKS: Dict[str, Tuple[Callable[[Any], Any], Callable[[], Any]]] = {
    "spec_methods": (lambda f: spec_methods_from(f.spec_methods), lambda: sglang_spec_methods()),
    "cache_report": (lambda f: f.has_flag("--enable-cache-report"), lambda: _cache_report_flag()),
    "hicache": (lambda f: hicache_features_from(f.flags, f.env_vars), lambda: _hicache_features()),
    "tier_details": (lambda f: None if f.request_fields is None else TIER_DETAILS_FIELD in f.request_fields,
                     lambda: _tier_details_flag()),
}


def feature_answer(name: str) -> Tuple[Any, str]:
    """(a feature check's answer, where it came from): the engine probe's (FEATURE_CHECKS), else the source's."""
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


def sglang_tier_launch(cfg: Config, port: int) -> Tuple[List[str], dict]:
    """Arguments and environment for cfg.kv_tier: a host pool of the tier's GiB in SGLang's decimal GB
    (rounded down), and for "+disk" the file backend in this engine's own directory (kvtier.trial_dir), always
    capped at the disk tier's size, which the planner checked against the free disk. Port 0 (a profile's
    recorded launch) writes kvtier.DIR_PLACEHOLDER for the directory; serving fills in its own. A disk tier
    that would land on a network filesystem is refused, never launched as host-only."""
    host_gib, disk_gib = cfg.tier_gib
    args = ["--enable-hierarchical-cache", "--hicache-size", str(int(host_gib * GiB // 10**9))]
    env = {}
    if disk_gib:
        reason = kvtier.refusal()
        if reason:
            raise ValueError(f"{cfg.kv_tier}: {reason}")
        args += ["--hicache-storage-backend", "file"]
        env[FILE_DIR_VAR] = str(kvtier.trial_dir(port)) if port else kvtier.DIR_PLACEHOLDER
        env[FILE_CAP_VAR] = str(disk_gib * GiB)
    return args, env


def head_context_env(args: List[str], ctx: int, target_ctx: int) -> Dict[str, str]:
    """The environment that lets a head launched by `args` run at the target's context `ctx` (LONGER_CONTEXT_VAR),
    read from the launch arguments, so a comparison row that names the head in its own flags gets it too. Empty
    unless the algorithm is a head's, and unless `ctx` is within the target's own limit as SGLang derives it
    (`target_ctx`: ArchInfo.sglang_context, else max_position_embeddings): the variable lifts the target's check as
    well, and must only ever lift the head's."""
    algorithms = [args[i + 1].upper() for i, a in enumerate(args[:-1]) if a == "--speculative-algorithm"]
    if not algorithms or algorithms[-1] not in HEAD_ALGORITHMS or ctx > target_ctx:
        return {}
    return {LONGER_CONTEXT_VAR: "1"}


def sglang_graph_bytes(cfg: Config, vram_bytes: int, version: Optional[str]) -> int:
    """CUDA-graph memory `cfg` adds by raising SGLang's decode cap (kernels.sglang_graph_extra_bytes); 0 for every
    config without that setting, before anything else is looked at."""
    if cfg.kernels is None:
        return 0
    # A draft, an EAGLE3 head, an assistant or the checkpoint's own MTP head: each runs a second worker that
    # captures decode graphs of its own. The lookup proposers (n-gram, suffix) do not.
    second = bool(cfg.spec_decode) and speculative.parse(cfg.spec_decode)[0] not in speculative.LOOKUP
    return kernels.sglang_graph_extra_bytes(kernels.parse(cfg.kernels), vram_bytes / (1024 * 1024), cfg.batch,
                                            cfg.tp, version, second_model=second)


class SglangBackend(BaseBackend):
    name = "sglang"
    runtime_workspace_bytes = int(2.0 * GiB)  # CUDA graphs for many batch sizes + radix cache metadata

    def facts(self, refresh: bool = False):
        if refresh:
            from rigtune.backends import facts

            python = sglang_python()
            facts.forget("sglang", python, _sglang_package_dir(python) if python else None)
            _engine_facts.cache_clear()
            _own_head.cache_clear()
        return _engine_facts()

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        if hw.gpu is None or hw.gpu.vendor != "nvidia" or hw.gpu.cc < (7, 5):
            return False
        try:
            arch = load_arch(model).architecture
        except Exception as exc:
            logger.warning("could not read config for %s: %s", model.hf_id, exc)
            return True
        # The registry of the SGLang that would be launched, listed in its own Python: the usual case, since rigtune
        # cannot import it when it lives behind $SGLANG_PYTHON. It can only add an architecture (see VllmBackend).
        if probe_answer(lambda f: None if f.archs is None else arch in f.archs):
            return True
        live = sglang_registry_archs()
        if live is not None:
            return arch in live
        return arch in FALLBACK_ARCHS or arch == "unknown"

    def precisions(self, hw: HardwareDescriptor) -> List[str]:
        cc = hw.gpu.cc if hw.gpu else (0, 0)
        out = ["bf16" if cc >= (8, 0) else "fp16"]
        if cc >= (8, 9):  # SGLang fp8 path targets Ada/Hopper
            out.append("fp8")
        if cc >= (7, 5):  # pre-quantized 4-bit checkpoints
            out += list(INT4_METHODS)
        return out

    def supported_quants(self, hw: HardwareDescriptor) -> List[str]:
        return self.precisions(hw)

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants: Optional[List[str]] = None) -> PreparedModel:
        arch = load_arch(model)
        wanted = as_stored(quants or self.precisions(hw), arch, model.hf_id)
        weights, paths = hf_weight_options(model, arch.num_params or 0, wanted,
                                           lambda p: 1 if p == "fp8" else dtype_bytes(p), arch.checkpoint_bytes)
        return PreparedModel(spec=model, backend=self.name, arch=arch, hf_path=model.hf_id, hf_paths=paths,
                             weights_bytes=weights)

    def find_second_models(self, prepared: PreparedModel, hw: Optional[HardwareDescriptor] = None) -> List[str]:
        """Under --speculative all: record the EAGLE3 head, the draft model and the checkpoint's own MTP head this
        SGLang could load, with their sizes, since the memory left outside the reservation depends on them
        (spec_fraction). One whose size is unknown is not offered. Returns notes for the profile when the Hub could
        not be asked."""
        methods = feature("spec_methods")
        notes: List[str] = []
        if "eagle3" in methods:
            try:
                found = speculative.find_eagle3_head(prepared.spec, prepared.arch, engine=self.name,
                                                         hub_off=prepared.hub_off, release=self.release(hw),
                                                         notes=notes)
            except Exception as exc:
                found = None
                notes.append(f"sglang: no EAGLE3 head tried, the Hub could not be asked ({exc})")
            if found and found[1]:
                prepared.spec_repos["eagle3"] = found[0]
                prepared.spec_bytes[found[0]] = int(found[1])
                notes += speculative.found_head_notes(self.name, prepared.spec, prepared.arch, found[0])
            elif found:
                notes.append(f"sglang: EAGLE3 head {found[0]} not tried, its size could not be read")
        if "draft" in methods:  # a release whose second-model behaviour was read
            kinds = [k for k in speculative.BLOCK_DRAFTERS if k in sglang_block_drafters()]
            notes += speculative.record_block_drafters(prepared, self.name, kinds, self.release(hw))
            for kind in kinds:  # SGLang offers no second model of unknown size
                repo = prepared.spec_repos.get(kind)
                if repo and not prepared.spec_bytes.get(repo):
                    del prepared.spec_repos[kind]
                    prepared.spec_facts.pop(repo, None)
                    prepared.spec_revisions.pop(repo, None)
                    notes.append(f"sglang: {kind} drafter {repo} not tried, its size could not be read")
        draft = speculative.draft_for(prepared.spec.hf_id)
        if "draft" in methods and draft and prepared.drafts_mode == "table":
            try:
                size = speculative.repo_bytes(draft)
            except Exception as exc:
                size = None
                notes.append(f"sglang: draft model {draft} not tried, the Hub could not be asked ({exc})")
            else:
                if not size:
                    notes.append(f"sglang: draft model {draft} not tried, its size could not be read")
            if size:
                prepared.spec_repos["draft"] = draft
                prepared.spec_bytes[draft] = int(size)
        notes += self._own_head_notes(prepared)
        if speculative.MTP_KEY in prepared.spec_repos and speculative.draft_bytes("mtp:1", prepared) is None:
            notes.append(f"sglang: MTP head of {prepared.spec.hf_id} not tried, its size could not be read")
        for n in notes:
            logger.warning(n)
        return notes

    def _own_head_notes(self, prepared: PreparedModel) -> List[str]:
        """Record the checkpoint's own MTP head when it holds one and this SGLang may load it, as the vLLM backend
        does: the checkpoint first, then SGLang, asked only when there is a head. SGLang's rules rename by
        architecture alone (a Qwen3-30B-A3B without MTP weights gets a draft class too), so the checkpoint's yes is
        what keeps a head that is not there from being offered. When SGLang is asked, the layers it says its draft
        reads (num_nextn_predict_layers, which many of its rules set to 1) replace the checkpoint's count for the
        planner: its KV pool is sized by them (sglang_hybrid_kv)."""
        if "draft" not in feature("spec_methods"):  # a release whose second-model behaviour was not read (mtp_archs_in)
            return []
        facts, notes = speculative.look_for_mtp_head(prepared.spec, self.name)
        if facts is None:
            return notes
        arch = prepared.arch.architecture
        verdict = _own_head(prepared.spec.hf_id, prepared.spec.revision)
        multi_layer = verdict.banded or verdict.draft_arch in SGLANG_MULTI_LAYER_DRAFTS or (
            verdict.asked and verdict.draft_arch is None and arch in SGLANG_MULTI_LAYER_TARGETS)
        if verdict.loads is not False and multi_layer:
            return [f"sglang: the checkpoint's MTP head not tried: SGLang runs {verdict.draft_arch or arch}'s MTP head "
                    "only on its multi-layer EAGLE worker, one model runner and one copy of the embedding and output "
                    "head per step, which rigtune does not plan yet"]
        if not verdict.asked:  # as before SGLang was asked: its source, read for the Qwen3.5 family only
            if arch not in sglang_mtp_archs():
                why = ("and this SGLang's source has no Qwen3.5 MTP draft" if arch in speculative.MTP_ARCHS
                       else "and its source is read for the Qwen3.5 family only")
                return [f"sglang: the checkpoint's MTP head not tried: SGLang was not asked whether it loads one "
                        f"({verdict.why}), {why}"]
            return speculative.record_mtp_head(prepared, self.name, found=facts)
        if verdict.loads is False:
            return [f"sglang: the checkpoint's MTP head not tried: {verdict.why}"]
        notes = speculative.record_mtp_head(prepared, self.name, found=facts)
        if verdict.layers:
            prepared.arch.mtp_layers = verdict.layers
        if verdict.loads is None:
            notes.append(f"sglang: the checkpoint's MTP head is tried although whether SGLang loads it could not be "
                         f"told ({verdict.why}); a start that fails costs one trial")
        if arch not in speculative.MTP_ARCHS:
            # sglang_hybrid_kv's MTP rule and speculative.draft_bytes were read for Qwen3_5ForCausalLMMTP only.
            notes.append(f"sglang: {arch}'s MTP head: its draft's KV cache is planned only on a model with "
                         "linear-attention layers, as Qwen3.5's is, and not otherwise, though SGLang reserves one; "
                         "the draft's copy of the embedding and output head is counted as a single-step draft holds "
                         "it; neither was checked for this family")
        return notes

    def find_pairs(self, prepared: PreparedModel, ctx: int = 4096, *, draft: Optional[str] = None,
                   hub: str = "on", hw: Optional[HardwareDescriptor] = None) -> List[str]:
        from rigtune.pairing import discover
        return discover(prepared, self.name, ctx, draft=draft, hub=hub,
                        version=self.version(hw) if hw else None)

    supports_tp = True

    def kv_dtypes(self, hw: HardwareDescriptor) -> List[str]:
        return ["fp8_e5m2"] if hw.gpu is not None and hw.gpu.cc >= (8, 0) else []

    def batch_ladder(self) -> Tuple[int, ...]:
        return (16, 64, 256, 512)

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        return model

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        total = hw.gpu.vram_total_bytes if hw.gpu else 0
        found = hw.backends.get(self.name)
        version = found.version if found is not None else None
        mm = self.calibrated_memory(hw, lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION), "gpu",
                                    hybrid_fn=lambda cfg, arch: sglang_hybrid_kv(cfg, arch, total, version),
                                    full_kv_fn=sglang_full_kv_reason, extra_weights_fn=sglang_drafter_weights,
                                    extra_layers_fn=sglang_drafter_layers)
        mm.extra_fn = lambda cfg: sglang_graph_bytes(cfg, total, version)
        return mm

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> List[Config]:
        ctxs = ctx_grid(model.arch.max_position_embeddings, min_ctx)
        out: List[Config] = []
        stored = stored_quant(model.weights_bytes)
        for quant in [stored] if stored else [q for q in self.precisions(hw) if q in model.weights_bytes]:
            for frac in (0.80, 0.88, 0.93):
                for ctx in ctxs:
                    for batch in (16, 64, 256):
                        out.append(
                            Config(backend=self.name, quant=quant, ctx=ctx, batch=batch, gpu_memory_utilization=frac)
                        )
        return out

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> Config:
        return Config(
            backend=self.name,
            quant=stored_quant(model.weights_bytes) or self.precisions(hw)[0],
            ctx=min(max(4096, min_ctx), model.arch.max_position_embeddings),
            batch=256,
            gpu_memory_utilization=0.88,
        )

    def kernels_same_as_default(self, cfg: Config, hw: HardwareDescriptor) -> Optional[str]:
        if cfg.kernels is None:
            return None
        vram_mib = (hw.gpu.vram_total_bytes if hw.gpu else 0) / (1024 * 1024)
        return kernels.sglang_same_as_default(kernels.parse(cfg.kernels), vram_mib, cfg.batch, cfg.tp,
                                              self.version(hw))

    def config_refusal(self, cfg: Config, model: PreparedModel) -> Optional[str]:
        if cfg.kernels is not None:  # each setting must be one the parser of the SGLang that will run reported
            return kernels.sglang_unconfirmed(kernels.parse(cfg.kernels), _engine_facts())
        return None

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        refusal = self.config_refusal(cfg, model)
        if refusal:
            raise ValueError(refusal)
        path = model.hf_paths.get(cfg.quant) or model.hf_path or model.spec.hf_id
        args = [
            sglang_python() or sys.executable, "-m", "sglang.launch_server",
            "--model-path", path,
            "--served-model-name", model.spec.hf_id,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--context-length", str(cfg.ctx),
            "--max-running-requests", str(cfg.batch),
        ]
        if cfg.gpu_memory_utilization is not None:
            args += ["--mem-fraction-static", f"{cfg.gpu_memory_utilization:.2f}"]
        if cfg.prefill_budget is not None:
            args += ["--chunked-prefill-size", str(cfg.prefill_budget)]
        if cfg.quant == "fp8":
            args += ["--quantization", "fp8"]
        elif cfg.quant in ("bf16", "fp16"):
            args += ["--dtype", "bfloat16" if cfg.quant == "bf16" else "float16"]
        # awq / gptq, and a checkpoint served as stored (ckpt-*): its quantization_config selects the kernel.
        if cfg.kv_dtype != "auto":
            args += ["--kv-cache-dtype", cfg.kv_dtype]
        if cfg.prefix_cache is False:
            args += ["--disable-radix-cache"]
        revision = model.spec.revision if cfg.quant not in INT4_METHODS else None
        if cfg.spec_decode:  # the path and revision are read by an MTP head only: the target's own
            kind, second, _ = speculative.parse(cfg.spec_decode)
            # A second model (a draft, an EAGLE3 head, an assistant) is loaded as its checkpoint stores it, as vLLM
            # loads one (config/speculative.py line 1425 at v0.33.0: the draft's quantization is the speculative
            # config's own, None). Unasked, SGLang gives it the target's --quantization (arg_groups/serving_hook.py
            # lines 629-633 at v0.5.19, 685-690 at v0.5.23), so an fp8 trial ran an fp8 head that vLLM ran in bf16, and
            # a quantized draft met a method its checkpoint does not declare; "unquant" resets it to None (lines
            # 648-653 / 704-708), and a draft with None reads its own quantization_config (configs/model_config.py
            # ModelConfig._verify_quantization). The checkpoint's own MTP head keeps the target's, as on vLLM (lines
            # 1265-1268 there).
            args += speculative.sglang_config(
                cfg.spec_decode, model_path=path, revision=revision,
                draft_revision=model.spec_revisions.get(second), draft_path=model.spec_paths.get(second),
                draft_quantization="unquant" if kind in speculative.SECOND_MODEL else None)
        if cfg.tp > 1:
            args += ["--tp-size", str(cfg.tp)]
        if revision:
            args += ["--revision", revision]
        args += render_extra(cfg.extra)
        if cfg.kernels is not None:
            args += kernels.sglang_args(kernels.parse(cfg.kernels))
        env = {}
        # PCIe-only GPUs: peer-to-peer can hang at start-up inside containers. Measured on a pair of
        # A40s: NCCL's P2P path hangs at init, and with only that disabled the engine's custom
        # all-reduce (CUDA IPC, also P2P) hangs next. With both off it starts in under a minute.
        if cfg.tp > 1 and nvlink_between(list(range(cfg.tp))) is False:
            env["NCCL_P2P_DISABLE"] = "1"
            args += ["--disable-custom-all-reduce"]
        # SGLang's own counter, sglang:cached_tokens_total, needs --enable-metrics too, so only the per-reply count.
        if self.report_reuse and feature("cache_report"):
            args += ["--enable-cache-report"]
        if cfg.kv_tier is not None:
            tier_args, tier_env = sglang_tier_launch(cfg, port)
            args += tier_args
            env.update(tier_env)
        env.update(head_context_env(args, cfg.ctx, model.arch.sglang_context or model.arch.max_position_embeddings))
        return LaunchSpec(args=args, env=env)

    # ---- KV-cache tier

    def tier_refusal(self, cfg: Config, model: PreparedModel) -> Optional[str]:
        features, origin = feature_answer("hicache")
        if "hicache" not in features:
            return f"the SGLang that would run has no --enable-hierarchical-cache with --hicache-size {origin}"
        if cfg.prefix_cache is False:
            return "SGLang's hierarchical cache refuses --disable-radix-cache"
        if cfg.tp > 1:
            return "not offered with tensor parallelism (--hicache-size is per rank; never run)"
        return None

    def tier_disk_refusal(self) -> Optional[str]:
        features = feature("hicache")
        if "file" not in features:
            return "the SGLang that would run has no file backend with a directory rigtune can set"
        if "file_cap" not in features:
            return "the SGLang that would run cannot cap its file backend's size; not offered"
        return None

    def tier_host_reserve(self) -> int:
        return max(HOST_TIER_RESERVE, SGLANG_HOST_RESERVE)

    def tier_details_field(self) -> Optional[str]:
        return TIER_DETAILS_FIELD if feature("tier_details") else None

    def tier_host_need(self, cfg: Config, host_bytes: int) -> int:
        """With an EAGLE3 head, a draft model or the checkpoint's own MTP head SGLang adds a host pool for the draft's
        cache, as many tokens as the target's (hybrid_cache/hybrid_pool_assembler.build_full_draft_pools, outside
        --hicache-size). Its size per token is the second model's, which the planner does not know, so it is
        counted as the target's (for an MTP head, one layer, that overstates it)."""
        kind = speculative.parse(cfg.spec_decode)[0] if cfg.spec_decode else None
        return 2 * host_bytes if kind in ("eagle3", "draft", "mtp", *speculative.BLOCK_DRAFTERS) else host_bytes

    # A start-up out-of-memory failure with a second model steps down from the lowered fraction, this far at most
    # (the search's own floor, 0.85, sits above it on a 24 GB card).
    SPEC_RETRY_DEPTH = 0.10

    def spec_fraction(self, spec: Optional[str], model: PreparedModel, hw: HardwareDescriptor,
                      batch: Optional[int] = None) -> Optional[float]:
        """The largest --mem-fraction-static that leaves a spec's second weights room outside the reservation:
        1 - (draft weights + runtime workspace + 512 MB) / total VRAM, rounded down to 0.01 (0.86 for an 850 MB
        EAGLE3 head on a 24 GB card, 0.79 for Llama 3.2 1B). None: no second model (no spec, a lookup method) or no
        GPU, so no limit. 0.0 for a second model of unknown size: no fraction is known to leave it room.

        SGLang 0.5.19 loads the draft after the target, sizes its KV pools after both, and scales the pool's cell
        size for the draft's KV cache, so that cache lives inside the fraction; the planner does not count it, so
        the pool holds fewer tokens than it estimates. Whether the first sizing subtracts the draft's weights was
        not verified, so they are assumed to need room outside the fraction.

        A DFlash or DSpark drafter was read at 0.5.23: its weights are inside the fraction (planned there,
        sglang_drafter_weights), and what needs room outside is the logits buffer its CUDA graphs take at `batch`
        (sglang_logits_bytes), and for DFlash its start-up sampling rehearsal (sglang_rehearsal_bytes)."""
        if not spec or hw.gpu is None or not hw.gpu.vram_total_bytes:
            return None
        extra = speculative.draft_bytes(spec, model)
        if extra == 0:
            return None
        if extra is None:
            return 0.0
        if speculative.parse(spec)[0] in speculative.BLOCK_DRAFTERS:
            if not batch or not model.arch.vocab_size:  # the buffer cannot be sized: no fraction is known to fit
                return 0.0
            extra = sglang_logits_bytes(spec, model, batch) + sglang_rehearsal_bytes(spec, model, batch)
        outside = extra + self.memory_model(hw).runtime_workspace + MIN_SAFETY_MARGIN
        return math.floor((1 - outside / hw.gpu.vram_total_bytes) * 100) / 100

    def spec_retry_floor(self, cfg: Config, model: PreparedModel, hw: HardwareDescriptor) -> Optional[float]:
        """How low an out-of-memory retry may take a second-model spec trial: SPEC_RETRY_DEPTH below its cap."""
        cap = self.spec_fraction(cfg.spec_decode, model, hw, cfg.batch)
        return round(cap - self.SPEC_RETRY_DEPTH, 2) if cap else None

    def spec_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                      all_methods: bool = False, min_batch: Optional[int] = None) -> List[Config]:
        """Under --speculative all only: n-gram lookup, the verified EAGLE3 head, the draft model, the DFlash and
        DSpark drafters (sglang_block_drafters) and the checkpoint's own MTP head, as far as the installed SGLang has
        them. A second model (the MTP head counts as
        one) needs `hw` to be left room. An EAGLE3 head or a draft model is offered only where the planner says the
        configuration still fits at the lower fraction; an MTP setting that does not fit is tried at a smaller batch
        that covers the workload, or not at all (_mtp_variants). Fractions start from the one before any speculation
        lowered it, so the variants of a draft-model leader are not measured at its 0.79."""
        if not all_methods:
            return []
        methods = feature("spec_methods")
        specs: List[str] = []
        if "ngram" in methods:
            specs.append(speculative.ngram(speculative.NGRAM_TOKENS_SGLANG))
        if hw is not None:
            if model.drafts_mode == "verified" and "eagle3" in methods and cfg.quant not in INT4_METHODS:
                from rigtune.pairing import eligible_pairs
                specs += [speculative.assist(p["repo"], 3) for p in eligible_pairs(model, "assistant", cfg.ctx)]
            if "eagle3" in methods and "eagle3" in model.spec_repos:
                specs.append(speculative.eagle3(model.spec_repos["eagle3"], speculative.EAGLE3_TOKENS))
            if "draft" in methods and model.drafts_mode == "verified" and cfg.quant not in INT4_METHODS:
                from rigtune.pairing import eligible_pairs
                specs += [speculative.draft(p["repo"], speculative.DRAFT_TOKENS_SGLANG)
                          for p in eligible_pairs(model, "draft", cfg.ctx)]
            elif "draft" in methods and "draft" in model.spec_repos:
                specs.append(speculative.draft(model.spec_repos["draft"], speculative.DRAFT_TOKENS_SGLANG))
            if "draft" in methods and cfg.quant not in INT4_METHODS:
                for kind in speculative.BLOCK_DRAFTERS:
                    repo = model.spec_repos.get(kind)
                    facts = model.spec_facts.get(repo or "", {})
                    if kind in sglang_block_drafters() and repo and facts.get("k") and facts.get("kv_elements"):
                        specs.append(speculative.block_drafter(kind, repo, facts["k"]))
        out: List[Config] = []
        for s in specs:
            if s == cfg.spec_decode:
                continue
            v, lowered = self._at_spec_fraction(cfg, s, model, hw)
            planned = speculative.parse(s)[0] in speculative.BLOCK_DRAFTERS  # its weights and pool are in the plan
            if lowered or planned:  # the configuration must still fit at the lower fraction
                try:
                    if not v.gpu_memory_utilization or not estimate(hw, model, v, self.memory_model(hw)).feasible:
                        continue
                except Exception:
                    continue
            out.append(v)
        return out + self._mtp_variants(cfg, model, hw, all_methods, min_batch)[0]

    def spec_notes(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                   all_methods: bool = False, min_batch: Optional[int] = None) -> List[str]:
        return self._mtp_variants(cfg, model, hw, all_methods, min_batch)[1]

    def _at_spec_fraction(self, cfg: Config, spec: str, model: PreparedModel,
                          hw: Optional[HardwareDescriptor]) -> Tuple[Config, bool]:
        """(`cfg` with `spec`, whether its fraction was lowered): at the fraction before any speculation lowered it,
        or at the spec's own cap where that is lower (gmu_before_spec then keeps the fraction it came from)."""
        base = cfg.gmu_before_spec if cfg.gmu_before_spec is not None else cfg.gpu_memory_utilization
        v = cfg.model_copy(update={"spec_decode": spec, "gpu_memory_utilization": base, "gmu_before_spec": None})
        cap = self.spec_fraction(spec, model, hw, cfg.batch) if hw is not None else None
        if cap is not None and (base is None or base > cap):
            return v.model_copy(update={"gpu_memory_utilization": cap, "gmu_before_spec": base}), True
        return v, False

    def _mtp_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor], all_methods: bool,
                      min_batch: Optional[int]) -> Tuple[List[Config], List[str]]:
        """The checkpoint's own MTP head at each of speculative.MTP_TOKENS (steps; top-k 1, steps + 1 draft tokens),
        at the fraction that leaves the draft room outside the reservation (spec_fraction: the head and the copy of
        the embedding and output head the draft holds while the pools are sized, speculative.draft_bytes), with
        notes. A head whose sizes are unknown gets a fraction of 0 and is not offered. A setting that does not fit at
        `cfg`'s batch is tried at the largest batch that fits and covers `min_batch`, or not at all
        (BaseBackend.own_head_fit)."""
        if not (all_methods and hw is not None and self._offers_mtp(cfg, model)):
            return [], []
        variants = []
        for k in speculative.MTP_TOKENS:
            if speculative.mtp(k) == cfg.spec_decode:
                continue
            v, lowered = self._at_spec_fraction(cfg, speculative.mtp(k), model, hw)
            if lowered and not v.gpu_memory_utilization:  # sizes unknown: no fraction is known to leave it room
                continue
            variants.append(v)
        return self.own_head_fit(variants, model, hw, min_batch)

    def _offers_mtp(self, cfg: Config, model: PreparedModel) -> bool:
        """The checkpoint's own MTP head: recorded by find_second_models (the checkpoint holds it, and SGLang did not
        say it builds no draft from it), and the launch reads the checkpoint it was found in, not a 4-bit
        repository's."""
        return speculative.MTP_KEY in model.spec_repos and cfg.quant not in INT4_METHODS

    PREFILL_BUDGETS = (2048, 8192, 16384)

    def prefill_variants(self, cfg: Config) -> List[Config]:
        return [cfg.model_copy(update={"prefill_budget": b}) for b in self.PREFILL_BUDGETS if b != cfg.prefill_budget]

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(
            health_path="/health",
            model_name=model.spec.hf_id,  # --served-model-name
            tokenizer_id=model.spec.hf_id,
            gpu_ids=[hw.gpu.index] if hw.gpu else [],
        )
