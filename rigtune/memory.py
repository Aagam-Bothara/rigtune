"""Memory planner (runs before any benchmark).

    estimated = weights + kv_cache(ctx, batch, dtype) + runtime_workspace + safety_margin
    keep config only if estimated <= 0.95 * available_memory
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from rigtune.hfconfig import kv_element_bytes
from rigtune.models import ArchInfo, Config, HardwareDescriptor, MemoryEstimate, MiB, PreparedModel

logger = logging.getLogger(__name__)

BUDGET_FRACTION = 0.95
MIN_SAFETY_MARGIN = 512 * MiB
DEFAULT_MARGIN_FRACTION = 0.05


def safety_margin(available: int, fraction: float = DEFAULT_MARGIN_FRACTION) -> int:
    """`fraction` of available (default 5%) or 512 MB, whichever is larger."""
    return max(int(fraction * available), MIN_SAFETY_MARGIN)


@dataclass(frozen=True)
class HybridKV:
    """How one engine sizes its cache for a hybrid-attention layout, for one Config.

    Only backends whose rules were read from the engine's source supply one (MemoryModel.hybrid_fn); every
    other backend keeps the every-layer-full estimate for hybrid models, with a logged note. The defaults
    describe an engine that allocates exactly per token and per layer; vLLM's fields add its block and layer
    granularity, its in-flight sliding tokens and its start-up check.
    """

    state_slots: int  # recurrent-state copies the engine holds so that cfg.batch sequences can run at once
    state_bytes: Optional[int] = None  # bytes of one state copy, when not arch.linear_state_bytes
    # SGLang's radix-cache SWA pool holds this many sliding tokens per full-attention token.
    swa_full_tokens_ratio: Optional[float] = None
    # SGLang's radix-cache mamba pool takes this share of the cache budget, the KV pool the rest.
    state_share: Optional[float] = None
    # A fixed sliding-window pool in tokens (SGLang without the radix cache); replaces min(per-seq, window) x seqs.
    swa_pool_tokens: Optional[int] = None
    full_block: int = 1  # a sequence's full-layer tokens are allocated in blocks of this many
    sliding_block: int = 1  # likewise its sliding-layer tokens
    full_layers: Optional[int] = None  # full-layer pages after the engine's layer-group padding
    sliding_layers: Optional[int] = None  # likewise for sliding layers
    swa_extra_tokens: int = 0  # sliding tokens held once across the batch (vLLM: scheduled, not yet settled)
    sliding_cap: Optional[int] = None  # the most sliding tokens one sequence can hold, in-flight ones included
    min_bytes: int = 0  # the cache the engine needs before it starts at all; the plan is never below it


def _round_up(n: float, block: int) -> int:
    return math.ceil(n / block) * block if block > 1 else math.ceil(n)


def kv_cache_bytes(model: PreparedModel, cfg: Config, kv_tokens: int,
                   hybrid: Optional[HybridKV] = None) -> int:
    """KV cache (plus recurrent state) for `kv_tokens` tokens resident across cfg.batch sequences.

    Dense models, and hybrid models on an engine without verified rules (`hybrid` None), get
    `kv_bytes_per_token x kv_tokens` exactly as before. For a recognised hybrid layout, per sequence:
      full layers     x kv_tokens / cfg.batch                (rounded up to the engine's block)
      sliding layers  x min(kv_tokens / cfg.batch, window)   (rounded up, + one block once past the window)
    over cfg.batch sequences, plus the engine's in-flight sliding tokens once, plus
      recurrent state x hybrid.state_slots (fixed per sequence, so the paged-KV fraction does not apply)
    and never less than hybrid.min_bytes.
    """
    arch = model.arch
    kv_dtype = cfg.kv_dtype if cfg.kv_dtype != "auto" else arch.torch_dtype
    elem = kv_element_bytes(kv_dtype)
    if hybrid is None or not arch.is_hybrid:
        return int(arch.kv_bytes_per_token(1) * elem * kv_tokens)
    seqs = max(cfg.batch, 1)
    per_seq = kv_tokens / seqs
    full_layers = arch.full_attention_layers or 0
    full_bpt = arch.full_kv_bytes_per_token(elem) * (hybrid.full_layers or full_layers) / max(full_layers, 1)
    full_tokens = _round_up(per_seq, hybrid.full_block) * seqs if hybrid.full_block > 1 else kv_tokens
    swa_tokens = 0
    sliding_bpt = 0.0
    if arch.sliding_layers:
        sliding_bpt = arch.sliding_kv_bytes_per_token(elem) * (hybrid.sliding_layers or arch.sliding_layers) \
            / arch.sliding_layers
        window = arch.sliding_window or 0
        held = _round_up(min(per_seq, window), hybrid.sliding_block)
        if hybrid.sliding_block > 1 and per_seq > window:
            held += hybrid.sliding_block  # a window that starts mid-block spans one block more
        swa_tokens = held * seqs + hybrid.swa_extra_tokens
        if hybrid.sliding_cap is not None:
            swa_tokens = min(swa_tokens, hybrid.sliding_cap * seqs)
        if hybrid.swa_pool_tokens is not None:
            swa_tokens = hybrid.swa_pool_tokens
        elif hybrid.swa_full_tokens_ratio and full_layers:
            # SGLang 0.5.19 model_executor/pool_configurator.HybridSWAPoolConfigurator._solve_pool_sizes:
            # swa_tokens = full_tokens x swa_full_tokens_ratio, so the full pool must also cover swa / ratio.
            full_tokens = max(full_tokens, math.ceil(swa_tokens / hybrid.swa_full_tokens_ratio))
            swa_tokens = int(full_tokens * hybrid.swa_full_tokens_ratio)
    kv = int(full_bpt * full_tokens + sliding_bpt * swa_tokens)
    # Recurrent state keeps its own dtype (conv: model dtype, temporal: mamba_ssm_dtype); --kv-cache-dtype does
    # not shrink it in either engine.
    per_state = hybrid.state_bytes if hybrid.state_bytes is not None else arch.linear_state_bytes
    state = per_state * hybrid.state_slots if arch.linear_layers else 0
    if state and hybrid.state_share:
        # SGLang 0.5.19 mem_cache/kv_cache_configurator._handle_max_mamba_cache: the mamba pool gets
        # share x budget and the KV pool the rest, so the budget must cover whichever needs more.
        total = max(math.ceil(state / hybrid.state_share), math.ceil(kv / (1 - hybrid.state_share)))
    else:
        total = kv + state
    return max(total, hybrid.min_bytes)


class MemoryModel:
    """Backend-provided knobs for the generic estimator."""

    def __init__(self, runtime_workspace: int, kv_tokens_fn, device: str = "gpu",
                 margin_fraction: float = DEFAULT_MARGIN_FRACTION, calibrated: bool = False,
                 hybrid_fn: Optional[Callable[[Config, ArchInfo], HybridKV]] = None,
                 full_kv_fn: Optional[Callable[[ArchInfo], Optional[str]]] = None):
        self.runtime_workspace = runtime_workspace
        self.kv_tokens_fn = kv_tokens_fn  # Config -> int tokens resident in KV cache
        self.device = device  # "gpu" | "cpu"
        self.margin_fraction = margin_fraction
        self.calibrated = calibrated  # constants came from measured trials on this machine
        # (Config, ArchInfo) -> HybridKV for engines whose hybrid-layout rules are verified; None elsewhere.
        self.hybrid_fn = hybrid_fn
        # ArchInfo -> why this engine keeps a full KV cache on every layer of a recognised hybrid layout
        # (SGLang for Gemma 2/3); None when it does not.
        self.full_kv_fn = full_kv_fn

    def full_kv_reason(self, arch: ArchInfo) -> Optional[str]:
        return self.full_kv_fn(arch) if (self.full_kv_fn is not None and arch.is_hybrid) else None

    def hybrid_for(self, cfg: Config, arch: ArchInfo) -> Optional[HybridKV]:
        if self.hybrid_fn is None or not arch.is_hybrid or self.full_kv_reason(arch):
            return None
        return self.hybrid_fn(cfg, arch)


def estimate(
    hw: HardwareDescriptor,
    model: PreparedModel,
    cfg: Config,
    mm: MemoryModel,
    available: Optional[int] = None,
) -> MemoryEstimate:
    weights = model.weights_for(cfg.quant)
    kv = kv_cache_bytes(model, cfg, mm.kv_tokens_fn(cfg), mm.hybrid_for(cfg, model.arch))

    # Partial GPU offload (llama.cpp): only the offloaded fraction lands on the device.
    if mm.device == "gpu" and cfg.n_gpu_layers is not None:
        frac = min(1.0, max(0.0, cfg.n_gpu_layers / max(model.arch.num_layers, 1)))
        weights = int(weights * frac)
        kv = int(kv * frac)

    # Tensor parallel: weights and KV are sharded across GPUs; workspace and margin stay per GPU.
    if cfg.tp > 1:
        weights //= cfg.tp
        kv //= cfg.tp

    if available is None:
        available = hw.gpu.vram_free_bytes if (mm.device == "gpu" and hw.gpu) else hw.cpu.ram_free_bytes
    margin = safety_margin(available, mm.margin_fraction)
    total = weights + kv + mm.runtime_workspace + margin

    # vLLM/SGLang cap their own allocation at gpu_memory_utilization x total VRAM.
    budget = int(BUDGET_FRACTION * available)
    if cfg.gpu_memory_utilization is not None and hw.gpu is not None and mm.device == "gpu":
        budget = min(budget, int(cfg.gpu_memory_utilization * hw.gpu.vram_total_bytes))
        # ... and cannot use more than what is actually free right now.
        budget = min(budget, int(BUDGET_FRACTION * available))

    return MemoryEstimate(
        config_key=cfg.key(),
        weights=weights,
        kv_cache=kv,
        runtime_workspace=mm.runtime_workspace,
        safety_margin=margin,
        total=total,
        budget=budget,
        feasible=total <= budget,
    )


def plan(
    hw: HardwareDescriptor,
    model: PreparedModel,
    configs: Sequence[Config],
    mm: MemoryModel,
) -> List[Tuple[Config, MemoryEstimate]]:
    """Return (config, estimate) pairs that fit. Also checks host RAM for CPU-offloaded layers."""
    if model.arch.is_hybrid and mm.hybrid_fn is None:
        logger.info("%s: %s has a hybrid attention layout, but this backend's hybrid KV rules are not verified; "
                    "planning KV as if every layer kept a full cache", model.backend, model.arch.architecture)
    elif mm.full_kv_reason(model.arch):
        logger.info("%s: %s", model.backend, mm.full_kv_reason(model.arch))
    kept: List[Tuple[Config, MemoryEstimate]] = []
    for cfg in configs:
        est = estimate(hw, model, cfg, mm)
        if not est.feasible:
            logger.debug("drop %s: %.2f GiB > budget %.2f GiB", cfg.key(), est.total / 2**30, est.budget / 2**30)
            continue
        if mm.device == "gpu" and cfg.n_gpu_layers is not None and cfg.n_gpu_layers < model.arch.num_layers:
            # Remainder lives in host RAM; make sure that fits too.
            frac_cpu = 1.0 - cfg.n_gpu_layers / max(model.arch.num_layers, 1)
            host_need = int(model.weights_for(cfg.quant) * frac_cpu) + safety_margin(hw.cpu.ram_free_bytes)
            if host_need > BUDGET_FRACTION * hw.cpu.ram_free_bytes:
                logger.debug("drop %s: host RAM insufficient for offload remainder", cfg.key())
                continue
        kept.append((cfg, est))
    return kept
