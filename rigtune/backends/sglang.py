"""SGLang backend (NVIDIA, compute capability >= 7.5)."""

from __future__ import annotations

import functools
import importlib.util
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import FrozenSet, List, Optional, Tuple

from rigtune import kvtier, speculative
from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid, render_extra
from rigtune.backends.vllm import KNOWN_ARCHS, PAGED_KV_FRACTION
from rigtune.hfconfig import dtype_bytes, load_arch
from rigtune.memory import HOST_TIER_RESERVE, MIN_SAFETY_MARGIN, HybridKV, MemoryModel, estimate
from rigtune.models import ArchInfo, Config, GiB, HardwareDescriptor, ModelSpec, PreparedModel
from rigtune.hardware import nvlink_between, sglang_python
from rigtune.quantized import INT4_METHODS, hf_weight_options

logger = logging.getLogger(__name__)

# SGLang 0.5.19 defaults behind its hybrid-model pools (all read from source; not yet run on hardware).
MAMBA_FULL_MEMORY_RATIO = 0.9  # server_args.ServerArgs.mamba_full_memory_ratio
MAMBA_SLOTS_PER_REQUEST = 3  # mem_cache/kv_cache_configurator.MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
MAMBA_EXTRA_BUFFER_SLOTS = 2  # kv_cache_configurator.MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
MAMBA_EXTRA_BUFFER_SLOTS_LAZY = 1  # MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY (and _NO_OVERLAP)
SWA_FULL_TOKENS_RATIO = 0.8  # server_args.ServerArgs.swa_full_tokens_ratio
SWA_EVICTION_INTERVAL = 128  # environ.SGLANG_SWA_EVICTION_INTERVAL
SWA_CHUNKS_IN_FLIGHT = 2  # pool_configurator.SWAChunkCapPoolConfigurator, under the default overlap scheduler
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


def sglang_default_chunk(vram_total_bytes: int) -> int:
    """chunked_prefill_size SGLang picks from GPU memory (memory_hook._handle_gpu_memory_settings)."""
    mib = vram_total_bytes / 2**20
    if not mib:
        return 4096
    for limit, chunk in ((35 * 1024, 2048), (60 * 1024, 4096), (160 * 1024, 8192)):
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


def sglang_hybrid_kv(cfg: Config, arch: ArchInfo, vram_total_bytes: int) -> HybridKV:
    """SGLang 0.5.19's pools for hybrid models, read from source (not yet run on hardware). Sliding layers only
    count for SGLANG_HYBRID_SWA_ARCHS; for others MemoryModel.full_kv_fn keeps every layer full.

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
      resolved), so the SWA pool is (window + eviction interval + 2 pages) x requests + 2 prefill chunks + 1 page.
    Speculative decoding (D = --speculative-num-draft-tokens, as speculative.sglang_config passes it):
    _handle_max_mamba_cache also reserves an intermediate state per draft token for every request slot,
    per_req x (requests + 1) x D, inside the mamba share with the radix cache (the joint solve for K) and beside it
    without. ReplaySSM, which would replace it, is off by default (server_args enable_linear_replayssm_spec).
    SGLang's GDN conv state does not grow with D (configs/mamba_utils.Mamba2StateShape.create: conv_kernel - 1).
    A second model's own KV cache is not counted, as for dense models (see spec_fraction).
    """
    draft = sglang_draft_tokens(cfg.spec_decode)
    if cfg.prefix_cache is not False:
        slots = sglang_mamba_slots_per_request(cfg, arch) * cfg.batch + 1
        return HybridKV(state_slots=slots + draft * (cfg.batch + 1),
                        swa_full_tokens_ratio=SWA_FULL_TOKENS_RATIO,
                        state_share=MAMBA_FULL_MEMORY_RATIO / (1 + MAMBA_FULL_MEMORY_RATIO))
    swa_pool = None
    if arch.sliding_layers and arch.full_attention_layers:
        page = 1  # CUDA default (arg_groups/overrides.py page_size resolution)
        chunk = cfg.prefill_budget or sglang_default_chunk(vram_total_bytes)
        swa_pool = ((arch.sliding_window or 0) + SWA_EVICTION_INTERVAL + 2 * page) * cfg.batch \
            + SWA_CHUNKS_IN_FLIGHT * chunk + page
    return HybridKV(state_slots=(cfg.batch + 1) * (1 + draft), swa_pool_tokens=swa_pool)


def sglang_draft_tokens(spec: Optional[str]) -> int:
    """--speculative-num-draft-tokens for a spec: k for n-gram, k + 1 for a second model (sglang_config)."""
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


def _server_args_source(pkg: Path) -> str:
    """srt/server_args.py, plus srt/arg_groups/*.py where a release has split its arguments out."""
    files = [pkg / "srt" / "server_args.py"] + sorted((pkg / "srt" / "arg_groups").glob("*.py"))
    texts = []
    for f in files:
        try:
            texts.append(f.read_text("utf-8"))
        except OSError:
            continue
    return "\n".join(texts)


@functools.lru_cache(maxsize=1)
def sglang_spec_methods() -> FrozenSet[str]:
    """rigtune's spec kinds ("ngram", "eagle3", "draft") the SGLang that would be launched offers. Empty when
    SGLang is not found or predates 0.5.2."""
    python = sglang_python()
    pkg = _sglang_package_dir(python) if python else None
    return spec_methods_in(_server_args_source(pkg)) if pkg else frozenset()


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


class SglangBackend(BaseBackend):
    name = "sglang"
    runtime_workspace_bytes = int(2.0 * GiB)  # CUDA graphs for many batch sizes + radix cache metadata

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        if hw.gpu is None or hw.gpu.vendor != "nvidia" or hw.gpu.cc < (7, 5):
            return False
        try:
            arch = load_arch(model).architecture
        except Exception as exc:
            logger.warning("could not read config for %s: %s", model.hf_id, exc)
            return True
        live = sglang_registry_archs()
        if live is not None:
            return arch in live
        return arch in KNOWN_ARCHS or arch == "unknown"

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
        weights, paths = hf_weight_options(model, arch.num_params or 0, list(quants or self.precisions(hw)),
                                           lambda p: 1 if p == "fp8" else dtype_bytes(p))
        return PreparedModel(spec=model, backend=self.name, arch=arch, hf_path=model.hf_id, hf_paths=paths,
                             weights_bytes=weights)

    def find_second_models(self, prepared: PreparedModel) -> List[str]:
        """Under --speculative all: record the EAGLE3 head and the draft model this SGLang could load, with their
        sizes, since the memory left outside the reservation depends on them (spec_fraction). One whose size is
        unknown is not offered. Returns notes for the profile when the Hub could not be asked."""
        methods = sglang_spec_methods()
        notes: List[str] = []
        if "eagle3" in methods:
            try:
                found = speculative.find_eagle3_head(prepared.spec, prepared.arch)
            except Exception as exc:
                found = None
                notes.append(f"sglang: no EAGLE3 head tried, the Hub could not be asked ({exc})")
            if found and found[1]:
                prepared.spec_repos["eagle3"] = found[0]
                prepared.spec_bytes[found[0]] = int(found[1])
            elif found:
                notes.append(f"sglang: EAGLE3 head {found[0]} not tried, its size could not be read")
        draft = speculative.draft_for(prepared.spec.hf_id)
        if "draft" in methods and draft:
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
        for n in notes:
            logger.warning(n)
        return notes

    supports_tp = True

    def kv_dtypes(self, hw: HardwareDescriptor) -> List[str]:
        return ["fp8_e5m2"] if hw.gpu is not None and hw.gpu.cc >= (8, 0) else []

    def batch_ladder(self) -> Tuple[int, ...]:
        return (16, 64, 256, 512)

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        return model

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        total = hw.gpu.vram_total_bytes if hw.gpu else 0
        return self.calibrated_memory(hw, lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION), "gpu",
                                      hybrid_fn=lambda cfg, arch: sglang_hybrid_kv(cfg, arch, total),
                                      full_kv_fn=sglang_full_kv_reason)

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> List[Config]:
        ctxs = ctx_grid(model.arch.max_position_embeddings, min_ctx)
        out: List[Config] = []
        for quant in [q for q in self.precisions(hw) if q in model.weights_bytes]:
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
            quant=self.precisions(hw)[0],
            ctx=min(max(4096, min_ctx), model.arch.max_position_embeddings),
            batch=256,
            gpu_memory_utilization=0.88,
        )

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        args = [
            sglang_python() or sys.executable, "-m", "sglang.launch_server",
            "--model-path", model.hf_paths.get(cfg.quant) or model.hf_path or model.spec.hf_id,
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
        if cfg.kv_dtype != "auto":
            args += ["--kv-cache-dtype", cfg.kv_dtype]
        if cfg.prefix_cache is False:
            args += ["--disable-radix-cache"]
        if cfg.spec_decode:
            args += speculative.sglang_config(cfg.spec_decode)
        if cfg.tp > 1:
            args += ["--tp-size", str(cfg.tp)]
        if model.spec.revision and cfg.quant not in INT4_METHODS:
            args += ["--revision", model.spec.revision]
        args += render_extra(cfg.extra)
        env = {}
        # PCIe-only GPUs: peer-to-peer can hang at start-up inside containers. Measured on a pair of
        # A40s: NCCL's P2P path hangs at init, and with only that disabled the engine's custom
        # all-reduce (CUDA IPC, also P2P) hangs next. With both off it starts in under a minute.
        if cfg.tp > 1 and nvlink_between(list(range(cfg.tp))) is False:
            env["NCCL_P2P_DISABLE"] = "1"
            args += ["--disable-custom-all-reduce"]
        # SGLang's own counter, sglang:cached_tokens_total, needs --enable-metrics too, so only the per-reply count.
        if self.report_reuse and _cache_report_flag():
            args += ["--enable-cache-report"]
        if cfg.kv_tier is not None:
            tier_args, tier_env = sglang_tier_launch(cfg, port)
            args += tier_args
            env.update(tier_env)
        return LaunchSpec(args=args, env=env)

    # ---- KV-cache tier

    def tier_refusal(self, cfg: Config, model: PreparedModel) -> Optional[str]:
        if "hicache" not in _hicache_features():
            return ("the SGLang that would run has no --enable-hierarchical-cache with --hicache-size (read "
                    "from its source)")
        if cfg.prefix_cache is False:
            return "SGLang's hierarchical cache refuses --disable-radix-cache"
        if cfg.tp > 1:
            return "not offered with tensor parallelism (--hicache-size is per rank; never run)"
        return None

    def tier_disk_refusal(self) -> Optional[str]:
        features = _hicache_features()
        if "file" not in features:
            return "the SGLang that would run has no file backend with a directory rigtune can set"
        if "file_cap" not in features:
            return "the SGLang that would run cannot cap its file backend's size; not offered"
        return None

    def tier_host_reserve(self) -> int:
        return max(HOST_TIER_RESERVE, SGLANG_HOST_RESERVE)

    def tier_details_field(self) -> Optional[str]:
        return TIER_DETAILS_FIELD if _tier_details_flag() else None

    def tier_host_need(self, cfg: Config, host_bytes: int) -> int:
        """With an EAGLE3 head or a draft model SGLang adds a host pool for the second model's cache, as many
        tokens as the target's (hybrid_cache/hybrid_pool_assembler.build_full_draft_pools, outside
        --hicache-size). Its size per token is the second model's, which the planner does not know, so it is
        counted as the target's."""
        kind = speculative.parse(cfg.spec_decode)[0] if cfg.spec_decode else None
        return 2 * host_bytes if kind in ("eagle3", "draft") else host_bytes

    # A start-up out-of-memory failure with a second model steps down from the lowered fraction, this far at most
    # (the search's own floor, 0.85, sits above it on a 24 GB card).
    SPEC_RETRY_DEPTH = 0.10

    def spec_fraction(self, spec: Optional[str], model: PreparedModel, hw: HardwareDescriptor) -> Optional[float]:
        """The largest --mem-fraction-static that leaves a spec's second weights room outside the reservation:
        1 - (draft weights + runtime workspace + 512 MB) / total VRAM, rounded down to 0.01 (0.86 for an 850 MB
        EAGLE3 head on a 24 GB card, 0.79 for Llama 3.2 1B). None: no second model (no spec, a lookup method) or no
        GPU, so no limit. 0.0 for a second model of unknown size: no fraction is known to leave it room.

        SGLang 0.5.19 loads the draft after the target, sizes its KV pools after both, and scales the pool's cell
        size for the draft's KV cache, so that cache lives inside the fraction; the planner does not count it, so
        the pool holds fewer tokens than it estimates. Whether the first sizing subtracts the draft's weights was
        not verified, so they are assumed to need room outside the fraction."""
        if not spec or hw.gpu is None or not hw.gpu.vram_total_bytes:
            return None
        extra = speculative.draft_bytes(spec, model)
        if extra == 0:
            return None
        if extra is None:
            return 0.0
        outside = extra + self.memory_model(hw).runtime_workspace + MIN_SAFETY_MARGIN
        return math.floor((1 - outside / hw.gpu.vram_total_bytes) * 100) / 100

    def spec_retry_floor(self, cfg: Config, model: PreparedModel, hw: HardwareDescriptor) -> Optional[float]:
        """How low an out-of-memory retry may take a second-model spec trial: SPEC_RETRY_DEPTH below its cap."""
        cap = self.spec_fraction(cfg.spec_decode, model, hw)
        return round(cap - self.SPEC_RETRY_DEPTH, 2) if cap else None

    def spec_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                      all_methods: bool = False) -> List[Config]:
        """Under --speculative all only: n-gram lookup, the verified EAGLE3 head and the draft model, as far as the
        installed SGLang has them. A second model needs `hw` to be left room, and is offered only where the
        planner says the configuration still fits at the lower fraction. Fractions start from the one before any
        speculation lowered it, so the variants of a draft-model leader are not measured at its 0.79."""
        if not all_methods:
            return []
        methods = sglang_spec_methods()
        specs: List[str] = []
        if "ngram" in methods:
            specs.append(speculative.ngram(speculative.NGRAM_TOKENS_SGLANG))
        if hw is not None:
            if "eagle3" in methods and "eagle3" in model.spec_repos:
                specs.append(speculative.eagle3(model.spec_repos["eagle3"], speculative.EAGLE3_TOKENS))
            if "draft" in methods and "draft" in model.spec_repos:
                specs.append(speculative.draft(model.spec_repos["draft"], speculative.DRAFT_TOKENS_SGLANG))
        base = cfg.gmu_before_spec if cfg.gmu_before_spec is not None else cfg.gpu_memory_utilization
        out: List[Config] = []
        for s in specs:
            if s == cfg.spec_decode:
                continue
            v = cfg.model_copy(update={"spec_decode": s, "gpu_memory_utilization": base, "gmu_before_spec": None})
            cap = self.spec_fraction(s, model, hw) if hw is not None else None
            if cap is not None and (base is None or base > cap):
                v = v.model_copy(update={"gpu_memory_utilization": cap, "gmu_before_spec": base})
                try:
                    if not cap or not estimate(hw, model, v, self.memory_model(hw)).feasible:
                        continue
                except Exception:
                    continue
            out.append(v)
        return out

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
