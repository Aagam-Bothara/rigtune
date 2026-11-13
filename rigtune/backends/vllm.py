"""vLLM backend (NVIDIA, compute capability >= 7.5)."""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from rigtune import speculative
from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid, render_extra
from rigtune.hfconfig import dtype_bytes, kv_element_bytes, load_arch
from rigtune.memory import HybridKV, MemoryModel
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

# Architectures known to load in vLLM; consulted only when the live registry is unavailable.
KNOWN_ARCHS = {
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
}

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
    """
    elem = kv_element_bytes(cfg.kv_dtype if cfg.kv_dtype != "auto" else arch.torch_dtype)
    k = speculative.parse(cfg.spec_decode)[2] if cfg.spec_decode else 0
    n_full, n_sliding, n_linear = arch.full_attention_layers or 0, arch.sliding_layers, arch.linear_layers
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


class VllmBackend(BaseBackend):
    name = "vllm"
    runtime_workspace_bytes = int(1.5 * GiB)  # CUDA context + CUDA graphs + activation workspace

    # ---- capability

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        if hw.gpu is None or hw.gpu.vendor != "nvidia" or hw.gpu.cc < (7, 5):
            return False
        try:
            arch = load_arch(model).architecture
        except Exception as exc:
            logger.warning("could not read config for %s: %s", model.hf_id, exc)
            return True
        live = vllm_registry_archs()
        if live is not None:
            return arch in live
        return arch in KNOWN_ARCHS or arch == "unknown"

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
        if cc >= (8, 9) and _vllm_version() >= INT8_KV_FROM:
            out.append(INT8_KV)
        return out

    def batch_ladder(self) -> Tuple[int, ...]:
        return (16, 64, 256, 512)

    def spec_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                      all_methods: bool = False) -> List[Config]:
        # n-gram lookup on the GPU where vLLM has it, since the CPU proposer turns async scheduling off.
        specs = [speculative.ngram_gpu() if speculative.vllm_supports_ngram_gpu() else speculative.ngram()]
        if speculative.vllm_supports_suffix():
            specs.append(speculative.suffix())
        draft = speculative.draft_for(model.spec.hf_id)
        if draft and speculative.vllm_supports_draft():
            specs.append(speculative.draft(draft, speculative.DRAFT_TOKENS_VLLM))
        head = model.spec_repos.get("eagle3")
        if all_methods and head and speculative.vllm_supports_eagle3():  # the head and tokens SGLang is offered
            specs.append(speculative.eagle3(head, speculative.EAGLE3_TOKENS))
        out = [cfg.model_copy(update={"spec_decode": s}) for s in specs if s != cfg.spec_decode]
        return [v for v in out if self.known_failure(v) is None]

    def known_failure(self, cfg: Config) -> Optional[str]:
        """Why `cfg` is expected to fail, when it is. On an A40, vLLM 0.29's draft model on the fp8 KV cache through
        FlashInfer (the Ampere route) failed most requests with "CUDA error: an illegal memory access", each working
        alone; those trials were measured and dropped as failures. An EAGLE3 head takes the same proposer path, so
        that pair is not tried at all (the draft-model pair is still tried, as before)."""
        if (cfg.spec_decode or "").startswith("eagle3:") and cfg.kv_dtype == AMPERE_FP8_KV:
            return "EAGLE3 with the fp8 KV cache through FlashInfer (the draft model failed so on an A40)"
        return None

    def find_second_models(self, prepared: PreparedModel) -> List[str]:
        """Under --speculative all: the verified EAGLE3 head for this model, the one SGLang gets, so the two engines
        are compared on one proposer. vLLM plans no memory for it, so its size is recorded but not needed."""
        if not speculative.vllm_supports_eagle3() or not speculative.eagle3_heads_for(prepared.spec.hf_id):
            return []
        try:
            found = speculative.find_eagle3_head(prepared.spec, prepared.arch)
        except Exception as exc:
            note = f"vllm: no EAGLE3 head tried, the Hub could not be asked ({exc})"
            logger.warning(note)
            return [note]
        if found:
            prepared.spec_repos["eagle3"] = found[0]
            if found[1]:
                prepared.spec_bytes[found[0]] = int(found[1])
        return []

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        return model  # vLLM pulls HF weights itself at launch

    # ---- memory

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return self.calibrated_memory(hw, lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION), "gpu",
                                      hybrid_fn=lambda cfg, arch: vllm_hybrid_kv(
                                          cfg, arch, vllm_default_batched_tokens(hw)))

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
            args += ["--speculative-config", json.dumps(speculative.vllm_config(cfg.spec_decode))]
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
            if _attention_backend_flag():
                args += ["--attention-backend", str(attention)]
        # PCIe-only GPUs: peer-to-peer can hang at start-up inside containers. Measured on a pair of
        # A40s: NCCL's P2P path hangs at init, and with only that disabled the engine's custom
        # all-reduce (CUDA IPC, also P2P) hangs next. With both off it starts in under a minute.
        if cfg.tp > 1 and nvlink_between(list(range(cfg.tp))) is False:
            env["NCCL_P2P_DISABLE"] = "1"
            args += ["--disable-custom-all-reduce"]
        return LaunchSpec(args=args, env=env)

    # Chunked-prefill token budgets to try. Small budgets interleave prefill with decode and protect
    # per-token latency; large ones finish long prompts in fewer steps and cut time to first token.
    PREFILL_BUDGETS = (2048, 8192, 16384)

    def prefill_variants(self, cfg: Config) -> List[Config]:
        # vLLM requires max_num_batched_tokens >= max_num_seqs.
        return [cfg.model_copy(update={"prefill_budget": b}) for b in self.PREFILL_BUDGETS
                if b >= cfg.batch and b != cfg.prefill_budget]

    def disagg_launch_spec(self, cfg: Config, model: PreparedModel, port: int, role: str,
                           kv_transfer_config: dict, gpu_index: int, side_channel_port: int) -> LaunchSpec:
        spec = self.launch_spec(cfg, model, port)
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
        )
