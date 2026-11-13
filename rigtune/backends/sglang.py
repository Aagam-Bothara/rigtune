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

from rigtune import speculative
from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, ctx_grid, render_extra
from rigtune.backends.vllm import KNOWN_ARCHS, PAGED_KV_FRACTION
from rigtune.hfconfig import dtype_bytes, load_arch
from rigtune.memory import MIN_SAFETY_MARGIN, MemoryModel, estimate
from rigtune.models import Config, GiB, HardwareDescriptor, ModelSpec, PreparedModel
from rigtune.hardware import nvlink_between, sglang_python
from rigtune.quantized import INT4_METHODS, hf_weight_options

logger = logging.getLogger(__name__)


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
        return self.calibrated_memory(hw, lambda cfg: int(cfg.ctx * cfg.batch * PAGED_KV_FRACTION), "gpu")

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
        return LaunchSpec(args=args, env=env)

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
