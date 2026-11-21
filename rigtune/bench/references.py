"""Reference configurations: what you get by *not* tuning.

- `<backend>-default`: the backend launched the way its own docs launch it (vLLM `vllm serve
  <model>`, SGLang `launch_server`, llama.cpp `llama-server -m model.gguf`).
- `ollama-default`: the real Ollama binary serving its own quant with its own settings.
- `<backend>-bare` (`compare --bare-stock` only): vLLM or SGLang launched with the model and port alone.

These run through the same trial runner and workload as rigtune's winner, so every row in a
results file was measured the same way at roughly the same time on the same GPU.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from rigtune.backends.base import BaseBackend, LaunchSpec, LlmtraceHooks, Process
from rigtune.memory import MemoryModel
from rigtune.models import Config, HardwareDescriptor, ModelSpec, PreparedModel
from rigtune.quantized import stored_quant

logger = logging.getLogger(__name__)


def reference_configs(hw: HardwareDescriptor, prepared: Dict[str, PreparedModel], reg) -> Dict[str, Config]:
    """Stock-default Config per candidate backend, keyed by '<backend>-default'."""
    out: Dict[str, Config] = {}
    for name, model in prepared.items():
        backend = reg[name]
        max_pos = model.arch.max_position_embeddings
        # The model's own weights: a checkpoint its author quantized is served as stored, as `vllm serve` does.
        stored = stored_quant(model.weights_bytes)
        if name == "vllm":
            # `vllm serve <model>`: max-model-len = model max, max-num-seqs 256, gpu-memory-utilization 0.9, dtype auto
            out[f"{name}-default"] = Config(
                backend=name, quant=stored or backend.precisions(hw)[0], ctx=max_pos, batch=256,
                gpu_memory_utilization=0.90,
            )
            # `vllm serve <model> --quantization fp8` and nothing else: isolates what the search adds
            # beyond simply picking 8-bit weights, which is otherwise the bulk of the measured gain.
            if "fp8" in backend.precisions(hw) and "fp8" in model.weights_bytes:
                out[f"{name}-fp8-default"] = Config(
                    backend=name, quant="fp8", ctx=max_pos, batch=256, gpu_memory_utilization=0.90
                )
        elif name == "sglang":
            # `python -m sglang.launch_server --model-path <model>`: context = model max, mem-fraction ~0.88
            out[f"{name}-default"] = Config(
                backend=name, quant=stored or backend.precisions(hw)[0], ctx=max_pos, batch=256,
                gpu_memory_utilization=0.88,
            )
        elif name.startswith("llamacpp"):
            # `llama-server -m model-Q4_K_M.gguf`: -c 4096, -np 1, -b 2048, all layers on GPU when built with CUDA
            quants = list(model.weights_bytes)
            quant = "Q4_K_M" if "Q4_K_M" in quants else quants[0]
            cfg = Config(
                backend=name, quant=quant, ctx=min(4096, max_pos), batch=1,
                n_gpu_layers=(model.arch.num_layers + 1) if backend.cuda else 0, n_batch=2048,
            )
            if not backend.cuda:
                cfg.extra["threads"] = hw.cpu.physical_cores
            out[f"{name}-default"] = cfg
        elif name == "vllm-cpu":
            out[f"{name}-default"] = Config(backend=name, quant=stored or "bf16", ctx=min(4096, max_pos), batch=256)
    return out


# --------------------------------------------------------------------------- engines at their own defaults

BARE_ENGINES = ("vllm", "sglang")
BARE_KEY = "engine defaults"  # the config key a bare row shows: its Config carries no settings


class BareStock(BaseBackend):
    """An engine launched with the model and the port alone, every other setting at the engine's own default
    (`compare --bare-stock`): `vllm serve <model>` and `python -m sglang.launch_server --model-path <model>`.

    The `<engine>-default` rows pass what rigtune takes the defaults to be (`--max-num-seqs 256`, the model's
    full context, SGLang's `--mem-fraction-static 0.88`), and those are not always the engine's own: vLLM's
    `max_num_seqs` is 1024 from 70 GiB except on an A100 (engine/arg_utils.py EngineArgs.get_batch_defaults),
    and SGLang computes its memory fraction from the card. Those rows stay as they are, so every earlier
    result keeps its meaning; a bare row stands beside them. The host is pinned to 127.0.0.1, because rigtune
    never serves on every interface, and the host changes nothing that is measured."""

    def __init__(self, inner: BaseBackend):
        self.inner = inner
        self.name = f"{inner.name}-bare"
        self.health_path = getattr(inner, "health_path", "/health")

    def available(self, hw: HardwareDescriptor) -> bool:
        return self.inner.available(hw)

    def version(self, hw: HardwareDescriptor) -> Optional[str]:
        return self.inner.version(hw)

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return self.inner.memory_model(hw)

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return self.inner.workload_hooks(hw, model)

    def known_failure(self, cfg: Config) -> Optional[str]:
        return None

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        path = model.spec.hf_id  # the model's own repository: it is also the name the engine serves it under
        if self.inner.name == "vllm":
            from rigtune.backends.vllm import vllm_serve_head

            args = list(vllm_serve_head(path))
        elif self.inner.name == "sglang":
            from rigtune.hardware import sglang_python

            args = [sglang_python() or sys.executable, "-m", "sglang.launch_server", "--model-path", path]
        else:
            raise ValueError(f"no bare launch for {self.inner.name}")
        args += ["--host", "127.0.0.1", "--port", str(port)]
        if model.spec.revision:
            args += ["--revision", model.spec.revision]
        return LaunchSpec(args=args, env={})


def bare_configs(prepared: Dict[str, PreparedModel]) -> Dict[str, Config]:
    """A row per engine this comparison has that can be launched bare, keyed `<engine>-bare`."""
    return {f"{name}-bare": Config(backend=f"{name}-bare", quant="engine-default", ctx=0, batch=0)
            for name in BARE_ENGINES if name in prepared}


# --------------------------------------------------------------------------- Ollama

_OLLAMA_TAGS = [
    (r"^(?:meta-llama|unsloth)/Llama-3\.2-(\d+)B", r"llama3.2:\1b"),
    (r"^(?:meta-llama|unsloth)/(?:Meta-)?Llama-3\.1-(\d+)B", r"llama3.1:\1b"),
    (r"^meta-llama/Meta-Llama-3-(\d+)B", r"llama3:\1b"),
    (r"^Qwen/Qwen2\.5-(\d+(?:\.\d+)?)B", r"qwen2.5:\1b"),
    (r"^Qwen/Qwen3-(\d+(?:\.\d+)?)B", r"qwen3:\1b"),
    (r"^mistralai/Mistral-7B", r"mistral:7b"),
    (r"^google/gemma-2-(\d+)b", r"gemma2:\1b"),
    (r"^google/gemma-3-(\d+)b", r"gemma3:\1b"),
    (r"^microsoft/Phi-3-mini", r"phi3:mini"),
    (r"^microsoft/Phi-4", r"phi4"),
]


def ollama_tag_for(hf_id: str) -> Optional[str]:
    """Best-effort HF id -> Ollama library tag (Ollama's default quant is Q4_K_M for these)."""
    for pat, repl in _OLLAMA_TAGS:
        if re.match(pat, hf_id, re.I):
            return re.sub(pat, repl, hf_id, flags=re.I).split("-")[0]
    return None


def ollama_binary() -> Optional[str]:
    env = os.environ.get("OLLAMA_BIN")
    if env and os.path.exists(env):
        return env
    return shutil.which("ollama")


class OllamaReference(BaseBackend):
    """Real Ollama as a comparison row. Not a serving candidate; never selected by the selector."""

    name = "ollama"
    health_path = "/"

    def __init__(self, tag: str):
        self.tag = tag

    def available(self, hw: HardwareDescriptor) -> bool:
        return ollama_binary() is not None

    def version(self, hw: HardwareDescriptor) -> Optional[str]:
        b = ollama_binary()
        if not b:
            return None
        try:
            out = subprocess.run([b, "--version"], capture_output=True, text=True, timeout=15)
            m = re.search(r"(\d+\.\d+\.\d+)", out.stdout + out.stderr)
            return m.group(1) if m else None
        except Exception:
            return None

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool:
        return self.available(hw)

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        return MemoryModel(runtime_workspace=0, kv_tokens_fn=lambda c: 0, device="gpu" if hw.gpu else "cpu")

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants=None) -> PreparedModel:
        from rigtune.hfconfig import load_arch

        return PreparedModel(spec=model, backend=self.name, arch=load_arch(model), hf_path=self.tag,
                             weights_bytes={"ollama": 0})

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> Config:
        # Ollama picks num_ctx / num_parallel / quant itself; the Config only labels the row.
        return Config(backend=self.name, quant="ollama", ctx=0, batch=0, extra={"tag": self.tag})

    def candidate_configs(self, hw, model, min_ctx: int = 0) -> List[Config]:
        return [self.default_config(hw, model, min_ctx)]

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        b = ollama_binary()
        if b is None:
            raise RuntimeError("ollama binary not found (set $OLLAMA_BIN)")
        return LaunchSpec(args=[b, "serve"], env={"OLLAMA_HOST": f"127.0.0.1:{port}", "OLLAMA_KEEP_ALIVE": "30m"})

    def launch(self, cfg: Config, model: PreparedModel, port: int, log_path: Optional[Path] = None) -> Process:
        proc = super().launch(cfg, model, port, log_path=log_path)
        if not proc.wait_ready(timeout=120, poll=0.5):
            return proc
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}")
        logger.info("ollama pull %s", self.tag)
        subprocess.run([ollama_binary(), "pull", self.tag], env=env, check=True, timeout=3600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return proc

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(
            health_path="/", model_name=self.tag, tokenizer_id=model.spec.hf_id,
            gpu_ids=[hw.gpu.index] if hw.gpu else [], process_memory=hw.gpu is None,
        )
