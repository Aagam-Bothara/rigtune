"""Backend interface (spec section 3) plus the subprocess wrapper every backend uses."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import httpx

from rigtune.memory import HOST_TIER_RESERVE, MemoryModel
from rigtune.models import Config, HardwareDescriptor, ModelSpec, PreparedModel

logger = logging.getLogger(__name__)


@dataclass
class LlmtraceHooks:
    """How the calibration driver should talk to (and measure) this backend."""

    completions_path: str = "/v1/completions"
    # Session files are replayed as messages, so the engine applies its own chat template and turn n+1 starts with
    # exactly the bytes it rendered for turn n.
    chat_path: str = "/v1/chat/completions"
    health_path: str = "/health"
    models_path: str = "/v1/models"
    stream_usage: bool = True  # backend reports usage in the final stream chunk when asked
    model_name: Optional[str] = None  # value to put in the "model" field of requests
    tokenizer_id: Optional[str] = None  # HF id whose tokenizer counts tokens when usage is absent
    gpu_ids: List[int] = field(default_factory=list)  # NVML indices llmtrace should sample
    process_memory: bool = False  # sample RSS of the backend process (CPU backends)
    # Prometheus endpoint with prefix-cache counters (vLLM), read around each level of a session file whose replies
    # carry no cached-token counts.
    metrics_path: Optional[str] = None
    # A trial with a KV-cache tier (Config.kv_tier): record where cached tokens came from and time to first
    # token on returning turns. Set by the trial runner; off for every other trial, whose metrics stay put.
    tier_report: bool = False
    # The request field asking the engine to split each reply's cached tokens by tier (SGLang's
    # return_cached_tokens_details, BaseBackend.tier_details_field), set with tier_report.
    tier_details_field: Optional[str] = None


@dataclass
class LaunchSpec:
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None


class Process:
    """A launched backend server: subprocess + readiness probe + logs."""

    def __init__(self, spec: LaunchSpec, port: int, health_url: str, log_path: Optional[Path] = None):
        self.spec = spec
        self.port = port
        self.health_url = health_url
        self.log_path = log_path
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self.started_at: Optional[float] = None
        self.ready_at: Optional[float] = None

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    def start(self) -> "Process":
        env = dict(os.environ)
        env.update(self.spec.env)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "ab")
            stdout = self._log_fh
        else:
            stdout = subprocess.DEVNULL
        logger.info("launch: %s", " ".join(self.spec.args))
        kwargs = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True  # own process group so we can kill children
        self._proc = subprocess.Popen(
            self.spec.args,
            env=env,
            cwd=self.spec.cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
        self.started_at = time.monotonic()
        return self

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self._proc.poll() if self._proc else None

    def wait_ready(self, timeout: float = 600.0, poll: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if not self.alive():
                    logger.error("backend exited during startup (rc=%s); see %s", self.returncode(), self.log_path)
                    return False
                try:
                    r = client.get(self.health_url)
                    if r.status_code < 500:
                        self.ready_at = time.monotonic()
                        return True
                except httpx.HTTPError:
                    pass
                time.sleep(poll)
        logger.error("backend not ready after %.0fs; see %s", timeout, self.log_path)
        return False

    def stop(self, grace: float = 15.0) -> None:
        if self._proc is None:
            return
        if self.alive():
            try:
                if sys.platform != "win32":
                    os.killpg(self._proc.pid, signal.SIGTERM)  # own session: group id == server pid
                else:
                    self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    if sys.platform != "win32":
                        os.killpg(self._proc.pid, signal.SIGKILL)
                    else:
                        self._proc.kill()
                except Exception:
                    pass
                self._proc.wait(timeout=10)
        if sys.platform != "win32":
            # The server can be gone while its children are not: a vLLM API server that died leaves
            # VLLM::EngineCore holding the GPU, and every later trial then sees less memory. The
            # children share the server's process group, so reap whatever is left of it.
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except OSError:
                pass
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def tail_log(self, n: int = 40) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""


CTX_GRID = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
CTX_GRID_WIDTH = 3  # sizes tried per calibration: the smallest that fits the workload and the next two


def ctx_grid(max_pos: int, min_ctx: int = 0) -> List[int]:
    """Context lengths to try: the three smallest standard sizes that fit the model and hold the workload.

    Returns [] if the model cannot hold `min_ctx` at all.
    """
    if min_ctx > max_pos:
        return []
    grid = [c for c in CTX_GRID if min_ctx <= c <= max_pos]
    if not grid:
        # Workload needs more than the largest standard size that fits: use the model max.
        grid = [max_pos]
    return grid[:CTX_GRID_WIDTH]


def render_extra(extra: Dict[str, object], skip: Tuple[str, ...] = ()) -> List[str]:
    """Extra launch flags. `True` renders as a bare flag; `False` and None are dropped."""
    out: List[str] = []
    for k, v in extra.items():
        if k in skip or v is None or v is False:
            continue
        flag = f"--{k.replace('_', '-')}"
        out += [flag] if v is True else [flag, str(v)]
    return out


def free_port(preferred: Optional[int] = None) -> int:
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@runtime_checkable
class Backend(Protocol):
    name: str

    def available(self, hw: HardwareDescriptor) -> bool: ...

    def supports(self, hw: HardwareDescriptor, model: ModelSpec) -> bool: ...

    def prepare(self, model: ModelSpec, hw: HardwareDescriptor, quants: Optional[List[str]] = None) -> PreparedModel: ...

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel: ...

    def estimate_memory(self, cfg: Config, model: PreparedModel, hw: HardwareDescriptor) -> int: ...

    def candidate_configs(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> List[Config]: ...

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec: ...

    def launch(self, cfg: Config, model: PreparedModel, port: int, log_path: Optional[Path] = None) -> Process: ...

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks: ...

    def default_config(self, hw: HardwareDescriptor, model: PreparedModel, min_ctx: int = 0) -> Config: ...

    def version(self, hw: HardwareDescriptor) -> Optional[str]: ...


class BaseBackend:
    """Shared plumbing. Concrete backends override the abstract-ish methods."""

    name: str = "base"
    runtime_workspace_bytes: int = 1024 * 1024 * 1024
    # Launch with the engine's flag for per-request cached-token counts. Fixed when the registry is built, from the
    # workload (registry(report_reuse=...)): only session files ask for it, so no other launch arguments change.
    report_reuse: bool = False

    def available(self, hw: HardwareDescriptor) -> bool:
        return hw.backend_available(self.name)

    def version(self, hw: HardwareDescriptor) -> Optional[str]:
        return hw.backend_version(self.name)

    def memory_model(self, hw: HardwareDescriptor) -> MemoryModel:
        raise NotImplementedError

    def calibrated_memory(self, hw: HardwareDescriptor, kv_tokens_fn, device: str,
                          hybrid_fn=None, full_kv_fn=None) -> MemoryModel:
        """MemoryModel using this machine's fitted workspace/margin when `rigtune memory-report --apply` ran.

        `hybrid_fn` (Config, ArchInfo) -> memory.HybridKV is given only by backends whose hybrid-attention KV
        rules were read from the engine's source; `full_kv_fn` ArchInfo -> reason, by those that keep a full
        KV cache on every layer of some hybrid layouts."""
        from rigtune.hardware import hardware_hash
        from rigtune.memcal import margin_override, workspace_override
        from rigtune.memory import DEFAULT_MARGIN_FRACTION

        hh = hardware_hash(hw)
        ws = workspace_override(hh, self.name)
        mf = margin_override(hh, self.name)
        return MemoryModel(
            runtime_workspace=ws if ws is not None else self.runtime_workspace_bytes,
            kv_tokens_fn=kv_tokens_fn,
            device=device,
            margin_fraction=mf if mf is not None else DEFAULT_MARGIN_FRACTION,
            calibrated=ws is not None or mf is not None,
            hybrid_fn=hybrid_fn,
            full_kv_fn=full_kv_fn,
        )

    def estimate_memory(self, cfg: Config, model: PreparedModel, hw: HardwareDescriptor) -> int:
        from rigtune.memory import estimate

        return estimate(hw, model, cfg, self.memory_model(hw)).total

    def materialize(self, model: PreparedModel, quants: List[str]) -> PreparedModel:
        """Download / convert weights for the quants the planner kept. Default: nothing to do."""
        return model

    def prefill_variants(self, cfg: Config) -> List[Config]:
        """Configs differing from `cfg` only in the prefill knob. Default: this backend has none."""
        return []

    def disagg_launch_spec(self, cfg: Config, model: PreparedModel, port: int, role: str,
                           kv_transfer_config: dict, gpu_index: int, side_channel_port: int) -> LaunchSpec:
        """Launch one engine of a disaggregated prefill/decode pair."""
        raise NotImplementedError(f"{self.name} does not support disaggregated prefill/decode")

    # ---- optional search dimensions (default: this backend offers none)

    supports_tp: bool = False

    def supported_quants(self, hw: HardwareDescriptor) -> List[str]:
        """Every weight precision this backend could run on `hw` (what --quant filters)."""
        return []

    def kv_dtypes(self, hw: HardwareDescriptor) -> List[str]:
        """Quantized KV-cache types worth trying on `hw`."""
        return []

    def batch_ladder(self) -> Tuple[int, ...]:
        """Batch sizes, smallest first, that a smaller KV cache may let the search step up to."""
        return ()

    def prefix_variants(self, cfg: Config) -> List[Config]:
        """Prefix-cache settings to try when the workload's prompts share a prefix."""
        return []

    def spec_variants(self, cfg: Config, model: PreparedModel, hw: Optional[HardwareDescriptor] = None,
                      all_methods: bool = False) -> List[Config]:
        """Speculative-decoding settings to try. `hw` lets a backend leave memory for a second model, and
        `all_methods` (--speculative all) adds the methods that have not run on hardware yet."""
        return []

    def known_failure(self, cfg: Config) -> Optional[str]:
        """Why `cfg` is expected to fail at start-up or under load, so the search does not try it. Default: never."""
        return None

    def text_only_refusal(self) -> Optional[str]:
        """Why a multimodal model whose multimodal profiling failed at start-up cannot be retried serving its language
        model alone (Config.text_only), or None when it can. Default: this backend has no such mode."""
        return f"rigtune has no text-only launch for {self.name}"

    # ---- KV-cache tier (--kv-tier; default: this backend offers none)

    def tier_refusal(self, cfg: Config, model: PreparedModel) -> Optional[str]:
        """Why `cfg` cannot get a host-RAM KV-cache tier (Config.kv_tier), or None when it can. The engines'
        rules are read from their source; a backend without them offers no tier."""
        return f"{self.name} has no KV-cache tier"

    def tier_disk_refusal(self) -> Optional[str]:
        """Why no disk tier can sit behind the host one (under rigtune.kvtier.trial_dir), or None when it can.
        A disk tier is offered only where the engine bounds its size."""
        return f"{self.name} has no disk tier"

    def tier_host_cap(self) -> Optional[int]:
        """Host bytes the engine can give a tier at most, beyond the planner's budget (vLLM: /dev/shm)."""
        return None

    def tier_host_reserve(self) -> int:
        """Host RAM the tier must leave free: the planner's reserve, or more where the engine checks it."""
        return HOST_TIER_RESERVE

    def tier_host_need(self, cfg: Config, host_bytes: int) -> int:
        """Host RAM a tier of `host_bytes` takes with `cfg`: more than that where the engine adds pools."""
        return host_bytes

    def tier_details_field(self) -> Optional[str]:
        """The request field that makes each reply say which tier its cached tokens came from (LlmtraceHooks
        .tier_details_field), or None: the engine's /metrics counters, if any, are all a tier trial gets."""
        return None

    def find_second_models(self, prepared: PreparedModel) -> List[str]:
        """Under --speculative all: look up second models (EAGLE3 heads, drafts) on the Hub and record them in
        `prepared`. Returns notes for the profile, such as a lookup the Hub could not answer. Default: none."""
        return []

    def replica_launch_spec(self, cfg: Config, model: PreparedModel, port: int, gpu_index: int) -> LaunchSpec:
        """One full engine pinned to one GPU, for the replicas layout. Without a KV-cache tier: its host RAM
        was budgeted for one engine, and every replica would take it again."""
        spec = self.launch_spec(cfg.model_copy(update={"kv_tier": None}), model, port)
        spec.env = {**spec.env, "CUDA_VISIBLE_DEVICES": str(gpu_index)}
        return spec

    def launch_spec(self, cfg: Config, model: PreparedModel, port: int) -> LaunchSpec:
        raise NotImplementedError

    health_path: str = "/health"

    def workload_hooks(self, hw: HardwareDescriptor, model: PreparedModel) -> LlmtraceHooks:
        return LlmtraceHooks(gpu_ids=[hw.gpu.index] if hw.gpu else [])

    def launch(self, cfg: Config, model: PreparedModel, port: int, log_path: Optional[Path] = None) -> Process:
        spec = self.launch_spec(cfg, model, port)
        return Process(spec, port, f"http://127.0.0.1:{port}{self.health_path}", log_path=log_path).start()
