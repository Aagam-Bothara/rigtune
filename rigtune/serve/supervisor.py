"""Keep the winning backend alive: launch, health-check, restart with backoff."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import httpx

from rigtune.backends.base import BaseBackend, Process, free_port
from rigtune.models import Config, PreparedModel

logger = logging.getLogger(__name__)

# Failed health polls in a row before a live backend is restarted, and before /health reports it not ready: one
# slow poll under load is not an outage, and readiness should not flap where the restart logic would wait.
MISSES_TOLERATED = 3


class Supervisor:
    def __init__(
        self,
        backend: BaseBackend,
        cfg: Config,
        model: PreparedModel,
        port: Optional[int] = None,
        log_path: Optional[Path] = None,
        startup_timeout: float = 900.0,
        health_interval: float = 5.0,
        max_restarts: int = 5,
        power: Optional[object] = None,
        gpu_index: Optional[int] = None,
    ):
        self.gpu_index = gpu_index  # set for one replica of the replicas layout
        self.backend = backend
        self.cfg = cfg
        self.model = model
        self.port = port or free_port()
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.health_interval = health_interval
        self.max_restarts = max_restarts
        self.process: Optional[Process] = None
        self.restarts = 0
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Held while a child is created or stopped, never while it starts up, so stop() can always reach it.
        self._lock = threading.Lock()
        self._ready = False  # started, and not MISSES_TOLERATED health polls in a row failed since
        self.power = power  # rigtune.power.PowerController when the profile carries a power setting
        self.power_error: Optional[str] = None
        self.tier_warning: Optional[str] = None  # set when a KV-cache tier pick no longer fits this host

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return self.base_url + self.backend.health_path

    @property
    def _tiered(self) -> bool:
        return self.cfg.kv_tier is not None and self.gpu_index is None  # replicas launch without a tier

    def _check_tier(self) -> None:
        """A KV-cache tier was sized for this host when it was calibrated. Check it again before serving, by
        the same rules calibration used (pipeline.tier_refusal: the engine's own, host RAM and /dev/shm, the
        disk tier's engine, filesystem and space). When it no longer fits, serve the same configuration
        without the tier, with a warning, rather than launch into a failure the check already predicts (or
        swap the host). That covers a profile with a vLLM disk tier, which is no longer offered, and one whose
        disk tier would now land on a network filesystem: neither is launched, nor launched host-only."""
        if not self._tiered:
            return
        from rigtune.hardware import probe_cpu
        from rigtune.models import HardwareDescriptor
        from rigtune.pipeline import tier_refusal

        try:
            hw = HardwareDescriptor(os="", python="", cpu=probe_cpu())
            reason = tier_refusal(hw, self.backend, self.model, self.cfg)
        except Exception as exc:
            reason = f"its host memory could not be checked ({exc})"
        if reason:
            self.tier_warning = f"serving without the {self.cfg.kv_tier} KV-cache tier calibrated: {reason}"
            logger.warning(self.tier_warning)
            self.cfg = self.cfg.model_copy(update={"kv_tier": None})

    def start(self) -> None:
        from rigtune import kvtier

        kvtier.sweep()  # disk tiers a killed rigtune left behind
        self._check_tier()
        if not self._launch():
            return  # stopped while starting
        with self._lock:
            if self._stop.is_set():
                return
            self._apply_power()
            self._thread = threading.Thread(target=self._watch, name="rigtune-supervisor", daemon=True)
            self._thread.start()

    def _apply_power(self) -> None:
        """Serve at the calibrated power setting. Failure is not fatal: the server runs uncapped."""
        if self.cfg.power_limit_w is None and self.cfg.sm_clock_mhz is None:
            return
        from rigtune.power import setting_of

        if self.power is None:
            self.power_error = "profile has a power setting but no power controller was provided"
            logger.warning(self.power_error)
            return
        try:
            self.power.apply(setting_of(self.cfg))  # type: ignore[attr-defined]
        except Exception as exc:
            self.power_error = str(exc)
            logger.warning("serving without the calibrated power setting: %s", exc)

    def _launch(self) -> bool:
        """Start the backend and wait until it is ready. False when stop() came first: no child is left running."""
        with self._lock:
            if self._stop.is_set():
                return False
            self._ready = False
            if self._tiered:
                from rigtune import kvtier

                kvtier.clear(self.port)  # a restarted engine starts with an empty disk tier
                if self.cfg.tier_gib[1]:
                    kvtier.prepare(self.port)
            if self.gpu_index is not None:
                spec = self.backend.replica_launch_spec(self.cfg, self.model, self.port, self.gpu_index)
                self.process = Process(spec, self.port, self.health_url, log_path=self.log_path).start()
            else:
                self.process = self.backend.launch(self.cfg, self.model, self.port, log_path=self.log_path)
            process = self.process
        # Startup takes minutes; stop() must be able to end it, so the lock is not held while waiting.
        ready = process.wait_ready(timeout=self.startup_timeout)
        with self._lock:
            if self._stop.is_set():
                process.stop()
                return False
            if not ready:
                tail = process.tail_log(30)
                process.stop()
                raise RuntimeError(f"{self.backend.name} failed to start on port {self.port}\n{tail}")
            self._ready = True
            self.last_error = None
            logger.info("%s ready on %s (pid %s)", self.backend.name, self.base_url, process.pid)
            return True

    def healthy(self) -> bool:
        """One health poll: the process is alive and its health endpoint answers 2xx."""
        if self._stop.is_set() or self.process is None or not self.process.alive():
            return False
        try:
            with httpx.Client(timeout=5.0) as c:
                return 200 <= c.get(self.health_url).status_code < 300
        except httpx.HTTPError:
            return False

    def _watch(self) -> None:
        misses = 0
        while not self._stop.wait(self.health_interval):
            if self.healthy():
                misses = 0
                self._ready = True
                continue
            misses += 1
            alive = self.process is not None and self.process.alive()
            if misses >= MISSES_TOLERATED or not alive:
                self._ready = False
            if misses < MISSES_TOLERATED and alive:
                continue  # transient; give it a couple of intervals
            if self.restarts >= self.max_restarts:
                self.last_error = f"backend died and restart budget ({self.max_restarts}) exhausted"
                logger.error(self.last_error)
                return
            self.restarts += 1
            delay = min(60.0, 2.0 ** self.restarts)
            logger.warning("backend unhealthy; restart %d/%d in %.0fs", self.restarts, self.max_restarts, delay)
            if self.process is not None:
                self.process.stop()
            if self._stop.wait(delay):  # a shutdown during the backoff ends it, and launches nothing
                return
            try:
                if not self._launch():
                    return
                misses = 0
            except Exception as exc:
                self.last_error = str(exc)
                logger.error("restart failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:  # a child created before this point is stopped here; _launch creates none after it
            self._ready = False
            if self.process is not None:
                self.process.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._tiered:
            from rigtune import kvtier

            kvtier.clear(self.port)  # the disk tier holds served prompts' KV cache: never left behind
        if self.power is not None and getattr(self.power, "applied", None) is not None:
            try:
                self.power.restore()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.error("power restore failed: %s (run `rigtune power reset`)", exc)

    def status(self) -> dict:
        return {
            "backend": self.backend.name,
            "gpu": self.gpu_index,
            "port": self.port,
            "pid": self.process.pid if self.process else None,
            "alive": bool(self.process and self.process.alive()),
            # Liveness says the process exists; readiness that it has started and still answers its health checks.
            "ready": bool(self._ready and not self._stop.is_set() and self.process and self.process.alive()),
            "restarts": self.restarts,
            "last_error": self.last_error,
            **({"kv_tier": {"tier": self.cfg.kv_tier, "warning": self.tier_warning}}
               if self._tiered or self.tier_warning else {}),
            "power": {
                "applied": (self.power.applied.label()  # type: ignore[attr-defined]
                            if self.power is not None and getattr(self.power, "applied", None) else None),
                "error": self.power_error,
            },
        }
