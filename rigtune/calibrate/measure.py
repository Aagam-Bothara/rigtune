"""Measure one trial: drive the workload over HTTP while sampling device telemetry.

Telemetry priority: llmtrace GPUSampler (NVML) -> bare pynvml -> psutil RSS (+ RAPL energy on Linux).
Request timing (TTFT / TPOT / tok/s) is measured at the HTTP client, which is the only vantage point
shared by all backends.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import queue
import re
import statistics
import threading
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from rigtune.backends.base import LlmtraceHooks
from rigtune.calibrate.datasets import CHARS_PER_TOKEN, Session, Step
from rigtune.calibrate.tokens import TokenCounter, worst_source
from rigtune.calibrate.workload import TURN_OVERHEAD, Workload
from rigtune.models import TrialMetrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- telemetry


@dataclass
class TelemetrySummary:
    source: str = "none"
    peak_mem_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    power_w: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    energy_j: Optional[float] = None
    samples: int = 0
    # The same window's energy from NVML's total-energy counter (Volta and newer, where the driver reports it),
    # or None. Sampled power on an A100 is a ~1 s board average read every 100 ms: its
    # integral is right over a long window and smears anything shorter, which the counter does not.
    counter_energy_j: Optional[float] = None


@dataclass
class _Sample:
    t: float
    mem_mb: Optional[float] = None
    util: Optional[float] = None
    power_w: Optional[float] = None
    clock_mhz: Optional[float] = None


def _summarize(samples: List[_Sample], source: str, extra_energy: Optional[float] = None) -> TelemetrySummary:
    s = TelemetrySummary(source=source, samples=len(samples))
    mems = [x.mem_mb for x in samples if x.mem_mb is not None]
    utils = [x.util for x in samples if x.util is not None]
    powers = [(x.t, x.power_w) for x in samples if x.power_w is not None]
    clocks = [x.clock_mhz for x in samples if x.clock_mhz is not None]
    if clocks:
        s.sm_clock_mhz = statistics.fmean(clocks)
    if mems:
        s.peak_mem_mb = max(mems)
    if utils:
        s.gpu_util_pct = statistics.fmean(utils)
    if powers:
        s.power_w = statistics.fmean(p for _, p in powers)
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(powers, powers[1:]):
            energy += 0.5 * (p0 + p1) * max(0.0, t1 - t0)
        s.energy_j = energy if len(powers) > 1 else None
    if extra_energy is not None:
        s.energy_j = extra_energy
    return s


@dataclass
class IdleSummary:
    """The card after the last request of a trial: its idle draw, and the wind-down ("hold") on the way to it.

    Closed-loop levels never let the GPU idle, so neither shows up in J/token; a server at a modest arrival rate
    pays both between requests. Measured on the tail of the window that measured the trial's last level.
    """

    idle_power_w: Optional[float] = None  # over the tail's last second, from the same source as hold_joules
    idle_power_sampled_w: Optional[float] = None  # the mean of the sampled power over that second, always
    hold_joules: Optional[float] = None  # energy above idle during the tail
    hold_s: Optional[float] = None  # time until sampled power came within 10% of the drop to idle
    truncated: bool = False  # power was still falling at the end of the tail, so idle reads high
    below_idle: bool = False  # the tail held less energy than idle x its length: noise; hold recorded as 0
    energy_source: str = "trapezoid"  # "nvml_counter" | "trapezoid": where idle_power_w and hold_joules came from
    tail_s: float = 0.0


IDLE_WINDOW_S = 1.0  # idle power is taken over the tail's final second
SETTLED = 0.10  # the hold is over once power is within this fraction of the drop from its start


def _idle_summary(points: List[Tuple[float, float]], t0: float, t_end: float, p_start: Optional[float],
                  counter_j: Optional[float], counter_idle_w: Optional[float] = None) -> Optional[IdleSummary]:
    """Idle power and hold energy from the power samples (t, W) taken after `t0`, the instant the last request
    finished. `p_start` is the last active sample's power; `counter_j` the energy counter over [t0, t_end] and
    `counter_idle_w` its mean power over the final second.

    Idle and hold come from one source, so the hold is a difference of like with like: the counter where the
    driver reports it (exact however short the hold), else the signed trapezoid of P - P_idle over the samples.
    NVML's power reading is a ~1 s average on A100-class boards, so the hold's duration as sampled is an upper
    bound while its energy is preserved. Nothing is clamped point by point, which would bias the hold upward by
    the noise above idle; a total below zero is recorded as 0 and flagged.
    """
    if len(points) < 2:
        return None
    tail_s = max(0.0, t_end - t0)
    last = [p for t, p in points if t >= t_end - IDLE_WINDOW_S] or [p for _, p in points[-2:]]
    sampled = statistics.fmean(last)
    p0 = p_start if p_start is not None else points[0][1]
    drop = max(0.0, p0 - sampled)
    settled_at = next((t - t0 for t, p in points if p - sampled <= SETTLED * drop), None)
    if counter_j is not None and counter_idle_w is not None:
        idle, source = counter_idle_w, "nvml_counter"
        hold = counter_j - idle * tail_s
    else:
        idle, source = sampled, "trapezoid"
        trace = [(t0, p0)] + points
        hold = sum(0.5 * ((a - idle) + (b - idle)) * max(0.0, tb - ta) for (ta, a), (tb, b) in zip(trace, trace[1:]))
    # Still falling: the first half of the final second sits clearly above the second half.
    window = [(t, p) for t, p in points if t >= t_end - IDLE_WINDOW_S]
    mid = t_end - IDLE_WINDOW_S / 2
    early, late = [p for t, p in window if t < mid], [p for t, p in window if t >= mid]
    falling = bool(early and late) and statistics.fmean(early) - statistics.fmean(late) > max(SETTLED * drop, 2.0)
    return IdleSummary(idle_power_w=idle, idle_power_sampled_w=sampled, hold_joules=max(0.0, hold),
                       hold_s=settled_at if settled_at is not None else tail_s, truncated=falling or settled_at is None,
                       below_idle=hold < 0, energy_source=source, tail_s=tail_s)


def _rapl_delta(start: Optional[float], end: Optional[float]) -> Optional[float]:
    return end - start if (start is not None and end is not None and end >= start) else None


class Telemetry:
    """Context manager sampling GPU (llmtrace/pynvml) or process (psutil) telemetry."""

    def __init__(self, hooks: LlmtraceHooks, pid: Optional[int] = None, interval_ms: int = 100,
                 clock: Callable[[], float] = time.monotonic):
        self.hooks = hooks
        self._clock = clock  # stamps the samples rigtune takes itself, and the marks; injectable for tests
        self.pid = pid
        self.interval_ms = interval_ms
        self._samples: List[_Sample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._llmtrace_sampler: Any = None
        self._source = "none"
        self._rapl_start: Optional[float] = None
        self._nvml: Any = None  # pynvml, while the energy counter is open
        self._counter_handles: List[Any] = []
        self._counter_start: Optional[float] = None
        self._mark: Optional[Tuple[float, float, Optional[float], Optional[float]]] = None
        self._idle_mark: Optional[Tuple[float, float, Optional[float]]] = None
        self.summary = TelemetrySummary()
        self.idle: Optional[IdleSummary] = None  # set when mark() opened an idle tail

    # ---- start/stop

    def __enter__(self) -> "Telemetry":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self.hooks.gpu_ids:
            self._open_counter()
        if self.hooks.gpu_ids and self._start_llmtrace():
            return
        if self.hooks.gpu_ids and self._start_pynvml():
            return
        self._start_process()

    def mark(self) -> None:
        """End the active part of the window here; what follows until stop() is the idle tail (see IdleSummary).

        The summary then describes the active part alone, exactly as if the window had closed at the mark, and
        `idle` describes the tail.
        """
        self._mark = (self._clock(), time.time(), self._counter_now(),
                      _rapl_energy_j() if self._rapl_start is not None else None)

    def mark_idle(self) -> None:
        """Start of the tail's final second: the counter's reading from here to stop() gives idle power."""
        self._idle_mark = (self._clock(), time.time(), self._counter_now())

    def stop(self) -> None:
        end_mono, end_wall, counter_end = self._clock(), time.time(), self._counter_now()
        samples, wall_clock = self._stop_sampling()
        rapl_end = _rapl_energy_j() if self._rapl_start is not None else None
        if self._nvml is not None:
            _shutdown(self._nvml)
            self._nvml = None
        if self._mark is None:
            self.summary = _summarize(samples, self._source, _rapl_delta(self._rapl_start, rapl_end))
            self.summary.counter_energy_j = _counter_delta_j(self._counter_start, counter_end)
            return
        mono, wall, counter_mid, rapl_mid = self._mark
        split, t_end = (wall, end_wall) if wall_clock else (mono, end_mono)
        active = [x for x in samples if x.t <= split]
        self.summary = _summarize(active, self._source, _rapl_delta(self._rapl_start, rapl_mid))
        self.summary.counter_energy_j = _counter_delta_j(self._counter_start, counter_mid)
        before = [x.power_w for x in active if x.power_w is not None]
        tail = [(x.t, x.power_w) for x in samples if x.t > split and x.power_w is not None]
        counter_idle_w = None
        if self._idle_mark is not None:
            i_mono, i_wall, counter_idle = self._idle_mark
            last_s = (end_wall - i_wall) if wall_clock else (end_mono - i_mono)
            last_j = _counter_delta_j(counter_idle, counter_end)
            counter_idle_w = last_j / last_s if (last_j is not None and last_s > 0) else None
        self.idle = _idle_summary(tail, split, t_end, before[-1] if before else None,
                                  _counter_delta_j(counter_mid, counter_end), counter_idle_w)

    # ---- the energy counter, read beside whichever sampler runs

    def _open_counter(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
        except Exception:
            return
        try:
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.hooks.gpu_ids]
            start = _counter_mj(pynvml, handles)
        except Exception:
            start = None
        if start is None:
            _shutdown(pynvml)
            return
        self._nvml, self._counter_handles, self._counter_start = pynvml, handles, start

    def _counter_now(self) -> Optional[float]:
        return _counter_mj(self._nvml, self._counter_handles) if self._nvml is not None else None

    def _stop_sampling(self) -> Tuple[List[_Sample], bool]:
        """Stop the sampler; its samples, and whether their clock is wall time rather than monotonic."""
        if self._llmtrace_sampler is not None:
            samp = self._llmtrace_sampler
            samp.stop()
            raw = samp.drain()
            samples = [
                _Sample(
                    t=(x.monotonic if x.monotonic is not None else x.timestamp),
                    mem_mb=x.memory_used_mb,
                    util=x.gpu_utilization_pct,
                    power_w=x.power_draw_watts,
                    clock_mhz=x.sm_clock_mhz,
                )
                for x in raw
            ]
            return samples, any(x.monotonic is None for x in raw)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self._samples, False

    # ---- backends

    def _start_llmtrace(self) -> bool:
        if importlib.util.find_spec("llmtrace") is None:
            return False
        try:
            from llmtrace.data_plane.gpu_sampler import GPUSampler  # type: ignore
            from llmtrace.models.config import GPUSamplerConfig  # type: ignore

            sampler = GPUSampler(GPUSamplerConfig(gpu_ids=list(self.hooks.gpu_ids), sample_interval_ms=self.interval_ms))
            sampler.start()
            if not sampler.available:
                logger.debug("llmtrace sampler unavailable: %s", sampler.unavailable_reason)
                return False
            self._llmtrace_sampler = sampler
            self._source = "llmtrace"
            return True
        except Exception as exc:
            logger.debug("llmtrace sampler failed: %s", exc)
            return False

    def _start_pynvml(self) -> bool:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.hooks.gpu_ids]
        except Exception:
            return False

        def read() -> _Sample:
            mem = util = power = clock = None
            try:
                mem = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles) / 2**20
                util = statistics.fmean(pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in handles)
                power = sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in handles) / 1000.0
                clock = statistics.fmean(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM) for h in handles)
            except Exception:
                pass
            return _Sample(t=self._clock(), mem_mb=mem, util=util, power_w=power, clock_mhz=clock)

        self._source = "pynvml"
        self._run_thread(read, on_stop=lambda: pynvml.nvmlShutdown())
        return True

    def _start_process(self) -> None:
        proc = None
        if self.pid is not None:
            try:
                import psutil

                proc = psutil.Process(self.pid)
            except Exception:
                proc = None
        self._rapl_start = _rapl_energy_j()
        if proc is None and self._rapl_start is None:
            self._source = "none"
            return

        def read() -> _Sample:
            mem = None
            if proc is not None:
                try:
                    rss = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except Exception:
                            pass
                    mem = rss / 2**20
                except Exception:
                    pass
            return _Sample(t=self._clock(), mem_mb=mem)

        self._source = "psutil" if proc is not None else "rapl"
        self._run_thread(read)

    def _run_thread(self, read, on_stop=None) -> None:
        interval = self.interval_ms / 1000.0

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self._samples.append(read())
                except Exception as exc:  # pragma: no cover
                    logger.debug("telemetry read failed: %s", exc)
                self._stop.wait(interval)
            if on_stop:
                try:
                    on_stop()
                except Exception:
                    pass

        self._thread = threading.Thread(target=loop, name="rigtune-telemetry", daemon=True)
        self._thread.start()


def _counter_mj(nvml: Any, handles: List[Any]) -> Optional[float]:
    """Total energy since the driver loaded, summed over the devices, in millijoules; None if unsupported."""
    if not handles:
        return None
    try:
        return float(sum(nvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles))
    except Exception:  # NVMLError_NotSupported: pre-Volta, or a driver that does not report it
        return None


def _counter_delta_j(start: Optional[float], end: Optional[float]) -> Optional[float]:
    """A counter that did not advance (a driver reload resets it) is not trusted: the trapezoid stands."""
    if start is None or end is None or end <= start:
        return None
    return (end - start) / 1000.0


def _shutdown(nvml: Any) -> None:
    try:
        nvml.nvmlShutdown()
    except Exception:
        pass


def _rapl_energy_j() -> Optional[float]:
    """Package energy from Intel/AMD RAPL sysfs (Linux). Needs read permission; often root-only."""
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None
    total = 0.0
    found = False
    try:
        for name in os.listdir(base):
            if not name.startswith("intel-rapl:") or name.count(":") != 1:
                continue
            path = os.path.join(base, name, "energy_uj")
            with open(path, "r") as fh:
                total += int(fh.read().strip()) / 1e6
                found = True
    except (OSError, ValueError):
        return None
    return total if found else None


# --------------------------------------------------------------------------- HTTP driver


@dataclass
class RequestOutcome:
    ok: bool
    ttft_s: float = 0.0
    duration_s: float = 0.0
    tokens: int = 0
    token_source: str = "none"  # "usage" | "tokenizer" | "chunks"
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None  # the engine's usage.prompt_tokens, when it reports one
    cached_tokens: Optional[int] = None  # usage.prompt_tokens_details.cached_tokens: served from the prefix cache
    # The cached tokens by where they came from (device, host, storage), for an engine that says so
    # (_cached_by_source; on tier trials only).
    cached_by_source: Dict[str, int] = field(default_factory=dict)
    # Session replay only (_SessionScheduler): which step of which session this was, when it was dispatched, and
    # what it shared with what the engine had already seen.
    session: Optional[str] = None
    turn: int = 0
    seq: int = -1  # dispatch order within the level
    distance: Optional[int] = None  # requests dispatched between this session's previous step and this one
    idle_gap: Optional[int] = None  # of those, the ones dispatched after its previous step had finished
    working_set: int = 0  # tokens the live sessions' last prompts and answers held when it was dispatched
    tag: str = ""  # Step.tag
    chained: bool = False  # sent the moment the step before it finished (Step.chain), ahead of the queue
    primer: bool = False  # a window's primer (Step.primer): it loads the GPU like any request, but is not measured
    lcp: Optional[float] = None  # an explicit step's Step.lcp and Step.new, from the file
    new: Optional[float] = None
    started_s: Optional[float] = None  # perf_counter when it was sent (session replay): for a chain's latency
    live: int = 0  # sessions live when it was dispatched
    request_chars: int = 0  # length of the request's messages, rendered for comparison
    reusable_chars: int = 0  # how much of that is a prefix of what the engine saw before (see _SessionScheduler)
    text: str = ""  # the answer, while the scheduler needs it for the next step's history
    skipped: Optional[str] = None  # not a request: "cut" (the next prompt would not fit) or "truncated" (budget spent)


def _parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def _one_request(
    client: httpx.AsyncClient, url: str, body: Dict[str, Any], timeout: float, counter: TokenCounter,
    keep_text: bool = False,
) -> RequestOutcome:
    t0 = time.perf_counter()
    first: Optional[float] = None
    chunks = 0
    pieces: List[str] = []
    usage_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    by_source: Dict[str, int] = {}
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="replace")[:300]
                return RequestOutcome(ok=False, error=f"HTTP {resp.status_code}: {text}")
            async for line in resp.aiter_lines():
                obj = _parse_sse_line(line)
                if obj is None:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    ch = choices[0]
                    text = ch.get("text")
                    if text is None and isinstance(ch.get("delta"), dict):
                        text = ch["delta"].get("content")
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        chunks += 1
                        pieces.append(text)
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    usage_tokens = int(usage["completion_tokens"])
                if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                    prompt_tokens = usage["prompt_tokens"]
                    details = usage.get("prompt_tokens_details")
                    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
                        cached_tokens = details["cached_tokens"]
                by_source = _cached_by_source(obj) or by_source
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        return RequestOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
    end = time.perf_counter()
    if first is None:
        return RequestOutcome(ok=False, error="no tokens produced")
    # Exact count: server usage > model tokenizer > chunk count (approximate).
    if usage_tokens:
        tokens, source = usage_tokens, "usage"
    else:
        counted = counter.count("".join(pieces))
        if counted:
            tokens, source = counted, "tokenizer"
        else:
            tokens, source = chunks, "chunks"
    return RequestOutcome(ok=True, ttft_s=first - t0, duration_s=end - t0, tokens=tokens, token_source=source,
                          prompt_tokens=prompt_tokens, cached_tokens=cached_tokens,
                          cached_by_source=by_source, text="".join(pieces) if keep_text else "")


TIER_SOURCES = ("device", "host", "storage")


def _cached_by_source(chunk: Dict[str, Any]) -> Dict[str, int]:
    """SGLang's split of a reply's cached tokens by tier, asked for with return_cached_tokens_details:
    0.5.19's serving_chat.py streams it in a chunk of its own, with no choices, as
    sglext.cached_tokens_details {device, host[, storage, storage_backend]} (entrypoints/openai/protocol.py
    CachedTokensDetails). Empty for any other chunk, and for every engine that does not send one."""
    ext = chunk.get("sglext")
    details = ext.get("cached_tokens_details") if isinstance(ext, dict) else None
    if not isinstance(details, dict):
        return {}
    return {k: int(details[k]) for k in TIER_SOURCES if isinstance(details.get(k), int)}


async def _drive(
    base_url: str,
    hooks: LlmtraceHooks,
    workload: Workload,
    concurrency: int,
    timeout: float,
    counter: Optional[TokenCounter] = None,
) -> List[RequestOutcome]:
    counter = counter or TokenCounter()
    if workload.sessions:
        return await _drive_sessions(base_url, hooks, workload, concurrency, timeout, counter,
                                     body_hook=_tier_body_hook(hooks))
    sem = asyncio.Semaphore(concurrency)
    url = base_url.rstrip("/") + hooks.completions_path
    results: List[RequestOutcome] = []

    async with httpx.AsyncClient() as client:

        async def worker(prompt: str) -> None:
            body: Dict[str, Any] = {
                "model": hooks.model_name or "model",
                "prompt": prompt,
                "max_tokens": workload.decode_tokens,
                "temperature": workload.temperature,
                "stream": True,
                # Fixed-length answers for synthetic prompts; real-text workloads stop when the model does.
                "ignore_eos": not workload.natural_stop,
            }
            if hooks.stream_usage:
                body["stream_options"] = {"include_usage": True}
            async with sem:
                results.append(await _one_request(client, url, body, timeout, counter))

        await asyncio.gather(*(worker(p) for p in workload.prompts))
    return results


Messages = List[Dict[str, str]]


def _render(messages: Messages) -> str:
    """Messages as one string, so the prefix two requests share can be measured without a chat template."""
    return "".join(f"\x00{m['role']}\x01{m['content']}" for m in messages)


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _count(text: str, counter: TokenCounter, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    return counter.count(text) or int(len(text) / chars_per_token) + 1


@dataclass
class _LiveSession:
    """A session being replayed: the conversation so far, and what the engine said it counted for the last step."""

    session: Session
    step: int = 0  # the next step to send
    history: Messages = field(default_factory=list)  # the conversation so far, without the system prompt
    seen: str = ""  # the last request and the engine's answer to it, rendered: what its cache can hold
    last_prompt_tokens: Optional[int] = None
    carried_tokens: int = 0  # tokens of the answer the next step carries
    last_seq: Optional[int] = None
    done_at: Optional[int] = None  # requests dispatched in the level when its last step finished
    footprint: int = 0  # its last prompt and answer, in tokens: what it holds in the cache while live

    @property
    def chained(self) -> bool:
        """Whether the step just finished asked for the next to follow at once (Step.chain)."""
        return self.step > 0 and self.session.steps[self.step - 1].chain

    def request(self) -> Messages:
        """The next step's messages: the conversation so far plus its user turn, or an explicit step's own."""
        step = self.session.steps[self.step]
        if step.explicit:
            return [{"role": r, "content": c} for r, c in step.messages]
        system = [{"role": "system", "content": self.session.system}] if self.session.system else []
        return system + self.history + [{"role": "user", "content": step.user}]

    def next_prompt_tokens(self, messages: Messages, counter: TokenCounter,
                           chars_per_token: float = CHARS_PER_TOKEN) -> int:
        """How long the next prompt will be: what the engine counted last step, plus the answer it carries and the
        new user turn; or, before the engine has reported anything (or for an explicit step), every message, at the
        file's own characters per token when no tokenizer counts them (Workload.chars_per_token)."""
        step = self.session.steps[self.step]
        if self.last_prompt_tokens is not None and not step.explicit:
            return self.last_prompt_tokens + self.carried_tokens + _count(step.user, counter) + 2 * TURN_OVERHEAD
        return sum(_count(m["content"], counter, chars_per_token) for m in messages) + TURN_OVERHEAD * len(messages)

    def advance(self, sent: Messages, out: "RequestOutcome", history: str, counter: TokenCounter) -> None:
        """Take in the answer to the step just sent. With recorded history the next step carries the file's answer;
        the engine still saw its own, so that is what `seen` keeps."""
        step = self.session.steps[self.step]
        recorded = step.answer if history == "recorded" and step.answer is not None else None
        carried = recorded if recorded is not None else out.text
        convo = sent[1:] if sent and sent[0]["role"] == "system" and not step.explicit and self.session.system \
            else list(sent)
        self.history = convo + [{"role": "assistant", "content": carried}]
        self.seen = _render(sent + [{"role": "assistant", "content": out.text}])
        self.last_prompt_tokens = out.prompt_tokens
        self.carried_tokens = _count(recorded, counter) if recorded is not None else out.tokens
        self.last_seq = out.seq
        self.step += 1


class _SessionScheduler:
    """Which session step goes next in a level of session replay. No I/O and no clock, so it is tested on its own.

    `pool` (P, at least the concurrency) sessions are live at once and take turns in a queue: a session whose step
    finishes goes to the back. In steady state P - 1 other requests are dispatched between two of its steps (its
    reuse distance), of which about P - c start while it sits idle, its previous step done (its idle gap); both are
    recorded per request. With P = c a session's next step follows its last at once, which keeps its cache entries
    the most recently used and can never show eviction. A session never has two steps in flight. A step marked
    `chain` (an agent's tool-call follow-up) puts its successor at the front of the queue instead.

    A level is a budget of steps, not of sessions: new sessions are admitted as others end for as long as the
    budget lasts, so the caller can keep its concurrency full until the budget is spent (a level that instead ran
    a fixed set of sessions to completion thinned out as its last sessions ended: in simulation it averaged 0.75 c
    at c = 8). Once the budget is spent no new step is issued, except to finish an open chain; sessions with steps
    left are counted truncated. The budget should give a pool's sessions several steps each (turn_budget).

    A step whose prompt `fits` rejects stops its session and is recorded as cut rather than sent to fail; a failed
    step ends its session too, since the next one would carry no answer to it. `shared_system`, a system prompt
    several sessions share (and the warm-up sent), is what a session's first step can reuse."""

    def __init__(self, sessions: List[Session], concurrency: int, pool: int, budget: int,
                 fits: Callable[[_LiveSession, Messages], bool], history: str = "generated",
                 counter: Optional[TokenCounter] = None, shared_system: str = ""):
        self.waiting = list(reversed(sessions))  # pop() from the end, in the order the sessions were drawn
        self.pool = max(concurrency, pool or 0)
        self.budget = budget
        self.fits = fits
        self.history = history
        self.counter = counter or TokenCounter()
        self.shared = _render([{"role": "system", "content": shared_system}]) if shared_system else ""
        self.ready: List[_LiveSession] = []  # a FIFO; small, so list.pop(0) is fine
        self.live = 0
        self.in_flight = 0
        self.dispatched = 0
        self.working_set = 0  # tokens the live sessions' last prompts and answers make up
        self.skipped: List[RequestOutcome] = []
        self._admit()

    def _admit(self) -> None:
        while self.waiting and self.live < self.pool and self.dispatched < self.budget:
            self.ready.append(_LiveSession(self.waiting.pop()))
            self.live += 1

    def _retire(self, state: _LiveSession) -> None:
        self.live -= 1
        self.working_set -= state.footprint
        self._admit()

    def next(self) -> Optional[Tuple[_LiveSession, Messages, RequestOutcome]]:
        """The next step to dispatch, with its messages and its outcome's bookkeeping filled in; None when no
        session is ready (all live ones are in flight) or the budget is spent and no chain is open."""
        while self.ready and (self.dispatched < self.budget or self.ready[0].chained):
            state = self.ready.pop(0)
            messages = state.request()
            if not self.fits(state, messages):
                self.skipped.append(RequestOutcome(ok=False, skipped="cut", session=state.session.id,
                                                   turn=state.step + 1, tag=state.session.steps[state.step].tag,
                                                   error="session stopped: its next prompt would not fit"))
                self._retire(state)
                continue
            rendered = _render(messages)
            reusable = _common_prefix(state.seen, rendered) if state.seen else (
                len(self.shared) if self.shared and rendered.startswith(self.shared) else 0)
            step = state.session.steps[state.step]
            meta = RequestOutcome(ok=False, session=state.session.id, turn=state.step + 1, seq=self.dispatched,
                                  distance=None if state.last_seq is None else self.dispatched - state.last_seq - 1,
                                  idle_gap=None if state.done_at is None else self.dispatched - state.done_at,
                                  live=self.live, request_chars=len(rendered), reusable_chars=reusable,
                                  working_set=self.working_set, tag=step.tag, chained=state.chained,
                                  primer=step.primer, lcp=step.lcp, new=step.new)
            self.dispatched += 1
            self.in_flight += 1
            return state, messages, meta

        return None

    def done(self, state: _LiveSession, messages: Messages, out: RequestOutcome) -> None:
        """A dispatched step finished: queue the session's next step (at the front when the step chains to it, even
        past the budget), or end the session."""
        self.in_flight -= 1
        if not out.ok or state.step + 1 >= state.session.turns:
            self._retire(state)
            return
        state.advance(messages, out, self.history, self.counter)
        footprint = (out.prompt_tokens or 0) + out.tokens
        self.working_set += footprint - state.footprint
        state.footprint, state.done_at = footprint, self.dispatched
        if state.chained:
            self.ready.insert(0, state)
        elif self.dispatched < self.budget:
            self.ready.append(state)
        else:
            self._truncate(state)

    def _truncate(self, state: _LiveSession) -> None:
        self.skipped.append(RequestOutcome(ok=False, skipped="truncated", session=state.session.id,
                                           turn=state.step + 1, error="turn budget spent"))
        self._retire(state)

    def close(self) -> List[RequestOutcome]:
        """Markers for every session stopped early, once nothing is in flight."""
        while self.ready:
            self._truncate(self.ready.pop(0))
        return self.skipped


def _shared_system(workload: Workload) -> str:
    """The system prompt sessions share: the override, or one that more than one session uses. For explicit
    requests, the common start of every session's first system message: an agent's first message holds fixed
    instructions and then its own memory, so two first messages are rarely equal although most of each is shared.
    (The warm-up's own sessions are not explicit; they all carry the shared prompt, which the count below finds.)"""
    if workload.session_system:
        return workload.session_system
    if workload.session_list and all(s.steps[0].explicit for s in workload.session_list):
        firsts = [s.steps[0].messages[0] for s in workload.session_list]
        if len(firsts) < 2 or any(role != "system" for role, _ in firsts):
            return ""
        return os.path.commonprefix([text for _, text in firsts])
    counts = Counter(s.system for s in workload.session_list if s.system)
    return next((text for text, n in counts.most_common(1) if n > 1), "")


BodyHook = Callable[[Dict[str, Any], "_LiveSession", Messages], None]


def _tier_body_hook(hooks: LlmtraceHooks) -> Optional[BodyHook]:
    """On a tier trial, ask each request's reply for its cached tokens by tier (hooks.tier_details_field);
    None otherwise, so no other trial's requests change."""
    name = hooks.tier_details_field if hooks.tier_report else None
    if not name:
        return None

    def hook(body: Dict[str, Any], state: "_LiveSession", messages: Messages) -> None:
        body[name] = True

    return hook


async def _drive_sessions(base_url: str, hooks: LlmtraceHooks, workload: Workload, concurrency: int,
                          timeout: float, counter: TokenCounter,
                          body_hook: Optional[BodyHook] = None) -> List[RequestOutcome]:
    """Replay a level of a session file through the chat endpoint, `concurrency` steps in flight (_SessionScheduler
    decides which). A step's messages are the conversation so far, so the prefix the engine can reuse is exactly
    what it rendered before. Returns one outcome per step sent, then markers for sessions cut or truncated.
    `body_hook(body, state, messages)` may add fields to each request (an engine's cache-detail switch, say)."""
    url = base_url.rstrip("/") + hooks.chat_path

    def fits(state: _LiveSession, messages: Messages) -> bool:
        return state.next_prompt_tokens(messages, counter, workload.chars_per_token) <= workload.prefill_tokens

    sched = _SessionScheduler(workload.session_list, concurrency, workload.pool_at(concurrency),
                              workload.turn_budget(concurrency), fits, workload.history, counter,
                              _shared_system(workload))
    results: List[RequestOutcome] = []

    async with httpx.AsyncClient() as client:

        async def send(state: _LiveSession, messages: Messages) -> RequestOutcome:
            step = state.session.steps[state.step]
            body: Dict[str, Any] = {
                "model": hooks.model_name or "model",
                "messages": messages,
                "max_tokens": step.max_tokens or workload.decode_tokens,
                "temperature": workload.temperature,
                "stream": True,
                "ignore_eos": not workload.natural_stop,
            }
            if hooks.stream_usage:
                body["stream_options"] = {"include_usage": True}
            if body_hook is not None:
                body_hook(body, state, messages)
            started = time.perf_counter()
            out = await _one_request(client, url, body, timeout, counter, keep_text=True)
            out.started_s = started
            return out

        pending: Dict[asyncio.Future, Tuple[_LiveSession, Messages, RequestOutcome]] = {}
        while True:
            while len(pending) < concurrency:
                item = sched.next()
                if item is None:
                    break
                pending[asyncio.ensure_future(send(item[0], item[1]))] = item
            if not pending:
                break
            finished, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in sorted(finished, key=lambda f: pending[f][2].seq):
                state, messages, meta = pending.pop(fut)
                out = fut.result()
                for name in ("session", "turn", "seq", "distance", "idle_gap", "live", "request_chars",
                             "reusable_chars", "working_set", "tag", "chained", "primer", "lcp", "new"):
                    setattr(out, name, getattr(meta, name))
                sched.done(state, messages, out)
                out.text = ""  # the history keeps the answer; the outcome need not
                results.append(out)
    return results + sched.close()


def _metrics_from(outcomes: List[RequestOutcome], wall_s: float, concurrency: int,
                  tier: bool = False) -> TrialMetrics:
    sessions = _session_stats(outcomes, wall_s)
    tiers = _tier_stats(outcomes) if tier else {}  # a tier trial only (LlmtraceHooks.tier_report)
    outcomes = [o for o in outcomes if o.skipped is None]  # a session stopped early sent nothing for its rest
    ok = [o for o in outcomes if o.ok]
    m = TrialMetrics(
        requests=len(outcomes),
        failed=len(outcomes) - len(ok),
        concurrency=concurrency,
        duration_s=wall_s,
        output_tokens=sum(o.tokens for o in ok),
        **sessions,
        **tiers,
    )
    # A few distinct errors, so "12/48 requests failed" says why.
    m.errors = list(dict.fromkeys((o.error or "")[:200] for o in outcomes if not o.ok and o.error))[:3]
    if not ok:
        return m
    m.token_count_source = worst_source(o.token_source for o in ok)
    timed = [o for o in ok if not o.primer] or ok  # a window's primer loads the engine but is not measured
    ttfts = sorted(o.ttft_s * 1000 for o in timed)
    m.ttft_ms = statistics.median(ttfts)
    m.ttft_p95_ms = ttfts[min(len(ttfts) - 1, int(round(0.95 * (len(ttfts) - 1))))]
    m.ttft_samples_ms = [round(t, 1) for t in ttfts]
    tpots = [(o.duration_s - o.ttft_s) * 1000 / (o.tokens - 1) for o in timed if o.tokens > 1]
    m.tpot_ms = statistics.fmean(tpots) if tpots else None
    m.tok_s = m.output_tokens / wall_s if wall_s > 0 else 0.0
    return m


def _session_stats(outcomes: List[RequestOutcome], wall_s: float) -> Dict[str, Any]:
    """What a level of session replay achieved, from the engine's own counts, beside what its structure allowed.

    Achieved: cached prompt tokens over prompt tokens. SGLang leaves the count out of a reply when nothing was
    cached, so once any reply in the level carried one a reply without it counts as zero; when none did, the level
    has no count at all (None), never a measured zero, and the engine's /metrics counters can stand in.
    Expected: each step's prompt tokens, scaled by the share of its messages that is a prefix of what the engine
    saw before (the previous step and the answer the engine generated, or for a first step a system prompt that
    sessions share; see _SessionScheduler). It is an estimate: it measures characters of the messages, not the
    engine's tokens, and ignores that engines cache whole blocks, so engines are better compared on the later
    steps' achieved share than on the gap between the two. With recorded history a later step carries the file's
    answer, which the engine never generated, so less is reusable. Both are also split into first steps (only a
    shared system prompt is reusable) and later steps (the history). A flat level (--report-reuse) records only the
    achieved share, from the replies that carry a count."""
    steps = [o for o in outcomes if o.session is not None]
    if not steps:
        flat = [o for o in outcomes if o.ok and o.prompt_tokens is not None]
        if not any(o.cached_tokens is not None for o in flat):
            return {}
        return _reuse(flat, "")
    sent = [o for o in steps if o.skipped is None]
    out: Dict[str, Any] = {"sessions": len({o.session for o in steps}),
                           "sessions_cut": sum(o.skipped == "cut" for o in steps),
                           "sessions_truncated": sum(o.skipped == "truncated" for o in steps)}
    if sent and wall_s > 0:
        out["mean_in_flight"] = round(sum(o.duration_s for o in sent) / wall_s, 2)  # time-weighted, by Little's law
    sent = [o for o in sent if not o.primer] or sent  # the rest describes measured requests, not windows' primers
    if sent and wall_s > 0:
        out["effective_pool"] = round(statistics.fmean(o.live for o in sent), 1)
        out["working_set_tokens"] = max(o.working_set for o in sent)
    for name in ("distance", "idle_gap"):
        values = [getattr(o, name) for o in sent if getattr(o, name) is not None]
        if values:
            out["reuse_distance" if name == "distance" else name] = statistics.median(values)
    served = [o for o in sent if o.ok and o.prompt_tokens is not None]
    if served:  # what live traffic will be compared with: a later step's prompt holds its history
        lengths = sorted(o.prompt_tokens for o in served)
        out["prompt_tokens_seen"] = {"p50": _nearest_rank(lengths, 50), "p90": _nearest_rank(lengths, 90),
                                     "max": lengths[-1], "mean": int(round(sum(lengths) / len(lengths)))}
    reached: Dict[str, int] = {}
    for o in sent:
        if o.ok:
            reached[o.session] = max(reached.get(o.session, 0), o.turn)
    if reached:
        out["turns_seen"] = {"mean": round(sum(reached.values()) / len(reached), 2), "max": max(reached.values())}
    out["prompt_tokens_total"] = 0
    if any(o.cached_tokens is not None for o in served):  # SGLang's way of saying zero, once it says anything
        out.update(_reuse(served, ""))
        for part, steps_ in (("_first", [o for o in served if o.turn == 1]),
                             ("_later", [o for o in served if o.turn > 1])):
            if steps_:
                out[f"prefix_reuse{part}"] = _reuse(steps_, part)[f"prefix_reuse{part}"]
    else:
        out["prompt_tokens_total"] = sum(o.prompt_tokens for o in served)
    for part, steps_ in (("", served), ("_first", [o for o in served if o.turn == 1]),
                         ("_later", [o for o in served if o.turn > 1])):
        total = sum(o.prompt_tokens for o in steps_)
        if total:
            expected = sum(o.prompt_tokens * o.reusable_chars / o.request_chars for o in steps_ if o.request_chars)
            out[f"reuse_expected{part}"] = round(expected / total, 4)
    out.update(_behind_edit(served))
    out.update(_by_kind(sent))
    ttfts: Dict[int, List[float]] = {}
    for o in sent:
        if o.ok:
            ttfts.setdefault(o.turn, []).append(o.ttft_s * 1000)
    out["ttft_by_turn_ms"] = {str(t): round(statistics.median(v), 1) for t, v in sorted(ttfts.items())}
    return out


def _behind_edit(served: List[RequestOutcome]) -> Dict[str, float]:
    """For explicit requests: the share of prompt tokens a session had sent before but placed behind a change, so
    the engine prefilled them again (1 - lcp - new of each request, from the file, weighted by the engine's prompt
    count). Not all of it is avoidable: an eviction refills what it kept whatever the layout, so only the difference
    between a rewriting layout and an appending one of the same events is what the layout costs. Agent replay only;
    nothing for a turn-by-turn level."""
    steps = [o for o in served if o.lcp is not None and o.new is not None]
    total = sum(o.prompt_tokens for o in steps)
    if not total:
        return {}
    refill = sum(o.prompt_tokens * max(0.0, 1.0 - o.lcp - o.new) for o in steps)
    return {"refill_behind_edit": round(refill / total, 4)}


def _by_kind(sent: List[RequestOutcome]) -> Dict[str, Any]:
    """For explicit requests, by tag (a request's kind unless the file tagged it): how many were sent, their prompt
    tokens, the reuse their structure allowed (reuse_expected) and the engine reported (prefix_reuse, when any reply
    in the level carried a count; a reply without one is zero, as in _session_stats), the avoidable share, and their
    median time to first token, so rewritten requests can be set beside appended ones. Also chain_latency_ms: the
    median time from a chain's first send to its last request's first token, which an agent waits through and no
    single request's time to first token shows. Report only: the constraint stays each request's TTFT."""
    steps = [o for o in sent if o.lcp is not None]
    if not steps:
        return {}
    served = [o for o in steps if o.ok and o.prompt_tokens is not None]
    counted = any(o.cached_tokens is not None for o in served)
    kinds: Dict[str, Dict[str, Any]] = {}
    for tag in sorted({o.tag for o in steps}):
        group = [o for o in served if o.tag == tag]
        total = sum(o.prompt_tokens for o in group)
        entry: Dict[str, Any] = {"requests": sum(o.tag == tag for o in steps), "prompt_tokens": total}
        ttfts = [o.ttft_s * 1000 for o in steps if o.tag == tag and o.ok]
        if ttfts:
            entry["ttft_ms"] = round(statistics.median(ttfts), 1)
        if total:
            entry["reuse_expected"] = round(sum(o.prompt_tokens * o.reusable_chars / o.request_chars
                                                for o in group if o.request_chars) / total, 4)
            if counted:
                entry["prefix_reuse"] = round(sum(o.cached_tokens or 0 for o in group) / total, 4)
            entry.update(_behind_edit(group))
        kinds[tag] = entry
    out: Dict[str, Any] = {"reuse_by_kind": kinds}
    by_session: Dict[str, List[RequestOutcome]] = {}
    for o in steps:
        by_session.setdefault(o.session, []).append(o)
    chains: List[float] = []
    for sid in sorted(by_session):
        run = sorted(by_session[sid], key=lambda o: o.turn)
        i = 0
        while i < len(run):
            j = i
            while j + 1 < len(run) and run[j + 1].chained and run[j + 1].turn == run[j].turn + 1:
                j += 1
            links = run[i:j + 1]
            if j > i and all(o.ok and o.started_s is not None for o in links):
                chains.append((links[-1].started_s + links[-1].ttft_s - links[0].started_s) * 1000)
            i = j + 1
    if chains:
        out["chain_latency_ms"] = round(statistics.median(chains), 1)
    return out


def _tier_stats(outcomes: List[RequestOutcome]) -> Dict[str, Any]:
    """What a level of a tier trial adds (TrialMetrics.cached_by_source, ttft_returning_ms, ttft_by_hit_ms):
    the replies' cached tokens by tier where the engine split them, and time to first token on returning steps
    (a session's second step on), overall and by where each one's cache hit came from. Returning steps are
    the ones a tier can speed up: a first step can reuse at most a shared system prompt."""
    served = [o for o in outcomes if o.ok and o.skipped is None and o.session is not None]
    out: Dict[str, Any] = {}
    split = [o for o in served if o.cached_by_source]
    if split:
        out["cached_by_source"] = {s: sum(o.cached_by_source.get(s, 0) for o in split) for s in TIER_SOURCES
                                   if any(s in o.cached_by_source for o in split)}
    returning = sorted(o.ttft_s * 1000 for o in served if o.turn > 1)
    if not returning:
        return out
    out["ttft_returning_ms"] = round(statistics.median(returning), 1)
    out["ttft_returning_p95_ms"] = round(returning[min(len(returning) - 1,
                                                       int(round(0.95 * (len(returning) - 1))))], 1)
    if split or any(o.cached_tokens is not None for o in served):  # SGLang leaves a zero out (_session_stats)
        kinds: Dict[str, List[float]] = {}
        for o in served:
            if o.turn > 1:
                kinds.setdefault(_hit_kind(o), []).append(o.ttft_s * 1000)
        out["ttft_by_hit_ms"] = {k: round(statistics.median(v), 1) for k, v in sorted(kinds.items())}
    return out


def _hit_kind(o: RequestOutcome) -> str:
    """Where a step's cache hit came from: the slowest tier any of its cached tokens came from, "hit" when the
    engine gives only the total, or "miss"."""
    if o.cached_by_source:
        return next((s for s in ("storage", "host", "device") if o.cached_by_source.get(s)), "miss")
    return "hit" if o.cached_tokens else "miss"


def _nearest_rank(ordered: List[int], pct: float) -> int:
    return int(ordered[min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))])


def _reuse(served: List[RequestOutcome], part: str) -> Dict[str, Any]:
    """Achieved reuse over replies with a prompt count; a missing cached count is zero (see _session_stats)."""
    total = sum(o.prompt_tokens for o in served)
    cached = sum(o.cached_tokens or 0 for o in served)
    if part:
        return {f"prefix_reuse{part}": round(cached / total, 4) if total else None}
    return {"prompt_tokens_total": total, "cached_tokens_total": cached, "reuse_source": "usage",
            "prefix_reuse": round(cached / total, 4) if total else None}


Counters = Dict[Tuple[str, str], float]  # (metric name without _total, its labels as exposed) -> value
PREFIX_QUERIES, PREFIX_HITS = "vllm:prefix_cache_queries", "vllm:prefix_cache_hits"


def _prefix_counters(base_url: str, hooks: LlmtraceHooks) -> Optional[Counters]:
    """Every counter on the engine's /metrics, keyed by name and labels (a KV-tier experiment can read the ones by
    source), or None when there is no endpoint or it cannot be read."""
    if not hooks.metrics_path:
        return None
    try:
        text = httpx.get(base_url.rstrip("/") + hooks.metrics_path, timeout=10.0).text
    except httpx.HTTPError:
        return None
    out: Counters = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        # prometheus_client exposes a counter as <name>_total, with labels in braces before the value
        head, _, value = line.rpartition(" ")
        name, _, labels = head.partition("{")
        try:
            out[(name.strip().removesuffix("_total"), labels.rstrip("}"))] = float(value)
        except ValueError:
            continue
    return out


def _counter_sum(counters: Counters, name: str) -> Optional[float]:
    """One counter summed over its label sets (engines), or None when the engine does not expose it."""
    values = [v for (n, _), v in counters.items() if n == name]
    return sum(values) if values else None


def _engine_reuse(m: TrialMetrics, before: Optional[Counters], after: Optional[Counters]) -> None:
    """Fall back to vLLM's prefix-cache counters for a level whose replies reported no cached tokens."""
    if m.reuse_source != "none" or before is None or after is None:
        return
    sums = [_counter_sum(c, n) for c in (before, after) for n in (PREFIX_QUERIES, PREFIX_HITS)]
    if any(s is None for s in sums):
        return
    queries, hits = sums[2] - sums[0], sums[3] - sums[1]
    if queries > 0 and hits >= 0:
        m.prefix_reuse, m.reuse_source = round(hits / queries, 4), "metrics"


PROMPT_BY_SOURCE = "vllm:prompt_tokens_by_source"
# vLLM 0.29 v1/metrics/stats.py PromptTokenStats.ALL_SOURCES, as rigtune names them: what came from the GPU's
# own prefix cache, and what a KV connector loaded (the offloading connector: host and disk tiers together).
VLLM_SOURCES = {"local_cache_hit": "device", "external_kv_transfer": "tier"}


def _engine_sources(m: TrialMetrics, before: Optional[Counters], after: Optional[Counters]) -> None:
    """A tier trial's cached tokens by source from vLLM's /metrics, for a level whose replies did not split
    them: the change over the level in vllm:prompt_tokens_by_source (v1/metrics/loggers.py, by source)."""
    if m.cached_by_source or before is None or after is None:
        return

    def by_source(counters: Counters) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (name, labels), value in counters.items():
            source = re.search(r'source="([^"]*)"', labels) if name == PROMPT_BY_SOURCE else None
            if source and source.group(1) in VLLM_SOURCES:
                key = VLLM_SOURCES[source.group(1)]
                out[key] = out.get(key, 0.0) + value
        return out

    was, now = by_source(before), by_source(after)
    if now:
        m.cached_by_source = {k: max(0, int(round(now[k] - was.get(k, 0.0)))) for k in sorted(now)}


_TIER_FIELDS = ("cached_by_source", "ttft_returning_ms", "ttft_returning_p95_ms", "ttft_by_hit_ms")
_REUSE_FIELDS = ("prompt_tokens_total", "cached_tokens_total", "prefix_reuse", "prefix_reuse_first",
                 "prefix_reuse_later", "reuse_expected", "reuse_expected_first", "reuse_expected_later",
                 "reuse_source", "sessions", "sessions_cut", "sessions_truncated", "mean_in_flight", "effective_pool",
                 "reuse_distance", "idle_gap", "working_set_tokens", "ttft_by_turn_ms", "prompt_tokens_seen",
                 "turns_seen")
_REUSE_FIELDS += ("refill_behind_edit", "reuse_by_kind", "chain_latency_ms")  # explicit requests (agent traces)


def _level_workload(workload: Workload, k: int, counter: TokenCounter) -> Workload:
    """The prompts for concurrency level k: fresh ones after the first level.

    Replaying the first level's prompts lets the engine's prefix cache serve every later level's
    prefill almost for free: on an A40 one configuration read 3638 tok/s with replayed prompts and
    1814 with fresh ones. A workload's shared prefix is kept; caching that is the point. A session file's levels
    get sessions of their own the same way, by offset into its shuffled pool.
    """
    if k == 0:
        return workload
    level = replace(workload, seed=workload.seed + 7_919 * k, prompts=[], fitted=False,
                    prefix_text=workload.prefix_text, prefix_fixed=True,
                    sample_offset=workload.sample_offset + k * workload.n_prompts).ensure_prompts()
    if counter.available:
        level.fit_prompts(counter)
    return level


def _session_warmup(workload: Workload, synthetic: Workload) -> Workload:
    """The warm-up for a session file: the synthetic prompts, each as a one-step session through the chat endpoint.

    They carry the system prompt sessions share, so it is cached as it would be in production, and nothing of any
    session: warming a session's own turns would flatter the level that replays it. A system prompt only one
    session uses is not shared, so it is left out too."""
    system = _shared_system(workload)
    users = [p[:max(64, min(512, workload.prefill_tokens) * CHARS_PER_TOKEN)] for p in synthetic.prompts]
    warm = [Session(id=f"warmup{i}", system=system, steps=(Step(user=u),)) for i, u in enumerate(users)]
    top = max(synthetic.concurrencies)
    return replace(workload, session_list=warm, prompts=[s.first_turn for s in warm], n_prompts=len(warm),
                   decode_tokens=synthetic.decode_tokens, concurrencies=synthetic.concurrencies, session_pool=0,
                   turns_per_slot=max(1, -(-len(warm) // top)), fitted=True)


def _client_main(index: int, base_url: str, hooks: LlmtraceHooks, workload: Workload, concurrency: int,
                 timeout: float, ready: Any, go: Any, out: Any) -> None:
    """One load-generator process (see _drive_clients). Module level so it pickles under spawn."""
    ready.put(index)
    go.wait()
    out.put((index, asyncio.run(_drive(base_url, hooks, workload, concurrency, timeout))))


def _drive_clients(base_url: str, hooks: LlmtraceHooks, workload: Workload, concurrency: int, timeout: float,
                   clients: int) -> Tuple[List[RequestOutcome], float]:
    """Drive one level from several processes, with the prompts and the concurrency split between them.

    One Python client tops out near 3000 streamed tokens/s (through the replica balancer on two A40s:
    2954 tok/s from one client, 3541 from four). The clock starts once every process is up, so
    process start-up is not measured. A process that dies counts its requests as failed.
    """
    import multiprocessing as mp

    units = workload.session_list if workload.sessions else workload.prompts  # a session is never split
    shards = [s for s in (units[i::clients] for i in range(clients)) if s]
    per = max(1, -(-concurrency // len(shards)))
    ctx = mp.get_context("spawn")
    ready, out, go = ctx.Queue(), ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_client_main, daemon=True,
                         args=(i, base_url, hooks, _shard(workload, s, len(shards), concurrency), per, timeout,
                               ready, go, out))
             for i, s in enumerate(shards)]
    for p in procs:
        p.start()
    started, deadline = 0, time.monotonic() + 180
    while started < len(procs) and time.monotonic() < deadline:
        try:
            ready.get(timeout=1)
            started += 1
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break
    t0 = time.perf_counter()
    go.set()
    results: Dict[int, List[RequestOutcome]] = {}
    while len(results) < len(procs):
        try:
            i, outcomes = out.get(timeout=5)
            results[i] = outcomes
        except queue.Empty:
            if not any(p.is_alive() for p in procs) and out.empty():
                break
    wall = time.perf_counter() - t0
    for p in procs:
        p.join(timeout=10)
    merged: List[RequestOutcome] = []
    for i, s in enumerate(shards):
        died = RequestOutcome(ok=False, error="load-generator process died")
        shard_budget = _shard(workload, s, len(shards), concurrency).level_turns if workload.sessions else 0
        merged += results.get(i) or [died] * (len(s) if not workload.sessions else
                                              min(sum(x.turns for x in s), shard_budget))
    return merged, wall


def _shard(workload: Workload, units: list, shards: int, concurrency: int = 1) -> Workload:
    """One load-generator process's part of a level: some of its prompts, or some of its whole sessions with its
    share of the pool and of the level's turn budget. The share is of the level's total, floor included: each process
    applying the floor of 8 to its own share would inflate a small level several times over."""
    if not workload.sessions:
        return replace(workload, prompts=units, n_prompts=len(units), fitted=True)
    return replace(workload, session_list=units, prompts=[s.first_turn for s in units], n_prompts=len(units),
                   fitted=True, session_pool=-(-workload.session_pool // shards) if workload.session_pool else 0,
                   level_turns=-(-workload.turn_budget(concurrency) // shards))


def run_trial(
    base_url: str,
    hooks: LlmtraceHooks,
    workload: Workload,
    pid: Optional[int] = None,
    request_timeout: float = 120.0,
    warmup: bool = True,
    counter: Optional[TokenCounter] = None,
    clients: int = 1,
    enough: Optional[Callable[[TrialMetrics], bool]] = None,
    close_call: Optional[Callable[[TrialMetrics], bool]] = None,
    idle_tail_s: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> TrialMetrics:
    """Run the workload at each concurrency level and fold into one TrialMetrics.

    Every level gets fresh prompts (see _level_workload). `clients` > 1 drives each level from that
    many processes, for layouts whose combined throughput a single Python client cannot keep up with.

    `enough` (see objectives.enough_level) measures the levels highest first and stops at the first one
    it accepts: when the objective scores a trial by its throughput, a lower level cannot beat a level
    that already meets the constraints. Re-scored this way, 115 recorded trials in six calibrations on
    an A40 kept every score and every pick. A level keeps its own prompts whatever the order.

    `close_call` (see objectives.close_call_level) flags a level whose tail is too close to its ceiling to call
    from its requests; that level is measured again with as many requests on fresh prompts, inside the same
    telemetry window, and judged on both runs together.

    `idle_tail_s` > 0 keeps the last measured level's telemetry window open that long after its last request,
    and records the card's idle power and the wind-down to it (IdleSummary) on the trial. The level's own
    numbers come from the part of the window before the tail, so they are what they would be without it.

    Summary rule: tok/s and TTFT/TPOT come from the concurrency level with the highest
    throughput (that is the load the server would actually be run at); energy per token
    is total joules / total tokens across all levels; peak memory is the max.
    """
    if counter is None:
        counter = TokenCounter.for_model(hooks.tokenizer_id)
    workload.ensure_prompts()
    if counter.available and not workload.fitted:
        workload.fit_prompts(counter)
    prompt_tokens = workload.measured_prompt_tokens(counter) or 0
    if warmup:
        # One short request per slot at the busiest level. Kernels that compile on first use (Triton attention,
        # the sampler, speculative decoding) otherwise compile during the first batched level measured: on an
        # A40 a draft model's first batched level had a p95 time to first token of 1.3-1.6 s and the level
        # after it 0.12-0.18 s, and vLLM 0.29 logged "Triton kernel JIT compilation during inference". Once
        # the busiest level ran first, that spike landed on the level that is scored.
        top = max(workload.concurrencies)
        small = Workload(
            n_prompts=max(2, top), prefill_tokens=workload.prefill_tokens,
            decode_tokens=min(16, workload.decode_tokens), concurrencies=(top,), seed=workload.seed + 1,
            # Same shared prefix as the real prompts, so the warmup leaves it cached as production would.
            shared_prefix_tokens=workload.shared_prefix_tokens, prefix_text=workload.prefix_text, prefix_fixed=True,
        )
        if workload.sessions:
            small = _session_warmup(workload, small)
        try:
            asyncio.run(_drive(base_url, hooks, small, top, request_timeout, counter))
        except Exception as exc:
            logger.debug("warmup failed: %s", exc)

    per_level: Dict[str, TrialMetrics] = {}
    total_energy = 0.0
    energy_known = True
    peak_mem: Optional[float] = None
    utils: List[float] = []
    powers: List[float] = []
    clocks: List[float] = []
    source = "none"

    idle: Optional[IdleSummary] = None
    idle_after: Optional[int] = None
    levels = list(enumerate(workload.concurrencies))
    if enough is not None:
        levels.sort(key=lambda kc: -kc[1])
    for i, (k, c) in enumerate(levels):
        level = _level_workload(workload, k, counter)
        n = level.level_requests(c)
        if n < len(level.prompts):
            level = level.head(n)
        def drive(lw: Workload) -> Tuple[List[RequestOutcome], float]:
            if clients > 1:
                return _drive_clients(base_url, hooks, lw, c, request_timeout, clients)
            t0 = time.perf_counter()
            got = asyncio.run(_drive(base_url, hooks, lw, c, request_timeout, counter))
            return got, time.perf_counter() - t0

        resampled = False
        counters = _prefix_counters(base_url, hooks) if workload.sessions else None  # outside the measured window
        with Telemetry(hooks, pid=pid) as tel:
            outcomes, wall = drive(level)
            if close_call is not None and close_call(_metrics_from(outcomes, wall, c, hooks.tier_report)):
                # Too close to the ceiling to call from these requests: as many again, on prompts of their own.
                extra = _level_workload(workload, k + len(workload.concurrencies), counter)
                if n < len(extra.prompts):
                    extra = extra.head(n)
                more, more_wall = drive(extra)
                outcomes, wall, resampled = outcomes + more, wall + more_wall, True
            if idle_tail_s > 0 and (i == len(levels) - 1 or _stops(enough, _metrics_from(outcomes, wall, c, hooks.tier_report))):
                # The trial's last level: stay in the window while the card winds down to idle.
                tel.mark()
                if idle_tail_s > IDLE_WINDOW_S:  # the final second, read on its own, is the idle power
                    sleep(idle_tail_s - IDLE_WINDOW_S)
                    tel.mark_idle()
                    sleep(IDLE_WINDOW_S)
                else:
                    sleep(idle_tail_s)
                idle_after = c
        m = _metrics_from(outcomes, wall, c, hooks.tier_report)
        if counters is not None:
            after = _prefix_counters(base_url, hooks)
            _engine_reuse(m, counters, after)
            if hooks.tier_report:
                _engine_sources(m, counters, after)
        m.resampled = resampled
        s = tel.summary
        if idle_after == c and tel.idle is not None:
            idle = tel.idle
        source = s.source if s.source != "none" else source
        m.telemetry_source = s.source
        m.prompt_tokens = prompt_tokens
        m.peak_mem_mb = s.peak_mem_mb
        m.gpu_util_pct = s.gpu_util_pct
        m.power_w = s.power_w
        m.sm_clock_mhz = s.sm_clock_mhz
        m.counter_energy_j = s.counter_energy_j
        if s.sm_clock_mhz is not None:
            clocks.append(s.sm_clock_mhz)
        if s.energy_j is not None and m.output_tokens:
            m.joules_per_token = s.energy_j / m.output_tokens
            total_energy += s.energy_j
        else:
            energy_known = False
        if s.peak_mem_mb is not None:
            peak_mem = max(peak_mem or 0.0, s.peak_mem_mb)
        if s.gpu_util_pct is not None:
            utils.append(s.gpu_util_pct)
        if s.power_w is not None:
            powers.append(s.power_w)
        per_level[str(c)] = m
        if enough is not None and m.ok and enough(m):
            break

    best = max(per_level.values(), key=lambda x: x.tok_s)
    total_tokens = sum(x.output_tokens for x in per_level.values())
    counted = [x.counter_energy_j for x in per_level.values()]
    summary = TrialMetrics(
        tok_s=best.tok_s,
        ttft_ms=best.ttft_ms,
        ttft_p95_ms=best.ttft_p95_ms,
        tpot_ms=best.tpot_ms,
        peak_mem_mb=peak_mem,
        gpu_util_pct=statistics.fmean(utils) if utils else None,
        power_w=statistics.fmean(powers) if powers else None,
        sm_clock_mhz=statistics.fmean(clocks) if clocks else None,
        joules_per_token=(total_energy / total_tokens) if (energy_known and total_tokens) else None,
        counter_energy_j=sum(counted) if all(e is not None for e in counted) else None,  # type: ignore[misc]
        duration_s=sum(x.duration_s for x in per_level.values()),
        requests=sum(x.requests for x in per_level.values()),
        failed=sum(x.failed for x in per_level.values()),
        output_tokens=total_tokens,
        concurrency=best.concurrency,
        telemetry_source=source,
        token_count_source=worst_source(x.token_count_source for x in per_level.values()),
        prompt_tokens=prompt_tokens,
        by_concurrency=per_level,
    )
    if workload.sessions or best.reuse_source != "none":  # the reuse of the level the summary reports, like its speed
        summary = summary.model_copy(update={f: getattr(best, f) for f in _REUSE_FIELDS})
    if hooks.tier_report:
        summary = summary.model_copy(update={f: getattr(best, f) for f in _TIER_FIELDS})
    if idle is not None:
        summary.idle_power_w = idle.idle_power_w
        summary.idle_power_sampled_w = idle.idle_power_sampled_w
        summary.hold_below_idle = idle.below_idle
        summary.hold_joules = idle.hold_joules
        summary.hold_s = idle.hold_s
        summary.hold_truncated = idle.truncated
        summary.hold_after_concurrency = idle_after
        summary.energy_source = idle.energy_source
        summary.idle_tail_s = idle.tail_s
    return summary


def _stops(enough: Optional[Callable[[TrialMetrics], bool]], m: TrialMetrics) -> bool:
    """Whether the early-stop rule ends the trial at this level (see run_trial's `enough`)."""
    return enough is not None and m.ok and enough(m)
