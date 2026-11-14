"""Client self-check: is rigtune's load generator fast enough to measure an engine at this many users?

Every number a calibration records is measured by one Python process (calibrate/measure.py::_drive) that parses
every streamed chunk on one event loop. Past some number of users that process, not the engine, sets the pace,
and nothing in a calibration would say so. The self-check drives the same code against a fake engine of known
speed (rigtune/bench/fake_engine.py, in a process of its own) and reports, per level, the client's numbers as a
calibration would record them (calibrate/measure.py::_metrics_from, wall time from asyncio.run included):

- time to first token error: the client's median and p95 minus the engine's own, taken the same way, measured on
  the engine's clock from reading a request to writing its first chunk;
- throughput ratio: the client's tokens per second over the engine's (tokens sent over the span from the first
  request read to the last chunk written);
- event-loop lag: the p99 lateness of a 5 ms ticker running beside the requests;
- with an inter-token latency of 0, the client's tokens per second are its ceiling: one token a chunk, sent as
  fast as the socket takes them.

A level passes when its p95 error is under max(5 ms, 5% of the engine's p95) and its throughput ratio is at least
0.97. A level that misses either is a level whose measurements are partly the client's.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)

TTFT_ERROR_MS = 5.0  # p95 error allowed in milliseconds, or ...
TTFT_ERROR_FRACTION = 0.05  # ... this share of the engine's own p95, whichever is larger
MIN_THROUGHPUT_RATIO = 0.97
USERS_PER_CLIENT = 64  # the suggestion on a miss: one load-generator process per this many users
TICK_S = 0.005
REQUESTS_PER_USER = 2
CALIBRATION_ABOVE = 32  # a calibration with a level above this checks the client at its busiest level


@dataclass
class LevelReport:
    users: int
    requests: int
    failed: int
    client_ttft_p50_ms: float
    client_ttft_p95_ms: float
    engine_ttft_p50_ms: float
    engine_ttft_p95_ms: float
    client_tok_s: float
    engine_tok_s: float
    loop_lag_p99_ms: float

    @property
    def ttft_error_p50_ms(self) -> float:
        return self.client_ttft_p50_ms - self.engine_ttft_p50_ms

    @property
    def ttft_error_p95_ms(self) -> float:
        return self.client_ttft_p95_ms - self.engine_ttft_p95_ms

    @property
    def ttft_allowance_ms(self) -> float:
        return max(TTFT_ERROR_MS, TTFT_ERROR_FRACTION * self.engine_ttft_p95_ms)

    @property
    def throughput_ratio(self) -> float:
        return self.client_tok_s / self.engine_tok_s if self.engine_tok_s else 0.0

    @property
    def passed(self) -> bool:
        return (self.failed == 0 and self.ttft_error_p95_ms < self.ttft_allowance_ms
                and self.throughput_ratio >= MIN_THROUGHPUT_RATIO)

    def problems(self) -> List[str]:
        out = []
        if self.failed:
            out.append(f"{self.failed} of {self.requests} requests failed")
        if self.ttft_error_p95_ms >= self.ttft_allowance_ms:
            out.append(f"p95 time to first token {self.ttft_error_p95_ms:.1f} ms over the engine's "
                       f"(allowed {self.ttft_allowance_ms:.1f})")
        if self.throughput_ratio < MIN_THROUGHPUT_RATIO:
            out.append(f"throughput {self.throughput_ratio:.3f} of the engine's (allowed {MIN_THROUGHPUT_RATIO})")
        return out

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(ttft_error_p50_ms=self.ttft_error_p50_ms, ttft_error_p95_ms=self.ttft_error_p95_ms,
                 throughput_ratio=self.throughput_ratio, passed=self.passed)
        return d


def _median_p95(values: Sequence[float]) -> Tuple[float, float]:
    """Median and p95 taken as calibrate/measure.py::_metrics_from takes a level's time to first token."""
    if not values:
        return 0.0, 0.0
    ordered = sorted(values)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def level_report(users: int, outcomes: Sequence[Any], wall_s: float, engine: Sequence[Dict[str, float]],
                 lags_s: Sequence[float]) -> LevelReport:
    """One level's report from the client's outcomes (calibrate/measure.py RequestOutcome), the wall time it took,
    the engine's per-request records (fake_engine /stats) and the ticker's lateness samples (seconds)."""
    from rigtune.calibrate.measure import _metrics_from

    m = _metrics_from(list(outcomes), wall_s, users)
    span = (max(r["start_s"] + r["duration_s"] for r in engine) - min(r["start_s"] for r in engine)) if engine else 0
    engine_p50, engine_p95 = _median_p95([r["ttft_s"] * 1000 for r in engine])
    lags = sorted(lags_s)
    return LevelReport(
        users=users, requests=m.requests, failed=m.failed,
        client_ttft_p50_ms=m.ttft_ms or 0.0, client_ttft_p95_ms=m.ttft_p95_ms or 0.0,
        engine_ttft_p50_ms=engine_p50, engine_ttft_p95_ms=engine_p95,
        client_tok_s=m.tok_s,
        engine_tok_s=sum(r["tokens"] for r in engine) / span if span > 0 else 0.0,
        loop_lag_p99_ms=lags[max(0, math.ceil(0.99 * len(lags)) - 1)] * 1000 if lags else 0.0,
    )


def suggestion(users: int) -> str:
    return (f"spread {users} users over {math.ceil(users / USERS_PER_CLIENT)} load-generator processes (one per "
            f"{USERS_PER_CLIENT} users); rigtune does so only for multi-GPU layouts so far")


def summary(reports: Sequence[LevelReport]) -> Optional[str]:
    """A line for the profile when a level missed a threshold, else None."""
    missed = [r for r in reports if not r.passed]
    if not missed:
        return None
    worst = max(missed, key=lambda r: r.users)
    return (f"client self-check: at {', '.join(str(r.users) for r in missed)} users this machine's load generator "
            f"is part of what is measured ({'; '.join(worst.problems())} at {worst.users}), so levels that busy may "
            f"read slower than the engine is. What would fix it: {suggestion(worst.users)}.")


# ---------------------------------------------------------------- running it


class EngineProcess:
    """fake_engine in a process of its own, run by path with -I: it needs only the standard library, and importing
    rigtune would cost seconds."""

    def __init__(self, ttft_ms: float, itl_ms: float, tokens: int):
        from rigtune.bench import fake_engine

        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(Path(fake_engine.__file__)), "--port", "0", "--ttft-ms", str(ttft_ms),
             "--itl-ms", str(itl_ms), "--tokens", str(tokens)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        line = self.proc.stdout.readline() if self.proc.stdout else ""  # "ready <port>" once it listens
        if not line.startswith("ready "):
            self.close()
            raise RuntimeError(f"the fake engine did not start (exit code {self.proc.returncode})")
        self.url = f"http://127.0.0.1:{int(line.split()[1])}"

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.proc.stdout:
            self.proc.stdout.close()

    def __enter__(self) -> "EngineProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


async def _level(url: str, workload, users: int, timeout_s: float, counter, lags: List[float]):
    """calibrate/measure.py::_drive with a ticker beside it that records its own lateness in `lags`."""
    from rigtune.backends.base import LlmtraceHooks
    from rigtune.calibrate.measure import _drive

    stop = asyncio.Event()

    async def ticker() -> None:
        loop = asyncio.get_running_loop()
        due = loop.time() + TICK_S
        while not stop.is_set():
            await asyncio.sleep(max(0.0, due - loop.time()))
            lags.append(max(0.0, loop.time() - due))
            due += TICK_S

    tick = asyncio.ensure_future(ticker())
    try:
        return await _drive(url, LlmtraceHooks(model_name="fake"), workload, users, timeout_s, counter)
    finally:
        stop.set()
        await tick


def run(users: Sequence[int] = (1, 32, 64, 128), ttft_ms: float = 20.0, itl_ms: float = 10.0, tokens: int = 128,
        timeout_s: float = 120.0) -> List[LevelReport]:
    """Drive each level in turn against one fake engine process, `users` at a time and twice that many requests,
    timed as calibrate/measure.py::run_trial times a level; one report per level."""
    from rigtune.calibrate.tokens import TokenCounter
    from rigtune.calibrate.workload import Workload

    counter = TokenCounter()
    reports = []
    with EngineProcess(ttft_ms, itl_ms, tokens) as engine:
        for u in users:
            n = max(8, REQUESTS_PER_USER * u)
            workload = Workload(n_prompts=n, prompts=[f"self-check {i}" for i in range(n)], decode_tokens=tokens,
                                concurrencies=(u,), name="selfcheck")
            httpx.post(engine.url + "/reset", timeout=10.0)
            lags: List[float] = []
            t0 = time.perf_counter()
            outcomes = asyncio.run(_level(engine.url, workload, u, timeout_s, counter, lags))
            wall = time.perf_counter() - t0
            stats = httpx.get(engine.url + "/stats", timeout=30.0).json()
            reports.append(level_report(u, outcomes, wall, stats, lags))
    return reports


def cached_note(users: int, path: Path, cpu: str, run_fn=run) -> Optional[str]:
    """The profile note for a calibration whose busiest level is `users`: run the self-check at that level once
    per CPU and rigtune version (cached in `path`), and say so only when it missed a threshold. Never raises: a
    check that cannot run is logged and adds nothing."""
    from rigtune import __version__

    try:
        key = f"{__version__}|{cpu}|{users}"
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        if key not in cache:
            report = run_fn([users])[0]
            cache[key] = report.to_dict()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        fields = {k: v for k, v in cache[key].items() if k in LevelReport.__dataclass_fields__}
        return summary([LevelReport(**fields)])
    except Exception as exc:
        logger.warning("client self-check did not run: %s", exc)
        return None
