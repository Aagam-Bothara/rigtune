"""One method over a set of problems, with what it cost: rigtune.reasoning's compute accounting.

Each method runs as one batch on a GPU that serves nothing else, with at most `sequences` answers being generated at
once. Every request asks for one answer (rigtune.reasoning.methods), so the limit counts what the engine generates.
With the engine kept full, a batch's wall time is the GPU time the method took, prompt reuse and the critique turn's
re-reading of the first answer included, which token counts alone leave out. A follow-up call (self-correction's
critique turn, adaptive's extra answers) goes before calls for new problems, so it does not wait out the whole batch
and leave a tail of second turns on an idle GPU. How full the limit stayed is recorded (utilisation): when the batch
is too small to keep the engine full, wall time is the slowest requests, not GPU time, and the report says so.

Grading happens after the clock stops (grade_attempt). A problem whose request failed (a context overflow on a long
first answer, a server error) is kept as wrong, with its error, never dropped: the failures fall on the hardest
problems, and dropping them would flatter the method that hits them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from rigtune.reasoning import methods as rm
from rigtune.reasoning.problems import Problem, grade

METHODS = ("direct", "self-consistency", "self-correction", "adaptive")


class SequenceLimit:
    """At most `limit` sequences in flight; a priority-1 request goes before any priority-0 one waiting. Keeps the
    time-weighted number in flight, for utilisation."""

    def __init__(self, limit: int, clock: Callable[[], float] = time.perf_counter):
        self.limit, self.used, self.peak = limit, 0, 0
        self._cond = asyncio.Condition()
        self._urgent = 0
        self._clock = clock
        self._last = clock()
        self.busy = 0.0  # integral of sequences in flight over time

    def _tick(self) -> None:
        now = self._clock()
        self.busy += self.used * (now - self._last)
        self._last = now

    async def acquire(self, n: int = 1, priority: int = 0) -> None:
        async with self._cond:
            if priority:
                self._urgent += 1
            try:
                await self._cond.wait_for(lambda: (self.used == 0 or self.used + n <= self.limit)
                                          and (priority or self._urgent == 0))
            finally:
                if priority:
                    self._urgent -= 1
            self._tick()
            self.used += n
            self.peak = max(self.peak, self.used)
            if priority and self.used < self.limit:  # waiters it held back may fit in what is left
                self._cond.notify_all()

    async def release(self, n: int = 1) -> None:
        async with self._cond:
            self._tick()
            self.used -= n
            self._cond.notify_all()


@dataclass
class Run:
    method: str
    k: Optional[int]
    settings: Dict[str, Any]
    sequences: int
    attempts: List[rm.Attempt]
    wall_seconds: float
    utilisation: float  # mean sequences in flight over the batch, as a share of `sequences`
    energy_joules: Optional[float] = None
    peak_sequences: int = 0
    problem_set: Optional[str] = None  # the problems' fingerprint (problems.load_math)
    server: Optional[str] = None  # what served it: engine, version, launch settings (the runner's fingerprint)
    errors: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.method if self.k is None else f"{self.method}-{self.k}"

    def to_json(self) -> Dict[str, Any]:
        return {"method": self.method, "k": self.k, "name": self.name, "settings": self.settings,
                "sequences": self.sequences, "wall_seconds": self.wall_seconds, "utilisation": self.utilisation,
                "energy_joules": self.energy_joules, "peak_sequences": self.peak_sequences,
                "problem_set": self.problem_set, "server": self.server, "errors": self.errors,
                "attempts": [a.to_json() for a in self.attempts]}


def limited(post: rm.Post, limit: SequenceLimit) -> rm.Post:
    """`post`, holding a place in the limit while each request runs, at its priority, and saying how long it ran."""
    async def wrapped(body: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(body)
        priority = int(body.pop(rm.PRIORITY, 0) or 0)
        n = int(body.get("n") or 1)
        await limit.acquire(n, priority)
        t0 = time.perf_counter()
        try:
            reply = dict(await post(body))
        finally:
            await limit.release(n)
        reply[rm.SERVICE_SECONDS] = time.perf_counter() - t0
        return reply
    return wrapped


def grade_attempt(attempt: rm.Attempt, problem: Problem) -> rm.Attempt:
    """Fill in self-consistency's and adaptive's majority (taken here, after the clock), then whether the attempt's
    final answer (and self-correction's first) is right."""
    if attempt.method.startswith(("self-consistency", "adaptive")) and attempt.final is None and not attempt.error:
        attempt.final = rm.majority(attempt.answers)
    if attempt.final is None:
        attempt.correct, attempt.grader = False, "error" if attempt.error else "no-answer"
    else:
        attempt.correct, _, attempt.grader = grade("\\boxed{" + attempt.final + "}", problem.answer)
    if attempt.method == "self-correction":
        attempt.first_correct = attempt.first is not None and grade(
            "\\boxed{" + attempt.first + "}", problem.answer)[0]
    return attempt


async def run_method(post: rm.Post, settings: rm.Settings, problems: Sequence[Problem], method: str,
                     k: Optional[int] = None, sequences: int = 256,
                     energy: Optional[Callable[[], Optional[float]]] = None,
                     problem_set: Optional[str] = None, server: Optional[str] = None) -> Run:
    """Every problem answered by `method` (self-consistency and adaptive with `k` answers at most), at most `sequences`
    at once, then graded. `energy` reads the GPU's energy counter in joules (nvml_energy)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; one of {', '.join(METHODS)}")
    if method == "self-consistency" and (not k or k < 2):
        raise ValueError("self-consistency needs k of 2 or more (k = 1 is direct)")
    if method == "adaptive" and (not k or k < 3):
        raise ValueError("adaptive needs k of 3 or more (it always takes 2)")
    limit = SequenceLimit(sequences)
    ask = limited(post, limit)

    async def one(problem: Problem) -> rm.Attempt:
        try:
            if method == "direct":
                return await rm.direct(ask, settings, problem)
            if method == "self-consistency":
                return await rm.self_consistency(ask, settings, problem, k)
            if method == "self-correction":
                return await rm.self_correction(ask, settings, problem)
            return await rm.adaptive(ask, settings, problem, k)
        except Exception as exc:  # kept, as wrong: never dropped
            name = method if method in ("direct", "self-correction") else f"{method}-{k}"
            return rm.Attempt(name, problem.id, [], None, error=f"{type(exc).__name__}: {exc}"[:300])

    e0 = energy() if energy else None
    limit._last = time.perf_counter()
    t0 = time.perf_counter()
    attempts = list(await asyncio.gather(*(one(p) for p in problems)))
    wall = time.perf_counter() - t0
    limit._tick()
    e1 = energy() if energy else None
    for attempt, problem in zip(attempts, problems):  # after the clock: grading is not the method's compute
        grade_attempt(attempt, problem)
    joules = (e1 - e0) if e0 is not None and e1 is not None else None
    return Run(method=method, k=k if method in ("self-consistency", "adaptive") else None,
               settings=settings.__dict__.copy(), sequences=sequences, attempts=attempts, wall_seconds=wall,
               utilisation=limit.busy / (sequences * wall) if wall > 0 else 0.0, energy_joules=joules,
               peak_sequences=limit.peak, problem_set=problem_set, server=server,
               errors=[f"{a.problem_id}: {a.error}" for a in attempts if a.error])


def nvml_energy(index: int = 0) -> Optional[Callable[[], Optional[float]]]:
    """A reader of GPU `index`'s total energy in joules (NVML, Volta and newer), or None without NVML."""
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception:
        return None

    def read() -> Optional[float]:
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0  # millijoules
        except Exception:
            return None
    return read
