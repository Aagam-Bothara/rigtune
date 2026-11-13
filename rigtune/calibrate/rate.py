"""Joules per request at an arrival rate, composed from a trial's closed-loop concurrency levels.

Calibration measures each level closed-loop: c requests always in flight, so the GPU never idles and J/token
contains neither the card's idle draw nor the wind-down after a burst. A server receiving λ requests a second
pays both. Requests are counted in units of L output tokens, one figure shared by every trial being compared
(objectives.py takes the median at the workload's lowest level), so a configuration cannot score better by
answering shorter: on real-text workloads answers stop when the model does, and their lengths differ between
configurations. From each measured level c, with T_c tokens in D_c seconds at mean power P_c:

    R_c = T_c / D_c / L                  requests of L tokens per second the level sustained
    W_c = c / R_c                        mean time such a request spends in the server (Little's law)

At rate λ the mean number in flight n solves n = λ * W(n), with W linear between measured levels and flat below
the lowest one. How many requests are in flight at a given moment varies about n: with Poisson arrivals and service
that does not depend on the count (the M/G/infinity picture, which the wind-down term below already uses), it is a
Poisson count k with mean n. The card draws P(k) with k in flight: P(0) = P_idle, and for k >= 1 the closed-loop
power at concurrency k, linear between measured levels and held at the nearest one outside them. Its mean power is

    P_mean(n) = sum over k of Pois(k; n) * P(k)

(the sum stops where the remaining Poisson mass is below 1e-6), the energy per second divided by requests per second
gives the energy per request, and with a hold of E_hold joules over τ seconds at the end of each busy period:

    J/request(λ) = P_mean(n) / λ  +  E_hold * exp(-n) * (1 - exp(-λτ)) / (λτ)

reported as P_idle / λ (idle) plus (P_mean - P_idle) / λ (serving) plus the hold. Treating the server as if it ran
at concurrency n all the time instead (P(n) / λ) would have it never idle at one request in flight or more, where
the idle term cancels: at n = 1.7 a Poisson count is zero 18% of the time. Power is interpolated rather than energy
per request because that is what predicts a level left out: from the 1- and 8-user levels of the recorded A40 and
RTX 4090 calibrations, power and W predict the 4-user level within a median of 6%, energy per request straight
across overshoots it 2.2-2.5x.

The idle term is the card's baseline over the mean gap between arrivals. A busy period ends, and a hold is paid,
when the server empties, which with n in flight on average happens for a share exp(-n) of arrivals (the
probability a Poisson count with mean n is zero); the next arrival cuts the hold short, E[min(T, τ)] / τ for
T ~ Exp(λ). The hold is measured once per trial, after a burst at its last level; it is charged per busy period,
most of which at a low rate are single requests, which assumes the wind-down after one request is the wind-down
after a burst. The rate check (search.py) is what tests that.

Nothing is extrapolated. A rate is sustainable only within `max_utilisation` of the lowest rate served by the level
that covers the 95th percentile of what arrivals are served with and every level above it: throughput that falls
under load caps the rate, not the busiest level's figure. Levels below that one are left out, because a server that
reaches four in flight is not limited to what one request at a time can serve; counting the 1-user level whenever
fewer than one request is in flight on average made rates between 0.8 and 1.0 of its rate unsustainable and rates
just above it fine again. A trial whose
lowest measured level is above the workload's lowest (it stopped early) says nothing about a quieter server. And
latency is judged at every measured level from the one covering n to the one covering the 95th percentile of what
an arrival meets: an arrival that finds k others in flight is served with k + 1, and k is a Poisson count with mean
n. A server averaging 0.6 in flight serves an arrival alongside two or more others about 1 time in 8. When that percentile is past the busiest
level measured, the latency there is unknown and the rate is not eligible.

Everything here is pure arithmetic over recorded TrialMetrics, so it re-scores old calibrations as readily as it
scores new ones (benchmarks/rescore_rate.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from rigtune.models import TrialMetrics

MAX_UTILISATION = 0.8  # past this, open-loop queueing tails outgrow anything the closed-loop levels show
LATENCY_QUANTILE = 0.95  # latency is judged up to where this share of arrivals is served


@dataclass(frozen=True)
class LevelPoint:
    """One measured concurrency level, reduced to what the queueing model needs."""

    concurrency: int
    rate_rps: float  # R_c, in requests of the shared length
    wait_s: float  # W_c
    power_w: float  # P_c, the level's mean power
    per_request_j: float  # E_c above idle, or the whole J/request when idle power is unknown
    tokens_per_request: float  # what this level's own answers averaged
    metrics: TrialMetrics


@dataclass
class Composition:
    """J/request at one arrival rate, and what it is made of."""

    arrival_rate: float
    total_j: Optional[float]  # None: no energy telemetry at all
    idle_j: float
    active_j: float
    hold_j: float
    in_flight: float  # n, the operating point
    utilisation: float  # λ / the capacity it is judged against
    operating: TrialMetrics  # the busiest level latency is judged at (objectives picks the worst of latency_levels)
    feasible: bool  # sustainable, and judged on levels this trial measured; latency is the objective's to judge
    violation: float  # requests/s past max_utilisation, plus requests in flight below what was measured
    latency_levels: List[TrialMetrics] = field(default_factory=list)  # every level from n's to the 95th percentile's
    tokens_per_request: Optional[float] = None  # the shared L, or None when each level used its own answers
    latency_in_flight: int = 1  # requests in flight that 95% of arrivals are served with, themselves included
    idle_w: Optional[float] = None  # the idle power used
    fallback: bool = False  # no idle power recorded: J/token x tokens per request, idle and hold not separated
    below_lowest: bool = False  # n fell below the lowest measured level, whose figures were used
    missing_low_levels: bool = False  # ...and the workload had lower levels this trial never measured
    latency_beyond_top: bool = False  # the 95th percentile in flight is past the busiest level measured
    unsustainable: bool = False  # the rate is past max_utilisation of the capacity
    hold_truncated: bool = False
    operating_floor: int = 1  # the trial's lowest measured level

    def note(self) -> str:
        rate = f"{self.arrival_rate:g} req/s"
        if self.total_j is None:
            return f"at {rate}: no energy telemetry, so no joules per request"
        per = f" of {self.tokens_per_request:.0f} tokens" if self.tokens_per_request else ""
        if self.fallback:
            text = (f"at {rate}: {self.total_j:.1f} J/request{per} from J/token; no idle power was recorded, so "
                    f"the idle draw between requests is not counted")
        else:
            share = self.idle_j / self.total_j * 100 if self.total_j else 0.0
            text = (f"at {rate}: {self.total_j:.1f} J/request{per} = {self.idle_j:.1f} idle ({share:.0f}%) + "
                    f"{self.active_j:.1f} serving + {self.hold_j:.1f} wind-down")
        judged = "/".join(str(m.concurrency) for m in self.latency_levels) or str(self.operating.concurrency)
        text += (f", {self.in_flight:.2f} requests in flight on average, utilisation {self.utilisation:.0%}; "
                 f"latency judged at {judged} users, up to the {self.latency_in_flight} in flight that 95% of "
                 f"arrivals are served with")
        if self.latency_beyond_top:
            text += (f"; not eligible: that is more than the busiest level measured, "
                     f"{max((m.concurrency for m in self.latency_levels), default=self.operating.concurrency)}")
        if self.missing_low_levels:
            text += (f"; not eligible: this trial was measured only from {self.operating_floor} users and the "
                     f"rate needs fewer in flight")
        elif self.below_lowest:
            text += f" (below the lowest level measured, {self.operating_floor}, whose figures were used)"
        if self.hold_truncated:
            text += "; the wind-down was still falling when its measurement ended, so idle reads high"
        if self.unsustainable:
            text += (f"; not sustainable: the levels from that percentile up served as little as "
                     f"{self.arrival_rate / self.utilisation:.2f} req/s and {self.utilisation:.0%} is past the limit")
        return text


def level_energy_j(m: TrialMetrics) -> Optional[float]:
    """A level's energy: the NVML counter where recorded, else sampled J/token x tokens, else mean power x time."""
    if m.counter_energy_j is not None:
        return m.counter_energy_j
    if m.joules_per_token is not None and math.isfinite(m.joules_per_token) and m.output_tokens:
        return m.joules_per_token * m.output_tokens
    if m.power_w is not None and m.duration_s > 0:
        return m.power_w * m.duration_s
    return None


def lowest_level(metrics: TrialMetrics) -> Optional[TrialMetrics]:
    measured = [m for m in metrics.by_concurrency.values() if m.ok] or ([metrics] if metrics.ok else [])
    return min(measured, key=lambda x: x.concurrency) if measured else None


def levels(metrics: TrialMetrics, idle_w: Optional[float], tokens_per_request: Optional[float] = None
           ) -> List[LevelPoint]:
    """The trial's successful levels, lowest concurrency first, in requests of `tokens_per_request` tokens (each
    level's own average when None). Levels without energy or tokens are left out."""
    measured = [m for m in metrics.by_concurrency.values() if m.ok] or ([metrics] if metrics.ok else [])
    out: List[LevelPoint] = []
    for m in sorted(measured, key=lambda x: x.concurrency):
        served = m.requests - m.failed
        energy = level_energy_j(m)
        if served <= 0 or m.duration_s <= 0 or m.output_tokens <= 0 or energy is None:
            continue
        own = m.output_tokens / served
        size = tokens_per_request or own
        rate = m.output_tokens / m.duration_s / size
        power = energy / m.duration_s
        above = energy - (idle_w or 0.0) * m.duration_s
        out.append(LevelPoint(concurrency=m.concurrency, rate_rps=rate, wait_s=m.concurrency / rate, power_w=power,
                              per_request_j=max(0.0, above) / m.output_tokens * size, tokens_per_request=own,
                              metrics=m))
    return out


def in_flight(points: Sequence[LevelPoint], rate: float) -> Optional[float]:
    """The n with n = rate x W(n), W linear between levels and flat below the lowest; None past the busiest."""
    first = points[0]
    if rate * first.wait_s <= first.concurrency:
        return rate * first.wait_s
    for a, b in zip(points, points[1:]):
        fa, fb = rate * a.wait_s - a.concurrency, rate * b.wait_s - b.concurrency
        if fb <= 0:  # f is linear on [a, b], positive at a: its root lies between
            return a.concurrency + fa / (fa - fb) * (b.concurrency - a.concurrency)
    return None


def poisson_quantile(mean: float, q: float) -> int:
    """The smallest k with P(N <= k) >= q for N ~ Poisson(mean)."""
    if mean <= 0:
        return 0
    k, term = 0, math.exp(-mean)
    total = term
    while total < q and k < 10_000:
        k += 1
        term *= mean / k
        total += term
    return k


def _between(points: Sequence[LevelPoint], n: float) -> Tuple[float, float]:
    """(P, W) at n, each linear between measured levels and clamped at the ends."""
    if n <= points[0].concurrency:
        return points[0].power_w, points[0].wait_s
    for a, b in zip(points, points[1:]):
        if n <= b.concurrency:
            w = (n - a.concurrency) / (b.concurrency - a.concurrency)
            return a.power_w + w * (b.power_w - a.power_w), a.wait_s + w * (b.wait_s - a.wait_s)
    return points[-1].power_w, points[-1].wait_s


def serving_energy(points: Sequence[LevelPoint], n: float, idle_w: Optional[float]) -> float:
    """Energy per request above idle of a closed-loop level at concurrency n, from interpolated power and time in the
    server, (P(n) - P_idle) * W(n) / n: what a measured level would read, and so what a left-out level is checked
    against. At or below the lowest level, that level's own figure. Used for trials without idle data; a rate with
    idle data is scored by mean_power."""
    if n <= points[0].concurrency:
        return points[0].per_request_j
    power, wait = _between(points, n)
    return max(0.0, power - (idle_w or 0.0)) * wait / n


TAIL_MASS = 1e-6  # the in-flight sum stops once less than this much Poisson mass is left


def power_at(points: Sequence[LevelPoint], k: float) -> float:
    """P(k) for k >= 1 in flight: linear between measured levels, the nearest level's outside them."""
    return _between(points, k)[0]


def mean_power(points: Sequence[LevelPoint], n: float, idle_w: float) -> float:
    """sum_k Pois(k; n) * P(k), with P(0) = idle_w: the card's mean power with n requests in flight on average."""
    if n <= 0:
        return idle_w
    term = math.exp(-n)
    mass, total, k = term, term * idle_w, 0
    while 1.0 - mass > TAIL_MASS and k < 100_000:
        k += 1
        term *= n / k
        mass += term
        total += term * power_at(points, k)
    return total / mass


def hold_factor(rate: float, hold_s: float) -> float:
    """E[min(T, τ)] / τ for T ~ Exp(rate): how much of a hold the next arrival leaves to be paid."""
    x = rate * max(0.0, hold_s)
    return 1.0 if x < 1e-9 else (1.0 - math.exp(-x)) / x


def compose(metrics: TrialMetrics, rate: float, max_utilisation: float = MAX_UTILISATION,
            tokens_per_request: Optional[float] = None, workload_lowest: Optional[int] = None,
            idle_w: Optional[float] = None) -> Composition:
    """J/request at `rate` requests per second for one trial (see the module docstring).

    `tokens_per_request` is the request size every compared trial is counted in; `workload_lowest` the lowest
    concurrency level the workload measures, below which a trial that skipped it cannot be judged; `idle_w` the idle
    power to use in place of the trial's own reading (objectives pools it per engine)."""
    if rate <= 0:
        raise ValueError("an arrival rate must be positive")
    idle_w = idle_w if idle_w is not None else metrics.idle_power_w
    points = levels(metrics, idle_w, tokens_per_request)
    if not points:
        return Composition(arrival_rate=rate, total_j=None, idle_j=0.0, active_j=0.0, hold_j=0.0, in_flight=0.0,
                           utilisation=0.0, operating=metrics, feasible=True, violation=0.0,
                           tokens_per_request=tokens_per_request, idle_w=idle_w)
    top, floor = points[-1], points[0].concurrency
    n = in_flight(points, rate)
    n = float(top.concurrency) if n is None else n
    # Latency: an arrival meeting k others is served with k + 1 in flight; judged on every level from n's up.
    q = poisson_quantile(n, LATENCY_QUANTILE) + 1
    low = next((p for p in points if p.concurrency >= n - 1e-9), top)
    high = next((p for p in points if p.concurrency >= q), None)
    beyond = high is None
    # Capacity: the least served by the level covering that percentile and every level above it.
    capacity = min(p.rate_rps for p in points if p.concurrency >= (high or top).concurrency)
    rho = rate / capacity
    violation = max(0.0, rate - max_utilisation * capacity)
    unsustainable = violation > 0
    below = n < floor and floor > 1
    missing = n < floor and workload_lowest is not None and floor > workload_lowest
    if missing:
        violation += floor - n
    judged = [p.metrics for p in points if low.concurrency <= p.concurrency <= (high or top).concurrency]
    common = dict(arrival_rate=rate, in_flight=n, utilisation=rho, operating=(high or top).metrics,
                  latency_levels=judged, feasible=violation == 0.0 and not beyond, violation=violation,
                  tokens_per_request=tokens_per_request, latency_in_flight=q, idle_w=idle_w, below_lowest=below,
                  missing_low_levels=missing, latency_beyond_top=beyond, unsustainable=unsustainable,
                  operating_floor=floor)
    if idle_w is None:
        active = serving_energy(points, n, None)  # no P(0): the closed-loop figure at n, idle not separated
        return Composition(total_j=active, idle_j=0.0, active_j=active, hold_j=0.0, fallback=True, **common)
    idle_j = idle_w / rate
    active = max(0.0, mean_power(points, n, idle_w) - idle_w) / rate
    hold_j = (metrics.hold_joules or 0.0) * math.exp(-n) * hold_factor(rate, metrics.hold_s or 0.0)
    return Composition(total_j=idle_j + active + hold_j, idle_j=idle_j, active_j=active, hold_j=hold_j,
                       hold_truncated=metrics.hold_truncated, **common)
