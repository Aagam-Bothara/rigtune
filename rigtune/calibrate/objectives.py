"""The four objectives. All are constrained argmax; no weighted score formula in v1.

| objective   | rule                                   |
|-------------|----------------------------------------|
| throughput  | max tok/s                              |
| latency     | min request latency (TTFT + TPOT x tokens) s.t. tok/s >= floor |
| balanced    | max tok/s  s.t. TTFT p95 <= ceiling    |
| efficiency  | min J/tok  s.t. tok/s >= floor         |
| efficiency at --arrival-rate λ | min J/request at λ  s.t. λ sustainable, TTFT p95 <= ceiling |

Floors default to a fraction of the best observed tok/s; the ceiling defaults to an absolute
TTFT. If nothing satisfies the constraint, the least-violating candidate is chosen and the
result is flagged as relaxed.

Each trial is measured at several concurrency levels. A trial is scored at whichever of its
levels best satisfies the objective (e.g. for `balanced`, the highest-throughput level whose
TTFT is still under the ceiling), so the winner is a (config, load) pair the server can run at.

Scores within `noise_tolerance` of the best are treated as ties and broken in favour of the
more capable configuration (larger context, then larger batch), since a 1% tok/s edge is
measurement noise and a bigger context window is a real capability.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rigtune.calibrate.rate import Composition, compose, lowest_level
from rigtune.calibrate.tail import quantile_interval
from rigtune.models import OBJECTIVES, TrialMetrics, TrialResult


@dataclass
class Constraints:
    ttft_ceiling_ms: float = 500.0  # balanced
    # Which time to first token the ceiling applies to: 95 = the 95th percentile (1 request in 20 is
    # slower), 50 = the median. In 8 of 20 recorded comparisons (an A40, an L4, a CPU) the load a
    # pick was served at by the median broke the ceiling at p95, so the tail is the default.
    ttft_percentile: int = 95
    tok_s_floor_frac: float = 0.5  # latency / efficiency: floor = frac x best tok/s
    tok_s_floor_abs: Optional[float] = None  # overrides the fraction when set
    noise_tolerance: float = 0.02  # scores within 2% are ties -> prefer larger ctx, then batch
    # Per-token decode latency ceiling. Applied under every objective when set: TTFT constrains the
    # prefill phase, this constrains the decode phase.
    tpot_ceiling_ms: Optional[float] = None
    # A power-capped or clock-locked variant of the leading config may cost up to this much of its
    # score and still be chosen, when it uses less energy. The default keeps savings "free":
    # within run-to-run noise of the uncapped result.
    power_max_loss: float = 0.02
    # efficiency only: score each trial by joules per request at this many requests per second, idle draw and
    # wind-down included (calibrate/rate.py), in place of J/token above a throughput floor. None = J/token.
    arrival_rate: Optional[float] = None
    max_utilisation: float = 0.8  # the arrival rate may use at most this share of the busiest level's rate
    # --strict-constraints: refuse to serve when the final pick misses the constraints, instead of serving the
    # least-violating configuration. Applied to the final choice only (meets); during the search the leader is
    # still the least-violating configuration, so the stages after it can find one that does meet them.
    strict: bool = False
    # objective "custom" only: the goal in its canonical text, `min joules_per_token; tok_s>=200` (format_goal). A
    # string, not a structure, because constraints are recorded in the profile as JSON and compared field by field;
    # None for the four presets, so it changes no preset's fingerprint.
    goal: Optional[str] = None


@dataclass
class Ranked:
    result: TrialResult
    metrics: TrialMetrics  # the concurrency level the trial is scored at
    score: float  # lower is better
    feasible: bool
    violation: float  # how far outside the constraint (0 if feasible)
    rate: Optional[Composition] = None  # efficiency at an arrival rate: the J/request breakdown behind the score
    # Efficiency at an arrival rate: (not eligible for want of idle data, past sustainability, past latency), ordered
    # lexicographically when nothing is feasible, since requests per second and milliseconds do not add up.
    violation_key: Optional[Tuple[float, ...]] = None

    @property
    def concurrency(self) -> int:
        return self.metrics.concurrency


def _ok_results(results: Sequence[TrialResult]) -> List[TrialResult]:
    # An open-loop rate check measures the pick at one rate; it checks the choice and is never part of it.
    return [r for r in results if r.ok and not r.metrics.open_loop]


def _ttft(m: TrialMetrics) -> float:
    t = m.ttft_ms
    return t if (t is not None and math.isfinite(t)) else math.inf


def _ttft_for(m: TrialMetrics, percentile: int) -> float:
    """The time to first token a ceiling is judged on: the 95th percentile when asked for and recorded."""
    if percentile >= 95 and m.ttft_p95_ms is not None and math.isfinite(m.ttft_p95_ms):
        return m.ttft_p95_ms
    return _ttft(m)


def _levels(r: TrialResult) -> List[TrialMetrics]:
    levels = [m for m in r.metrics.by_concurrency.values() if m.ok]
    return levels or [r.metrics]


def _e2e_latency(m: TrialMetrics) -> float:
    """Median request latency: time to first token plus the rest of the answer at TPOT.

    Counting decode makes the latency objective see speculative decoding, which leaves TTFT
    alone and cuts time per token.
    """
    ttft = _ttft(m)
    if not math.isfinite(ttft):
        return ttft
    per_request = (m.output_tokens / m.requests) if m.requests else 0.0
    tpot = m.tpot_ms if (m.tpot_ms is not None and math.isfinite(m.tpot_ms)) else 0.0
    return ttft + tpot * max(0.0, per_request - 1)


def _finite(v: Optional[float]) -> Optional[float]:
    return v if (v is not None and math.isfinite(v)) else None


@dataclass(frozen=True)
class Metric:
    """One thing a goal can optimise or limit, read from one measured concurrency level."""

    name: str
    sense: str  # "max" or "min": which way is better
    # How it moves as load rises, for one configuration: "improves", "worsens" or "unknown". Only a metric that
    # improves lets a trial stop at the busiest level that meets the limits (enough_level).
    load: str
    read: Callable[[TrialMetrics], Optional[float]]  # None: not measured at that level
    optimisable: bool = True  # False: a limit only


# Every metric a goal names, in its own unit (ms, W, MiB). tok_s is read as recorded, never replaced by None, so a
# throughput floor judges a trial exactly as the presets always have.
METRICS: Dict[str, Metric] = {m.name: m for m in (
    Metric("tok_s", "max", "improves", lambda m: m.tok_s),
    Metric("ttft_p95_ms", "min", "worsens", lambda m: _finite(_ttft_for(m, 95))),
    Metric("ttft_p50_ms", "min", "worsens", lambda m: _finite(_ttft(m))),
    Metric("tpot_ms", "min", "worsens", lambda m: _finite(m.tpot_ms)),
    Metric("request_latency_ms", "min", "worsens", lambda m: _finite(_e2e_latency(m))),
    # Energy per token can fall as load rises (the card's draw is shared by more tokens) or rise (a busier card
    # draws more than it adds tokens): no level order settles it.
    Metric("joules_per_token", "min", "unknown", lambda m: _finite(m.joules_per_token)),
    # Limits only: the least power or memory is never worth having for itself, only as a bound on the rest.
    Metric("power_w", "min", "unknown", lambda m: _finite(m.power_w), optimisable=False),
    Metric("mem_mb", "min", "unknown", lambda m: _finite(m.peak_mem_mb), optimisable=False),
)}


@dataclass(frozen=True)
class Limit:
    """`metric op bound`: a ceiling (<=) or a floor (>=), in the metric's own unit.

    of_best: the bound is a fraction of the best trial's tok/s among those ranked (tok_s >= 50%best), the
    throughput floor latency and efficiency have always had. missing: what a level that did not measure the metric
    counts as. "fail" counts it as the worst value, so it misses the limit; "zero" counts it as 0, the TPOT
    ceiling's convention (a level with no decode step to time passes it)."""

    metric: str
    op: str  # "<=" or ">="
    bound: float
    of_best: bool = False
    missing: str = "fail"

    def resolve(self, results: Sequence[TrialResult]) -> float:
        """The bound in the metric's unit, among `results` for a fraction of the best."""
        if not self.of_best:
            return self.bound
        best = max((r.metrics.tok_s for r in results), default=0.0)
        return self.bound * best

    def value(self, m: TrialMetrics) -> float:
        v = METRICS[self.metric].read(m)
        if v is not None:
            return v
        if self.missing == "zero":
            return 0.0
        return math.inf if self.op == "<=" else -math.inf

    def excess(self, m: TrialMetrics, bound: float) -> float:
        """How far the level is outside the limit, in the metric's unit; 0 when it meets it."""
        v = self.value(m)
        return max(0.0, v - bound) if self.op == "<=" else max(0.0, bound - v)

    def met(self, m: TrialMetrics, bound: float) -> bool:
        """Whether one level meets the limit, for stopping a trial there. A "zero" ceiling is met at or under its
        bound and any other limit when nothing exceeds it: the two tests enough_level has always made. They differ
        only for a NaN bound."""
        if self.missing == "zero":
            v = self.value(m)
            return v <= bound if self.op == "<=" else v >= bound
        return not self.excess(m, bound) > 0


@dataclass(frozen=True)
class Goal:
    """One metric to optimise within any number of limits: still a constrained argmax, with no weighted score."""

    optimise: str
    limits: Tuple[Limit, ...] = ()
    # efficiency's own fallback: a level without energy readings is ordered by tok/s, so the preset still resolves
    # where NVML cannot read power.
    energy_fallback: bool = False
    # A goal stated in its own terms (objective "custom"): a level that misses limits is ranked by how many it
    # misses, then by how far past them it is relative to each bound, since milliseconds and watts do not add up.
    # The presets keep summing each limit's excess in its own unit, so no preset pick moves.
    custom: bool = False

    def score(self, m: TrialMetrics) -> float:
        """Lower is better; a level that did not measure the optimised metric scores worst."""
        v = METRICS[self.optimise].read(m)
        if v is None:
            return 1e9 - m.tok_s if self.energy_fallback else math.inf
        return -v if METRICS[self.optimise].sense == "max" else v

    def bounds(self, results: Sequence[TrialResult]) -> Tuple[float, ...]:
        return tuple(lim.resolve(results) for lim in self.limits)

    def violation(self, m: TrialMetrics, bounds: Sequence[float]) -> float:
        """The limits' excesses summed in order, each in its own unit: how the presets have always measured how
        far outside their constraints a level is."""
        total = 0.0
        for lim, bound in zip(self.limits, bounds):
            total = total + lim.excess(m, bound)
        return total

    def meets(self, m: TrialMetrics, bounds: Sequence[float]) -> bool:
        return all(lim.met(m, b) for lim, b in zip(self.limits, bounds))

    def violation_key(self, m: TrialMetrics, bounds: Sequence[float]) -> Tuple[float, float]:
        """(limits missed, the sum of each excess over its bound): how a custom goal orders levels that miss."""
        missed, relative = 0, 0.0
        for lim, bound in zip(self.limits, bounds):
            excess = lim.excess(m, bound)
            if excess > 0:
                missed += 1
                relative += excess / abs(bound) if bound else excess
        return float(missed), relative

    @property
    def improves_with_load(self) -> bool:
        """Whether the optimised metric gets better as load rises, so the busiest level that meets the limits is
        the trial's best (enough_level)."""
        return METRICS[self.optimise].load == "improves"

    @property
    def scores_tok_s(self) -> bool:
        """Whether a level's score is its tok/s, negated: what dropping a losing engine on its raw tok/s needs."""
        return self.optimise == "tok_s"

    @property
    def ttft_limit(self) -> Optional[Limit]:
        """The ceiling on a time-to-first-token percentile, if the goal has one."""
        return next((lim for lim in self.limits if lim.metric in ("ttft_p95_ms", "ttft_p50_ms")), None)


def ttft_metric(percentile: int) -> str:
    return "ttft_p95_ms" if percentile >= 95 else "ttft_p50_ms"


CUSTOM = "custom"  # the objective of a goal no preset expresses: Constraints.goal says what it is

_LIMIT = re.compile(r"(?P<metric>[a-z0-9_]+)\s*(?P<op><=|>=)\s*"
                    r"(?P<num>\d+(?:\.\d*)?(?:e[+-]?\d+)?|\.\d+(?:e[+-]?\d+)?)(?P<best>%best)?")


def _number(x: float) -> str:
    """A bound as goal text writes it: the shortest form that reads back as the same float."""
    x = float(x)  # an int has is_integer() only from Python 3.12
    return str(int(x)) if x.is_integer() and abs(x) < 1e15 else repr(x)


def format_goal(goal: Goal) -> str:
    """The canonical text of a custom goal: `min joules_per_token; tok_s>=50%best; ttft_p95_ms<=800`, the limits
    in the order of their names, so one goal is always one string (and one profile fingerprint)."""
    parts = [f"{METRICS[goal.optimise].sense} {goal.optimise}"]
    for lim in goal.limits:
        number = _number(round(lim.bound * 100, 10)) + "%best" if lim.of_best else _number(lim.bound)
        parts.append(f"{lim.metric}{lim.op}{number}")
    return "; ".join(parts)


@functools.lru_cache(maxsize=64)  # rank runs on every progress line
def parse_goal(text: str) -> Goal:
    """A custom goal from its text (format_goal). Raises ValueError, naming what is wrong, for anything else."""
    head, *rest = [part.strip() for part in text.split(";")]
    words = head.split()
    if len(words) != 2 or words[0] not in ("min", "max"):
        raise ValueError(f"a goal starts with `min <metric>` or `max <metric>`, not {head!r}")
    sense, optimise = words
    metric = METRICS.get(optimise)
    if metric is None or not metric.optimisable:
        raise ValueError(f"cannot optimise {optimise!r}; choose from "
                         f"{', '.join(m.name for m in METRICS.values() if m.optimisable)}")
    if sense != metric.sense:
        raise ValueError(f"{optimise} is better {'higher' if metric.sense == 'max' else 'lower'}: "
                         f"`{metric.sense} {optimise}`")
    limits: Dict[str, Limit] = {}
    for part in rest:
        match = _LIMIT.fullmatch(part)
        if not match:
            raise ValueError(f"not a limit: {part!r} (`metric<=number` or `metric>=number`)")
        name, op = match["metric"], match["op"]
        known = METRICS.get(name)
        if known is None or name == "tpot_ms":
            raise ValueError(f"unknown limit {name!r}" + (" (the TPOT ceiling is Constraints.tpot_ceiling_ms)"
                                                         if name == "tpot_ms" else ""))
        wanted = "<=" if known.sense == "min" else ">="
        if op != wanted:
            raise ValueError(f"{name} is better {'lower' if known.sense == 'min' else 'higher'}, so its limit is "
                             f"{wanted}, not {op}")
        if match["best"] and name != "tok_s":
            raise ValueError(f"%best is a share of the best tok/s; {name} takes a number")
        if name in limits:
            raise ValueError(f"{name} is limited twice")
        bound = float(match["num"])
        if not math.isfinite(bound):  # a number too long for a float reads as infinity, which never writes back
            raise ValueError(f"{name}'s bound is not a finite number")
        if match["best"] and not 0 < bound <= 100:
            raise ValueError(f"{name}>={match['num']}%best: a share of the best tok/s is above 0% and at most 100%")
        limits[name] = Limit(name, op, bound / 100 if match["best"] else bound, of_best=bool(match["best"]))
    return Goal(optimise, tuple(limits[k] for k in sorted(limits)), custom=True)


def resolve_goal(objective: str, cons: Constraints) -> Goal:
    """The goal an objective name stands for, built from the constraints' fields (the docstring table above), or
    for `custom` read from Constraints.goal.

    The workload's TPOT ceiling, when set, is the last limit of every one, a missing TPOT counting as 0."""
    tpot = (Limit("tpot_ms", "<=", cons.tpot_ceiling_ms, missing="zero"),) if cons.tpot_ceiling_ms is not None else ()
    if objective == CUSTOM:
        if cons.goal is None:
            raise ValueError("the custom objective needs a goal (Constraints.goal)")
        stated = parse_goal(cons.goal)
        return dataclasses.replace(stated, limits=stated.limits + tpot)
    if cons.goal is not None:
        raise ValueError(f"a goal is stated for the custom objective only, not {objective}")
    floor = (Limit("tok_s", ">=", cons.tok_s_floor_abs) if cons.tok_s_floor_abs is not None
             else Limit("tok_s", ">=", cons.tok_s_floor_frac, of_best=True))
    if objective == "throughput":
        return Goal("tok_s", tpot)
    if objective == "latency":
        return Goal("request_latency_ms", (floor, *tpot))
    if objective == "balanced":
        return Goal("tok_s", (Limit(ttft_metric(cons.ttft_percentile), "<=", cons.ttft_ceiling_ms), *tpot))
    if objective == "efficiency":
        return Goal("joules_per_token", (floor, *tpot), energy_fallback=True)
    raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")


def enough_level(objective: str, cons: Optional[Constraints] = None) -> Optional[Callable[[TrialMetrics], bool]]:
    """Whether one concurrency level settles a trial's score, for measuring levels highest first.

    A goal whose metric improves with load (tok/s: `throughput` and `balanced`) scores a trial by its best level
    that meets the limits, so no lower level can beat the first one that meets them. (Not always: n-gram
    speculation lost throughput as load rose, but only on configurations that lost anyway.) Any other metric
    (`latency`, `efficiency`) can prefer a lower level, so it needs every one measured: None.
    """
    if objective not in (*OBJECTIVES, CUSTOM):
        return None
    goal = resolve_goal(objective, cons or Constraints())
    if not goal.improves_with_load:
        return None
    bounds = goal.bounds([])

    def enough(m: TrialMetrics) -> bool:
        return m.ok and goal.meets(m, bounds)

    return enough


def close_call_level(objective: str, cons: Optional[Constraints] = None,
                     confidence: float = 0.95) -> Optional[Callable[[TrialMetrics], bool]]:
    """Whether a level's tail is too close to its ceiling to judge from its own requests: the ceiling lies inside
    the interval of the percentile it applies to (calibrate/tail.py). Such a level is measured again.
    Only a goal with a TTFT ceiling (`balanced`) judges a tail against one, so the rest get None.

    This is a screen, not a guarantee. It uses the widest interval the order statistics can express, whose real
    confidence is `attained_confidence(n, q)` — about 81% for a p95 from 32 requests, because the slowest of 32
    samples exceeds the true p95 only 80.6% of the time. Demanding a true 95% upper bound would take 59 requests
    per level (72 for a two-sided interval), so a level that passes this screen has not been certified at 95%.
    """
    if objective not in (*OBJECTIVES, CUSTOM):
        return None
    ceiling = resolve_goal(objective, cons or Constraints()).ttft_limit
    if ceiling is None:
        return None
    q = 0.95 if ceiling.metric == "ttft_p95_ms" else 0.5

    def close(m: TrialMetrics) -> bool:
        if not m.ok or not m.ttft_samples_ms:
            return False
        low, high = quantile_interval(m.ttft_samples_ms, q, confidence, unbounded=False)
        return low <= ceiling.bound <= high

    return close


def _sort_key(x: Ranked) -> Tuple[bool, Tuple[float, ...], float]:
    if x.feasible:
        return (False, (), x.score)
    return (True, x.violation_key if x.violation_key is not None else (x.violation,), x.score)


def _capability(r: TrialResult) -> Tuple[int, int, float]:
    c = r.config
    mem = c.gpu_memory_utilization if c.gpu_memory_utilization is not None else float(c.n_gpu_layers or 0)
    return (c.ctx, c.batch, mem)


def _variant_key(r: TrialResult) -> str:
    """What makes two trials 'the same configuration at different power': everything but power."""
    d = getattr(r, "disagg", None)
    reps = getattr(r, "replicas", 1)
    return (r.config.base_key() + (f"|pd:{d.prefill.key()}" if d is not None else "")
            + (f"|x{reps}" if reps > 1 else ""))


def _strategies(r: TrialResult) -> int:
    """Optional strategies a config switches on beyond the engine's defaults.

    Within the noise band the config with fewer of them wins, so a strategy is adopted only when
    it measurably helps. A quantized KV cache or speculative decoding that merely ties is complexity
    (and, for the cache, a quality risk) with nothing to show for it. So is a KV-cache tier that ties: it
    holds host RAM (and perhaps disk) the machine could use for something else.
    """
    c = r.config
    return sum((c.kv_dtype != "auto", c.spec_decode is not None, c.prefill_budget is not None,
                bool({"cache_reuse", "kv_unified"} & set(c.extra)), c.kv_tier is not None))


def _joules(x: Ranked) -> float:
    j = x.metrics.joules_per_token
    return j if (j is not None and math.isfinite(j)) else math.inf


def _break_ties(ranked: List[Ranked], cons: Constraints) -> List[Ranked]:
    """Within the leading cluster of feasible, near-equal scores: fewer optional strategies first, then
    the larger config, then less energy.

    Two things join the leader's cluster: anything within `noise_tolerance` of its score, and any
    power-capped or clock-locked variant of the same configuration within `power_max_loss`.
    Variants share context, batch and memory settings, so the energy key decides between them,
    and a cap that saves joules at no measurable throughput cost wins over running uncapped.
    """
    if not ranked or not ranked[0].feasible:
        return ranked
    lead = ranked[0]
    span = abs(lead.score) * max(cons.noise_tolerance, 0.0)
    power_span = abs(lead.score) * max(cons.power_max_loss, cons.noise_tolerance, 0.0)
    lead_base = _variant_key(lead.result)

    def joins(x: Ranked) -> bool:
        if not x.feasible:
            return False
        gap = abs(x.score - lead.score)
        return gap <= span or (_variant_key(x.result) == lead_base and gap <= power_span)

    cluster = [x for x in ranked if joins(x)]
    if len(cluster) < 2:
        return ranked
    cluster.sort(key=lambda x: (_strategies(x.result), tuple(-v for v in _capability(x.result)), _joules(x)))
    rest = [x for x in ranked if x not in cluster]
    return cluster + rest


def _rank_at_rate(ok: Sequence[TrialResult], cons: Constraints) -> List[Ranked]:
    """`efficiency` at an arrival rate: one score per trial, its joules per request at that rate.

    A trial is not scored at one of its levels but at the operating point the rate puts it at, composed from
    every level (calibrate/rate.py). Latency is judged on every measured level from that point up to what 95% of
    arrivals are served with, and `metrics` is the worst of them. The throughput floor does not apply: what the
    rate needs is the trial's capacity, which `max_utilisation` checks.

    Every trial is counted in one request size (shared_request_size), so a configuration cannot win by answering
    shorter. Idle power is pooled per engine and power setting (idle_pools), so two configurations of one engine
    differ in idle only by more than its readings spread. And when any trial carries an idle measurement, one
    without it is not eligible: its fallback leaves out the idle draw, about P_idle / λ per request, and would beat
    every trial that counts it. Replica and disaggregated trials never carry one, so at a rate a layout stage
    cannot pick them on that omission.
    """
    assert cons.arrival_rate is not None
    size = shared_request_size(ok)
    lowest = min((x.concurrency for r in ok for x in _levels(r)), default=None)
    pools = idle_pools(ok)
    comps = [compose(r.metrics, cons.arrival_rate, cons.max_utilisation, tokens_per_request=size[0] if size else None,
                     workload_lowest=lowest, idle_w=pooled_idle(r, pools)) for r in ok]
    any_idle = any(c.total_j is not None and not c.fallback for c in comps)
    out: List[Ranked] = []
    for r, comp in zip(ok, comps):
        judged = comp.latency_levels or [comp.operating]
        worst = max(judged, key=lambda m: _latency_excess(m, cons))
        latency = _latency_excess(worst, cons) + (LATENCY_UNKNOWN if comp.latency_beyond_top else 0.0)
        no_idle = 1.0 if (any_idle and comp.fallback) else 0.0
        key = (no_idle, comp.violation, latency)
        # No energy telemetry: order by tok/s, as J/token efficiency does.
        score = comp.total_j if comp.total_j is not None else 1e9 - worst.tok_s
        out.append(Ranked(result=r, metrics=worst, score=score, feasible=key == (0.0, 0.0, 0.0),
                          violation=no_idle * NO_IDLE_VIOLATION + comp.violation + latency, rate=comp,
                          violation_key=key))
    out.sort(key=_sort_key)
    return out


NO_IDLE_VIOLATION = 1e6  # a trial without idle data, ranked beside trials with it: last among the infeasible
LATENCY_UNKNOWN = 1e6  # ms: the rate needs a level that was never measured, so its latency is not known
IDLE_BAND_FLOOR_W = 2.0  # NVML reads power in whole watts or so; two configurations closer than this idle alike
OUTPUT_CHANGE = 0.25  # answers this much longer or shorter than the median: the configuration changes the output


def _latency_excess(m: TrialMetrics, cons: Constraints) -> float:
    tpot = m.tpot_ms if (m.tpot_ms is not None and math.isfinite(m.tpot_ms)) else 0.0
    return (max(0.0, _ttft_for(m, cons.ttft_percentile) - cons.ttft_ceiling_ms)
            + (max(0.0, tpot - cons.tpot_ceiling_ms) if cons.tpot_ceiling_ms is not None else 0.0))


@dataclass(frozen=True)
class IdlePool:
    median_w: float
    low_w: float
    high_w: float
    band_w: float  # a reading this far from the median is the configuration's own, not noise
    trials: int


def _idle_group(r: TrialResult) -> Tuple[str, Optional[int], Optional[int]]:
    return (r.config.backend, r.config.power_limit_w, r.config.sm_clock_mhz)


def idle_pools(ok: Sequence[TrialResult]) -> Dict[Tuple[str, Optional[int], Optional[int]], IdlePool]:
    """Idle power per (engine, power setting), pooled over every trial that read it: one 1-second reading per trial
    is noisy, and the idle draw belongs to the card and the loaded engine far more than to one configuration of it.
    The band is three robust standard deviations (1.4826 x the median absolute deviation), at least 2 W."""
    groups: Dict[Tuple[str, Optional[int], Optional[int]], List[float]] = {}
    for r in ok:
        if r.metrics.idle_power_w is not None:
            groups.setdefault(_idle_group(r), []).append(r.metrics.idle_power_w)
    pools = {}
    for g, vals in groups.items():
        med = statistics.median(vals)
        mad = statistics.median(abs(v - med) for v in vals)
        pools[g] = IdlePool(median_w=med, low_w=min(vals), high_w=max(vals),
                            band_w=max(IDLE_BAND_FLOOR_W, 3 * 1.4826 * mad), trials=len(vals))
    return pools


def pooled_idle(r: TrialResult, pools: Dict[Tuple[str, Optional[int], Optional[int]], IdlePool]) -> Optional[float]:
    """The idle power a trial is scored with: its engine's pooled median, unless its own reading is outside the band."""
    own = r.metrics.idle_power_w
    pool = pools.get(_idle_group(r))
    if own is None or pool is None or pool.trials < 2:
        return own
    return own if abs(own - pool.median_w) > pool.band_w else pool.median_w


def _answer_tokens(r: TrialResult) -> Optional[float]:
    low = lowest_level(r.metrics)
    return low.output_tokens / (low.requests - low.failed) if low and low.requests > low.failed else None


def shared_request_size(ok: Sequence[TrialResult]) -> Optional[Tuple[float, int, int]]:
    """(tokens per request, the concurrency it was read at, trials it came from): the median of the answers'
    mean length at the lowest concurrency level the trials measured. Real-text answers stop when the model does,
    and on Dolly prompts on an A40, scored by its own answers' length, the J/request pick answered in 194 tokens where the J/token pick took 373;
    counted per request of its own length, the shorter answers looked like saved energy."""
    lowest = min((x.concurrency for r in ok for x in _levels(r)), default=None)
    sizes = []
    for r in ok:
        at = [x for x in _levels(r) if x.concurrency == lowest and x.requests > x.failed and x.output_tokens]
        if at:
            sizes.append(at[0].output_tokens / (at[0].requests - at[0].failed))
    if not sizes or lowest is None:
        return None
    return statistics.median(sizes), lowest, len(sizes)


def rank(results: Sequence[TrialResult], objective: str, cons: Optional[Constraints] = None) -> List[Ranked]:
    """Best first. Feasible candidates always precede infeasible ones."""
    cons = cons or Constraints()
    ok = _ok_results(results)
    if not ok:
        return []
    if objective == "efficiency" and cons.arrival_rate is not None:
        return _break_ties(_rank_at_rate(ok, cons), cons)
    goal = resolve_goal(objective, cons)
    bounds = goal.bounds(ok)  # a floor relative to the best is relative to these trials
    ranked: List[Ranked] = []
    for r in ok:
        # Score the trial at its best concurrency level for this objective.
        per_level = []
        for m in _levels(r):
            if goal.custom:
                key = goal.violation_key(m, bounds)
                per_level.append(Ranked(result=r, metrics=m, score=goal.score(m), feasible=key[0] == 0.0,
                                        violation=key[1], violation_key=key))
                continue
            violation = goal.violation(m, bounds)
            per_level.append(Ranked(result=r, metrics=m, score=goal.score(m), feasible=violation == 0.0,
                                    violation=violation))
        per_level.sort(key=_sort_key)
        ranked.append(per_level[0])
    ranked.sort(key=_sort_key)
    return _break_ties(ranked, cons)


def capacity_level(result: Optional[TrialResult], objective: str, cons: Optional[Constraints]) -> Optional[int]:
    """The busiest concurrency level measured for `result` at which it met the constraints, each level judged on
    its own; None when no measured level met them."""
    if result is None or not result.ok:
        return None
    best: Optional[int] = None
    for m in _levels(result):
        alone = result.model_copy(update={"metrics": m.model_copy(update={"by_concurrency": {}})})
        ranked = rank([alone], objective, cons)
        if ranked and ranked[0].feasible and m.concurrency:
            best = max(best or 0, m.concurrency)
    return best


def ranked_level(result: TrialResult, results: Sequence[TrialResult], objective: str,
                 cons: Optional[Constraints]) -> Optional[int]:
    """The concurrency level `result` was scored at, ranked among `results`."""
    mine = [x for x in rank(results, objective, cons) if x.result is result]
    return mine[0].metrics.concurrency if mine else None


def strict_refusal(winner: Optional[TrialResult], results: Sequence[TrialResult], objective: str,
                   cons: Optional[Constraints], what: str = "configuration") -> Optional[str]:
    """Why `winner`, picked from `results`, may not be served under --strict-constraints: it misses them. None when
    it meets them, or strict mode is off. Judged among the same results it was picked from, since a throughput floor
    relative to the best depends on them."""
    if cons is None or not cons.strict or winner is None:
        return None
    mine = [x for x in rank(results, objective, cons) if x.result is winner]
    if mine and mine[0].feasible:
        return None
    if objective == CUSTOM:
        by = f" (the closest {_missed(mine[0])})" if mine else ""
        return (f"strict constraints: no measured {what} met every limit of the goal {cons.goal}{by}; refusing to "
                "serve the one that missed the least")
    by = f" (the closest missed by {mine[0].violation:.1f})" if mine else ""
    return (f"strict constraints: no measured {what} met the {objective} constraints{by}; refusing to serve the "
            "least-violating one")


def pick(
    results: Sequence[TrialResult], objective: str, cons: Optional[Constraints] = None
) -> Tuple[Optional[TrialResult], List[str]]:
    """Return (winner, notes). Winner is None only if no trial succeeded."""
    ranked = rank(results, objective, cons)
    if not ranked:
        return None, ["no successful trials"]
    top = ranked[0]
    notes: List[str] = []
    if not top.feasible and objective == CUSTOM:
        notes.append(f"no config met every limit of the goal {(cons or Constraints()).goal}; picked the one that "
                     f"missed the least ({_missed(top)})")
    elif not top.feasible:
        notes.append(
            f"no config satisfied the {objective} constraint; picked the least-violating one "
            f"(violation {top.violation:.1f})"
        )
    no_energy = top.rate.total_j is None if top.rate is not None else top.metrics.joules_per_token is None
    if objective == "efficiency" and no_energy:
        notes.append("no energy telemetry available; efficiency objective fell back to tok/s ordering")
    m = top.metrics
    notes.append(
        f"winner scored at concurrency {m.concurrency}: {m.tok_s:.1f} tok/s, "
        f"TTFT {m.ttft_ms:.0f} ms" + (f" (p95 {m.ttft_p95_ms:.0f})" if m.ttft_p95_ms is not None else "")
        + (f", {m.joules_per_token:.3f} J/tok" if m.joules_per_token else "")
    )
    if top.rate is not None:
        notes.append(top.rate.note() + _size_note(results) + _own_length_note(top, cons) + _spread(ranked)
                     + _idle_advice(top.rate) + _no_idle_note(ranked))
        for extra in (_idle_pool_note(results), _output_change_note(ranked, results)):
            if extra:
                notes.append(extra)
    return top.result, notes


def _missed(x: Ranked) -> str:
    """How a custom goal's least-missing level missed: how many limits, and how far past them in all."""
    missed = int(x.violation_key[0]) if x.violation_key else 0
    return f"missed {missed} limit{'s' if missed != 1 else ''}, {x.violation:.0%} past {'them' if missed != 1 else 'it'}"


def _own_length_note(top: Ranked, cons: Optional[Constraints]) -> str:
    """The winner's J/request at its own answers' length: what its requests cost, beside the normalised score."""
    c, own = top.rate, _answer_tokens(top.result)
    if c is None or c.total_j is None or not c.tokens_per_request or own is None or cons is None:
        return ""
    mine = compose(top.result.metrics, c.arrival_rate, cons.max_utilisation, tokens_per_request=own, idle_w=c.idle_w)
    if mine.total_j is None:
        return ""
    return f"; at its own answers' length, {own:.0f} tokens, that is {mine.total_j:.1f} J/request"


def _idle_pool_note(results: Sequence[TrialResult]) -> str:
    pools = idle_pools(_ok_results(results))
    if not pools:
        return ""
    parts = []
    for (backend, cap, clock), p in sorted(pools.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, kv[0][2] or 0)):
        label = backend + (f" capped at {cap} W" if cap else "") + (f" at {clock} MHz" if clock else "")
        parts.append(f"{label} {p.median_w:.1f} W, the median of {p.trials} trials ({p.low_w:.1f}-{p.high_w:.1f}; "
                     f"a configuration within {p.band_w:.1f} W of it is scored at the median)")
    return "idle power pooled per engine: " + "; ".join(parts)


def _output_change_note(ranked: Sequence[Ranked], results: Sequence[TrialResult]) -> str:
    """Configurations whose answers are far longer or shorter than the median: they change the output, so their
    energy per request compares different work, and their answers deserve a quality check."""
    size = shared_request_size(_ok_results(results))
    if size is None:
        return ""
    changed = []
    for x in ranked:
        t = _answer_tokens(x.result)
        if t is not None and abs(t / size[0] - 1) > OUTPUT_CHANGE:
            changed.append((x, t))
    if not changed:
        return ""
    shown = "; ".join(f"{x.result.config.key()} ({t:.0f} tokens)" for x, t in changed[:3])
    more = f"; and {len(changed) - 3} more" if len(changed) > 3 else ""
    lead = " The winner is one of them." if any(x is ranked[0] for x, _ in changed) else ""
    return (f"output-changing: {len(changed)} configuration{'s answer' if len(changed) > 1 else ' answers'} more "
            f"than {OUTPUT_CHANGE:.0%} longer or shorter than the median {size[0]:.0f} tokens at {size[1]} users, so "
            f"they change what is generated, not only how fast: {shown}{more}.{lead}")


def _size_note(results: Sequence[TrialResult]) -> str:
    size = shared_request_size(_ok_results(results))
    if size is None:
        return ""
    return (f"; every configuration counted in requests of {size[0]:.0f} tokens, the median answer at "
            f"{size[1]} users across {size[2]} trials")


def _no_idle_note(ranked: Sequence[Ranked]) -> str:
    skipped = [x for x in ranked if x.rate is not None and x.rate.fallback and x.violation >= NO_IDLE_VIOLATION]
    if not skipped:
        return ""
    return (f"; {len(skipped)} configuration{'s were' if len(skipped) > 1 else ' was'} measured without an idle "
            f"tail (replica or disaggregated layouts record none) and not eligible at this rate")


def _idle_advice(c: Composition) -> str:
    if c.total_j is None or c.fallback or not c.total_j or c.idle_j / c.total_j < 0.5:
        return ""
    return ("; most of each request's energy is the card idling between requests, which no configuration "
            "changes and a power cap cannot lower (it only shortens a wind-down drawing more than the cap)")


def _spread(ranked: Sequence[Ranked]) -> str:
    """What the user can act on: how far the configurations measured differ in idle draw and in serving energy.

    Idle draw is the card's, not the configuration's, so it barely moves between configurations; when it is
    most of each request, no setting rigtune can choose will move the total much. A power cap cannot lower it.
    """
    comps = [x.rate for x in ranked if x.rate is not None and x.rate.total_j is not None and not x.rate.fallback]
    if len(comps) < 2:
        return ""
    idle = [c.idle_j for c in comps]
    active = [c.active_j + c.hold_j for c in comps]
    return (f"; across the {len(comps)} configurations measured, idle ranged {min(idle):.1f}-{max(idle):.1f} "
            f"J/request and serving plus wind-down {min(active):.1f}-{max(active):.1f}")
