"""A frozen copy of how the four objectives scored, ranked and stopped early before goals generalised them.

tests/test_preset_pins.py compares today's rigtune.calibrate.objectives against this, with ==, on every recorded
calibration table and on seeded synthetic trials, so a refactor that is meant to move nothing shows it moved
nothing. Never edit it to follow the live code: a difference is what the pins exist to catch. Only the rate path
(efficiency at an arrival rate) is taken from the live module, as the goals work leaves it alone.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

from rigtune.calibrate.objectives import Constraints, Ranked, _rank_at_rate
from rigtune.calibrate.tail import quantile_interval
from rigtune.models import OBJECTIVES, TrialMetrics, TrialResult


def _ok_results(results: Sequence[TrialResult]) -> List[TrialResult]:
    return [r for r in results if r.ok and not r.metrics.open_loop]


def _ttft(m: TrialMetrics) -> float:
    t = m.ttft_ms
    return t if (t is not None and math.isfinite(t)) else math.inf


def _ttft_for(m: TrialMetrics, percentile: int) -> float:
    if percentile >= 95 and m.ttft_p95_ms is not None and math.isfinite(m.ttft_p95_ms):
        return m.ttft_p95_ms
    return _ttft(m)


def _levels(r: TrialResult) -> List[TrialMetrics]:
    levels = [m for m in r.metrics.by_concurrency.values() if m.ok]
    return levels or [r.metrics]


def _e2e_latency(m: TrialMetrics) -> float:
    ttft = _ttft(m)
    if not math.isfinite(ttft):
        return ttft
    per_request = (m.output_tokens / m.requests) if m.requests else 0.0
    tpot = m.tpot_ms if (m.tpot_ms is not None and math.isfinite(m.tpot_ms)) else 0.0
    return ttft + tpot * max(0.0, per_request - 1)


def _tok_s_floor(cons: Constraints, results: Sequence[TrialResult]) -> float:
    if cons.tok_s_floor_abs is not None:
        return cons.tok_s_floor_abs
    best = max((r.metrics.tok_s for r in results), default=0.0)
    return cons.tok_s_floor_frac * best


def _score_and_constraint(
    objective: str, cons: Constraints, results: Sequence[TrialResult]
) -> Tuple[Callable[[TrialMetrics], float], Callable[[TrialMetrics], float]]:
    if objective == "throughput":
        return (lambda m: -m.tok_s), (lambda m: 0.0)
    if objective == "latency":
        floor = _tok_s_floor(cons, results)
        return _e2e_latency, (lambda m: max(0.0, floor - m.tok_s))
    if objective == "balanced":
        return (lambda m: -m.tok_s), (lambda m: max(0.0, _ttft_for(m, cons.ttft_percentile) - cons.ttft_ceiling_ms))
    if objective == "efficiency":
        floor = _tok_s_floor(cons, results)

        def score(m: TrialMetrics) -> float:
            j = m.joules_per_token
            if j is None or not math.isfinite(j):
                return 1e9 - m.tok_s
            return j

        return score, (lambda m: max(0.0, floor - m.tok_s))
    raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")


def enough_level(objective: str, cons: Optional[Constraints] = None) -> Optional[Callable[[TrialMetrics], bool]]:
    if objective not in ("throughput", "balanced"):
        return None
    cons = cons or Constraints()
    _, violation = _score_and_constraint(objective, cons, [])
    tpot_ceiling = cons.tpot_ceiling_ms

    def enough(m: TrialMetrics) -> bool:
        if not m.ok or violation(m) > 0:
            return False
        tpot = m.tpot_ms if (m.tpot_ms is not None and math.isfinite(m.tpot_ms)) else 0.0
        return tpot_ceiling is None or tpot <= tpot_ceiling

    return enough


def close_call_level(objective: str, cons: Optional[Constraints] = None,
                     confidence: float = 0.95) -> Optional[Callable[[TrialMetrics], bool]]:
    if objective != "balanced":
        return None
    cons = cons or Constraints()
    q = 0.95 if cons.ttft_percentile >= 95 else 0.5

    def close(m: TrialMetrics) -> bool:
        if not m.ok or not m.ttft_samples_ms:
            return False
        low, high = quantile_interval(m.ttft_samples_ms, q, confidence, unbounded=False)
        return low <= cons.ttft_ceiling_ms <= high

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
    d = getattr(r, "disagg", None)
    reps = getattr(r, "replicas", 1)
    return (r.config.base_key() + (f"|pd:{d.prefill.key()}" if d is not None else "")
            + (f"|x{reps}" if reps > 1 else ""))


def _strategies(r: TrialResult) -> int:
    c = r.config
    return sum((c.kv_dtype != "auto", c.spec_decode is not None, c.prefill_budget is not None,
                bool({"cache_reuse", "kv_unified"} & set(c.extra)), c.kv_tier is not None))


def _joules(x: Ranked) -> float:
    j = x.metrics.joules_per_token
    return j if (j is not None and math.isfinite(j)) else math.inf


def _break_ties(ranked: List[Ranked], cons: Constraints) -> List[Ranked]:
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


def rank(results: Sequence[TrialResult], objective: str, cons: Optional[Constraints] = None) -> List[Ranked]:
    cons = cons or Constraints()
    ok = _ok_results(results)
    if not ok:
        return []
    if objective == "efficiency" and cons.arrival_rate is not None:
        return _break_ties(_rank_at_rate(ok, cons), cons)
    score, violation = _score_and_constraint(objective, cons, ok)
    if cons.tpot_ceiling_ms is not None:
        phase_violation, ceiling = violation, cons.tpot_ceiling_ms

        def violation(m: TrialMetrics) -> float:  # noqa: F811
            tpot = m.tpot_ms if (m.tpot_ms is not None and math.isfinite(m.tpot_ms)) else 0.0
            return phase_violation(m) + max(0.0, tpot - ceiling)
    ranked: List[Ranked] = []
    for r in ok:
        per_level = [
            Ranked(result=r, metrics=m, score=score(m), feasible=violation(m) == 0.0, violation=violation(m))
            for m in _levels(r)
        ]
        per_level.sort(key=_sort_key)
        ranked.append(per_level[0])
    ranked.sort(key=_sort_key)
    return _break_ties(ranked, cons)
