"""What a goal's limits cost, read from the trials a calibration recorded: no launch, never stored in the profile.

Every row is a measured (configuration, concurrency) point. The table shows the pick at the level it was scored at,
the best point on each metric with the limits ignored (and the limits it breaks), the points among those meeting
the limits that nothing beats on every goal metric at once, and, for each limit, the best point once that one
limit is dropped. Only configurations the search tried are here, and only the levels it measured, so every cost
is a floor: an untried configuration might have done better without the limit.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rigtune.calibrate.objectives import (
    CUSTOM,
    METRICS,
    Constraints,
    Goal,
    _levels,
    _ok_results,
    format_goal,
    rank,
    resolve_goal,
)
from rigtune.goals import limit_text
from rigtune.models import Profile, TrialMetrics, TrialResult

PARETO_ROWS = 8
# The table's columns: (metric, heading, decimals).
COLUMNS: Tuple[Tuple[str, str, int], ...] = (
    ("tok_s", "tok/s", 1), ("ttft_p95_ms", "p95 TTFT ms", 0), ("tpot_ms", "TPOT ms", 1),
    ("request_latency_ms", "request ms", 0), ("joules_per_token", "J/tok", 3), ("power_w", "W", 0),
    ("mem_mb", "peak MiB", 0), ("cost_per_mtok", "$/Mtok", 3),
)


@dataclass
class Point:
    result: TrialResult
    metrics: TrialMetrics  # the level
    values: Dict[str, Optional[float]]
    score: float  # the goal's, lower better
    misses: List[str]  # the limits this level misses, as the goal writes them


@dataclass
class Row:
    why: str
    point: Point
    within_noise: bool = False  # within the noise tolerance of the pick's score


@dataclass
class TradeOffs:
    goal: str  # the goal the table is scored under, as text
    columns: List[Tuple[str, str, int]]
    rows: List[Row] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _point(r: TrialResult, m: TrialMetrics, goal: Goal, bounds: Sequence[float], priced: Goal) -> Point:
    values = {name: priced.reader(name)(m) for name, _, _ in COLUMNS}
    misses = [limit_text(lim) for lim, b in zip(goal.limits, bounds) if lim.excess(m, b, goal.reader(lim.metric)) > 0]
    return Point(r, m, values, goal.score(m), misses)


def _best(points: Sequence[Point]) -> Optional[Point]:
    return min(points, key=lambda p: p.score, default=None)


def _dominates(a: Point, b: Point, names: Sequence[str]) -> bool:
    better = False
    for name in names:
        va, vb = a.values.get(name), b.values.get(name)
        if va is None or vb is None:
            return False
        if METRICS[name].sense == "max":
            va, vb = -va, -vb
        if va > vb:
            return False
        better = better or va < vb
    return better


def tradeoffs(table: Sequence[TrialResult], objective: str, cons: Optional[Constraints] = None,
              accuracy: Optional[Sequence[str]] = None,
              prices: Optional[Tuple[Optional[float], Optional[float]]] = None,
              winner: Optional[str] = None) -> TradeOffs:
    """The trade-off table of `objective` under `cons`, over the recorded trials. Pure: nothing is measured or
    changed. `accuracy` is what the accuracy gate's refusals cost (AccuracyGate.price_notes), shown under the
    table; `prices` (per GPU-hour, per kWh) add a $/Mtok column to a goal that does not count cost itself.

    `winner` is the configuration the calibration served (its key), when the table is scored under the goal it was
    calibrated for: the search chose it on its fresh confirm runs, or the accuracy gate narrowed the choice, so the
    best-ranked recorded point is not always it. Without one, the pick is the best-ranked point under this goal.
    Every run of the picked configuration is the pick: another of its runs is never offered as a trade."""
    cons = cons or Constraints()
    goal = resolve_goal(objective, cons)
    priced = goal if goal.gpu_hourly_usd is not None or prices is None else dataclasses.replace(
        goal, gpu_hourly_usd=prices[0], kwh_usd=prices[1])
    columns = [c for c in COLUMNS if c[0] != "cost_per_mtok" or priced.gpu_hourly_usd is not None]
    out = TradeOffs(goal=_goal_text(objective, goal), columns=columns)
    if cons.arrival_rate is not None:
        out.notes.append("not shown at an arrival rate: the pick is scored at an operating point composed from "
                         "every level, not at one measured level")
        return out
    ok = _ok_results(table)
    ranked = rank(ok, objective, cons)
    if not ranked:
        out.notes.append("no successful trials")
        return out
    bounds = goal.bounds(ok)
    points = [_point(r, m, goal, bounds, priced) for r in ok for m in _levels(r)]
    served = [x for x in ranked if _key(x.result) == winner] if winner is not None else []
    lead = served[0] if served else ranked[0]
    pick = next(p for p in points if p.result is lead.result and p.metrics is lead.metrics)
    label = "the pick" if served else "the pick under this goal"
    out.rows.append(Row(label + ("" if lead.feasible else ", missing limits"), pick))
    picked = _key(pick.result)
    others = [p for p in points if _key(p.result) != picked]
    for name, heading, _ in columns:
        have = [p for p in points if p.values[name] is not None]
        if not have:
            continue
        sign = -1.0 if METRICS[name].sense == "max" else 1.0
        best = min(have, key=lambda p: sign * p.values[name])  # type: ignore[operator]
        out.rows.append(Row(f"best {heading}, limits ignored", best))
    meeting = [pick, *(p for p in others if not p.misses)] if not pick.misses else [p for p in others
                                                                                       if not p.misses]
    names = list(dict.fromkeys([goal.optimise, *(lim.metric for lim in goal.limits)]))
    front = sorted((p for p in meeting if not any(_dominates(q, p, names) for q in meeting)), key=lambda p: p.score)
    for p in front[:PARETO_ROWS]:
        out.rows.append(Row("meets the limits, unbeaten on " + ", ".join(names), p))
    if len(front) > PARETO_ROWS:
        out.notes.append(f"{len(front) - PARETO_ROWS} more points meet the limits and are unbeaten on "
                         f"{', '.join(names)}; not shown")
    for lim in goal.limits:
        # What this limit alone kept out: the best point that misses it and meets every other limit. A point that
        # meets this limit too owes nothing to dropping it, so no gain is claimed for it.
        text = limit_text(lim)
        best = _best([p for p in others if p.misses == [text]])
        if best is not None and best.score < pick.score:
            out.rows.append(Row(f"without {text}{_gain(goal, pick, best)}", best))
        else:
            out.notes.append(f"without {text}: no recorded configuration does better on {goal.optimise}")
    tol = max(cons.noise_tolerance, 0.0)
    for row in out.rows:
        row.within_noise = row.point is not pick and _close(row.point.score, pick.score, tol)
    out.notes += _caveats(points, table)
    out.notes += list(accuracy or [])
    return out


def _close(a: float, b: float, tol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= abs(b) * tol


def _gain(goal: Goal, pick: Point, best: Point) -> str:
    """How much better the optimised metric is than the pick's, as `: +12.0%` (blank when not comparable)."""
    a = goal.reader(goal.optimise)(pick.metrics)
    b = goal.reader(goal.optimise)(best.metrics)
    if a is None or b is None or not a:
        return ""
    change = (b - a) / abs(a) if METRICS[goal.optimise].sense == "max" else (a - b) / abs(a)
    return f": {change:+.1%} {'more' if METRICS[goal.optimise].sense == 'max' else 'less'} {goal.optimise}"


def _goal_text(objective: str, goal: Goal) -> str:
    text = format_goal(goal)
    return text if objective == CUSTOM else f"{objective}: {text}"


def _caveats(points: Sequence[Point], table: Sequence[TrialResult]) -> List[str]:
    notes = ["recorded points only: configurations the search never tried are absent, so each cost is a floor"]
    if any(r.stage == "confirm" for r in table):
        notes.append("search-stage rows are each configuration's run during the search (the best of noisy runs "
                     "picked it out); confirm rows are fresh re-measurements")
    full = max((len(r.metrics.by_concurrency) for r in table if r.ok), default=0)
    if any(0 < len(p.result.metrics.by_concurrency) < full for p in points):
        notes.append("levels a trial stopped before (measured busiest first) are absent; --all-levels measures them")
    return notes


def to_lines(t: TradeOffs) -> List[str]:
    """The table as Markdown, rows in the order above."""
    heads = ["why", "configuration", "users", "stage"] + [h for _, h, _ in t.columns] + ["misses"]
    lines = [f"Trade-offs under {t.goal}", "", "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for row in t.rows:
        p = row.point
        cells = [row.why + (" (within noise of the pick)" if row.within_noise else ""), _key(p.result),
                 str(p.metrics.concurrency), p.result.stage]
        cells += ["-" if p.values[name] is None else f"{p.values[name]:.{nd}f}" for name, _, nd in t.columns]
        cells.append(", ".join(p.misses) or "-")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [""] + [f"- {n}" for n in t.notes]
    return lines


def _key(r: TrialResult) -> str:
    if r.disagg is not None:
        return r.disagg.key()
    return f"{r.config.key()} x{r.replicas}" if r.replicas > 1 else r.config.key()


def profile_constraints(p: Profile) -> Constraints:
    """The constraints a profile was scored with. One written before rigtune recorded them carries its workload's
    ceilings, and the TTFT percentile its options name."""
    if p.calibration_constraints:
        from rigtune.goals import recorded_constraints

        return recorded_constraints(p.calibration_constraints)
    spec = p.workload_spec or {}
    return Constraints(ttft_ceiling_ms=spec.get("ttft_ceiling_ms") or Constraints().ttft_ceiling_ms,
                       tpot_ceiling_ms=spec.get("tpot_ceiling_ms"),
                       ttft_percentile=95 if p.options.get("ttft") == "p95" else 50)
