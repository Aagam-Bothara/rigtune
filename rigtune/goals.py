"""Goals as the command line states them: `--optimise METRIC` and `--limit "METRIC<=N[unit]"`.

A stated goal that says what a preset says IS that preset: the same objective name, the same constraints and so
the same cached profile, so `--optimise tok_s --limit "ttft_p95_ms<=300"` is `--objective balanced --ttft-ceiling
300`, and no question is ever cached twice. Only a goal no preset expresses becomes the `custom` objective, whose
text (objectives.format_goal) the constraints carry.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from rigtune.calibrate.objectives import (
    CUSTOM,
    METRICS,
    Constraints,
    Goal,
    Limit,
    format_goal,
    resolve_goal,
    ttft_metric,
)

# Units a limit may be written in (in any case), by kind, as the multiplier to the metric's own unit. A bare number
# is in that unit.
UNITS: Dict[str, Dict[str, float]] = {
    "time": {"ms": 1.0, "s": 1000.0},
    "power": {"W": 1.0},
    "memory": {"MiB": 1.0, "GiB": 1024.0},
}
KIND = {"ttft_p95_ms": "time", "ttft_p50_ms": "time", "tpot_ms": "time", "request_latency_ms": "time",
        "power_w": "power", "mem_mb": "memory"}

# The accuracy gate's limit (accuracy_gate.py): not a per-level metric but a loss certified on --eval-set, so it
# sets --max-accuracy-loss rather than joining the goal. In accuracy points; `1%` is one point too.
ACCURACY = "accuracy_loss"
DEFAULT_MAX_LOSS = 3.0  # --max-accuracy-loss's default (pipeline.SearchOptions)

_LIMIT = re.compile(r"\s*(?P<metric>[A-Za-z0-9_]+)\s*(?P<op><=|>=|=<|=>|<|>|=)?\s*(?P<rest>.*?)\s*")
_VALUE = re.compile(r"(?P<num>\d+(?:\.\d*)?|\.\d+)\s*(?P<unit>[a-z%]*)")


class GoalError(ValueError):
    """A goal stated on the command line that cannot be calibrated as written."""


def _names(optimisable: bool = False) -> str:
    return ", ".join(m.name for m in METRICS.values() if m.optimisable or not optimisable)


def check_optimise(metric: str) -> str:
    metric = metric.lower()  # as --limit reads its metric
    if metric not in METRICS:
        raise GoalError(f"unknown metric {metric!r}; --optimise takes one of {_names(True)}")
    if not METRICS[metric].optimisable:
        raise GoalError(f"{metric} is a limit only (--limit \"{metric}<=N\"); --optimise takes one of "
                        f"{_names(True)}")
    return metric


def parse_limit(text: str) -> Limit:
    """One --limit, in the metric's own unit. Refused with the reason for anything the goal could not mean."""
    m = _LIMIT.fullmatch(text)
    if m is None:
        raise GoalError(f"{text!r} is not a limit: write METRIC<=N or METRIC>=N, e.g. \"ttft_p95_ms<=800\"")
    name, op, rest = m["metric"].lower(), m["op"], m["rest"]
    if op is None and not rest:
        raise GoalError(f"{text!r} has no bound: quote the limit, --limit \"{text}>=N\"; unquoted, the shell took "
                        "`>` (or `<`) as a redirect")
    if name == ACCURACY:
        return _accuracy_limit(text, op, rest)
    if name not in METRICS:
        raise GoalError(f"unknown metric {name!r}; --limit takes one of {_names()}, {ACCURACY}")
    metric = METRICS[name]
    wanted = "<=" if metric.sense == "min" else ">="
    if op in ("<", ">", "=", None):
        raise GoalError(f"{text!r}: a limit is {wanted} (it includes its bound), not {op or 'a bare number'}")
    op = {"=<": "<=", "=>": ">="}.get(op, op)
    if op != wanted:
        raise GoalError(f"{text!r}: {name} is better {'lower' if metric.sense == 'min' else 'higher'}, so its "
                        f"limit is {wanted}")
    v = _VALUE.fullmatch(rest.lower())
    if v is None:
        raise GoalError(f"{text!r}: the bound must be a number, with a unit if you like (ms, s, W, MiB, GiB)")
    number, unit = float(v["num"]), v["unit"]
    if not math.isfinite(number):  # a number too long for a float reads as infinity, which no goal writes back
        raise GoalError(f"{text!r}: the bound is not a finite number")
    if unit == "%best":
        if name != "tok_s":
            raise GoalError(f"{text!r}: %best is a share of the best tok/s measured, for tok_s only")
        if not 0 < number <= 100:
            raise GoalError(f"{text!r}: a share of the best tok/s is above 0% and at most 100%")
        return Limit(name, op, number / 100, of_best=True)
    if not unit:
        return Limit(name, op, number)
    scale = {u.lower(): x for u, x in UNITS.get(KIND.get(name, ""), {}).items()}.get(unit)
    if scale is None:
        units = ", ".join(UNITS[KIND[name]]) if name in KIND else ("%best" if name == "tok_s" else "none")
        raise GoalError(f"{text!r}: {name} does not take {unit!r} (units: {units})")
    # Rounded, so one bound written two ways is one goal: 1.001 s is 1000.9999999999999 ms unrounded.
    return Limit(name, op, round(number * scale, 9))


def _accuracy_limit(text: str, op: Optional[str], rest: str) -> Limit:
    if op != "<=" and op != "=<":
        raise GoalError(f"{text!r}: {ACCURACY} is better lower, so its limit is <=")
    v = _VALUE.fullmatch(rest.lower())
    if v is None or v["unit"] not in ("", "%", "points", "point"):
        raise GoalError(f"{text!r}: {ACCURACY} is in accuracy points, e.g. \"{ACCURACY}<=1\" (1% is one point)")
    points = float(v["num"])
    if not 0 < points < 100:
        raise GoalError(f"{text!r}: a loss in accuracy points, above 0 and below 100")
    return Limit(ACCURACY, "<=", points)


@dataclass(frozen=True)
class Resolved:
    objective: str  # a preset's name, or custom
    constraints: Constraints
    ttft_percentile: int  # for the search options, whose cache key names it under balanced
    line: str  # what the run says before it starts: the goal, and whether a preset says the same
    # The accuracy gate's tolerance, in points: --max-accuracy-loss, or an accuracy_loss limit.
    max_accuracy_loss: float = DEFAULT_MAX_LOSS


def _same(stated: Dict[str, Limit], lim: Limit, how: str) -> None:
    """Add a limit, refusing a metric stated twice with different bounds."""
    had = stated.get(lim.metric)
    if had is not None and had != lim:
        raise GoalError(f"{lim.metric} is limited twice, differently ({how})")
    stated[lim.metric] = lim


def resolve(objective: Optional[str], optimise: Optional[str], limits: Sequence[str], *,
            ttft_ceiling: Optional[float], ttft_percentile: int, tok_s_floor: Optional[float],
            tpot_ceiling: Optional[float], workload_ttft_ms: float, workload_tpot_ms: Optional[float],
            arrival_rate: Optional[float] = None, strict: bool = False, eval_set: Optional[str] = None,
            max_accuracy_loss: float = DEFAULT_MAX_LOSS) -> Resolved:
    """The objective and constraints of a goal stated with --optimise or --limit.

    With --optimise, the older --ttft-ceiling, --tok-s-floor and --tpot-ceiling state limits of that goal, and a goal a
    preset says is that preset. With --objective (balanced when neither is given) the older flags keep the meaning
    they have always had for that preset, and each --limit becomes the preset's own setting where it has one (_added).

    The workload's TPOT ceiling applies to every goal unless a limit or --tpot-ceiling replaces it. An accuracy_loss
    limit is the accuracy gate's tolerance, --max-accuracy-loss with --eval-set, and no part of the goal: a --limit
    that holds only it leaves the preset and the older flags exactly as without it."""
    if optimise is not None and objective is not None:
        raise GoalError("--optimise states the goal; leave out --objective, which names a preset one")
    stated: Dict[str, Limit] = {}
    for text in limits:
        _same(stated, parse_limit(text), "two --limit options")
    accuracy = stated.pop(ACCURACY, None)
    if accuracy is not None:
        if not eval_set:
            raise GoalError(f"{ACCURACY} is certified on labelled items: add --eval-set gsm8k (or a JSONL file of "
                            "them)")
        if max_accuracy_loss not in (DEFAULT_MAX_LOSS, accuracy.bound):
            raise GoalError(f"{ACCURACY} is limited twice, differently (--limit and --max-accuracy-loss)")
        max_accuracy_loss = accuracy.bound

    def finish(r: Resolved) -> Resolved:
        graded = (f", with answers graded on {eval_set} to lose at most {max_accuracy_loss:g} points"
                  if accuracy is not None else "")
        return dataclasses.replace(r, line=r.line + graded, max_accuracy_loss=max_accuracy_loss)

    if ttft_percentile < 95 and "ttft_p95_ms" in stated:
        raise GoalError("--ttft-percentile 50 judges the median, and the limit names the 95th percentile: limit "
                        "ttft_p50_ms, or leave out --ttft-percentile")
    tpot = stated.pop("tpot_ms", None)
    if tpot is not None and tpot_ceiling is not None and tpot.bound != tpot_ceiling:
        raise GoalError("tpot_ms is limited twice, differently (--limit and --tpot-ceiling)")
    tpot_ms = tpot.bound if tpot is not None else (tpot_ceiling if tpot_ceiling is not None else workload_tpot_ms)
    from_workload = tpot is None and tpot_ceiling is None
    base = dict(ttft_ceiling_ms=workload_ttft_ms, ttft_percentile=ttft_percentile, tpot_ceiling_ms=tpot_ms,
                arrival_rate=arrival_rate, strict=strict)
    if optimise is None:
        return finish(_added(objective or "balanced", stated, base, ttft_ceiling, tok_s_floor, from_workload))
    if ttft_ceiling is not None:
        _same(stated, Limit(ttft_metric(ttft_percentile), "<=", ttft_ceiling), "--limit and --ttft-ceiling")
    if tok_s_floor is not None:
        _same(stated, Limit("tok_s", ">=", tok_s_floor), "--limit and --tok-s-floor")
    goal = Goal(check_optimise(optimise), tuple(stated[k] for k in sorted(stated)))
    preset_name = _preset(goal)
    pct = ttft_percentile
    if preset_name == "balanced":
        ceiling = goal.limits[0]
        pct = 95 if ceiling.metric == "ttft_p95_ms" else 50
        base.update(ttft_ceiling_ms=ceiling.bound, ttft_percentile=pct)
    elif preset_name in ("latency", "efficiency"):
        floor = goal.limits[0]
        base["tok_s_floor_frac" if floor.of_best else "tok_s_floor_abs"] = floor.bound
    elif preset_name is None:
        base.update(goal=format_goal(goal))
    said = f"the {preset_name} preset" if preset_name is not None else "a custom goal, cached under its own name"
    return finish(Resolved(preset_name or CUSTOM, Constraints(**base), pct,
                           _line(goal, tpot_ms, from_workload, preset_name, said)))


def _added(preset: str, stated: Dict[str, Limit], base: Dict[str, Any], ttft_ceiling: Optional[float],
           tok_s_floor: Optional[float], from_workload: bool) -> Resolved:
    """--limit added to a preset (--objective, or balanced). The older flags mean what they always meant for it;
    each limit becomes the preset's own setting where it has one: a TTFT ceiling under balanced, and under
    efficiency at an arrival rate, which judges latency against it; a throughput floor under latency and efficiency;
    the TPOT ceiling under all four. A named preset never turns into another. A limit it has no setting for makes
    the goal custom, said in one line, unless that goal is another preset's (refused: ask for that preset) or an
    arrival rate is given (refused: the rate scores efficiency's joules per request)."""
    rate = base["arrival_rate"]
    cons = Constraints(ttft_ceiling_ms=ttft_ceiling if ttft_ceiling is not None else base["ttft_ceiling_ms"],
                       tok_s_floor_abs=tok_s_floor, tpot_ceiling_ms=base["tpot_ceiling_ms"],
                       ttft_percentile=base["ttft_percentile"], arrival_rate=rate, strict=base["strict"])
    at_rate = preset == "efficiency" and rate is not None
    folded_ttft = False
    left: List[Limit] = []  # what the preset has no setting for
    for lim in (stated[k] for k in sorted(stated)):
        if lim.metric in ("ttft_p95_ms", "ttft_p50_ms") and (preset == "balanced" or at_rate):
            if ttft_ceiling is not None and lim.metric == ttft_metric(base["ttft_percentile"]):
                if lim.bound != ttft_ceiling:
                    raise GoalError(f"{lim.metric} is limited twice, differently (--limit and --ttft-ceiling)")
            elif ttft_ceiling is None and not folded_ttft:
                cons = dataclasses.replace(cons, ttft_ceiling_ms=lim.bound,
                                           ttft_percentile=95 if lim.metric == "ttft_p95_ms" else 50)
                folded_ttft = True
            else:
                left.append(lim)
        elif lim.metric == "tok_s" and preset in ("latency", "efficiency") and not at_rate:
            if tok_s_floor is not None and (lim.of_best or lim.bound != tok_s_floor):
                raise GoalError("tok_s is limited twice, differently (--limit and --tok-s-floor)")
            cons = dataclasses.replace(cons, tok_s_floor_abs=None if lim.of_best else lim.bound,
                                       tok_s_floor_frac=lim.bound if lim.of_best else cons.tok_s_floor_frac)
        else:
            left.append(lim)
    goal = resolve_goal(preset, cons)
    if not left:
        said = f"the {preset} preset"
        return Resolved(preset, cons, cons.ttft_percentile, _line(goal, None, from_workload, preset, said))
    texts = ", ".join(format_goal(Goal("tok_s", (lim,))).split("; ", 1)[1] for lim in left)
    if rate is not None:
        raise GoalError(f"--objective {preset} at --arrival-rate has no setting for {texts}: at a rate efficiency "
                        "scores joules per request within its TTFT and TPOT ceilings. Leave the limit out, or "
                        "leave out --arrival-rate to calibrate a custom goal" if at_rate else
                        f"--arrival-rate applies to --objective efficiency only, not {preset}")
    custom = Goal(goal.optimise, tuple(sorted((*(lim for lim in goal.limits if lim.metric != "tpot_ms"), *left),
                                              key=lambda lim: lim.metric)))
    other = _preset(custom)
    if other is not None:
        raise GoalError(f"--objective {preset} has no setting for {texts}; with it the goal ({format_goal(custom)}) "
                        f"is what --objective {other} calibrates: ask for that preset")
    said = (f"a custom goal: --objective {preset} has no setting for {texts}, so it is calibrated as a goal of "
            "its own, cached under its own name")
    return Resolved(CUSTOM, Constraints(**base, goal=format_goal(custom)), base["ttft_percentile"],
                    _line(custom, base["tpot_ceiling_ms"], from_workload, None, said))


def _line(goal: Goal, tpot_ms: Optional[float], from_workload: bool, preset: Optional[str], said: str) -> str:
    """What the run says before it starts: the goal, its TPOT ceiling, and whether a preset says the same."""
    if tpot_ms is not None and all(lim.metric != "tpot_ms" for lim in goal.limits):
        goal = dataclasses.replace(goal, limits=tuple(sorted((*goal.limits, Limit("tpot_ms", "<=", tpot_ms)),
                                                             key=lambda lim: lim.metric)))
    shown = dataclasses.replace(goal, limits=tuple(sorted(goal.limits, key=lambda lim: lim.metric)))
    workload = any(lim.metric == "tpot_ms" for lim in shown.limits) and from_workload
    text = format_goal(shown) + (" (the TPOT ceiling is the workload's)" if workload else "")
    if preset == "efficiency":
        said += " (without energy readings it orders by tok/s)"
    return f"goal: {text} = {said}"


def recorded_constraints(record: Dict[str, Any]) -> Constraints:
    """The constraints a profile recorded (Profile.calibration_constraints); a field it lacks is at its default."""
    names = {f.name for f in dataclasses.fields(Constraints)}
    return Constraints(**{k: v for k, v in record.items() if k in names})


def _preset(goal: Goal) -> Optional[str]:
    """The preset a goal is, or None when no preset expresses it."""
    names: List[str] = [lim.metric for lim in goal.limits]
    if goal.optimise == "tok_s" and not names:
        return "throughput"
    if goal.optimise == "tok_s" and names in (["ttft_p95_ms"], ["ttft_p50_ms"]):
        return "balanced"
    if goal.optimise == "request_latency_ms" and names == ["tok_s"]:
        return "latency"
    if goal.optimise == "joules_per_token" and names == ["tok_s"]:
        return "efficiency"
    return None
