"""What a set of runs says: accuracy against GPU time per method, paired against direct answers, with uncertainty.

- Per run: accuracy (with a Wilson interval) over every problem, failures counted wrong; GPU-seconds per problem (the
  batch's wall time on one dedicated GPU); tokens; answers cut off at max_tokens; failed requests; how full the
  sequence limit stayed (below UTILISATION_FLOOR, wall time measures the slowest requests rather than GPU time, and
  the row says so); prompt tokens served from the prefix cache; energy when read.
- Paired against direct answers: the difference in accuracy with a paired bootstrap interval, and McNemar's exact test.
- Self-correction's four transitions from its first answer to its final one.
- Best of the answers, judged by the reference, for self-consistency and adaptive: an upper bound on any way of
  choosing among them, never a method.
- The plan's criterion (plan-reasoning-harness.md), decided from the data: self-correction beats its own first answer,
  and beats self-consistency at EQUAL GPU time per problem, each with the paired 95% interval above zero. Equal time
  is read off the self-consistency curve, direct answers as its k = 1: self-correction is compared with the mix of the
  two runs whose cost brackets its own, weighted to its cost, problem by problem (and, beside it, with the dearer of
  the two alone, since a straight line under a concave curve flatters self-correction a little). Undecided when no
  runs bracket it, when any compared run had failed requests or kept the GPU less than UTILISATION_FLOOR full, or
  when the runs answered other problems, ran other settings, other sequence limits or another server.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from rigtune.accuracy import wilson
from rigtune.calibrate.paired import mcnemar_p
from rigtune.reasoning.problems import grade

BOOTSTRAP = 2000
UTILISATION_FLOOR = 0.8


@dataclass
class Row:
    name: str
    n: int
    accuracy: float
    interval: Tuple[float, float]
    gpu_seconds: float  # per problem
    completion_tokens: float  # per problem
    truncated: int  # answers cut off at max_tokens
    errors: int
    utilisation: Optional[float]
    cached_share: Optional[float]  # of prompt tokens
    joules: Optional[float]  # per problem
    preemptions: Optional[float] = None  # requests the engine preempted and recomputed during the run


def _correct(run: Dict[str, Any]) -> Dict[str, float]:
    return {a["problem_id"]: float(bool(a["correct"])) for a in run["attempts"]}


def row(run: Dict[str, Any]) -> Row:
    attempts = run["attempts"]
    n = len(attempts)
    right = sum(1 for a in attempts if a["correct"])
    calls = [c for a in attempts for c in a["calls"]]
    prompt = sum(c["prompt_tokens"] for c in calls)
    joules = run.get("energy_joules")
    return Row(name=run["name"], n=n, accuracy=right / n if n else 0.0, interval=wilson(right, n),
               gpu_seconds=run["wall_seconds"] / n if n else 0.0,
               completion_tokens=sum(c["completion_tokens"] for c in calls) / n if n else 0.0,
               truncated=sum(1 for c in calls if c.get("finish_reason") == "length"),
               errors=sum(1 for a in attempts if a.get("error")), utilisation=run.get("utilisation"),
               cached_share=sum(c.get("cached_tokens", 0) for c in calls) / prompt if prompt else None,
               joules=joules / n if joules is not None and n else None, preemptions=run.get("preemptions"))


def _bootstrap(diffs: Sequence[float], seed: int) -> Tuple[float, float]:
    rng = random.Random(seed)
    m = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(BOOTSTRAP))
    return boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP) - 1]


def paired(a: Dict[str, float], b: Dict[str, float], seed: int = 0) -> Dict[str, Any]:
    """b against a on the problems both answered: difference in accuracy (b - a) with a 95% paired bootstrap interval,
    and McNemar's exact p on the discordant problems (for 0/1 outcomes)."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    diffs = [b[i] - a[i] for i in ids]
    lost = sum(1 for d in diffs if d < 0)
    gained = sum(1 for d in diffs if d > 0)
    return {"n": len(ids), "difference": sum(diffs) / len(ids), "interval": _bootstrap(diffs, seed),
            "gained": gained, "lost": lost, "mcnemar_p": mcnemar_p(lost, gained)}


def transitions(run: Dict[str, Any]) -> Dict[str, int]:
    """Self-correction's first answer to its final one."""
    out = {"wrong->right": 0, "right->wrong": 0, "wrong->wrong": 0, "right->right": 0}
    for a in run["attempts"]:
        first = "right" if a["first_correct"] else "wrong"
        final = "right" if a["correct"] else "wrong"
        out[f"{first}->{final}"] += 1
    return out


def best_of(run: Dict[str, Any], references: Dict[str, str]) -> float:
    """Share of problems where any of the run's answers is right, judged by the reference (an upper bound)."""
    attempts = run["attempts"]
    hit = sum(1 for a in attempts if any(x is not None and grade("\\boxed{" + x + "}", references[a["problem_id"]])[0]
                                         for x in a["answers"]))
    return hit / len(attempts) if attempts else 0.0


def _comparable(runs: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Why the runs cannot be compared, or None: other problems, or other settings than their method's."""
    ids = {frozenset(a["problem_id"] for a in r["attempts"]) for r in runs}
    if len(ids) > 1:
        return "the runs answered different problems"
    sets = {r.get("problem_set") for r in runs}
    if len(sets) > 1:
        return "the runs came from different problem sets"
    if len({r.get("sequences") for r in runs}) > 1:
        return "the runs ran at different sequence limits"
    if len({r.get("server") for r in runs}) > 1:
        return "the runs were served differently"
    keys = {tuple(sorted((k, str(v)) for k, v in (r.get("settings") or {}).items() if k != "keep_texts"))
            for r in runs}
    if len(keys) > 1:
        return "the runs used different model settings"
    return None


def criterion(runs: Sequence[Dict[str, Any]], seed: int = 0) -> Dict[str, Any]:
    """The plan's success criterion from the runs given: one self-correction run, direct answers, and
    self-consistency runs."""
    sc = next((r for r in runs if r["method"] == "self-correction"), None)
    if sc is None:
        return {"decided": False, "why": "no self-correction run"}
    curve = sorted((r for r in runs if r["method"] in ("direct", "self-consistency")), key=lambda r: row(r).gpu_seconds)
    why = _comparable([sc] + curve)
    if why:
        return {"decided": False, "why": why}
    failed = [r["name"] for r in [sc] + curve if row(r).errors]
    if failed:
        return {"decided": False, "why": f"failed requests in {', '.join(failed)}"}
    idle = [r["name"] for r in [sc] + curve if (r.get("utilisation") or 0) < UTILISATION_FLOOR]
    if idle:  # their wall time is the slowest requests, not GPU time
        return {"decided": False, "why": f"utilisation below {UTILISATION_FLOOR:.0%} in {', '.join(idle)}"}
    final = _correct(sc)
    first = {a["problem_id"]: float(bool(a["first_correct"])) for a in sc["attempts"]}
    vs_first = paired(first, final, seed)
    cost = row(sc).gpu_seconds
    below = [r for r in curve if row(r).gpu_seconds <= cost]
    above = [r for r in curve if row(r).gpu_seconds >= cost]
    if not below or not above:
        return {"decided": False, "why": "no self-consistency runs bracket self-correction's GPU time",
                "vs_first": vs_first}
    lo, hi = below[-1], above[0]
    span = row(hi).gpu_seconds - row(lo).gpu_seconds
    w = 1.0 if span == 0 else (row(hi).gpu_seconds - cost) / span  # the weight on the cheaper run
    a, b = _correct(lo), _correct(hi)
    rival = {i: w * a[i] + (1 - w) * b[i] for i in final}
    vs_rival = paired(rival, final, seed)
    vs_rival.pop("mcnemar_p", None)  # not defined for fractional differences
    # A straight line between the brackets lies below a concave accuracy curve, which flatters self-correction a
    # little; the comparison with the dearer bracket alone, which costs at least as much, is reported beside it.
    vs_dearer = paired(b, final, seed)
    met = vs_first["interval"][0] > 0 and vs_rival["interval"][0] > 0
    return {"decided": True, "met": met, "vs_first": vs_first, "rival": f"{lo['name']} and {hi['name']}",
            "rival_weight": w, "vs_rival": vs_rival, "dearer": hi["name"], "vs_dearer": vs_dearer}


def markdown(runs: Sequence[Dict[str, Any]], references: Dict[str, str], title: str = "Reasoning harness") -> str:
    """The report as Markdown: the table, the pairs against direct, the transitions, the bounds and the criterion."""
    rows = sorted((row(r) for r in runs), key=lambda x: x.gpu_seconds)
    lines = [f"# {title}", "",
             "| Method | Problems | Accuracy | 95% interval | GPU-s / problem | Tokens / problem | Cut off | Failed | "
             "Utilisation | Cached prompt | J / problem | Preempted |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        util = "—" if x.utilisation is None else f"{x.utilisation:.0%}" + (
            " (low: wall time, not GPU time)" if x.utilisation < UTILISATION_FLOOR else "")
        cached = "—" if x.cached_share is None else f"{x.cached_share:.0%}"
        joules = f"{x.joules:.0f}" if x.joules is not None else "—"
        pre = f"{x.preemptions:.0f}" if x.preemptions is not None else "—"
        lines.append(f"| {x.name} | {x.n} | {x.accuracy:.1%} | {x.interval[0]:.1%}–{x.interval[1]:.1%} | "
                     f"{x.gpu_seconds:.2f} | {x.completion_tokens:.0f} | {x.truncated} | {x.errors} | {util} | "
                     f"{cached} | {joules} | {pre} |")
    base = next((r for r in runs if r["method"] == "direct"), None)
    if base is not None:
        lines += ["", "Against direct answers, paired (descriptive; the criterion below is the test):", ""]
        for r in runs:
            if r is base:
                continue
            p = paired(_correct(base), _correct(r))
            if p.get("n"):
                lines.append(f"- {r['name']}: {p['difference']:+.1%} ({p['interval'][0]:+.1%} to "
                             f"{p['interval'][1]:+.1%}), {p['gained']} gained, {p['lost']} lost, "
                             f"McNemar p = {p['mcnemar_p']:.3g}")
    for r in runs:
        if r["method"] == "self-correction":
            t = transitions(r)
            lines += ["", f"Self-correction, first answer to final: {t['wrong->right']} repaired, "
                          f"{t['right->wrong']} harmed, {t['wrong->wrong']} still wrong, {t['right->right']} kept."]
        if r["method"] in ("self-consistency", "adaptive"):
            lines.append(f"\n{r['name']}: best of its answers by the reference {best_of(r, references):.1%} "
                         "(an upper bound, not a method)")
    c = criterion(runs)
    if c["decided"]:
        verdict = "met" if c["met"] else "not met"
        lines += ["", f"Criterion: {verdict}. Final against first answer {c['vs_first']['difference']:+.1%} "
                      f"({c['vs_first']['interval'][0]:+.1%} to {c['vs_first']['interval'][1]:+.1%}); against "
                      f"{c['rival']} at equal GPU time ({c['rival_weight']:.0%} on the cheaper) "
                      f"{c['vs_rival']['difference']:+.1%} ({c['vs_rival']['interval'][0]:+.1%} to "
                      f"{c['vs_rival']['interval'][1]:+.1%}); against {c['dearer']} alone, which costs at least as "
                      f"much, {c['vs_dearer']['difference']:+.1%} ({c['vs_dearer']['interval'][0]:+.1%} to "
                      f"{c['vs_dearer']['interval'][1]:+.1%})."]
    else:
        lines += ["", f"Criterion not decided: {c['why']}."]
    return "\n".join(lines) + "\n"
