"""Re-score recorded calibrations by joules per request at an arrival rate, under assumed idle and wind-down.

Recorded profiles keep every measured level's power, duration, requests and tokens, which is all the rate model
needs (rigtune/calibrate/rate.py) except the card's idle draw and its wind-down after a burst. No recorded
calibration measured those, so they are assumptions here, stated in every row; a trial that did record them keeps
its own. Only trials measured at every one of the workload's levels are scored: a trial that stopped at its
busiest level says nothing about a quiet server. For each rate the table names the configuration plain J/token
picks and the one J/request picks, and whether they differ. This answers, before any GPU time, whether the rate
can move a pick at all on these cards; only a calibration that measures idle can say by how much.

    PYTHONUTF8=1 python benchmarks/rescore_rate.py \
        benchmarks/strategies/results-llama/profiles/4090/balanced-file-dolly-d03d5896-levels-all-ttft-p95.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from rigtune.calibrate.objectives import Constraints, pick, rank, shared_request_size
from rigtune.calibrate.rate import lowest_level
from rigtune.models import TrialResult


def _trials(d: Dict[str, object], idle_w: float, hold_j: float, hold_s: float):
    """(trials measured at every level, with assumptions filled in where nothing was recorded; skipped count)."""
    spec = d.get("workload_spec") or {}
    want = {str(c) for c in spec.get("concurrencies") or []}
    out, skipped = [], 0
    for row in d.get("calibration_table") or []:
        t = TrialResult.model_validate(row)
        if not t.ok:
            continue
        if want and not want <= {k for k, m in t.metrics.by_concurrency.items() if m.ok}:
            skipped += 1
            continue
        if t.metrics.idle_power_w is None:
            t.metrics = t.metrics.model_copy(update={"idle_power_w": idle_w, "hold_joules": hold_j, "hold_s": hold_s})
        out.append(t)
    return out, skipped


def _key(t: TrialResult) -> str:
    return t.config.key()


def rescore(path: Path, rates: Sequence[float], idle_w: float, hold_j: float,
            hold_s: float) -> Optional[Dict[str, object]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if "calibration_table" not in d:
        return None
    spec = d.get("workload_spec") or {}
    trials, skipped = _trials(d, idle_w, hold_j, hold_s)
    gpus = (d.get("hardware") or {}).get("gpus") or []
    out: Dict[str, object] = {"file": path.name, "gpu": str(gpus[0]["name"]).replace("NVIDIA ", "") if gpus else "CPU",
                              "trials": len(trials), "skipped": skipped, "rows": []}
    if len(trials) < 2:
        return out
    base = dict(ttft_ceiling_ms=float(spec.get("ttft_ceiling_ms") or 500.0), tpot_ceiling_ms=spec.get("tpot_ceiling_ms"))
    per_token, _ = pick(trials, "efficiency", Constraints(**base))
    size = shared_request_size(trials)
    out["size"] = size
    for rate in rates:
        cons = Constraints(arrival_rate=rate, **base)
        ranked = rank(trials, "efficiency", cons)
        top = ranked[0]
        theirs = next(x for x in ranked if x.result is per_token)  # the J/token pick, scored the same way
        c = top.rate
        out["rows"].append({"rate": rate, "per_token": _key(per_token), "per_request": _key(top.result),
                            "changed": _key(top.result) != _key(per_token), "feasible": top.feasible,
                            "total": c.total_j, "idle": c.idle_j, "active": c.active_j, "hold": c.hold_j,
                            "in_flight": c.in_flight, "judged_at": "/".join(str(m.concurrency) for m in c.latency_levels),
                            "per_token_total": theirs.score, "per_token_feasible": theirs.feasible,
                            "tokens": _answer_tokens(top.result), "per_token_tokens": _answer_tokens(per_token)})
    return out


def _answer_tokens(t: TrialResult) -> Optional[float]:
    """The trial's own mean answer length at its lowest level: what J/request would have counted it in."""
    low = lowest_level(t.metrics)
    return low.output_tokens / (low.requests - low.failed) if low and low.requests > low.failed else None


def table(results: Sequence[Dict[str, object]], idle_w: float, hold_j: float, hold_s: float) -> str:
    lines = [f"Assumed where not recorded: idle {idle_w:g} W, wind-down {hold_j:g} J over {hold_s:g} s.", "",
             "| GPU | profile | req/s | J/token pick (its tokens/request) | J/request pick (its tokens/request) "
             "| its J/request: idle + serving + wind-down | in flight, latency judged at | J/token pick's J/request "
             "| pick changed? |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for row in r["rows"]:
            lines.append(
                f"| {r['gpu']} | {r['file']} | {row['rate']:g} | {row['per_token']} ({_n(row['per_token_tokens'])}) | "
                f"{row['per_request']} ({_n(row['tokens'])}) | "
                f"{row['total']:.0f} = {row['idle']:.0f} + {row['active']:.0f} + {row['hold']:.0f}"
                f"{'' if row['feasible'] else ' (infeasible)'} | {row['in_flight']:.2f}, {row['judged_at']} users | "
                f"{row['per_token_total']:.0f}{'' if row['per_token_feasible'] else ' (infeasible at this rate)'} | "
                f"{'yes' if row['changed'] else 'no'} |")
    lines.append("")
    for r in results:
        size = r.get("size")
        counted = (f"; counted in requests of {size[0]:.0f} tokens (median at {size[1]} users over {size[2]} trials)"
                   if size else "")
        lines.append(f"{r['file']}: {r['trials']} trials scored, {r['skipped']} skipped for missing levels{counted}")
    return "\n".join(lines)


def _n(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.0f}"


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("profiles", nargs="+", type=Path, help="profile JSON files, or folders of them")
    ap.add_argument("--rates", nargs="+", type=float, default=[0.05, 0.3, 1.0], help="arrival rates, requests/s")
    ap.add_argument("--idle-w", type=float, default=60.0, help="assumed idle power, W (A100 prediction: 50-80)")
    ap.add_argument("--hold-j", type=float, default=30.0, help="assumed wind-down energy above idle, J")
    ap.add_argument("--hold-s", type=float, default=1.0, help="assumed wind-down duration, s")
    args = ap.parse_args(argv)
    files = sorted({f for p in args.profiles for f in ([p] if p.is_file() else Path(p).glob("*.json"))})
    results = [r for r in (rescore(f, args.rates, args.idle_w, args.hold_j, args.hold_s) for f in files) if r]
    print(table(results, args.idle_w, args.hold_j, args.hold_s))


if __name__ == "__main__":
    main()
