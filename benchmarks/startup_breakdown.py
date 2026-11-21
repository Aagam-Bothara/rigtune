"""Where calibration time goes: engine start-up against measuring, from trial ledgers and engine logs (no GPU).

    python benchmarks/startup_breakdown.py ledger <rigtune-home>/ledger [...]   # each calibration's time, split
    python benchmarks/startup_breakdown.py logs <rigtune-home>/logs [...]       # each start-up, split into phases

`ledger` reads every *.jsonl under the paths. For each calibration it splits the total (the profile's
calibration_seconds) into the launched trials' start-up (launch to the first healthy answer, with the parts the
engine's log timed: Triton warm-up, torch.compile, CUDA-graph capture), their measuring (the load levels' wall time),
the rest of each trial (warm-up requests, telemetry, teardown) and the time outside trials. A ledger written before
a part was read from the log leaves that part inside "the rest".

`logs` reads every *.log under the paths whose name starts with its launch time (rigtune names each engine log
<unix seconds>_<config>.log) and splits each start-up at lines the engine prints, by their timestamps: a phase ends
at its marker, and one whose marker is missing merges into the next. It prints the median of each phase per folder
and engine. vLLM 0.33 and SGLang 0.5.23 markers; benchmarks/sessions/2026-11/startup-breakdown has the results.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# --- ledger --------------------------------------------------------------------------------------------------------


def _card(trial: dict) -> str:
    card = trial.get("card")
    if isinstance(card, dict):
        return ", ".join(g.get("name", "?") for g in card.get("gpus", [])) or "?"
    return str(card)


def calibration_split(records: List[dict]) -> Optional[Dict[str, float]]:
    """One calibration's seconds by part, or None when it launched nothing."""
    cal = next((r for r in records if r["type"] == "calibration"), None)
    launched = [r for r in records if r["type"] == "trial" and r["outcome"]["launched"] and r["duration_s"]]
    if not launched:
        return None
    starts = [r.get("startup") or {} for r in launched]
    wall = sum(r["duration_s"] for r in launched)
    split = {
        "start-up": sum(s.get("seconds") or 0 for s in starts),
        "  Triton warm-up": sum(s.get("warmup_s") or 0 for s in starts),
        "  torch.compile": sum(s.get("compile_s") or 0 for s in starts),
        "  CUDA-graph capture": sum(s.get("capture_s") or 0 for s in starts),
        "measuring": sum(r["trial"]["metrics"]["duration_s"] or 0 for r in launched),
    }
    split["  the rest of start-up"] = split["start-up"] - sum(split[k] for k in split if k.startswith("  "))
    split["rest of each trial"] = wall - split["start-up"] - split["measuring"]
    total = cal["profile"]["calibration_seconds"] if cal else wall
    split["outside trials"] = total - wall
    split["total"], split["launches"] = total, len(launched)
    return split


def report_ledgers(paths: List[Path]) -> None:
    order = ["start-up", "  Triton warm-up", "  torch.compile", "  CUDA-graph capture", "  the rest of start-up",
             "measuring", "rest of each trial", "outside trials"]
    for path in sorted(p for root in paths for p in ([root] if root.is_file() else root.rglob("*.jsonl"))):
        by_cal: Dict[str, List[dict]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                by_cal[record["calibration_id"]].append(record)
        for cid, records in by_cal.items():
            split = calibration_split(records)
            if split is None:
                continue
            first = next(r for r in records if r["type"] == "trial")
            engines = sorted({f"{r['engine']['name']} {r['engine']['version']}" for r in records if r["type"] == "trial"})
            print(f"\n=== {first['model']['id']} on {_card(first)}, {', '.join(engines)} (calibration {cid[:8]}): "
                  f"{split['total']:.0f} s, {split['launches']:.0f} launches")
            for part in order:
                print(f"    {part:<26}{split[part]:7.0f} s  {100 * split[part] / split['total']:5.1f}%")


# --- logs ----------------------------------------------------------------------------------------------------------

_VLLM_TS = [
    (re.compile(r"^[WIE](\d\d)(\d\d) (\d\d):(\d\d):(\d\d)\.\d+"), True),  # torch's own warnings: no year
    (re.compile(r"(?:INFO|WARNING|ERROR|DEBUG) (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) "), True),  # vLLM: no year
    (re.compile(r"^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d),\d+ - "), False),  # FlashInfer
]
_SGLANG_TS = [(re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)"), False)]

# (phase, the line that ends it, whether its first or last match counts)
VLLM_PHASES = [
    ("python + API server imports", r"non-default args", "first"),
    ("engine process spawn + imports", r"Initializing a V1 LLM engine", "first"),
    ("distributed + model init", r"Loading model from scratch|Starting to load model", "first"),
    ("weights load", r"Model loading took", "first"),
    ("KV layout + encoder profiling", r"Starting torch\.compile|\[backbone\] Using cache directory", "first"),
    ("torch.compile", r"torch\.compile took|Directly load the compiled graph", "first"),
    ("profiling/warm-up run (Triton)", r"Initial profiling/warmup run took", "first"),
    ("CUDA graphs, memory-sizing pass", r"Available KV cache memory", "first"),
    ("kernel warm-up + CUDA graph capture", r"init engine \(profile, create kv cache, warmup model\) took", "first"),
    ("API server up", r"Starting vLLM server on|Application startup complete", "first"),
]
SGLANG_PHASES = [
    ("python + imports, args", r"server_args=", "first"),
    ("tokenizer/template + scheduler spawn", r"Init parallel begin|Init torch distributed begin", "first"),
    ("distributed + model init", r"Load weight begin", "first"),
    ("weights load", r"Load weight end", "last"),
    ("KV/mamba pool allocation", r"Memory pool end", "last"),
    ("CUDA graph capture (all sets)", r"CUDA graph end", "last"),
    ("scheduler finish", r"Engine startup timings|max_total_num_tokens=", "last"),
    ("HTTP server up", r"Uvicorn running on|Application startup complete", "first"),
    ("warm-up request, ready", r"The server is fired up and ready to roll", "first"),
]
TOTAL = "launch to the ready line"


def _stamp(line: str, patterns, year: int) -> Optional[float]:
    for rx, yearless in patterns:
        m = rx.search(line)
        if m:
            parts = [year, *map(int, m.groups())] if yearless else list(map(int, m.groups()))
            return datetime(*parts, tzinfo=timezone.utc).timestamp()
    return None


def start_up_phases(path: Path) -> Optional[Dict[str, float]]:
    """Seconds per phase of one engine log, or None when its name holds no launch time or no marker was found."""
    m = re.search(r"(?:^|__)(\d{10})_", path.name)
    if not m:
        return None
    t0 = int(m.group(1))
    sglang = "_sglang_" in path.name
    phases, patterns = (SGLANG_PHASES, _SGLANG_TS) if sglang else (VLLM_PHASES, _VLLM_TS)
    year = datetime.fromtimestamp(t0, timezone.utc).year
    found: Dict[str, float] = {}
    now = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        now = _stamp(line, patterns, year) or now
        if now is None:
            continue
        for name, rx, which in phases:
            if (which == "last" or name not in found) and re.search(rx, line):
                found[name] = now
    out: Dict[str, float] = {}
    prev = t0
    for name, _, _ in phases:
        if name in found and found[name] >= prev:
            out[name], prev = found[name] - prev, found[name]
    if not out:
        return None
    out[TOTAL] = prev - t0
    return out


def report_logs(paths: List[Path]) -> None:
    groups: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
    for root in paths:
        for path in sorted(root.rglob("*.log")):
            phases = start_up_phases(path)
            if phases:
                groups[(str(path.parent.relative_to(root.parent)), "sglang" if "_sglang_" in path.name else "vllm")]\
                    .append(phases)
    for (folder, engine), runs in sorted(groups.items()):
        names = [n for n, _, _ in (SGLANG_PHASES if engine == "sglang" else VLLM_PHASES)] + [TOTAL]
        print(f"\n=== {folder}, {engine}: {len(runs)} start-ups, median seconds [min-max]")
        for name in names:
            vals = [r[name] for r in runs if name in r]
            if vals:
                print(f"    {name:<38}{statistics.median(vals):6.0f}   [{min(vals):.0f}-{max(vals):.0f}]  n={len(vals)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["ledger", "logs"])
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    (report_ledgers if args.what == "ledger" else report_logs)(args.paths)


if __name__ == "__main__":
    main()
