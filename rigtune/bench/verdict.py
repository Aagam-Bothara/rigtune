"""Score a comparison as a win, tie or loss against every row (`rigtune verdict`, `compare --verdict`).

The rule extends the one written down before the 14 November 2026 head-to-head. For each row against
rigtune's:

- Only runs that succeeded and met the comparison's limits count (each run's RunRecord says, under
  `compare --verdict`). A row stands for the median of those runs, its range is their lowest to highest, and
  each round pairs rigtune's run with the row's run of the same round.
- **win**: rigtune's median is at least the row's threshold above the row's, so is rigtune's worst paired
  round (the lowest ratio over rounds where both ran), and every rigtune run is above every run of the row.
  Thresholds: +20% over a stock row (`*-default`), +10% over a tuner row (another tool's pick) or an expert
  row (a documentation's setting); all configurable.
- **loss**: the row's median is ahead (by at least the loss margin, default 0), every paired round is ahead,
  and every one of its runs is above every rigtune run. Ranges apart is kept as well as every round behind:
  with three rounds of exchangeable noise, all three rounds fall behind by chance one time in eight, all three
  of one row's runs above all three of the other's one time in twenty.
- **ahead on speed, answers unchecked**: what would be a win on speed, while rigtune's answers are not known
  to be checked and unchanged. A win means faster with the same answers, so speed alone is never called one.
- **provisional**: what would be a win or a loss, but with fewer than 3 such runs on a side, or fewer than 3
  paired rounds; one that would be a win on speed with the answers unchecked says it would be ahead on speed,
  answers unchecked.
- **behind**: not a loss, but rigtune's median and most paired rounds are behind by more than the row's own
  spread (its highest counted run minus its lowest), with 3 runs on each side and 3 paired rounds; one high
  rigtune run can keep the ranges overlapping without making the deficit noise.
- **behind, within noise**: rigtune's median is behind and it is neither of those. **tie**: anything else.
- **not applicable**: the row could not start or failed, never met the limits, measured nothing, or was not
  run (an added row skipped with a reason, such as a tool that produced no pick). Never a win. If rigtune's
  own row met the limits in no run, a row that met them in 3 or more is a loss.

Results written before RunRecords existed hold each run's tok/s but not its limit status: a successful run is
then counted as meeting the limits when the row's median run did, and the verdict says so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rigtune.bench.compare import ComparisonResult, ComparisonRow, read_runs

WIN, TIE, BEHIND, LOSS, PROVISIONAL, NA = ("win", "tie", "behind, within noise", "loss", "provisional",
                                           "not applicable")
SPEED_ONLY = "ahead on speed, answers unchecked"
BEHIND_CLEAR = "behind"
INCOMPLETE = "incomplete"  # overall only: a comparison read from a runs file that stopped before it finished
MIN_RUNS = 3


@dataclass
class Thresholds:
    """Fractions: 0.20 is +20%."""

    stock: float = 0.20
    tuner: float = 0.10
    expert: float = 0.10
    loss_margin: float = 0.0

    def for_kind(self, kind: str) -> float:
        return {"stock": self.stock, "tuner": self.tuner, "expert": self.expert}[kind]


@dataclass
class RowVerdict:
    label: str
    kind: str
    config_key: Optional[str]
    threshold: float
    outcome: str
    reason: str
    median_tok_s: Optional[float] = None
    runs_tok_s: List[Optional[float]] = field(default_factory=list)  # every run, None where it did not count
    range_tok_s: Optional[Tuple[float, float]] = None
    gain: Optional[float] = None  # rigtune's median / this row's, minus 1
    worst_paired_gain: Optional[float] = None  # the lowest rigtune / row ratio over paired rounds, minus 1
    overlap: Optional[bool] = None
    counted_runs: int = 0  # runs that succeeded and met the limits
    lean: Optional[str] = None  # what a provisional row would be with enough runs


def kind_of(row: ComparisonRow) -> str:
    if row.kind:
        return row.kind
    return "stock" if row.label.endswith("-default") else "tuner"


def counted(row: ComparisonRow) -> Dict[int, float]:
    """{round: tok/s} of the runs that succeeded and met the limits. Without RunRecords, every successful run
    counts when the row's median run met the limits (results written before they existed)."""
    if row.run_records:
        return {r.round: r.scored_tok_s for r in row.run_records
                if r.ok and r.meets_slo and r.scored_tok_s is not None and r.scored_tok_s > 0}
    if not row.ok or not row.meets_slo:
        return {}
    runs = row.runs_tok_s or [row.scored_tok_s]
    return {i: v for i, v in enumerate(runs, 1) if v is not None and v > 0}


def _every_run(row: ComparisonRow) -> List[Optional[float]]:
    if row.run_records:
        return [r.scored_tok_s if r.ok and r.meets_slo else None for r in row.run_records]
    return list(row.runs_tok_s) or [row.scored_tok_s]


def median(values: List[float]) -> float:
    """The middle value, the lower of the two middle ones for an even count: the run compare's combine_runs
    lets a row stand for, so a verdict's median is always a run that was measured."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def _why_not(row: ComparisonRow) -> str:
    records = row.run_records
    if records and not any(r.ok for r in records) or not records and not row.ok:
        error = next((r.error for r in records if r.error), None) if records else row.error
        return f"could not start or failed: {(error or 'no error recorded').splitlines()[0][:120]}"
    if records and any(r.ok and r.meets_slo for r in records) or not records and row.meets_slo:
        return "measured 0 tok/s"
    return "never met the limits at any level"


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def judge(rig: ComparisonRow, row: ComparisonRow, t: Thresholds) -> RowVerdict:
    kind = kind_of(row)
    bar = t.for_kind(kind)
    out = RowVerdict(label=row.label, kind=kind, config_key=row.config_key, threshold=bar, outcome=NA,
                     reason="", runs_tok_s=_every_run(row))
    b, r = counted(row), counted(rig)
    out.counted_runs = len(b)
    if not b:
        out.reason = _why_not(row)
        return out
    mb = median(list(b.values()))
    out.median_tok_s, out.range_tok_s = mb, (min(b.values()), max(b.values()))
    if not r:
        lean = f"rigtune's row met the limits in no run ({_why_not(rig)}); this row met them in {len(b)}"
        if len(b) >= MIN_RUNS:
            out.outcome, out.reason = LOSS, lean
        else:
            out.outcome, out.lean = PROVISIONAL, LOSS
            out.reason = f"would be a loss: {lean}, fewer than {MIN_RUNS}"
        return out
    mr = median(list(r.values()))
    gain = mr / mb - 1
    paired = sorted(r.keys() & b.keys())
    pairs = [r[i] / b[i] for i in paired]
    worst = min(pairs) - 1 if pairs else None
    apart_win, apart_loss = min(r.values()) > max(b.values()), min(b.values()) > max(r.values())
    out.gain, out.worst_paired_gain, out.overlap = gain, worst, not (apart_win or apart_loss)
    worst_text = f", worst paired round {_pct(worst)}" if worst is not None else ", no paired round"
    if gain >= bar and apart_win and worst is not None and worst >= bar:
        would, why = WIN, f"{_pct(gain)}{worst_text}, both at least +{bar * 100:.0f}%, ranges apart"
    elif mb >= mr * (1 + t.loss_margin) and apart_loss and pairs and all(p < 1 for p in pairs):
        would, why = LOSS, f"{_pct(gain)}{worst_text}: this row is ahead in every round and the ranges are " \
                           "apart"
    elif gain < 0:  # ranges apart with rigtune behind puts every round behind: then only the margin is short
        noise = "ranges overlap" if out.overlap else "under the loss margin"
        spread = max(b.values()) - min(b.values())
        clear = sum(r[i] < b[i] - spread for i in paired)
        if len(b) >= MIN_RUNS and len(r) >= MIN_RUNS and len(paired) >= MIN_RUNS and mb - mr > spread \
                and 2 * clear > len(paired):
            would = BEHIND_CLEAR
            why = (f"{_pct(gain)}{worst_text}: behind by more than this row's own spread ({spread:.1f} "
                   f"tok/s) in the median and in {clear} of {len(paired)} rounds; not a loss, {noise}")
        else:
            would, why = BEHIND, f"{_pct(gain)}{worst_text}, {noise}"
    elif out.overlap:
        would, why = TIE, f"{_pct(gain)}{worst_text}, ranges overlap"
    elif worst is None:
        would, why = TIE, f"{_pct(gain)}{worst_text}: no round in which both met the limits"
    else:
        short = "the median" if gain < bar else "the worst paired round"
        would, why = TIE, f"{_pct(gain)}{worst_text}: {short} is under the +{bar * 100:.0f}% bar"
    few = [f"{name} met the limits in {n} run{'s' if n != 1 else ''}"
           for name, n in (("rigtune", len(r)), ("this row", len(b))) if n < MIN_RUNS]
    if len(pairs) < MIN_RUNS and not few:
        few.append(f"only {len(pairs)} round{'s' if len(pairs) != 1 else ''} paired")
    if would in (WIN, LOSS) and few:
        out.outcome, out.lean = PROVISIONAL, would
        out.reason = f"would be a {would}: {why}; but {', '.join(few)}, fewer than {MIN_RUNS}"
    else:
        out.outcome, out.reason = would, why
    missed = sum(v is None for v in out.runs_tok_s)
    if missed:
        out.reason += f"; {missed} of {len(out.runs_tok_s)} runs failed or missed the limits"
    return out


@dataclass
class Verdict:
    results: str
    model_id: str
    gpu: Optional[str]
    workload: str
    calibrated_on: Optional[str]
    objective: str
    ttft_ceiling_ms: float
    tpot_ceiling_ms: Optional[float]
    thresholds: Thresholds
    rigtune: Dict[str, Any]
    rows: List[RowVerdict]
    overall: Dict[str, Any]
    notes: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1) + "\n"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else ("es" if word.endswith("s") else "s"))


def _overall(rows: List[RowVerdict]) -> Dict[str, Any]:
    n = {o: sum(r.outcome == o for r in rows)
         for o in (WIN, SPEED_ONLY, TIE, BEHIND, BEHIND_CLEAR, LOSS, PROVISIONAL, NA)}
    judged = len(rows) - n[NA]
    na = f"; {n[NA]} not applicable" if n[NA] else ""
    tally = f"{_plural(n[WIN], 'win')}, {_plural(n[TIE], 'tie')}"
    if n[SPEED_ONLY]:
        tally += f", {n[SPEED_ONLY]} ahead on speed with answers unchecked"
    ahead = n[WIN] + n[SPEED_ONLY]
    if n[LOSS]:
        outcome, text = LOSS, _plural(n[LOSS], "loss") + na
    elif n[PROVISIONAL]:
        outcome = PROVISIONAL
        text = f"{n[PROVISIONAL]} provisional: too few runs met the limits to call a win or a loss" + na
    elif not judged:
        outcome, text = NA, "no row could be compared"
    elif n[WIN] == judged and n[NA]:
        outcome = "win on the rows that could be compared"
        text = (f"win on the {_plural(n[WIN], 'row')} that could be compared; {n[NA]} not applicable, so "
                "not a win against every row")
    elif n[WIN] == judged:
        outcome, text = WIN, f"win against every row ({n[WIN]} of {len(rows)})"
    elif ahead == judged:  # ahead of every row it could be compared with, but not every answer was checked
        scope = (f"the {_plural(ahead, 'row')} that could be compared" + na if n[NA]
                 else f"every row ({ahead} of {len(rows)})")
        outcome = SPEED_ONLY
        text = (f"ahead on speed on {scope}, with rigtune's answers unchecked against "
                f"{_plural(n[SPEED_ONLY], 'row')}: not a win")
    elif n[BEHIND_CLEAR]:
        within = f", within noise on {n[BEHIND]} more" if n[BEHIND] else ""
        outcome = BEHIND_CLEAR
        text = f"behind on {_plural(n[BEHIND_CLEAR], 'row')} by more than its own spread{within}; {tally}{na}"
    elif n[BEHIND]:
        outcome, text = BEHIND, f"behind on {_plural(n[BEHIND], 'row')}, within noise; {tally}" + na
    else:
        outcome, text = TIE, f"no row ahead of rigtune: {tally}" + na
    return {"outcome": outcome, "text": text, "wins": n[WIN], "ahead_answers_unchecked": n[SPEED_ONLY],
            "ties": n[TIE], "behind": n[BEHIND], "behind_beyond_spread": n[BEHIND_CLEAR], "losses": n[LOSS],
            "provisional": n[PROVISIONAL], "not_applicable": n[NA]}


def answers_unchecked(result: ComparisonResult, row: RowVerdict) -> Optional[str]:
    """Why rigtune's answers are not known to be checked and unchanged against this row, or None when they
    are. A win needs them to be: on speed alone the row is ahead on speed, answers unchecked. These results
    hold nothing about answers, so it is never None."""
    return "rigtune's answers were not checked"


def score(result: ComparisonResult, thresholds: Optional[Thresholds] = None, results: str = "") -> Verdict:
    t = thresholds or Thresholds()
    rig = result.rigtune_row
    if rig is None:
        raise ValueError("nothing to score: the comparison stopped before rigtune's row was measured"
                         if result.incomplete else "the results have no rigtune row")
    rows = [judge(rig, r, t) for r in result.other_rows()]
    for row in rows:
        lean_win = row.outcome == PROVISIONAL and row.lean == WIN
        why = answers_unchecked(result, row) if row.outcome == WIN or lean_win else None
        if why and lean_win:  # too few runs, and even with enough it would be speed alone
            row.lean = SPEED_ONLY
            row.reason = row.reason.replace("would be a win:", f"would be {SPEED_ONLY}, not a win:", 1)
            row.reason += f"; {why}"
        elif why:  # faster, but nothing shows the answers stayed the same: not a win
            row.outcome, row.reason = SPEED_ONLY, f"{row.reason}; {why}"
    for s in result.skipped:
        rows.append(RowVerdict(label=s.label, kind=s.kind, config_key=None, threshold=t.for_kind(s.kind),
                               outcome=NA, reason=f"not run: {s.reason}"))
    notes: List[str] = []
    if not all(r.run_records for r in result.rows):
        notes.append("these results do not record each run's limit status (they were written before it was "
                     "kept, or without --verdict): a successful run counts as meeting the limits when its "
                     "row's median run did")
    for row in result.rows:
        levels = sorted({x.scored_concurrency for x in row.run_records
                         if x.ok and x.meets_slo and x.scored_concurrency is not None})
        if len(levels) > 1:
            listed = ", ".join(map(str, levels))
            notes.append(f"{row.label}'s counted runs were scored at different levels ({listed} users): each "
                         "run counts at the busiest level at which it met the limits")
    notes += result.notes  # the comparison's own: where each added row came from, held-out prompts, ...
    overall = _overall(rows)
    if result.incomplete:  # whatever the rows say, a comparison that did not finish settles nothing
        overall.update(outcome=INCOMPLETE, text=f"incomplete: {result.incomplete} (the rows so far: "
                                                f"{overall['text']})")
    r = counted(rig)
    return Verdict(
        results=results, model_id=result.model_id, gpu=result.gpu, workload=result.workload,
        calibrated_on=result.calibrated_on, objective=result.objective,
        ttft_ceiling_ms=result.ttft_ceiling_ms, tpot_ceiling_ms=result.tpot_ceiling_ms, thresholds=t,
        rigtune={"config_key": rig.config_key, "median_tok_s": median(list(r.values())) if r else None,
                 "scored_concurrency": rig.scored_concurrency, "counted_runs": len(r),
                 "runs_tok_s": _every_run(rig),
                 "range_tok_s": (min(r.values()), max(r.values())) if r else None, "met_limits": bool(r)},
        rows=rows, overall=overall, notes=notes)


def _f(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1f}"


def _range(r: Optional[Tuple[float, float]]) -> str:
    return "-" if r is None else f"{_f(r[0])}-{_f(r[1])}"


def to_markdown(v: Verdict) -> str:
    t, rg = v.thresholds, v.rigtune
    limits = f"TTFT <= {v.ttft_ceiling_ms:.0f} ms"
    if v.tpot_ceiling_ms:
        limits += f", TPOT <= {v.tpot_ceiling_ms:.0f} ms"
    where = f"measured on `{v.workload}`"
    if v.calibrated_on:
        where += f", calibrated on `{v.calibrated_on}`"
    margin = f" by at least {t.loss_margin * 100:.0f}%" if t.loss_margin else ""
    level = f" at {rg['scored_concurrency']} users" if rg["scored_concurrency"] else ""
    failed = "" if rg["met_limits"] else " (**met the limits in no run**)"
    lines = [
        f"## Verdict: {v.model_id} on {v.gpu or 'CPU'}",
        "",
        f"{where}; objective `{v.objective}` ({limits}).",
        "",
        f"Rule: only runs that met the limits count. A win needs rigtune's median and its worst paired round "
        f"at least +{t.stock * 100:.0f}% above a stock row (+{t.tuner * 100:.0f}% above a tuner row, "
        f"+{t.expert * 100:.0f}% above an expert row) and every rigtune run above every run of that row; a "
        f"loss is a row ahead{margin} in its median and in every round, with every one of its runs above "
        f"every rigtune run. Either needs {MIN_RUNS} runs on each side and {MIN_RUNS} paired rounds, or it "
        "is provisional. A win also needs rigtune's answers checked and unchanged; ahead on speed without "
        "that is \"ahead on speed, answers unchecked\", never a win. Rigtune behind otherwise is behind when "
        "its median and most paired rounds trail by more than the row's own spread, else behind within "
        "noise; anything else is a tie. A row that could not start, never met the limits or was not run is "
        "not applicable.",
        "",
        f"rigtune `{rg['config_key']}`: median {_f(rg['median_tok_s'])} tok/s{level}, runs "
        f"{_range(rg['range_tok_s'])} ({rg['counted_runs']} of {len(rg['runs_tok_s'])} met the limits)"
        f"{failed}",
        "",
        "| Row | Kind | Needs | Median tok/s | Runs | Gain | Worst round | Ranges | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in v.rows:
        name = f"{r.label} (`{r.config_key}`)" if r.config_key else r.label
        gain = "-" if r.gain is None else _pct(r.gain)
        worst = "-" if r.worst_paired_gain is None else _pct(r.worst_paired_gain)
        ranges = "-" if r.overlap is None else ("overlap" if r.overlap else "apart")
        runs = f"{_range(r.range_tok_s)} ({r.counted_runs} of {len(r.runs_tok_s)})"
        lines.append(f"| {name} | {r.kind} | +{r.threshold * 100:.0f}% | {_f(r.median_tok_s)} | {runs} | "
                     f"{gain} | {worst} | {ranges} | **{r.outcome}**: {r.reason} |")
    lines += ["", f"**Overall: {v.overall['text']}**"] + [f"_{n}_" for n in v.notes]
    return "\n".join(lines) + "\n"


def verdict_paths(results_path: Path, out: Optional[Path] = None) -> Tuple[Path, Path]:
    """Beside the results: <name>.verdict.json and <name>.verdict.md, for <name>.json or <name>.runs.jsonl;
    <name>.verdict-2.json and so on when a verdict is already there, so none is written over."""
    stem = results_path.name.removesuffix(".runs.jsonl").removesuffix(".json")
    folder, n = out or results_path.parent, 1
    while True:
        tag = "verdict" if n == 1 else f"verdict-{n}"
        js, md = folder / f"{stem}.{tag}.json", folder / f"{stem}.{tag}.md"
        if not js.exists() and not md.exists():
            return js, md
        n += 1


def write(results_path: Path, thresholds: Optional[Thresholds] = None,
          out: Optional[Path] = None) -> Tuple[Verdict, Path, Path]:
    """Score a results file, or the runs file written beside it as the comparison ran, and write the verdict
    as JSON and Markdown beside it (or in `out`)."""
    if results_path.name.endswith(".jsonl"):
        result = read_runs(results_path)
    else:
        result = ComparisonResult.model_validate_json(results_path.read_text(encoding="utf-8"))
    v = score(result, thresholds, results=results_path.name)
    js, md = verdict_paths(results_path, out)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(v.to_json(), encoding="utf-8")
    md.write_text(to_markdown(v), encoding="utf-8")
    return v, js, md
