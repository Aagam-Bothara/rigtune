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
- Answers. A row whose answers changed on the workload from its engine's stock row's (workload_answers),
  failed grading (compare --check-answers), or, when grading ran, were not checked, counts neither way: not
  applicable, or for failed grading **behind on speed, row's answers worse** when the speed rule calls it a
  loss or behind and **row's answers worse** otherwise. When rigtune's own answers changed, failed or were
  not checked, no row counts; with failed answers a row rigtune would win against on speed reads **ahead on
  speed, answers worse** and any other **rigtune's answers worse**.

Results written before RunRecords existed hold each run's tok/s but not its limit status: a successful run is
then counted as meeting the limits when the row's median run did, and the verdict says so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rigtune.bench.compare import ComparisonResult, ComparisonRow, read_runs
from rigtune.calibrate.objectives import parse_goal

WIN, TIE, BEHIND, LOSS, PROVISIONAL, NA = ("win", "tie", "behind, within noise", "loss", "provisional",
                                           "not applicable")
SPEED_ONLY = "ahead on speed, answers unchecked"
# Answers that failed the check. The speed is claimed only when the speed rule makes the claim: rigtune ahead
# is what would otherwise be a win, the row ahead is a loss or "behind".
RIG_WORSE = "ahead on speed, answers worse"  # rigtune would win on speed, but its answers failed the check
ROW_WORSE = "behind on speed, row's answers worse"  # the row is clearly faster, but its answers failed
RIG_WORSE_ONLY = "rigtune's answers worse"  # its answers failed, with no clear lead on speed
ROW_WORSE_ONLY = "row's answers worse"  # the row's answers failed, and it is not clearly faster
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
    answers: Optional[Dict[str, Any]] = None  # its answers at rigtune's scored level (answers_of)
    workload_answers: Optional[str] = None  # against its engine's stock row's answers (workload_answers)
    graded: Optional[Dict[str, Any]] = None  # its AnswerCheck (--check-answers), less the per-item outcomes


# A row's answers on the workload count as changed from its engine's stock row's, on the same prompts at the
# same level and temperature 0, when their mean length differs by more than objectives.OUTPUT_CHANGE (25%),
# when the share cut at max tokens differs by more than CUT_POINTS, or when the share that loops exceeds the
# stock row's by more than LOOPING_POINTS.
CUT_POINTS = 0.10
LOOPING_POINTS = 0.05
EPS = 1e-9  # a share exactly on a bar is within it: 0.4 - 0.3 is 0.10000000000000003 in floating point
SAME, CHANGED, UNKNOWN = "same", "changed", "unknown"


def _level_stats(row: ComparisonRow, level: int) -> List[Any]:
    if row.run_records:
        out = []
        for r in row.run_records:
            if not r.ok:
                continue
            got = r.level_answers.get(str(level))
            if got is None and r.scored_concurrency == level:
                got = r.answers  # a record that kept only its scored level's
            if got is not None:
                out.append(got)
        return out
    metrics = row.metrics.by_concurrency.get(str(level))
    return [metrics.answers] if metrics is not None and metrics.answers is not None else []


def answers_of(row: ComparisonRow, level: Optional[int]) -> Optional[Dict[str, Any]]:
    """The row's answer statistics at `level` users (rigtune's scored level), over every run that answered
    there: every row is sent the same prompts at a level, so these compare like with like. From the median
    run when it has no run records; None when none were recorded."""
    stats = [s for s in _level_stats(row, level) if s.answers] if level else []
    if not stats:
        return None
    n = sum(s.answers for s in stats)
    finish: Dict[str, int] = {}
    for s in stats:
        for k, v in s.finish_reasons.items():
            finish[k] = finish.get(k, 0) + v
    judged = sum(s.judged_for_repetition for s in stats)
    repeated = (sum((s.repeated_4gram_share or 0) * s.judged_for_repetition for s in stats) / judged
                if judged else None)
    return {"answers": n, "mean_tokens": sum((s.mean_tokens or 0) * s.answers for s in stats) / n,
            "finish_reasons": dict(sorted(finish.items())), "repeated_4gram_share": repeated,
            "looping_answers": sum(s.looping_answers for s in stats), "judged_for_repetition": judged}


def stock_label(row: ComparisonRow) -> str:
    """The stock row a row's answers are compared with: its own engine's (a bare row's and a disaggregated or
    replicas pick's too)."""
    return f"{row.config.backend.removesuffix('-bare')}-default"


def _share(a: Dict[str, Any], what: str) -> float:
    return a["finish_reasons"].get(what, 0) / a["answers"]


def workload_answers(result: ComparisonResult, row: ComparisonRow) -> Tuple[str, str]:
    """(SAME | CHANGED | UNKNOWN, why) for the row's answers on the workload against its engine's stock row's,
    at rigtune's scored level: the same prompts, greedy. A stock row is its own reference."""
    from rigtune.calibrate.objectives import OUTPUT_CHANGE

    rig = result.rigtune_row
    level = rig.scored_concurrency if rig is not None else None
    temperature = ((result.provenance or {}).get("workload") or {}).get("temperature")
    if temperature:  # sampled answers differ by chance: the bars below are for greedy ones
        return UNKNOWN, f"the workload was sampled at temperature {temperature:g}, not greedily"
    if not level:
        return UNKNOWN, "rigtune met the limits at no level, so there is no level to compare answers at"
    mine = answers_of(row, level)
    if mine is None:
        return UNKNOWN, f"{row.label} recorded no answers at {level} users"
    ref = stock_label(row)
    if row.label == ref:
        return SAME, ""
    base_row = next((r for r in result.rows if r.label == ref), None)
    base = answers_of(base_row, level) if base_row is not None else None
    if base is None:
        why = "is not in this comparison" if base_row is None else f"recorded no answers at {level} users"
        return UNKNOWN, f"{ref}, whose answers its own are compared with, {why}"
    changes, unknown = [], []
    if not base["mean_tokens"]:
        unknown.append(f"{ref}'s answers average 0 tokens, so a change in length cannot be measured")
    elif abs(mine["mean_tokens"] / base["mean_tokens"] - 1) > OUTPUT_CHANGE + EPS:
        changes.append(f"they average {mine['mean_tokens']:.0f} tokens against {base['mean_tokens']:.0f} for "
                       f"{ref} ({mine['mean_tokens'] / base['mean_tokens']:.2f}x)")
    cut, base_cut = _share(mine, "length"), _share(base, "length")
    if abs(cut - base_cut) > CUT_POINTS + EPS:
        changes.append(f"{cut * 100:.0f}% were cut at max tokens against {base_cut * 100:.0f}%")
    judged, base_judged = mine["judged_for_repetition"], base["judged_for_repetition"]
    if judged and base_judged:
        loops, base_loops = mine["looping_answers"] / judged, base["looping_answers"] / base_judged
        if loops - base_loops > LOOPING_POINTS + EPS:
            changes.append(f"{mine['looping_answers']} of {judged} look like loops ({loops * 100:.0f}%) "
                           f"against {base_loops * 100:.0f}%")
    else:  # no text kept (session replay), or answers too short to judge: a loop would go unseen
        neither = not judged and not base_judged
        unknown.append(f"repetition was measured on neither {row.label} nor {ref}" if neither else
                       f"repetition was measured on only one of {row.label} and {ref}")
    if changes:
        return CHANGED, f"at {level} users, on the same prompts as {ref}: " + "; ".join(changes)
    if unknown:
        return UNKNOWN, "; ".join(unknown)
    return SAME, ""


def _graded(row: ComparisonRow) -> Optional[Dict[str, Any]]:
    return row.answer_check.model_dump(exclude={"outcomes"}) if row.answer_check is not None else None


def binding_notes(result: ComparisonResult) -> List[str]:
    """For each tuner or expert row, the settings it changes from its engine's stock row that cannot act at
    the levels measured: a batch cap at or above the busiest level, a batched-token budget at or above what a
    step can hold (busiest level x mean prompt tokens + one decode token per user), a context at or above the
    mean prompt plus the longest answer the workload asks for. On those settings such a row measures like the
    stock engine here, whatever it would do at more users. Memory reserved and prefix caching are not judged:
    whether they act depends on how full the cache gets, which these results do not record."""
    levels = result.workload_spec.get("concurrencies") or []
    rig = result.rigtune_row
    if not levels or rig is None:
        return []
    users = max(levels)
    prompt = rig.metrics.prompt_tokens or int(result.workload_spec.get("prefill_tokens") or 0)
    answer = int(result.workload_spec.get("decode_tokens") or 0)
    step = users * prompt + users
    stock = {r.config.backend: r.config for r in result.rows if r.label == f"{r.config.backend}-default"}
    out = []
    for row in result.rows:
        if kind_of(row) not in ("tuner", "expert") or row.label == "rigtune":
            continue
        cfg, base = row.config, stock.get(row.config.backend)
        idle = []
        if (base is None or cfg.batch != base.batch) and cfg.batch >= users:
            idle.append(f"a batch cap of {cfg.batch} (it binds above {cfg.batch} users)")
        if cfg.prefill_budget and (base is None or cfg.prefill_budget != base.prefill_budget) and prompt:
            if cfg.prefill_budget >= step:
                idle.append(f"a batched-token budget of {cfg.prefill_budget} (a step here holds at most "
                            f"{users} x {prompt} + {users} = {step} tokens)")
        if (base is None or cfg.ctx != base.ctx) and prompt and cfg.ctx >= prompt + answer:
            idle.append(f"a context of {cfg.ctx} (prompts here average {prompt} tokens, answers at most "
                        f"{answer})")
        if idle:
            out.append(f"{row.label}: at up to {users} users, {'; '.join(idle)} cannot act, so on these "
                       "settings it measures like its engine's stock row; measure at more users "
                       "(compare --levels) to see them act")
    return out


def kind_of(row: ComparisonRow) -> str:
    if row.kind:
        return row.kind
    return "stock" if row.label.endswith(("-default", "-bare")) else "tuner"


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
                     reason="", runs_tok_s=_every_run(row), graded=_graded(row))
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
    provenance: Optional[Dict[str, Any]] = None  # what the comparison ran on, as its results recorded
    results_sha256: Optional[str] = None  # of the file this verdict scored, so it can be matched to it

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1) + "\n"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else ("es" if word.endswith("s") else "s"))


def _overall(rows: List[RowVerdict]) -> Dict[str, Any]:
    n = {o: sum(r.outcome == o for r in rows)
         for o in (WIN, SPEED_ONLY, TIE, BEHIND, BEHIND_CLEAR, LOSS, PROVISIONAL, NA, ROW_WORSE, RIG_WORSE,
                   ROW_WORSE_ONLY, RIG_WORSE_ONLY)}
    out = n[NA] + n[ROW_WORSE] + n[RIG_WORSE] + n[ROW_WORSE_ONLY] + n[RIG_WORSE_ONLY]  # count neither way
    judged = len(rows) - out
    na = f"; {n[NA]} not applicable" if n[NA] else ""
    if n[ROW_WORSE]:
        na += f"; {n[ROW_WORSE]} ahead of rigtune on worse answers"
    if n[ROW_WORSE_ONLY]:
        na += f"; {n[ROW_WORSE_ONLY]} whose answers are worse"
    if n[RIG_WORSE]:
        na += f"; {n[RIG_WORSE]} behind rigtune, whose answers are worse"
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
    elif n[WIN] == judged and out:
        outcome = "win on the rows that could be compared"
        text = (f"win on the {_plural(n[WIN], 'row')} that could be compared{na}, so not a win against every "
                "row")
    elif n[WIN] == judged:
        outcome, text = WIN, f"win against every row ({n[WIN]} of {len(rows)})"
    elif ahead == judged:  # ahead of every row it could be compared with, but not every answer was checked
        scope = (f"the {_plural(ahead, 'row')} that could be compared" + na if out
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
            "provisional": n[PROVISIONAL], "not_applicable": n[NA],
            "row_ahead_on_worse_answers": n[ROW_WORSE], "row_answers_worse": n[ROW_WORSE_ONLY],
            "ahead_on_worse_answers": n[RIG_WORSE], "rigtune_answers_worse": n[RIG_WORSE_ONLY]}


def answers_unchecked(result: ComparisonResult, row: RowVerdict) -> Optional[str]:
    """Why rigtune's answers are not known to be checked and unchanged against this row, or None when they
    are. A win needs both: its answers graded and holding against its engine's stock row's (compare
    --check-answers; under it, a row whose own answers were not checked is not applicable before this is
    asked), and its answers on the workload unchanged from that row's (workload_answers)."""
    rig = result.rigtune_row
    check = rig.answer_check
    # Asked only of a row that would be a win: under grading, rigtune's answers that failed or went
    # unchecked have already made every row count for nothing, so here they were not graded or they hold.
    assert check is None or check.passed is True, "rigtune's failed or unchecked answers reached a win"
    graded = "rigtune's answers were not graded" if check is None else None
    state, why = workload_answers(result, rig)
    if state != SAME and graded:
        return f"{graded}, nor matched against its stock row's on the workload ({why})"
    if state != SAME:
        return f"rigtune's answers on the workload were not matched against its stock row's ({why})"
    return graded


def score(result: ComparisonResult, thresholds: Optional[Thresholds] = None, results: str = "") -> Verdict:
    metric = parse_goal(result.goal).optimise if result.goal is not None else "tok_s"
    if metric != "tok_s":  # every win, tie and loss below counts scored tok/s
        raise ValueError(f"a verdict weighs tok/s, and this comparison's goal optimises {metric} ({result.goal})")
    t = thresholds or Thresholds()
    rig = result.rigtune_row
    if rig is None:
        raise ValueError("nothing to score: the comparison stopped before rigtune's row was measured"
                         if result.incomplete else "the results have no rigtune row")
    others = result.other_rows()
    rows = [judge(rig, r, t) for r in others]
    level = rig.scored_concurrency
    mine, mine_why = workload_answers(result, rig)
    graded = any(row.answer_check is not None for row in result.rows)  # compare --check-answers ran
    for verdict, row in zip(rows, others):
        verdict.answers = answers_of(row, level)
        state, why = workload_answers(result, row)
        verdict.workload_answers = f"{state}: {why}" if why else state
        if verdict.outcome == NA:
            continue
        check = row.answer_check
        if graded and (check is None or check.passed is None):  # grading ran: unchecked counts for nothing
            verdict.outcome, verdict.lean = NA, None
            verdict.reason = f"answers not checked: {check.reason if check else 'it has no answer check'}"
        elif check is not None and check.passed is False:  # faster on worse answers wins nothing either way
            verdict.lean = None
            if verdict.outcome in (LOSS, BEHIND_CLEAR):  # the speed rule says the row is ahead
                verdict.outcome = ROW_WORSE
                verdict.reason = f"{verdict.reason}; but its answers are worse: {check.reason}"
            else:
                verdict.outcome, verdict.reason = ROW_WORSE_ONLY, f"answers differ: {check.reason}"
        elif state == CHANGED:  # speed on different work counts neither way
            verdict.outcome, verdict.lean = NA, None
            verdict.reason = f"answers changed on the workload {why}"
    check = rig.answer_check
    failed = None  # (for each row, for the overall): why nothing rigtune measured counts
    if graded and (check is None or check.passed is None):
        why = check.reason if check else "it has no answer check"
        failed = (f"rigtune's answers were not checked: {why}", f"rigtune's answers were not checked ({why})")
    elif check is not None and check.passed is False:
        failed = (f"rigtune's answers differ: {check.reason}", f"rigtune's answers differ ({check.reason})")
    elif mine == CHANGED:
        failed = (f"rigtune's answers changed on the workload {mine_why}",
                  f"rigtune's answers changed on the workload ({mine_why})")
    worse = check is not None and check.passed is False
    if failed:  # rigtune's speed came with answers that are worse, changed or unknown: no row counts
        for verdict in rows:
            if verdict.outcome == NA:
                continue
            verdict.lean = None
            if worse and verdict.outcome == WIN:  # the speed rule says rigtune is ahead
                verdict.outcome, verdict.reason = RIG_WORSE, f"{verdict.reason}; but {failed[0]}"
            else:
                verdict.outcome, verdict.reason = RIG_WORSE_ONLY if worse else NA, failed[0]
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
    notes += binding_notes(result)
    if not any(answers_of(row, level) for row in result.rows):
        notes.append("no answer statistics were recorded at rigtune's scored level (compare --verdict "
                     "records them): answer length, finish reasons and repetition are unknown, so no row is "
                     "a win")
    if not graded:
        notes.append("answers were not graded (compare --check-answers), so no row is a win: rigtune ahead "
                     "of a row is ahead on speed, answers unchecked")
    for row in result.rows:
        if row.answer_check is not None and row.answer_check.passed is None:
            notes.append(f"{row.label}'s answers were not checked: {row.answer_check.reason}")
    notes += result.notes  # the comparison's own: where each added row came from, held-out prompts, ...
    overall = _overall(rows)
    if failed:
        overall["outcome"] = (RIG_WORSE if any(r.outcome == RIG_WORSE for r in rows) else
                              RIG_WORSE_ONLY if worse else NA)
        overall["text"] = f"{failed[1]}, so no row counts"
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
                 "range_tok_s": (min(r.values()), max(r.values())) if r else None, "met_limits": bool(r),
                 "answers": answers_of(rig, level),
                 "workload_answers": f"{mine}: {mine_why}" if mine_why else mine, "graded": _graded(rig)},
        rows=rows, overall=overall, notes=notes, provenance=result.provenance)


def _f(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.1f}"


def _f0(x: Optional[int]) -> str:
    return "-" if x is None else str(x)


def _range(r: Optional[Tuple[float, float]]) -> str:
    return "-" if r is None else f"{_f(r[0])}-{_f(r[1])}"


def _provenance_lines(v: Verdict) -> List[str]:
    p, out = v.provenance or {}, []
    if p:
        gpus = "; ".join(f"{g['name']} (driver {g['driver_version'] or '?'}, CUDA {g['cuda_version'] or '?'})"
                         for g in p.get("gpus", [])) or "no GPU"
        def engine(name: str, e: Dict[str, Any]) -> str:
            libs = ", ".join(f"{k} {x}" for k, x in e["libs"].items())
            return f"{name} {e['version']}" + (f" ({libs})" if libs else "")

        engines = "; ".join(engine(name, e) for name, e in p.get("engines", {}).items())
        commit = p.get("rigtune_commit") or "not run from a checkout"
        out.append(f"Ran on {gpus}, {p.get('os')}, Python {p.get('python')}; engines: {engines or 'none'}; "
                   f"rigtune {p.get('rigtune_version')} ({commit}).")
        if p.get("rigtune_diff_sha256"):
            out[-1] = out[-1][:-1] + f"; its local changes hash to sha256 {p['rigtune_diff_sha256']}."
        files = "; ".join(f"{role} `{f['name']}` sha256 {f['sha256']}"
                          for role, f in p.get("files", {}).items())
        if files:
            out.append(f"Prompts: {files}.")
        model, wl = p.get("model"), p.get("workload")
        if model:
            revision = model["revision"] or "main (none asked for)"
            snapshot = model["snapshot"] or "not found in the local Hugging Face cache"
            out.append(f"Weights: `{model['hf_id']}` at revision {revision}, cached snapshot {snapshot}.")
        if wl:
            out.append(f"Workload `{wl['name']}` from `{wl['source']}`, seed {wl['seed']}, temperature "
                       f"{wl['temperature']:g}; to get the prompts again: {wl['prompts']}.")
        if p.get("inputs"):
            out.append("Prompt files copied to " + ", ".join(f"`{x}`" for x in p["inputs"].values()) + ".")
        rows = [f"`{f['name']}` sha256 {f['sha256']}" for f in p.get("rows_files", [])]
        if p.get("expert_rows"):
            rows.append(f"`expert_rows.json` sha256 {p['expert_rows']['sha256']}")
        if rows:
            out.append(f"Added rows from {', '.join(rows)} (their content is in the verdict's JSON).")
        if p.get("command"):
            out.append(f"Command: `{' '.join(p['command'])}`.")
        if p.get("options"):
            shown = [f"{k} {v}" for k, v in p["options"].items() if v not in (None, [], False, "")]
            out.append("Options: " + "; ".join(shown) + ".")
        if p.get("engine_logs"):
            out.append(f"Engine logs: {p['engine_logs']}; each run's log names are in its run record.")
    if v.results_sha256:
        out.append(f"Scored `{v.results}`, sha256 {v.results_sha256}.")
    return out + ([""] if out else [])


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
    ]
    lines += _provenance_lines(v)
    lines += [
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
    answered = [("rigtune", rg["answers"], rg["workload_answers"])] + [
        (r.label, r.answers, r.workload_answers) for r in v.rows if r.config_key]
    if any(a for _, a, _ in answered):
        lines += ["", f"Answers at {rg['scored_concurrency']} users, rigtune's scored level, over every run "
                      "that answered there (every row is sent the same prompts at a level). Changed from "
                      "its engine's stock row means a mean length off by more than 25%, a share cut at max "
                      "tokens off by more than 10 points, or a share that loops above the stock row's by "
                      "more than 5 points:", "",
                  "| Row | Answers | Mean tokens | Ended on their own | Cut at max tokens | "
                  "Repeated 4-grams | Looping | Against its stock row |", "|---|---|---|---|---|---|---|---|"]
        for label, a, state in answered:
            if not a:
                lines.append(f"| {label} | not recorded | | | | | | {state or '-'} |")
                continue
            stop, cut = a["finish_reasons"].get("stop", 0), a["finish_reasons"].get("length", 0)
            rep = "-" if a["repeated_4gram_share"] is None else f"{a['repeated_4gram_share'] * 100:.0f}%"
            lines.append(f"| {label} | {a['answers']} | {a['mean_tokens']:.0f} | {stop} | {cut} | {rep} | "
                         f"{a['looping_answers']} of {a['judged_for_repetition']} | {state or '-'} |")
    graded = [("rigtune", rg["graded"])] + [(r.label, r.graded) for r in v.rows if r.config_key]
    if any(g for _, g in graded):
        first = next(g for _, g in graded if g)
        lines += ["", f"Answers graded on `{first['eval_set']}` ({first['items']} items, greedy), each "
                      "against its engine's stock row, allowed to lose at most "
                      f"{first['tolerance'] * 100:g} points. The stock rows answer twice; the floor is how "
                      "many items their two passes disagreed on:", "",
                  "| Row | Right | Against | Lost | Gained | Lower bound | Floor | Check |",
                  "|---|---|---|---|---|---|---|---|"]
        for label, g in graded:
            if not g:
                lines.append(f"| {label} | not graded | | | | | | |")
                continue
            bound = "-" if g["lower_bound"] is None else f"{g['lower_bound'] * 100:+.1f} pts"
            floor = "-" if g.get("noise_flips") is None else f"{g['noise_flips']} of {g['noise_paired']}"
            state = {True: "holds", False: "**fails**", None: "not checked"}[g["passed"]]
            lines.append(f"| {label} | {g['correct']} of {g['answered']} | {g['reference'] or '-'} | "
                         f"{_f0(g['lost'])} | {_f0(g['gained'])} | {bound} | {floor} | {state} |")
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
    v.results_sha256 = hashlib.sha256(results_path.read_bytes()).hexdigest()
    js, md = verdict_paths(results_path, out)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(v.to_json(), encoding="utf-8")
    md.write_text(to_markdown(v), encoding="utf-8")
    return v, js, md
