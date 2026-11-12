"""The accuracy gate: serve a cheaper weight precision only when labelled answers certify what it costs.

Speed is what calibration measures, and quantization can cost accuracy that no throughput number shows. On
Qwen2.5-14B on a 24 GB RTX 4090, 4-bit GPTQ served 1412 tok/s against 365 for the best fp8 configuration and
lost 4 of 200 GSM8K problems that fp8 got right, gaining none; on Qwen2.5-3B, 4-bit cost 4-5 points. Whether
3.9x is worth that is the operator's call, so the operator states a tolerance (`--max-accuracy-loss`, in points)
and labelled items (`--eval-set`), and the gate admits a precision only when the evidence rules out a loss larger
than the tolerance.

How it decides. During stage 1 each (backend, quant) answers the items once, greedily, on the engine its first
successful trial already launched. Each engine's reference is its most precise *faithful* format (FAITHFUL) that
was graded; it is planned from the feasible set, run first, and answers twice to show its own noise. At the end of
stage 1 each other precision is compared with a reference item by item, and admitted only if the lower end of the
95% score interval for (its accuracy minus the reference's) lies above minus the tolerance (calibrate/paired.py).
That is a non-inferiority bound, not a significance test: 4 lost and 0 gained on 200 items is "not significant"
(McNemar p = 0.125), and its bound is -5.03 points, so a significance test would admit it at any tolerance while
the bound refuses it at 3 and admits it at 6. Small samples therefore refuse rather than admit.

Which reference. A faithful format below a more precise faithful one on the same engine (fp8 under bf16) is gated
against it. A pre-quantized checkpoint or a lower GGUF quant is gated against its own engine's reference, or, when
its engine has none graded, against the most precise graded reference on another engine, and the note says the
comparison is cross-engine. Faithful formats are never gated across engines, since the engine and its chat template
change answers too; those gaps are only noted. One cross-engine reference for everything got this wrong both ways:
llama.cpp's Q8_0 ranks above fp8, so vLLM's fp8 would have been gated against another engine's template; and a
plain per-engine rule would have let SGLang's GPTQ, on a 14B whose SGLang fp8 failed to start, be its own reference.

What it does not do. It fails closed: a precision that ran but could not be graded is refused, and a pre-quantized
or low-bit format with no faithful reference graded anywhere is refused. A precision that never ran is not graded
and cannot win; the notes list it separately. Refusals go to StagedSearch.refused, which every later stage and the
final pick respect; dropping a precision from stage 1's kept list alone let a refused 4-bit checkpoint win once.
KV-cache quantization and speculative decoding, which later stages choose, are not gated.

Every graded precision's per-item outcomes ('1' right, '0' wrong, '-' unanswered, in eval-set order) are kept in
the profile with the set's sha256, so the same evidence can be re-decided at another tolerance without a GPU.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import httpx

from rigtune.accuracy import wilson
from rigtune.calibrate.paired import certifies, items_needed, mcnemar_p, score_bounds
from rigtune.evalset import EvalItem
from rigtune.models import Config, EvalSetInfo, PrecisionAccuracy
from rigtune.quality import precision_rank

logger = logging.getLogger(__name__)

Key = Tuple[str, str]
# (base_url, model name, items, concurrency, max_tokens, wall seconds) -> one response per item, None if unanswered
AnswerFn = Callable[[str, Optional[str], Sequence[EvalItem], int, int, float], List[Optional[str]]]

MIN_CONCURRENCY, MAX_CONCURRENCY = 8, 64

# The formats a reference may be: the model's own weights at 16 or 32 bits, and the 8-bit formats each engine makes
# from them itself (fp8 on vLLM and SGLang, Q8_0 on llama.cpp). Pre-quantized checkpoints (AWQ, GPTQ, W8A8) are
# separate artifacts somebody else produced, and the lower GGUF quants measurably lose (Q4_K_M +10% perplexity
# against Q8_0 on a 3B), so none of them may stand in as the thing the others are compared with.
FAITHFUL = frozenset({"fp32", "bf16", "fp16", "fp8", "Q8_0"})


def faithful(quant: str) -> bool:
    return quant in FAITHFUL


def pass_wall_seconds(items: int) -> float:
    """The time a grading pass may take: 60 s plus 2 s per item (660 s for 300 items, 2698 s for 1319). A fixed cap
    would leave the tail of a large set unanswered on a slow reference, and unanswered items shrink the sample."""
    return 60.0 + 2.0 * items


async def _answer_all(base_url: str, model: Optional[str], items: Sequence[EvalItem], concurrency: int,
                      max_tokens: int, wall_s: float, request_timeout: float) -> List[Optional[str]]:
    out: List[Optional[str]] = [None] * len(items)
    sem = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(base_url=base_url, timeout=request_timeout) as client:

        async def one(i: int, item: EvalItem) -> None:
            body = {"model": model or "model", "messages": item.messages, "temperature": 0.0,
                    "max_tokens": item.max_tokens or max_tokens, "stream": False}
            async with sem:
                try:
                    resp = await client.post("/v1/chat/completions", json=body)
                    if resp.status_code != 200:
                        logger.debug("accuracy gate: HTTP %s", resp.status_code)
                        return
                    choice = (resp.json().get("choices") or [{}])[0]
                    text = (choice.get("message") or {}).get("content")
                    out[i] = text if isinstance(text, str) else None
                except (httpx.HTTPError, ValueError, KeyError, IndexError, AttributeError) as exc:
                    logger.debug("accuracy gate request failed: %s", exc)

        tasks = [asyncio.ensure_future(one(i, item)) for i, item in enumerate(items)]
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=wall_s)
            for t in pending:  # past the wall cap: what has not answered counts as unanswered
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    return out


def chat_answers(base_url: str, model: Optional[str], items: Sequence[EvalItem], concurrency: int,
                 max_tokens: int = 512, wall_s: float = 600.0, request_timeout: float = 180.0) -> List[Optional[str]]:
    """Each item's greedy answer from a running engine's chat endpoint, `concurrency` at a time, stopping
    naturally; None for any request that failed or had not finished when `wall_s` ran out.

    Through /v1/chat/completions, so each engine applies its model's chat template, as benchmarks/task_quality.py
    does with vLLM's llm.chat."""
    return asyncio.run(_answer_all(base_url, model, items, concurrency, max_tokens, wall_s, request_timeout))


def _counts(ref: str, cand: str) -> Tuple[int, int, int]:
    """(lost, gained, n) over the items both passes answered."""
    lost = gained = n = 0
    for a, b in zip(ref, cand):
        if a == "-" or b == "-":
            continue
        n += 1
        lost += a == "1" and b == "0"
        gained += a == "0" and b == "1"
    return lost, gained, n


def _pts(x: float) -> str:
    return f"{x * 100:+.1f}" if abs(x) >= 0.0005 else "0.0"


def _label(key: Key) -> str:
    return f"{key[0]}/{key[1]}"


def _labels(keys) -> str:
    return ", ".join(_label(k) for k in sorted(keys)) or "none"


def _rank(key: Key) -> int:
    return precision_rank(key[1])


CROSS_ENGINE = ("cross-engine: its engine has no faithful format graded, so the engine and its chat template are "
                "confounded with the precision")


@dataclass
class AccuracyGate:
    items: List[EvalItem]
    max_loss: float = 3.0  # points: the largest loss in accuracy against the reference a precision may carry
    info: Optional[EvalSetInfo] = None
    max_tokens: int = 512
    wall_s: Optional[float] = None  # per grading pass; None: pass_wall_seconds(items)
    request_timeout: float = 180.0
    answer: AnswerFn = chat_answers
    clock: Callable[[], float] = time.monotonic
    keys: List[Key] = field(default_factory=list)
    planned: Dict[str, Key] = field(default_factory=dict)  # engine -> its most precise feasible faithful format
    outcomes: Dict[Key, str] = field(default_factory=dict)
    repeats: Dict[Key, str] = field(default_factory=dict)  # each planned reference's second pass
    attempted: Set[Key] = field(default_factory=set)  # engines that were up and asked to answer
    seconds: Dict[Key, float] = field(default_factory=dict)
    seconds_total: float = 0.0  # grading time, which the budget's typical-trial estimate leaves out
    seconds_repeat: float = 0.0  # of which the references' second passes
    decided: bool = False
    references: Dict[str, Key] = field(default_factory=dict)  # engine -> the reference the decision used
    refused: Set[Key] = field(default_factory=set)
    admitted: Set[Key] = field(default_factory=set)
    unrun: Set[Key] = field(default_factory=set)  # never ran in stage 1: not graded, and blocked like a refusal
    _records: Dict[Key, PrecisionAccuracy] = field(default_factory=dict)
    _against: Dict[Key, Key] = field(default_factory=dict)  # what each graded precision was compared with

    @property
    def tolerance(self) -> float:
        return self.max_loss / 100.0

    @property
    def set_name(self) -> str:
        return self.info.name if self.info is not None else "eval"

    @property
    def pass_wall_s(self) -> float:
        return self.wall_s if self.wall_s is not None else pass_wall_seconds(len(self.items))

    # ---- during stage 1

    def plan(self, keys: Sequence[Key]) -> List[Key]:
        """Choose each engine's reference before any trial: its most precise faithful feasible format (the first on
        a tie). Returned in key order, so the search runs them first and each is graded twice."""
        self.keys = list(keys)
        self.planned = {}
        for k in self.keys:
            if faithful(k[1]) and (k[0] not in self.planned or _rank(k) > _rank(self.planned[k[0]])):
                self.planned[k[0]] = k
        chosen = set(self.planned.values())
        return [k for k in self.keys if k in chosen]

    def cannot_be_refused(self, key: Key) -> bool:
        """Whether a stage-1 trial of `key` may set the pace other engines are dropped against: a planned reference,
        or a precision already admitted. A precision the gate may yet refuse must not push another engine out."""
        return key in self.planned.values() or key in self.admitted

    def wanted(self, backend: str, quant: str) -> bool:
        return bool(self.items) and not self.decided and (backend, quant) not in self.outcomes

    def _pass(self, base_url: str, model: Optional[str], concurrency: int) -> str:
        try:
            responses = self.answer(base_url, model, self.items, concurrency, self.max_tokens, self.pass_wall_s)
        except Exception as exc:  # grading must never take the trial down with it
            logger.warning("accuracy gate: grading failed: %s", exc)
            responses = [None] * len(self.items)
        return "".join("-" if r is None else ("1" if item.grade(r) else "0")
                       for item, r in zip(self.items, list(responses) + [None] * len(self.items)))

    def record(self, base_url: str, model: Optional[str], cfg: Config) -> None:
        """Grade this running engine's precision once (twice for a planned reference), unless it has grades.

        Concurrency follows the trial's batch, clamped to 8-64, so grading neither crawls on a batch-1 config
        nor floods one. A pass that answered nothing is not recorded, so a later trial of the same precision in
        stage 1 can try again; one that never answers is refused at the decision."""
        key = (cfg.backend, cfg.quant)
        if not self.wanted(*key):
            return
        self.attempted.add(key)
        concurrency = min(MAX_CONCURRENCY, max(MIN_CONCURRENCY, cfg.batch))
        t0 = self.clock()
        first = self._pass(base_url, model, concurrency)
        again = None
        if key in self.planned.values() and first.strip("-"):
            t1 = self.clock()
            again = self._pass(base_url, model, concurrency)
            self.seconds_repeat += self.clock() - t1
        elapsed = self.clock() - t0
        self.seconds[key] = self.seconds.get(key, 0.0) + elapsed
        self.seconds_total += elapsed
        if not first.strip("-"):
            logger.warning("accuracy gate: %s answered none of %d items; not recorded", _label(key), len(self.items))
            return
        self.outcomes[key] = first
        if again is not None and again.strip("-"):
            self.repeats[key] = again

    # ---- at the end of stage 1

    def _record(self, key: Key, outcomes: str, reference: bool = False) -> PrecisionAccuracy:
        answered = sum(c != "-" for c in outcomes)
        correct = outcomes.count("1")
        lo, hi = wilson(correct, answered)
        rec = PrecisionAccuracy(backend=key[0], quant=key[1], items=len(self.items), answered=answered,
                                correct=correct, accuracy=correct / answered if answered else 0.0, ci95=(lo, hi),
                                reference=reference, seconds=round(self.seconds.get(key, 0.0), 1),
                                outcomes=outcomes)
        if reference:
            rec.outcomes_repeat = self.repeats.get(key)
            if rec.outcomes_repeat is not None:
                rec.self_lost, rec.self_gained, _ = _counts(outcomes, rec.outcomes_repeat)
        return rec

    def _compare(self, rec: PrecisionAccuracy, against: Key) -> None:
        """Fill in the paired comparison of `rec`'s outcomes with `against`'s first pass."""
        lost, gained, n = _counts(self.outcomes[against], rec.outcomes)
        lo_d, hi_d = score_bounds(lost, gained, n)
        rec.against, rec.cross_engine = _label(against), against[0] != rec.backend
        rec.paired, rec.lost, rec.gained = n, lost, gained
        rec.delta_points = (gained - lost) / n * 100 if n else 0.0
        rec.delta_ci95 = (lo_d * 100, hi_d * 100)
        rec.mcnemar_p = mcnemar_p(lost, gained)

    def _accuracy_text(self, rec: PrecisionAccuracy) -> str:
        return (f"{rec.accuracy:.1%} ({rec.correct}/{rec.answered}, CI {rec.ci95[0]:.1%}-{rec.ci95[1]:.1%}; "
                f"answered {rec.answered} of {rec.items})")

    def _needed_text(self, rec: PrecisionAccuracy) -> str:
        need = items_needed(rec.lost or 0, rec.gained or 0, rec.paired or 0, self.tolerance)
        if need is None:
            return f"; at this rate of loss no number of items would certify {self.max_loss:.1f} points"
        return (f"; about {need} items at this rate would certify {self.max_loss:.1f} points with 80% probability"
                + (f" (the set has {self.info.total})" if self.info is not None and self.info.total < need else ""))

    def decide(self) -> Tuple[Set[Key], List[str]]:
        """Refuse every gated precision whose loss is not certified to be within the tolerance.

        Returns the pairs the search must not choose (refused, plus those that never ran) and one note per
        graded precision."""
        self.decided = True
        notes: List[str] = []
        self.unrun = {k for k in self.keys if k not in self.attempted}
        refs: Dict[str, Key] = {}
        for k in self.keys:
            if k in self.outcomes and faithful(k[1]) and (k[0] not in refs or _rank(k) > _rank(refs[k[0]])):
                refs[k[0]] = k
        self.references = refs
        best: Optional[Key] = None
        for k in refs.values():
            if best is None or _rank(k) > _rank(best):
                best = k
        for engine, planned in self.planned.items():
            ref = refs.get(engine)
            if ref == planned:
                continue
            why = "did not run" if planned in self.unrun else "answered none of the items"
            if ref is not None:
                notes.append(f"accuracy gate: {engine}'s planned reference {_label(planned)} {why}, so {_label(ref)} is "
                             f"its reference instead; it was answered once, so there is no measure of its own noise")
            else:
                notes.append(f"accuracy gate: {engine}'s planned reference {_label(planned)} {why}, and no other "
                             f"faithful format of {engine} was graded")
        refused: Set[Key] = set()
        for key in self.keys:
            if key in self.unrun:
                continue
            if key not in self.outcomes:  # it launched, and answered nothing
                rec = self._records[key] = self._record(key, "-" * len(self.items))
                rec.admitted, rec.reason = False, "ran, but answered none of the items"
                refused.add(key)
                notes.append(f"accuracy gate: refused {_label(key)}: it ran but answered none of the "
                             f"{len(self.items)} items, so its accuracy is unknown")
                continue
            engine, quant = key
            if refs.get(engine) == key:
                rec = self._records[key] = self._record(key, self.outcomes[key], reference=True)
                rec.reason = "reference: the most precise faithful format graded on this engine"
                notes.append(self._reference_note(rec))
                if best is not None and best != key:
                    self._compare(rec, best)
                    self._against[key] = best
                    notes.append(self._candidate_note(key, rec, best, gated=False))
                continue
            rec = self._records[key] = self._record(key, self.outcomes[key])
            if faithful(quant):
                against, gated = refs[engine], _rank(key) < _rank(refs[engine])
            elif engine in refs:
                against, gated = refs[engine], True
            elif best is not None:
                against, gated = best, True
            else:
                rec.admitted, rec.reason = False, "no faithful reference was graded on any engine"
                refused.add(key)
                notes.append(f"accuracy gate: refused {_label(key)}: no faithful format (bf16, fp16, fp8, Q8_0) was "
                             "graded on any engine, so there is nothing to compare it with")
                continue
            self._compare(rec, against)
            self._against[key] = against
            notes.append(self._candidate_note(key, rec, against, gated))
            if gated:
                (self.admitted if rec.admitted else refused).add(key)
        if self.unrun:
            notes.append(f"accuracy gate: not graded (never ran): {_labels(self.unrun)}")
        self.refused = refused
        return refused | self.unrun, notes

    def _reference_note(self, rec: PrecisionAccuracy) -> str:
        key = (rec.backend, rec.quant)
        text = (f"accuracy gate: reference {_label(key)} scored {self._accuracy_text(rec)} on {self.set_name} "
                f"items")
        if rec.outcomes_repeat is None:
            planned = key in self.planned.values()
            return text + ("; its second pass answered nothing, so there is no measure of its own noise" if planned
                           else "")
        lost, gained, n = _counts(rec.outcomes, rec.outcomes_repeat)
        again = sum(c != "-" for c in rec.outcomes_repeat)
        text += (f"; answered twice (the second pass answered {again} of {rec.items}), it changed {lost + gained} of "
                 f"{n} answers' grades ({lost} right then wrong, {gained} wrong then right)")
        if not certifies(lost, gained, n, self.tolerance):
            lo = score_bounds(lost, gained, n)[0]
            need = items_needed(lost, gained, n, self.tolerance)
            text += (f". Against itself its lower bound is {lo * 100:.1f} points, so at {n} items this set cannot "
                     f"certify {self.max_loss:.1f} points even for identical weights"
                     + (f"; about {need} items would" if need is not None else ""))
        return text

    def _candidate_note(self, key: Key, rec: PrecisionAccuracy, against: Key, gated: bool) -> str:
        lo, hi = rec.delta_ci95 or (-100.0, 100.0)
        cross = against[0] != key[0]
        evidence = (f"{self._accuracy_text(rec)} against {_label(against)} on {rec.paired} items answered by both "
                    f"({rec.lost} lost, {rec.gained} gained, McNemar p = {rec.mcnemar_p:.3g}); the difference is "
                    f"{_pts((rec.delta_points or 0) / 100)} points, 95% CI {lo:.1f} to {hi:.1f}")
        if not gated:
            rec.admitted = None
            if rec.reference:
                rec.reason += "; compared with another engine's reference, not gated"
                return f"accuracy gate: {_label(key)} and {_label(against)} are each their engine's reference, so " \
                       f"the gap is noted, not gated: {evidence}"
            rec.reason = "as precise as its engine's reference: compared, not gated"
            return f"accuracy gate: {_label(key)} is as precise as {_label(against)}: compared, not gated: {evidence}"
        rec.admitted = certifies(rec.lost or 0, rec.gained or 0, rec.paired or 0, self.tolerance)
        where = f" ({CROSS_ENGINE})" if cross else ""
        if rec.admitted:
            rec.reason = f"lower bound {lo:.1f} points is above -{self.max_loss:.1f}" + (" (cross-engine)" if cross
                                                                                        else "")
            return (f"accuracy gate: admitted {_label(key)}{where}: {evidence}, within the {self.max_loss:.1f}-point "
                    "tolerance")
        rec.reason = f"lower bound {lo:.1f} points is not above -{self.max_loss:.1f}" + (" (cross-engine)" if cross
                                                                                         else "")
        return (f"accuracy gate: refused {_label(key)}{where}: {evidence}, so a loss of more than "
                f"{self.max_loss:.1f} points cannot be ruled out" + self._needed_text(rec))

    # ---- after the search

    def records(self) -> List[PrecisionAccuracy]:
        """One record per graded precision, references first, for the profile."""
        return sorted(self._records.values(), key=lambda r: (not r.reference, -precision_rank(r.quant),
                                                             r.backend, r.quant))

    def summary(self) -> str:
        if not self.decided:
            return "accuracy gate: stage 1 never finished, so nothing was graded or decided"
        twice = len(self.repeats)
        return (f"accuracy gate ({len(self.items)} {self.set_name} items, tolerance {self.max_loss:.1f} points): "
                f"references {_labels(self.references.values())}; admitted {_labels(self.admitted)}; refused "
                f"{_labels(self.refused)}; not graded (never ran) {_labels(self.unrun)}; grading took "
                f"{self.seconds_total:.0f}s, of which {self.seconds_repeat:.0f}s was the second pass that measures "
                f"each reference's own noise ({twice} reference{'' if twice == 1 else 's'})")

    def final_notes(self, winner: Optional[object], results: Sequence[object], objective: str,
                    constraints: object) -> List[str]:
        """Notes for the end of the search."""
        return [self.summary()]


def best_certifiable(n: int) -> float:
    """The smallest tolerance, in points, that n items can certify at all: the bound when no answer changed."""
    return -score_bounds(0, 0, n)[0] * 100 if n > 0 else 100.0
