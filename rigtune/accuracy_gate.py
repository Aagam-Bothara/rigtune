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
import math
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


def grading_template_kwargs(spec, inspect: Optional[Callable] = None) -> Tuple[Optional[dict], Optional[str]]:
    """The chat-template arguments every graded request sends, and a note saying why; (None, None) for a model
    whose template has no thinking switch rigtune reads.

    A model whose template thinks (Qwen3, Qwen3.5) spends its tokens reasoning before it answers. Graded at
    max_tokens 512, Qwen3.5-9B on vLLM 0.33 stopped at the limit on 60 of 60 GSM8K items, mid-thought, and scored
    about 18% on every configuration, stock included, so no row could be told from another. With
    `enable_thinking` false the same model and configurations scored 87% on all 1319 items, and rigtune's pick
    was certified against stock (19 lost, 16 gained, lower bound -1.2 points). So a template that has the switch
    is graded with it false, whatever its default (introspect.thinking_switch): false turns thinking off under
    either way a template writes it. This grades the configuration's numerics in non-thinking mode; it does not
    grade the model's thinking mode, which needs a budget of thousands of tokens per item."""
    from rigtune import introspect

    look = inspect or (lambda s: introspect.inspect_model(s, features=("thinking",)))
    try:
        switch = look(spec).thinking
    except Exception as exc:  # inspect_model never raises; an injected one might
        logger.warning("answer grading: could not read %s's chat template (%s); graded as it is", spec.hf_id, exc)
        return None, None
    if switch is None:
        return None, None
    default = " (it thinks by default)" if switch.default == "on" else ""
    return ({"chat_template_kwargs": {switch.variable: False}},
            f"graded with {switch.variable} false{default}: its non-thinking answers, which fit in the answer budget")


async def _answer_all(base_url: str, model: Optional[str], items: Sequence[EvalItem], concurrency: int,
                      max_tokens: int, wall_s: float, request_timeout: float,
                      details: Optional[List[Optional[dict]]] = None,
                      extra_body: Optional[dict] = None) -> List[Optional[str]]:
    out: List[Optional[str]] = [None] * len(items)
    sem = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(base_url=base_url, timeout=request_timeout) as client:

        async def one(i: int, item: EvalItem) -> None:
            body = {"model": model or "model", "messages": item.messages, "temperature": 0.0,
                    "max_tokens": item.max_tokens or max_tokens, "stream": False, **(extra_body or {})}
            async with sem:
                try:
                    resp = await client.post("/v1/chat/completions", json=body)
                    if resp.status_code != 200:
                        logger.debug("accuracy gate: HTTP %s", resp.status_code)
                        return
                    data = resp.json()
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    text = message.get("content")
                    out[i] = text if isinstance(text, str) else None
                    if details is not None:  # how the answer ended, so a failed check can be read item by item
                        reasoning = message.get("reasoning") or message.get("reasoning_content")
                        details[i] = {"finish_reason": choice.get("finish_reason"),
                                      "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
                                      "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0}
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
                 max_tokens: int = 512, wall_s: float = 600.0, request_timeout: float = 180.0,
                 details: Optional[List[Optional[dict]]] = None,
                 extra_body: Optional[dict] = None) -> List[Optional[str]]:
    """Each item's greedy answer from a running engine's chat endpoint, `concurrency` at a time, stopping
    naturally; None for any request that failed or had not finished when `wall_s` ran out. Given `details`, a list
    as long as `items`, each answered item's entry becomes {finish_reason, completion_tokens, reasoning_chars}.
    `extra_body` is merged into every request (grading_template_kwargs' chat_template_kwargs).

    Through /v1/chat/completions, so each engine applies its model's chat template, as benchmarks/task_quality.py
    does with vLLM's llm.chat."""
    return asyncio.run(_answer_all(base_url, model, items, concurrency, max_tokens, wall_s, request_timeout,
                                   details, extra_body))


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
    # Grades of exact configurations (Config.key()): every stage-1 grading, and each configuration graded after the
    # decision. A precision's grade cannot certify a configuration whose KV cache, speculation or other launch
    # settings the later stages changed: those are graded again (certified), and the pick must pass.
    config_outcomes: Dict[str, str] = field(default_factory=dict)
    reference_configs: Set[str] = field(default_factory=set)  # stage-1 configurations of faithful formats
    config_records: Dict[str, PrecisionAccuracy] = field(default_factory=dict)
    config_seconds: Dict[str, float] = field(default_factory=dict)  # each final grading pass, in its record
    fallback_from: List[str] = field(default_factory=list)  # faster configurations refused before the pick
    # The configuration being re-measured for its final grade (StagedSearch sets it), graded once on that run.
    final_target: Optional[str] = None
    final_record: Optional[PrecisionAccuracy] = None

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
        stage 1 can try again; one that never answers is refused at the decision.

        After the decision, only `final_target` is graded, once, on its own configuration: the KV cache and
        speculation the later stages chose are part of what it serves, and stage 1 graded neither. A pass that
        answers nothing is kept as such, and certifies nothing."""
        if self.decided:
            if self.final_target is not None and cfg.base_key() == self.final_target and self.items \
                    and cfg.base_key() not in self.config_outcomes:
                t0 = self.clock()
                outcomes = self._pass(base_url, model, min(MAX_CONCURRENCY, max(MIN_CONCURRENCY, cfg.batch)))
                elapsed = self.clock() - t0
                self.config_seconds[cfg.base_key()] = elapsed
                self.seconds_total += elapsed
                self.config_outcomes[cfg.base_key()] = outcomes
            return
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
        self.config_outcomes[cfg.base_key()] = first
        if faithful(key[1]):  # it may become its engine's reference at the decision
            self.reference_configs.add(cfg.base_key())
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

    def _compare(self, rec: PrecisionAccuracy, against: Key, unanswered_lost: bool = False) -> None:
        """Fill in the paired comparison of `rec`'s outcomes with `against`'s first pass. With `unanswered_lost`,
        an item `rec` left unanswered is paired as a wrong answer instead of dropping out (certified)."""
        outcomes = rec.outcomes.replace("-", "0") if unanswered_lost else rec.outcomes
        lost, gained, n = _counts(self.outcomes[against], outcomes)
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

    def graded(self, cfg: Config) -> bool:
        return cfg.base_key() in self.config_outcomes

    def certified(self, cfg: Config) -> bool:
        """Whether this exact configuration may be served: its own grades certify a loss within the tolerance
        against its engine's reference, or it is that reference's own stage-1 configuration.

        Compared over the items the reference answered, as the decision compares, but an item this configuration
        left unanswered counts as lost where the reference was right: a missing answer must never make a check
        easier. So one unanswered item costs one item's worth of evidence, not the certificate, and an item the
        reference itself missed drops out rather than blocking every configuration."""
        outcomes = self.config_outcomes.get(cfg.base_key())
        if outcomes is None or not self.decided:
            return False
        key = (cfg.backend, cfg.quant)
        if cfg.base_key() in self.reference_configs and key in self.references.values():
            rec = self._record(key, outcomes)
            rec.config, rec.admitted, rec.reason = cfg.key(), True, "reference configuration"
            self.config_records[cfg.base_key()] = rec
            return True
        # A reference is compared with its own first pass; anything else with what stage 1 compared it with.
        against = key if key in self.references.values() else self._against.get(key)
        rec = self._record(key, outcomes)
        rec.config = cfg.key()
        if cfg.base_key() in self.config_seconds:  # graded after the decision: that pass's time, not stage 1's
            rec.seconds = round(self.config_seconds[cfg.base_key()], 1)
        if against is None or against not in self.outcomes:
            rec.admitted, rec.reason = False, "final configuration: no reference to compare it with"
            self.config_records[cfg.base_key()] = rec
            return False
        self._compare(rec, against, unanswered_lost=True)
        rec.admitted = certifies(rec.lost or 0, rec.gained or 0, rec.paired or 0, self.tolerance)
        lo = (rec.delta_ci95 or (-100.0, 100.0))[0]
        rec.reason = (f"final configuration: lower bound {lo:.1f} points is "
                      + ("above" if rec.admitted else "not above") + f" -{self.max_loss:.1f}")
        self.config_records[cfg.base_key()] = rec
        return rec.admitted

    def records(self) -> List[PrecisionAccuracy]:
        """One record per graded precision, references first, then each configuration graded for the final
        choice, the pick's own last."""
        ordered = sorted(self._records.values(), key=lambda r: (not r.reference, -precision_rank(r.quant),
                                                                r.backend, r.quant))
        checked = [r for r in self.config_records.values() if r is not self.final_record]
        return ordered + checked + ([self.final_record] if self.final_record is not None else [])

    def summary(self) -> str:
        if not self.decided:
            return "accuracy gate: stage 1 never finished, so nothing was graded or decided"
        twice = len(self.repeats)
        return (f"accuracy gate ({len(self.items)} {self.set_name} items, tolerance {self.max_loss:.1f} points): "
                f"references {_labels(self.references.values())}; admitted {_labels(self.admitted)}; refused "
                f"{_labels(self.refused)}; not graded (never ran) {_labels(self.unrun)}; grading took "
                f"{self.seconds_total:.0f}s, of which {self.seconds_repeat:.0f}s was the second pass that measures "
                f"each reference's own noise ({twice} reference{'' if twice == 1 else 's'})")

    def admitting_tolerance(self, key: Key) -> Optional[float]:
        """The smallest tolerance, to a tenth of a point, that would have admitted `key` on this evidence: just
        above minus its lower bound. None when it was never compared."""
        rec = self._records.get(key)
        if rec is None or rec.delta_ci95 is None or not rec.paired:
            return None
        return (math.floor(-rec.delta_ci95[0] * 10 + 1e-9) + 1) / 10

    def price_notes(self, winner: Optional[object], results: Sequence[object], objective: str,
                    constraints: object) -> List[str]:
        """What each refusal cost in speed, against the pick, and what would have changed the answer.

        A refusal that cost 3.9x is a different decision from one that cost 2%, and the operator can only weigh
        it with both the speed and the evidence in view: the tolerance that would have admitted the precision on
        these grades, and how many items would certify the tolerance asked for at its observed rate of loss. The
        refused precision was measured only in stage 1, never tuned, so its ratio is a floor."""
        from rigtune.calibrate.objectives import rank

        if winner is None or not self.refused:
            return []
        scored = rank([winner], objective, constraints)  # type: ignore[list-item, arg-type]
        if not scored:
            return []
        pick_tok = scored[0].metrics.tok_s
        notes: List[str] = []
        for key in sorted(self.refused):
            mine = [r for r in results if getattr(r, "ok", False)
                    and (r.config.backend, r.config.quant) == key]  # type: ignore[attr-defined]
            best = rank(mine, objective, constraints)  # type: ignore[arg-type]
            if not best:
                continue
            tok = best[0].metrics.tok_s
            ratio = tok / pick_tok if pick_tok else 0.0
            measured = f"its stage-1 trial measured {tok:.0f} tok/s against the pick's {pick_tok:.0f}"
            text = (f"accuracy gate: refusing {_label(key)} cost at least {ratio:.2f}x: {measured}, and it was never "
                    "tuned, so that ratio is a floor" if ratio > 1 else
                    f"accuracy gate: refusing {_label(key)} cost no measured speed: {measured}, untuned")
            admit = self.admitting_tolerance(key)
            rec = self._records.get(key)
            if admit is None or rec is None:
                text += "; it could not be graded, so no tolerance would have admitted it"
            else:
                text += f"; --max-accuracy-loss {admit:.1f} would have admitted it on this evidence"
                need = items_needed(rec.lost or 0, rec.gained or 0, rec.paired or 0, self.tolerance)
                text += (f", or about {need} items at its rate of loss would certify {self.max_loss:.1f} points"
                         if need is not None else
                         f", and no number of items would certify {self.max_loss:.1f} at its rate of loss")
            notes.append(text)
        return notes

    def final_note(self, winner: Optional[object]) -> Optional[str]:
        """What certified the pick's own configuration (StagedSearch.stage_final_quality serves nothing else), and
        whether it was a fallback from faster configurations that were refused."""
        if not self.decided:
            return None
        if winner is None:
            return ("accuracy gate: no configuration could be served: none whose own grades certify it was "
                    "available within the attempts and budget allowed")
        cfg: Config = winner.config  # type: ignore[attr-defined]
        if not self.certified(cfg):  # not reached through StagedSearch; kept so a caller cannot misread a pick
            return f"accuracy gate: the pick's final configuration {cfg.key()} is not certified"
        rec = self.final_record = self.config_records[cfg.base_key()]
        self.final_target = cfg.base_key()
        if rec.reason == "reference configuration":
            text = (f"accuracy gate: the pick's final configuration {cfg.key()} is its engine's reference, graded in "
                    f"stage 1: {self._accuracy_text(rec)}")
        else:
            lo, hi = rec.delta_ci95 or (-100.0, 100.0)
            text = (f"accuracy gate: the pick's final configuration {cfg.key()} is certified: it scored "
                    f"{self._accuracy_text(rec)} against {rec.against} ({rec.lost} lost, unanswered counted as "
                    f"lost, {rec.gained} gained; {_pts((rec.delta_points or 0) / 100)} points, 95% CI {lo:.1f} to "
                    f"{hi:.1f}), within the {self.max_loss:.1f}-point tolerance")
        if self.fallback_from:
            rec.reason += "; a fallback"
            text += (f". It is a fallback: {', '.join(self.fallback_from)} ranked higher but could not be "
                     "certified")
        return text

    def final_notes(self, winner: Optional[object], results: Sequence[object], objective: str,
                    constraints: object) -> List[str]:
        """Notes for the end of the search: the summary, what each refusal cost, and the pick's own grades."""
        final = self.final_note(winner)
        return [self.summary()] + self.price_notes(winner, results, objective, constraints) + ([final] if final else [])


def best_certifiable(n: int) -> float:
    """The smallest tolerance, in points, that n items can certify at all: the bound when no answer changed."""
    return -score_bounds(0, 0, n)[0] * 100 if n > 0 else 100.0
