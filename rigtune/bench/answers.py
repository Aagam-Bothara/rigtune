"""Grade every compared row's answers on a labelled set (`compare --check-answers`), so that no row wins on
answers that got worse.

Why grading, not agreement with the stock row. An agreement check at temperature 0 was considered and not
built, because rigtune measured one and withdrew it (rigtune/quality.py): on a 14B on an RTX 4090, identical
fp8 weights answered the same prompts twice and differed on 10% of answer openings, while 4-bit and int8
checkpoints differed on 83-100%, yet GSM8K put those 4-bit weights only about 2 points behind. Batched greedy
decoding is not bit-reproducible, and a quantized KV cache or speculation changes the bits further, so
agreement would refuse rows whose answers are as good. Labels ask the right question: are the answers as often
right?

How. While each row's engine is up after its first run that answered, the row's configuration answers the
items once, greedily, through the chat endpoint (accuracy_gate.chat_answers, the same path calibration's gate
uses), 8 to 64 at a time by its batch. Its reference is its own engine's stock row, graded the same way. A row
passes when the lower end of the 95% score interval for its accuracy minus the reference's lies above minus
the tolerance (calibrate/paired.py certifies, the accuracy gate's non-inferiority bound): 4 lost and 0 gained
of 200 is "not significant" by McNemar, and its bound is -5.0 points, so it fails at 3 and passes at 6. The
stock row passes by being the reference; it answers twice, back to back on the same engine, and every check
against it reports how many items its two passes disagreed on: batched greedy decoding is not bit-exact, and
that count is the floor below which no loss means anything. A row on an engine whose stock row was not
graded is not checked; under --check-answers the verdict counts an unchecked row as not applicable, and
counts nothing rigtune measured when its own answers fail or were not checked. A disaggregated or replicas
pick answers through its router or balancer (KeyedGate), graded under its own row's key.

A model whose chat template thinks is graded with thinking off (accuracy_gate.grading_template_kwargs, every row
alike, and each row's check says so): on the A40 in November 2026, Qwen3.5-9B graded with its template's default
spent all 512 tokens of 60 of 60 GSM8K answers thinking, and every row, stock included, scored about 18%.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

from rigtune.accuracy_gate import (ANSWER_MAX_TOKENS, MAX_CONCURRENCY, MIN_CONCURRENCY, _counts, chat_answers,
                                   pass_wall_seconds)
from rigtune.calibrate.paired import certifies, score_bounds
from rigtune.evalset import EvalItem
from rigtune.models import Config

logger = logging.getLogger(__name__)


@dataclass
class RowGrader:
    """The runner's `gate`: grades each configuration once, on its warm engine, keyed by Config.key() (or the
    key a KeyedGate gives); a key in `twice`, a stock row's, is graded twice to show its own noise."""

    items: Sequence[EvalItem]
    set_name: str
    sha256: str
    max_tokens: int = ANSWER_MAX_TOKENS
    answer: Callable[..., List[Optional[str]]] = chat_answers
    grades: Dict[str, str] = field(default_factory=dict)  # key -> per item '1' right, '0' wrong, '-'
    twice: Set[str] = field(default_factory=set)  # keys graded a second time: the stock rows'
    again: Dict[str, str] = field(default_factory=dict)  # key -> the second pass's per-item outcomes
    # Every graded answer, one JSON line each ({key, pass, item, label, right, response, finish_reason,
    # completion_tokens, reasoning_chars}), so a check that fails can be read answer by answer. Set only with the
    # default `answer` (chat_answers), which fills in how each answer ended.
    answers_file: Optional[Path] = None
    # Merged into every graded request, and why (accuracy_gate.grading_template_kwargs: a template that thinks is
    # graded with thinking off); passed to `answer` only when set.
    extra_body: Optional[dict] = None
    graded_with: Optional[str] = None
    # key -> how many of its first pass's answers stopped at max_tokens (finish_reason "length"), which grades them
    # wrong whatever they would have said; known only when `answer` reports how each answer ended (chat_answers).
    cut_off: Dict[str, int] = field(default_factory=dict)

    def _pass(self, base_url: str, model: Optional[str], cfg: Config, key: str, n: int = 1) -> Optional[str]:
        concurrency = min(MAX_CONCURRENCY, max(MIN_CONCURRENCY, cfg.batch or MIN_CONCURRENCY))
        details: List[Optional[dict]] = [None] * len(self.items)
        try:
            more = {"details": details} if self.answers_file is not None or self.answer is chat_answers else {}
            if self.extra_body:
                more["extra_body"] = self.extra_body
            responses = self.answer(base_url, model, self.items, concurrency, self.max_tokens,
                                    pass_wall_seconds(len(self.items)), **more)
        except Exception as exc:  # grading never takes the run down with it; an empty pass grades again later
            logger.warning("answer check: grading %s failed: %s", key, exc)
            return None
        responses = list(responses) + [None] * len(self.items)
        outcomes = "".join("-" if r is None else ("1" if item.grade(r) else "0")
                           for item, r in zip(self.items, responses))
        if not outcomes.strip("-"):
            return None
        if n == 1 and any(d is not None for d in details):
            self.cut_off[key] = sum(1 for d in details if d is not None and d.get("finish_reason") == "length")
        if self.answers_file is not None:
            self._keep(key, n, responses, outcomes, details)
        return outcomes

    def _keep(self, key: str, n: int, responses: List[Optional[str]], outcomes: str,
              details: List[Optional[dict]]) -> None:
        try:
            with open(self.answers_file, "a", encoding="utf-8") as f:
                for i, (item, said, got) in enumerate(zip(self.items, responses, outcomes)):
                    f.write(json.dumps({"key": key, "pass": n, "item": item.id or i, "label": item.answer,
                                        "right": None if got == "-" else got == "1", "response": said,
                                        **(details[i] or {})}) + "\n")
        except OSError as exc:  # the record is for reading later; the grade stands without it
            logger.warning("answer check: could not keep %s's answers in %s: %s", key, self.answers_file, exc)

    def record(self, base_url: str, model: Optional[str], cfg: Config, key: Optional[str] = None) -> None:
        key = key or cfg.key()
        if key not in self.grades:
            got = self._pass(base_url, model, cfg, key)
            if got is None:
                return
            self.grades[key] = got
        if key in self.twice and key not in self.again:  # on the same warm engine, straight after
            got = self._pass(base_url, model, cfg, key, n=2)
            if got is not None:
                self.again[key] = got


@dataclass
class KeyedGate:
    """A RowGrader seen through a runner's `gate.record(url, model, cfg)` under a key of the caller's: a
    disaggregated pair or a set of replicas answers through its router or balancer, and is graded under its
    row's config key rather than one engine's."""

    grader: RowGrader
    key: str

    def record(self, base_url: str, model: Optional[str], cfg: Config) -> None:
        self.grader.record(base_url, model, cfg, key=self.key)


def reference_label(backend: str) -> str:
    """The stock row a row's answers are compared with: its own engine's (a bare row's too)."""
    return f"{backend.removesuffix('-bare')}-default"


def check(label: str, backend: str, key: str, grader: RowGrader, keys: Dict[str, str], tolerance: float):
    """The AnswerCheck of one row, from the grades and the other rows' config keys ({label: key})."""
    from rigtune.bench.compare import AnswerCheck

    outcomes = grader.grades.get(key)
    base = AnswerCheck(eval_set=grader.set_name, sha256=grader.sha256, items=len(grader.items),
                       tolerance=tolerance, outcomes=outcomes or "", graded_with=grader.graded_with)
    if outcomes is None:
        base.reason = "not graded: its engine never answered the items"
        return base
    base.correct, base.answered = outcomes.count("1"), len(outcomes) - outcomes.count("-")
    base.cut_off = grader.cut_off.get(key)
    cut = (f"; {base.cut_off} of its {base.answered} answers stopped at the {grader.max_tokens}-token limit and are "
           "graded wrong" if base.cut_off else "")
    ref_label = reference_label(backend)
    ref_key = keys.get(ref_label, "")
    again = grader.again.get(ref_key)
    if again is not None:  # the reference's own noise: its two passes, item by item
        lost, gained, n = _counts(grader.grades[ref_key], again)
        base.noise_flips, base.noise_paired = lost + gained, n
    noise = (f"; {ref_label} itself differed on {base.noise_flips} of {base.noise_paired} items between two "
             "passes" if base.noise_flips is not None else "")
    if label == ref_label:
        base.passed, base.reason = True, "the reference for its engine" + noise + cut
        return base
    ref = grader.grades.get(ref_key)
    base.reference = ref_label
    if ref is None:
        base.reason = f"not checked: {ref_label} was not graded, so there is no reference on {backend}"
        return base
    lost, gained, n = _counts(ref, outcomes)
    lower = score_bounds(lost, gained, n)[0] if n else -1.0
    base.lost, base.gained, base.paired, base.lower_bound = lost, gained, n, lower
    base.passed = certifies(lost, gained, n, tolerance)
    verb = "holds" if base.passed else "fails"
    base.reason = (f"{verb}: {lost} lost and {gained} gained of {n} items against {ref_label}; the lower "
                   f"bound of the difference is {lower * 100:+.1f} points, against -{tolerance * 100:g}"
                   f"{noise}{cut}")
    return base
