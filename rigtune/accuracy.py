"""Graders for labelled answers: is this response right?

`benchmarks/task_quality.py` graded GSM8K at each weight precision by the final number of a greedy answer, and
that grading is what put 4-bit GPTQ about 2 points behind fp8 on Qwen2.5-14B at 24 GB. The same functions live
here so calibration can grade with them too, and the script imports them back, so its numbers and the
package's stay comparable.

A grader takes the model's response and the item's labelled answer and says right or wrong:

- `number`: the final number of the response (after the last "The answer is", else the last number anywhere)
  equals the answer's. GSM8K.
- `exact`: the whole response equals the answer once surrounding and repeated whitespace is ignored.
- `contains`: the answer appears in the response, ignoring case and whitespace runs.
- `regex`: the answer is a regular expression that `re.search` finds in the response.
- `choice`: the answer is one of the item's `choices` (labels such as A-D, matched exactly); the response picks
  one by saying "the answer is B" (or "Answer: (B)"), or by opening with it. Anything else picks nothing and is
  wrong.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, Optional, Sequence, Tuple

# The GSM8K instruction benchmarks/task_quality.py appends to every problem. `final_number` looks for the phrase
# it asks for, so the two belong together.
INSTRUCTION = "Solve the problem step by step. End your answer with a line of the form 'The answer is N'."
_ANSWER = re.compile(r"answer is\s*:?\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_WHITESPACE = re.compile(r"\s+")


def final_number(text: str) -> Optional[float]:
    """The number after the last 'The answer is', else the last number anywhere in the answer."""
    found = _ANSWER.findall(text)
    candidate = found[-1] if found else (_NUMBER.findall(text) or [None])[-1]
    if candidate is None:
        return None
    try:
        return float(candidate.replace(",", "").rstrip("."))
    except ValueError:
        return None


def reference(answer: str) -> float:
    """GSM8K reference answers end with '#### N'."""
    return float(answer.split("####")[-1].strip().replace(",", ""))


def wilson(correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for an accuracy."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "").strip())


def labelled_number(answer: str) -> float:
    """A labelled numeric answer: '18', '1,072', or a whole GSM8K solution ending in '#### 18'."""
    return reference(str(answer))


def grade_number(response: str, answer: str, choices: Optional[Sequence[str]] = None) -> bool:
    got = final_number(response or "")
    return got is not None and abs(got - labelled_number(answer)) < 1e-6


def grade_exact(response: str, answer: str, choices: Optional[Sequence[str]] = None) -> bool:
    return _normalise(response) == _normalise(answer)


def grade_contains(response: str, answer: str, choices: Optional[Sequence[str]] = None) -> bool:
    return _normalise(answer).casefold() in _normalise(response).casefold()


def grade_regex(response: str, answer: str, choices: Optional[Sequence[str]] = None) -> bool:
    return re.search(answer, response or "") is not None


def chosen(response: str, choices: Sequence[str]) -> Optional[str]:
    """Which of `choices` a response picks: the last one it states ("the answer is B", "Answer: (B)"), else
    the one it opens with, alone or bracketed ("B.", "(B) because ..."), else None.

    Labels match exactly and must stand alone, so neither "Because" nor "the answer is a bit tricky" picks B or
    A. Scanning the whole response for any standalone label would let the article "A" choose option A, which is
    also why an opening label counts only when punctuation or the end of a line follows it."""
    text = response or ""
    alternatives = "|".join(re.escape(c) for c in sorted(choices, key=len, reverse=True))
    if not alternatives:
        return None
    stated = re.findall(rf"(?i:answer\s*(?:is|:))\s*:?\s*\(?({alternatives})\)?(?!\w)", text)
    if stated:
        return stated[-1]
    lead = re.match(rf"\s*(?:\(({alternatives})\)|({alternatives})(?=\s*$|[.):,\n]))", text)
    return (lead.group(1) or lead.group(2)) if lead else None


def grade_choice(response: str, answer: str, choices: Optional[Sequence[str]] = None) -> bool:
    picked = chosen(response, choices or [])
    return picked is not None and picked == str(answer)


GRADERS: Dict[str, Callable[[str, str, Optional[Sequence[str]]], bool]] = {
    "exact": grade_exact,
    "number": grade_number,
    "contains": grade_contains,
    "regex": grade_regex,
    "choice": grade_choice,
}


def grade(grader: str, response: str, answer: str, choices: Optional[Sequence[str]] = None) -> bool:
    """Whether `response` is right under the named grader."""
    try:
        fn = GRADERS[grader]
    except KeyError:
        raise ValueError(f"unknown grader {grader!r}; expected one of {', '.join(sorted(GRADERS))}") from None
    return fn(response, answer, choices)
