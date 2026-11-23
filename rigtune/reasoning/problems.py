"""Verifiable maths problems and how an answer to one is graded.

The problems are the Hendrycks MATH test set (MIT licence; EleutherAI/hendrycks_math on the Hub, one Parquet file per
subject), levels 4 and 5 by default: GSM8K is at its ceiling for the models in question (Qwen3.5-9B answered 1,249 of
1,319 on the A100, 18 Nov 2026), which leaves too few wrong answers to see one repaired. Each problem's reference is
the last \\boxed{} of its solution, as the dataset's authors graded it.

A response is graded by its last \\boxed{} answer, against the reference, by math-verify (Hugging Face's maths
answer checker, which compares LaTeX expressions as maths) when it is installed (`pip install "rigtune[reason]"`) and
passes a self-test, else by the string normalisation the MATH authors published (their `is_equiv`). math-verify
decides whenever it can, because the authors' normalisation accepts answers it should not: it cuts everything after
`\\text{ `, so 19 passes for the reference "19 \\text{ and }43" and "12 \\text{ or } 13" (a hedge) passes for 12, and
it drops a short left side, so "y = 4" passes for "x = 4". Every grade says which decided it ("math-verify",
"normalised", or "math-verify-error" when math-verify failed on that answer and the normalisation decided), so a
report counts them. A response with no \\boxed{} answer is wrong.

Problems split into a development set and a held-out set by a hash of their text, so the split is the same on every
machine and for every run, and a problem's side never depends on which others were loaded.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

MATH_REPO = "EleutherAI/hendrycks_math"
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus")
DEV_SHARE = 0.04  # about 100 of the 2,500-odd level 4-5 test problems choose prompts; the rest are held out
# The dataset commit read on 22 Nov 2026: ids and the split are hashes of the exact problem text, so an upstream edit
# would move problems between the sets; pinned, and recorded in each set's fingerprint.
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Problem:
    id: str  # the first 16 hex digits of the problem text's sha256
    subject: str
    level: int
    problem: str
    answer: str  # the reference: the last \boxed{} of the official solution

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def problem_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def split_of(problem: Problem, dev_share: float = DEV_SHARE) -> str:
    """"dev" or "heldout", from the problem's id alone."""
    return "dev" if int(problem.id, 16) / 16 ** 16 < dev_share else "heldout"


def _braced(rest: str) -> Optional[str]:
    r"""The contents of the brace group `rest` opens (after spaces), braces matched; None when it opens none or never
    closes. Escaped braces (\{ \}) count as braces, as LaTeX nests them."""
    rest = rest.lstrip()
    if not rest.startswith("{"):
        return None
    depth = 0
    for i, ch in enumerate(rest):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return rest[1:i]
    return None


def last_boxed(text: str) -> Optional[str]:
    """The contents of the last \\boxed{...} or \\fbox{...} in `text` that holds one, braces matched, so a later
    mention ("I used \\boxed as asked") does not hide the answer; else the bare `\\boxed 5` form of the last mention.
    None when there is no answer."""
    starts = list(re.finditer(r"\\(?:boxed|fbox)\b", text or ""))
    for m in reversed(starts):
        inside = _braced(text[m.end():])
        if inside is not None and inside.strip():
            return inside
    if starts:
        rest = text[starts[-1].end():]
        if rest.startswith(" ") and not rest.lstrip().startswith("{"):  # \boxed 5
            token = rest.strip().split()
            return token[0].rstrip("$.,") if token else None
    return None


# The MATH authors' answer normalisation (hendrycks/math, modeling/math_equivalence.py), kept as published so a
# grade here matches theirs.
def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if substr and substr[0] == "{":
            new_str += substr
            continue
        if len(substr) < 2:
            return string
        a, b = substr[0], substr[1]
        if b != "{":
            new_str += "{" + a + "}{" + b + "}" + substr[2:]
        else:
            new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}" if string == f"{a}/{b}" else string


def _remove_right_units(string: str) -> str:
    return string.split("\\text{ ")[0] if "\\text{ " in string else string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def normalise(string: str) -> str:
    """An answer in the MATH authors' normal form (their strip_string)."""
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


MAX_ANSWER_CHARS = 400  # longer answers are not handed to math-verify, whose sympy can stall on them


@functools.lru_cache(maxsize=1)
def _math_verify() -> Optional[Tuple[Any, Any]]:
    """(parse, verify) from math-verify when it is installed and passes a self-test (2/4 is 1/2, 1/3 is not), else
    None. Its timeouts use SIGALRM on Linux, main thread only, and a process per call on Windows, which crashed there
    (WinError 6) and made every answer read as wrong; the self-test catches that, and grading runs on the main thread
    (rigtune.reasoning.methods)."""
    try:
        from math_verify import parse, verify  # type: ignore[import-not-found]

        ok = verify(parse("$\\frac{1}{2}$"), parse("$\\frac{2}{4}$")) and not verify(
            parse("$\\frac{1}{2}$"), parse("$\\frac{1}{3}$"))
    except Exception as exc:
        logger.warning("math-verify not used: %s: %s", type(exc).__name__, exc)
        return None
    if not ok:
        logger.warning("math-verify not used: it failed its self-test (2/4 against 1/2) on this machine")
        return None
    return parse, verify


def grade(response: str, reference: str) -> Tuple[bool, Optional[str], str]:
    """(correct, the response's answer, the grader that decided: "math-verify", "normalised", "math-verify-error" or
    "no-answer")."""
    answer = last_boxed(response)
    if answer is None or not answer.strip():
        return False, None, "no-answer"
    authors = normalise(answer) == normalise(reference)
    mv = _math_verify()
    if mv is None:
        return authors, answer, "normalised"
    if len(answer) > MAX_ANSWER_CHARS:
        return authors, answer, "math-verify-error"
    parse, verify = mv
    try:
        gold, guess = parse(f"${reference}$"), parse(f"${answer}$")
        if not gold or not guess:  # read as nothing: not a verdict
            return authors, answer, "math-verify-error"
        return bool(verify(gold, guess)), answer, "math-verify"  # the reference first, as math-verify expects
    except Exception:  # a failure, not a verdict: the authors' normalisation decides, and the label says so
        return authors, answer, "math-verify-error"


def equivalent(a: str, b: str) -> bool:
    """Whether two answers are the same as maths: math-verify when it works, else the MATH authors' normal form. For
    comparing a model's answers with each other (votes, agreement); never given the reference."""
    if normalise(a) == normalise(b):
        return True
    mv = _math_verify()
    if mv is None or len(a) > MAX_ANSWER_CHARS or len(b) > MAX_ANSWER_CHARS:
        return False
    _, verify = mv
    try:
        x, y = _parsed(a), _parsed(b)
        return bool(x and y and verify(x, y))
    except Exception:
        return False


@functools.lru_cache(maxsize=65536)
def _parsed(answer: str) -> Any:
    """math-verify's reading of an answer, once per answer however many it is compared with."""
    mv = _math_verify()
    return mv[0](f"${answer}$") if mv is not None else None


def problems_from_rows(rows: Iterable[Dict[str, Any]], subject: str, levels: Sequence[int] = (4, 5)) -> List[Problem]:
    """Problems from MATH rows ({problem, level: "Level 5", type, solution}) at the given levels; a row whose level or
    reference cannot be read is left out."""
    out = []
    for row in rows:
        m = re.fullmatch(r"Level (\d)", str(row.get("level", "")).strip())
        reference = last_boxed(str(row.get("solution", "")))
        if not m or reference is None or int(m.group(1)) not in levels:
            continue
        text = str(row["problem"])
        out.append(Problem(id=problem_id(text), subject=subject, level=int(m.group(1)), problem=text,
                           answer=reference))
    return out


ALL_LEVELS = (1, 2, 3, 4, 5)


def load_math(levels: Sequence[int] = (4, 5), split: Optional[str] = None, revision: str = MATH_REVISION,
              cache_dir: Optional[Path] = None) -> Tuple[List[Problem], str]:
    """The MATH test problems at `levels` (one `split`, "dev" or "heldout", or all), sorted by id, and a sha256 of the
    dataset commit and their JSON, which a report records. Downloads EleutherAI/hendrycks_math's test Parquet files at
    `revision` (pyarrow, from the `reason` extra); `cache_dir` keeps every level of that commit as JSON lines, so a
    later run needs neither and any levels can be asked of it."""
    cache = cache_dir / f"math-test-{revision[:12]}.jsonl" if cache_dir is not None else None
    if cache is not None and cache.is_file():
        problems = [Problem(**json.loads(line)) for line in cache.read_text(encoding="utf-8").splitlines() if line]
    else:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
        from huggingface_hub import hf_hub_download

        problems = []
        for subject in MATH_SUBJECTS:
            path = hf_hub_download(MATH_REPO, f"{subject}/test-00000-of-00001.parquet", repo_type="dataset",
                                   revision=revision)
            problems += problems_from_rows(pq.read_table(path).to_pylist(), subject, ALL_LEVELS)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("".join(json.dumps(p.to_json(), sort_keys=True) + "\n" for p in problems),
                             encoding="utf-8")
    problems = sorted((p for p in problems if p.level in levels), key=lambda p: p.id)
    if split is not None:
        problems = [p for p in problems if split_of(p) == split]
    body = revision + "".join(json.dumps(p.to_json(), sort_keys=True) for p in problems)
    return problems, hashlib.sha256(body.encode()).hexdigest()
