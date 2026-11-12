"""Labelled eval sets: the items the accuracy gate grades each precision on.

A set is a JSONL file, one item per line:

    {"prompt": "What is 2 + 3?", "answer": "5", "grader": "number"}
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
     "answer": "B", "grader": "choice", "choices": ["A", "B", "C", "D"], "max_tokens": 16, "id": "q2"}

`prompt` (one user message) or `messages` (a chat), `answer`, and `grader` (see rigtune/accuracy.py) are
required; `choices` is required by the `choice` grader; `max_tokens` and `id` are optional. The whole file is
checked when it is loaded, before any engine starts, and an error names the line: a set that fails on item 212
after an hour of calibration is a set that was never checked.

`gsm8k` names the bundled set, rigtune/data/gsm8k.jsonl: all 1319 GSM8K test problems with the instruction
benchmarks/task_quality.py uses, graded on the final number (built by benchmarks/make_eval_set.py; MIT licence
beside it). Its accuracies are therefore comparable with that script's. GSM8K is also public and likely in
many models' training data, which makes changed answers rarer and a bound flattering: for a decision about your
own traffic, grade on items from your own task.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

from rigtune.accuracy import GRADERS, grade, labelled_number

BUNDLED: Dict[str, str] = {"gsm8k": "gsm8k.jsonl"}
ROLES = ("system", "user", "assistant")


class EvalSetError(ValueError):
    """An eval set that cannot be used, with the file and line that say why."""


@dataclass
class EvalItem:
    messages: List[Dict[str, str]]
    answer: str
    grader: str
    choices: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    id: Optional[str] = None

    def grade(self, response: str) -> bool:
        return grade(self.grader, response, self.answer, self.choices)

    @property
    def chars(self) -> int:
        return sum(len(m["content"]) for m in self.messages)


@dataclass
class EvalSet:
    name: str  # "gsm8k", or the file's stem
    path: str
    sha256: str  # of the whole file, line endings normalised: a profile is reused only on the same set
    total: int  # items in the file
    items: List[EvalItem] = field(default_factory=list)  # the first `limit` of them


def _item(raw: Any, where: str) -> EvalItem:
    if not isinstance(raw, dict):
        raise EvalSetError(f"{where}: expected a JSON object")
    if "grader" not in raw:
        raise EvalSetError(f"{where}: no grader (expected one of {', '.join(sorted(GRADERS))})")
    grader = raw["grader"]
    if grader not in GRADERS:
        raise EvalSetError(f"{where}: unknown grader {grader!r} (expected one of {', '.join(sorted(GRADERS))})")
    if ("prompt" in raw) == ("messages" in raw):
        raise EvalSetError(f"{where}: give exactly one of prompt or messages")
    if "prompt" in raw:
        if not isinstance(raw["prompt"], str) or not raw["prompt"].strip():
            raise EvalSetError(f"{where}: prompt must be a non-empty string")
        messages = [{"role": "user", "content": raw["prompt"]}]
    else:
        messages = raw["messages"]
        if (not isinstance(messages, list) or not messages
                or not all(isinstance(m, dict) and m.get("role") in ROLES and isinstance(m.get("content"), str)
                           for m in messages)):
            raise EvalSetError(f"{where}: messages must be a list of {{role, content}} with role one of "
                               f"{', '.join(ROLES)}")
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    if "answer" not in raw or raw["answer"] is None or isinstance(raw["answer"], (dict, list, bool)):
        raise EvalSetError(f"{where}: no answer")
    answer = str(raw["answer"])
    choices = raw.get("choices")
    if grader == "number":
        try:
            labelled_number(answer)
        except ValueError:
            raise EvalSetError(f"{where}: the number grader needs a numeric answer, not {answer!r}") from None
    elif grader == "regex":
        try:
            re.compile(answer)
        except re.error as exc:
            raise EvalSetError(f"{where}: answer is not a valid regular expression: {exc}") from None
    elif grader == "choice":
        if (not isinstance(choices, list) or len(choices) < 2 or len(set(map(str, choices))) != len(choices)
                or not all(isinstance(c, str) and c.strip() for c in choices)):
            raise EvalSetError(f"{where}: the choice grader needs choices, a list of two or more distinct labels")
        if answer not in choices:
            raise EvalSetError(f"{where}: answer {answer!r} is not one of the choices {choices}")
    max_tokens = raw.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool)
                                   or max_tokens <= 0):
        raise EvalSetError(f"{where}: max_tokens must be a positive integer")
    item_id = raw.get("id")
    return EvalItem(messages=messages, answer=answer, grader=grader,
                    choices=list(choices) if grader == "choice" else None, max_tokens=max_tokens,
                    id=None if item_id is None else str(item_id))


def parse_items(text: str, source: str = "<eval set>") -> List[EvalItem]:
    """Every item of a JSONL eval set, checked; blank lines are skipped. Errors name `source` and the line."""
    items: List[EvalItem] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{source} line {number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalSetError(f"{where}: not valid JSON ({exc.msg})") from None
        items.append(_item(raw, where))
    if not items:
        raise EvalSetError(f"{source}: no items")
    return items


def bundled_text(name: str) -> str:
    """A bundled set, read through the installed package (importlib.resources), not the source tree."""
    return resources.files("rigtune").joinpath("data", BUNDLED[name]).read_text(encoding="utf-8")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def load_eval_set(spec: str, limit: int = 0) -> EvalSet:
    """The eval set `spec` names (a bundled name such as gsm8k, or a path), keeping its first `limit` items
    (0: all of them)."""
    if limit < 0:
        raise EvalSetError(f"--eval-items must be 0 (all) or more, not {limit}")
    if spec in BUNDLED:
        text, path, name = bundled_text(spec), f"rigtune/data/{BUNDLED[spec]}", spec
    else:
        p = Path(spec).expanduser()
        if not p.is_file():
            raise EvalSetError(f"no eval set {spec!r}: not a bundled set ({', '.join(sorted(BUNDLED))}) and no such "
                               "file")
        text, path, name = p.read_text(encoding="utf-8"), str(p), p.stem
    items = parse_items(text, path)
    return EvalSet(name=name, path=path, sha256=_sha256(text), total=len(items),
                   items=items[:limit] if limit else items)


def eval_set_sha256(spec: str) -> Optional[str]:
    """The sha256 the profile records for `spec`, or None when it cannot be read (the caller recalibrates)."""
    try:
        if spec in BUNDLED:
            return _sha256(bundled_text(spec))
        return _sha256(Path(spec).expanduser().read_text(encoding="utf-8"))
    except (OSError, KeyError):
        return None
