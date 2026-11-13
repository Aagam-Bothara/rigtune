#!/usr/bin/env python
"""Write held-out prompts from OpenAssistant (oasst2, Apache-2.0): the opening user turn of English conversations.

    python benchmarks/make_oasst_prompts.py --out oasst.jsonl
    rigtune compare <model> --workload chat --workload-file dolly.jsonl --eval-workload-file oasst.jsonl --repeats 3

Nothing else in the repository uses this dataset, so a pick calibrated on other prompts meets these for the
first time. The first 300 distinct openers after a seed-0 shuffle; they are not committed, this script
rebuilds the same file.

    python benchmarks/make_oasst_prompts.py --sessions --turns 6 --system system.txt --out oasst-sessions.jsonl
    rigtune bench <model> --workload chat --workload-file oasst-sessions.jsonl
    rigtune bench <model> --workload chat --workload-file oasst-sessions-flat.jsonl --report-reuse

--sessions writes whole conversations instead, for session replay: from each English opener, the thread that
follows the best-ranked reply at every step, kept when it reaches at least three user turns (so most sessions
have a third turn to reuse two turns of history), cut to `--turns` user turns. Beside it, it writes the same
conversations flattened (<out>-flat.jsonl: no "session" key, so each is one prompt), with the same `--system`
prompt in both, so a calibration on each differs only in whether the history is resent turn by turn.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def first_turns(rows: Iterable[Dict[str, object]], n: int, seed: int = 0, lang: str = "en") -> List[str]:
    """Up to n distinct conversation openers (user messages with no parent, in `lang`), after a seeded shuffle."""
    seen, openers = set(), []
    for r in rows:
        text = str(r.get("text") or "").strip()
        if (r.get("parent_id") is None and r.get("role") == "prompter" and r.get("lang") == lang
                and text and text not in seen):
            seen.add(text)
            openers.append(text)
    random.Random(seed).shuffle(openers)
    return openers[:n]


def _best(children: List[Dict[str, object]]) -> Dict[str, object]:
    """The reply OpenAssistant's raters ranked first (rank 0), unranked ones last, ties by id so it is stable."""
    return min(children, key=lambda r: (r.get("rank") is None, r.get("rank") or 0, str(r.get("message_id"))))


def sessions(rows: Iterable[Dict[str, object]], n: int, seed: int = 0, lang: str = "en",
             min_turns: int = 3, max_turns: Optional[int] = None, system: str = "") -> List[Dict[str, object]]:
    """Up to n conversations as session-file lines: from each distinct opener in `lang`, the thread that takes the
    best-ranked reply at every step, kept when it has at least `min_turns` user turns and cut to `max_turns`.
    `system`, when given, is every session's system prompt. Seeded shuffle, as above."""
    rows = [r for r in rows if r.get("lang") == lang and str(r.get("text") or "").strip()]
    children: Dict[object, List[Dict[str, object]]] = {}
    for r in rows:
        children.setdefault(r.get("parent_id"), []).append(r)
    seen, out = set(), []
    for root in sorted(children.get(None, []), key=lambda r: str(r.get("message_id"))):
        opener = str(root["text"]).strip()
        if root.get("role") != "prompter" or opener in seen:
            continue
        seen.add(opener)
        messages, node = [], root
        while True:
            messages.append({"role": "user" if node.get("role") == "prompter" else "assistant",
                             "content": str(node["text"]).strip()})
            replies = children.get(node.get("message_id"))
            if not replies:
                break
            node = _best(replies)
        if sum(m["role"] == "user" for m in messages) < min_turns:
            continue
        if max_turns is not None:
            users = [i for i, m in enumerate(messages) if m["role"] == "user"]
            messages = messages[:users[max_turns]] if len(users) > max_turns else messages
        while messages and messages[-1]["role"] == "assistant":  # nothing follows it, so neither file carries it
            messages = messages[:-1]
        line: Dict[str, object] = {"session": str(root.get("message_id")), "messages": messages}
        if system:
            line["system"] = system
        out.append(line)
    random.Random(seed).shuffle(out)
    return out[:n]


def flattened(session_lines: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """The same conversations as flat lines: no "session" key, the system prompt as a leading system message, so
    rigtune sends each as one prompt, exactly as a multi-turn messages file always was."""
    out = []
    for line in session_lines:
        system = [{"role": "system", "content": line["system"]}] if line.get("system") else []
        out.append({"messages": system + list(line["messages"])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("oasst.jsonl"))
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--sessions", action="store_true",
                    help="write whole conversations, for session replay, and the same ones flattened beside them")
    ap.add_argument("--turns", type=int, default=None, help="--sessions: keep each conversation's first N user turns")
    ap.add_argument("--system", type=Path, default=None, help="--sessions: a system prompt for every conversation")
    args = ap.parse_args()

    from datasets import load_dataset

    rows = load_dataset("OpenAssistant/oasst2", split="train")
    if not args.sessions:
        _write(args.out, [{"prompt": p} for p in first_turns(rows, args.n)], "prompts")
        return 0
    system = args.system.read_text(encoding="utf-8-sig").strip() if args.system else ""
    lines = sessions(rows, args.n, max_turns=args.turns, system=system)
    _write(args.out, lines, "sessions")
    _write(args.out.with_name(f"{args.out.stem}-flat{args.out.suffix}"), flattened(lines), "flattened conversations")
    return 0


def _write(path: Path, lines: List[Dict[str, object]], what: str) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    print(f"wrote {len(lines)} {what} to {path}")


if __name__ == "__main__":
    sys.exit(main())
