#!/usr/bin/env python
"""Write the bundled GSM8K eval set, rigtune/data/gsm8k.jsonl, that `--eval-set gsm8k` grades with.

    python benchmarks/make_eval_set.py --out rigtune/data/gsm8k.jsonl

Every problem of the GSM8K test split (openai/gsm8k, config "main", MIT licence; the licence is beside the
file), in dataset order, as one eval item each: the question followed by the instruction
benchmarks/task_quality.py appends, a chat prompt, and the reference's final number as the answer, graded by
`number`. The prompts and grading are task_quality.py's, so the gate's accuracies can be read against its
results. Rerunning this rebuilds the same file byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def answer_text(value: float) -> str:
    """The final number as plain text: '18', '-3', '0.5' (never '1e+06')."""
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def to_items(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    from rigtune.accuracy import INSTRUCTION, reference

    return [{"id": f"gsm8k-test-{i}", "prompt": f"{row['question']}\n\n{INSTRUCTION}",
             "answer": answer_text(reference(row["answer"])), "grader": "number"}
            for i, row in enumerate(rows)]


def write(items: List[Dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("rigtune/data/gsm8k.jsonl"))
    ap.add_argument("--parquet", type=Path, default=None,
                    help="read a downloaded copy of the split (main/test-00000-of-00001.parquet) instead of "
                         "loading it through `datasets`, whose cache lock names can pass Windows' path limit")
    args = ap.parse_args()

    if args.parquet is not None:
        import pyarrow.parquet as pq

        rows = pq.read_table(args.parquet).to_pylist()
    else:
        from datasets import load_dataset

        rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    items = to_items(rows)
    write(items, args.out)
    print(f"wrote {len(items)} items to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
