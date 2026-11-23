# The reasoning harness's first run: Qwen3-4B on the 98 development problems

Run on 23 November 2026 with [modal_reasoning.py](../../../modal_reasoning.py) on a Modal L4 (24 GB): vLLM 0.33.0
stock, Qwen/Qwen3-4B, thinking off, answers up to 4,096 tokens, 32 answers in flight, rigtune at fbc4be9. The 98
problems are MATH levels 4-5's development split (commit 21a5633873b6), graded by math-verify. [report.md](report.md)
is the harness's own report; [manifest.json](manifest.json) what was asked. Every reply's text is kept in the Modal
volume `rigtune-triton-cache` under `runs/qwen3-4b-dev-off/` (runs.jsonl, 5 MB), with the server's log. About $1.

This run exists to debug the harness on real output and to fix its prompts before the held-out set: 98 problems
cannot decide anything, and the criterion stayed undecided, as it should, because direct answers and self-correction
could not fill 32 places (utilisation 60-62%).

| Method | Accuracy | GPU-s / problem | Tokens / problem |
|---|---|---|---|
| direct | 74.5% | 3.10 | 1,194 |
| self-correction | 73.5% | 5.61 | 1,804 |
| self-consistency-3 | 77.6% | 6.53 | 3,418 |
| self-consistency-5 | 76.5% | 10.73 | 5,771 |
| adaptive-8 | 76.5% | 12.14 | 5,777 |

What it suggests, to be tested on the held-out set:

- Self-correction kept its answer: 2 repaired, 1 harmed, 25 still wrong, 70 kept right; it changed its answer on 11
  problems, mostly from one wrong answer to another, at nearly twice direct's GPU time. Its critique turns are real
  step-by-step reviews (median 1,368 characters).
- More samples gained 2-3 points over direct, within noise at 98 problems.
- The best of self-consistency-5's answers by the reference was right on 84.7% of problems against 76.5% for its
  majority: 8 points a better way of choosing (a trained verifier) could take.
- Adaptive escalated on 30 of 98 problems, whose first two answers disagreed, and cost more than self-consistency-5.

What it found in the harness, fixed in the commit that adds this record:

- The prompt-cache column read 0% because vLLM reports cached tokens only with --enable-prompt-tokens-details (its own
  log showed about 70% cache hits): unreported is now shown as unknown, and the runner turns the flag on.
- Progress lines were buffered until the end: the runner now runs unbuffered.
- 3-4% of answers stopped at 4,096 tokens (23 of adaptive's 376); the held-out run should allow more.
