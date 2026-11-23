# The held-out run: Qwen3.5-9B on 500 held-out problems, thinking off

Run on 23 November 2026 with [modal_reasoning.py](../../../modal_reasoning.py) on a Modal A100 SXM 80 GB: vLLM 0.33.0
serving Qwen/Qwen3.5-9B with its own MTP head (3 speculative tokens), thinking off, answers up to 8,192 tokens, a
20,480-token context, 128 answers in flight, rigtune at a29922a. The 500 problems are the first 500 of MATH levels
4-5's held-out split (commit 21a5633873b6, set f9160b38959e), graded by math-verify. [report.md](report.md) is the
harness's own report; [manifest.json](manifest.json) what was asked. Every reply's text is kept in the Modal volume
`rigtune-triton-cache` under `runs/qwen35-9b-heldout500-off/` (runs.jsonl, 27 MB), with the server's log. About $2.

| Method | Accuracy | 95% interval | GPU-s / problem | Tokens / problem | Cut off | Utilisation |
|---|---|---|---|---|---|---|
| direct | 92.6% | 90.0-94.6% | 0.49 | 1,915 | 33 | 73% |
| self-correction | 96.6% | 94.6-97.9% | 1.03 | 2,995 | 36 | 73% |
| self-consistency-3 | 95.4% | 93.2-96.9% | 1.37 | 5,692 | 96 | 91% |
| self-consistency-5 | 95.6% | 93.4-97.1% | 2.23 | 9,424 | 168 | 95% |

The criterion is undecided: direct answers and self-correction kept only 73% of the 128 places busy, so their GPU
seconds are wall time. Paired against direct answers, self-correction gained 22 problems and lost 2 (McNemar
p = 3.6e-5), and it moved 16 of its own first answers from wrong to right against 1 the other way.

**That gain is mostly a second budget, not a correction.** Splitting the transitions by whether the first answer
stopped at 8,192 tokens:

| Self-correction | First answer finished | First answer cut off |
|---|---|---|
| repaired (wrong to right) | 4 | 12 |
| harmed (right to wrong) | 1 | 0 |
| still wrong | 8 | 8 |
| kept right | 456 | 11 |

- 12 of the 16 repairs were first answers cut off before any answer; the critique turn, with 8,192 tokens of its own,
  finished them. Only 4 finished wrong answers were corrected, against 1 harmed: 4 to 1 is not evidence.
- 15 of the 22 problems self-correction gained over direct answers were ones direct's answer was cut off on.
- On the 414 problems no method was cut off on, the four are within noise: direct 405, self-consistency-3 407,
  self-consistency-5 409, self-correction 409.
- Of the 333 cut-off replies, 231 read as long unfinished solutions and 102 as loops (the same lines or text repeated
  in their last 3,000 characters).
- Thinking off, this model is near the ceiling here: 98-99% on the problems nothing was cut off on. The best of
  self-consistency-5's answers by the reference was right on 96.8%, only 1.2 points above its majority.

What it means for the plan:

- The cheaper rival self-correction has to beat is direct answers that may continue when cut off (or a larger
  budget), not more samples: that is what most of its gain is.
- A test of self-correction needs problems this model gets wrong after finishing: harder ones (Omni-MATH, or MATH
  level 5 alone), or a smaller model.
- 500 problems for 128 places gave 73% utilisation (98 for 32 gave 60-62% in the dev run): the harness's warning at
  fewer than 4 problems a place is too lenient, and a decided criterion needs fewer places for the same problems.
