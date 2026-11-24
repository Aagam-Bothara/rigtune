# Inkling-2-Mini session 1: the recipe, vLLM's memory facts, a reasoning probe (H100 NVL, 23-24 Nov 2026)

thinkingmachines/Inkling-2-Mini (snapshot 7a67ae23) on one RunPod H100 NVL (95,830 MiB, driver 595.112.07), vLLM's
nightly wheel 0.34.0rc1.dev1211+g5a215c27f (torch 2.14.0+cu132), the only kind of vLLM that serves it (no release
has models/inkling2). No rigtune calibration: it plans Inkling 2 as every layer full until vLLM main's allocator is
measured, which this session does. [setup.sh](setup.sh) and [session.sh](session.sh) are what ran; results in
[results.json](results.json), reasoning reports in [reasoning/](reasoning/). $3.75 of RunPod: the pod ran about 1 h
40 min, of which about 30 minutes measured (23:45-00:13 UTC); the rest went to an SSH start-up that never came up
(fixed by the image's own /start.sh), setup, and a benchmark dependency (pandas) the nightly wheel leaves out.

## Speed: Dolly prompts, 256 output tokens each, one run per point

`vllm bench serve` (openai-chat, the 300 Dolly prompts of round C, --ignore-eos, 4 x users prompts, 16-256). The
recipe's H100 command (recipes/models/thinkingmachines/Inkling-2-Mini.yaml): TP1, `--reasoning-parser inkling`,
`--max-model-len 262144`, the bundled DSpark drafter at 7 tokens with probabilistic drafting.

| Setting | 1 user | 8 | 32 | 64 | TTFT p50 at 64 | KV tokens (GiB) |
|---|---|---|---|---|---|---|
| recipe, first (cold kernels) | 156 | 438 | 1,367 | 1,633 | 442 ms | 504,926 (65.87) |
| **recipe, warm** | **228** | **805** | 1,557 | **2,492** | 290 ms | 504,926 (65.87) |
| no drafter | 147 | 685 | | 2,370 | 843 ms | 754,411 (70.69) |
| recipe + --max-num-batched-tokens 2048 | 229 | 793 | **1,844** | 2,232 | 379 ms | **1,218,842** (69.08) |
| recipe + --max-model-len 32768 | | 782 | | 2,167 | 370 ms | 92,481 (65.98) |

Output tokens/s. Mean acceptance length 2.5-2.7 on these prompts (vLLM's PR measured 3.57 on GSM8K).

- **The recipe's first run was cold**: p99 time to first token 2.5-3.7 s (median 54-442 ms), the Triton recompiles
  per prompt length vLLM has open (#69288, #69451). The same command warm, 15 minutes later: 228 / 805 / 1,557 /
  2,492. Every other row ran after it, so only the warm row compares.
- **Warm, the recipe is the fastest setting measured at 1, 8 and 64 users**: 1.55x no drafter at one user,
  1.05-1.18x at 8-64. Nothing in this table beats it on short chat prompts.
- **--max-num-batched-tokens 2048 holds 2.4x the KV tokens** (1,218,842 against 504,926; 4.65 against 1.93
  concurrent 262k requests, by vLLM's own count) at the same speed at 1 and 8 users, +18% at 32 and -10% at 64.
  One run per point: the 32- and 64-user differences are within what an unrepeated run can show.
- **The drafter costs 33% of the KV tokens** (504,926 against 754,411): 4.8 GiB of memory, and the rest the
  grouping vLLM logs as "Add 7 padding layers, may waste at most 350.00% KV cache memory" (with two more steps at
  12.5%), present whenever it drafts (vLLM #69280).
- **A 32k context cap does not raise concurrency**: 2.82 concurrent 32k requests, 92,481 tokens, from the same 66
  GiB. vLLM's per-request reserve does not shrink with the cap; it follows the token budget (the conv-state
  admission reserve, vLLM #54703 / #69181 / #69444).
- **--max-num-seqs 32** (the reasoning server) gave 71.63 GiB and 681,920 tokens: the batch cap is a memory knob too
  (the probabilistic drafter's draft_logits scale with it).

## Reasoning probe: MATH levels 4-5, the 98 development problems

`python -m rigtune.reasoning --split dev --methods direct --thinking default --reasoning-effort <e> --max-tokens 16384
--sequences 16` against the recipe with --max-num-seqs 32. Inkling 2's template takes a reasoning_effort (default
0.9, "high"), not a thinking switch.

| Effort | Accuracy | GPU-s / problem | Tokens / problem | Cut off |
|---|---|---|---|---|
| low | 99.0% | 0.66 | 1,131 | 0 |
| medium | 98.0% | 0.98 | 1,712 | 0 |
| high (the template's default) | 96.9% | 1.62 | 2,691 | 3 |

- MATH levels 4-5 are too easy for it: 97 of 98 at low effort. The reasoning comparison needs competition problems
  (its model card evaluates HMMT February 2026).
- More effort cost 2.5x the GPU time and lost answers: the 3 cut-offs at 16,384 tokens are all inside the thinking,
  one a loop ("x=... x=... x=..."), two 54-56k characters of deliberation on diagram problems that doubt the problem
  itself ("maybe the user prompt has a typo"), the model card's "re-checked its own work without making progress".
  98 problems decide nothing; it is the sign to test on harder ones.

## What it means for rigtune

- The win on Inkling-2-Mini is not speed on short prompts: the recipe is already fastest there. It is memory: 2.4x
  the KV tokens from the token budget, at the speed the recipe has up to 8 users and within noise above, which
  matters for long prompts and many sessions. Session 2 should measure long-context, many-user traffic, with
  rigtune's repeated, interleaved rounds instead of one run per point, and a warm-up before every row.
- F3b (vLLM's memory for Inkling 2) can now be planned from these figures: per setting, vLLM's own KV tokens and
  concurrency, which depend on the token budget, the drafter and the batch cap, not on the context cap.
- Reasoning: harder problems, effort as a knob beside samples at equal GPU time, and loop/deliberation cut-offs.
