# Inkling-2-Mini session 2: rigtune against stock vLLM and the recipe, long prompts, many users (H100 PCIe, 24 Nov 2026)

thinkingmachines/Inkling-2-Mini (snapshot 7a67ae23) on one RunPod H100 PCIe (81,559 MiB, driver 595.112.07), vLLM's
nightly wheel 0.34.0rc1.dev1219+gf08216eb6 (torch 2.14.0+cu132), rigtune 0.3.0 at 8c86f06 with `HF_HUB_OFFLINE=1`.
[setup.sh](setup.sh) and [session.sh](session.sh) are what ran, with the recipe's row in
[recipe-rows.json](recipe-rows.json); rigtune's plan in [plan.txt](plan.txt), the comparison in
[compare.json](compare.json), reasoning reports in [reasoning/](reasoning/). About $6.75 of RunPod (2 h 20 min at
$2.89/h). The reasoning manifests say "H100 NVL": session.sh's `--server` label was copied from session 1 by mistake;
the card was this H100 PCIe.

## Speed: rigtune's calibration, then three interleaved rounds

Workload `long-concurrent`: 128 prompts of 8,192 tokens, 256 output tokens each, at 8, 32 and 64 users, TTFT under
20 s and TPOT under 150 ms. rigtune calibrated for 35 minutes (7 trials, 2,124 s), then `rigtune compare` ran its
pick, stock vLLM (only the model named) and the recipe's H100 command three times each, interleaved.

| Row | tok/s at 32 users (3 runs) | TTFT p50 / p95 | TPOT | Peak memory |
|---|---|---|---|---|
| rigtune: ctx 65,536, batch 64, memory 0.95 | **398** (397.2, 397.7, 399.0) | 2,746 / 12,335 ms | 65.3 ms | 74.8 GB |
| stock vLLM: ctx 1,048,576, batch 256, memory 0.90 | **398** (397.8, 397.8, 400.8) | 2,743 / 12,352 ms | 64.9 ms | 71.4 GB |
| the recipe: ctx 262,144, batch 1,024, DSpark at 7 | failed to start, 3 of 3: out of memory | | | |

- **A tie.** The runs overlap (rigtune's 397-399, stock's 398-401); rigtune's own table says so. At 32 users with
  8,192-token prompts this card is bound by prefill (vLLM logged about 17,000 prompt tokens/s), and stock's defaults
  already keep it there. Calibration's other trials: batch 16 352 tok/s, n-gram drafting (4 tokens) 399,
  `--max-num-batched-tokens 2048` 369; rigtune kept the plain setting, the n-gram one being within noise of it.
- **The recipe does not start on an 80 GB H100.** vLLM gave its KV cache 51.96 GiB, then its warm-up asked for one
  more 4.2 GB block for the sampler at 1,024 requests x 8 tokens (`logits / self.soft_cap`, during the FlashInfer
  autotune) with 0.6-1.6 GB free. The same command started on session 1's 94 GB H100 NVL. The recipe says "H100"
  for both. vLLM sizes its pool before this warm-up and leaves too little for it here; not yet reported upstream.
- **rigtune did not predict that failure.** Its plan refuses the recipe on both cards, but only because it holds
  every one of the 1,024 slots at a full 262,144-token context (2.6 TB of KV), not because of the warm-up.

## What the plan got wrong: the Hub was off, and the cached snapshot was not read

With `HF_HUB_OFFLINE=1` rigtune's plan sized the weights from the parameter count, **8.96 GiB against the 18.24
GiB vLLM loaded** (21.75 GiB with the drafter), and **offered no DSpark drafter**, the recipe's own: both come from the
Hub's file listing (`hfconfig.load_arch`), which an offline Hub does not give, although the whole snapshot was in
the local cache. So calibration never tried the model's own drafter, and its memory plan was 9.3 GiB short (the
pick still started: vLLM sizes its pool from what is free, not from rigtune's plan). Fixed in the commit after this
one: offline, the snapshot's own files are read instead.

## Reasoning probe: Omni-MATH, difficulty 5 and up, the 101 development problems

`python -m rigtune.reasoning --dataset omni --min-level 5 --split dev --thinking default --max-tokens 16384
--sequences 16` against the recipe with `--max-num-seqs 32`.

| Run | Accuracy (95% interval) | GPU-s / problem | Tokens / problem | Cut off at 16,384 |
|---|---|---|---|---|
| effort low, one answer | 43.6% (34.3-53.3) | 5.45 | 8,098 | 30 of 101 |
| effort high, one answer | 30.7% (22.5-40.3) | 7.16 | 11,499 | 53 of 101 |
| effort low, 3 votes | **50.5%** (40.9-60.0) | 14.73 | 23,981 | 87 of 303 answers |

- **The token cap decides these runs.** Every cut-off answer was wrong. Low effort finished 71 problems and got 44
  right (62%); high effort finished 48 and got 31 (65%). Where both finished (48), low got 32 and high 31.
- **High effort lost 13 problems and gained none**; 12 of the 13 were cut off. At 16,384 tokens, more effort means
  more answers that never finish, not better ones. This says nothing yet about effort with room to finish.
- **Three votes: +8 problems, -1, against one answer** (exact two-sided McNemar p = 0.039), at 2.7x the GPU time.
  7 of the 8 gains were problems the single answer ran out of tokens on, and 4 of them were won by the one vote out
  of three that finished. As with Qwen3.5-9B's held-out run, most of what sampling more buys here is a finished
  answer.

## What it means

- On long-prompt, many-user traffic on an 80 GB card, rigtune ties stock vLLM and the recipe fails to start. A tie
  with stock is the honest result: the workload is prefill-bound and leaves little to tune.
- The calibration was incomplete: offline, rigtune saw neither the checkpoint's size nor its drafter. Session 3,
  after the fix, should rerun the calibration with the drafter in reach.
- Reasoning: a 16,384-token cap is too short for Omni-MATH at difficulty 5+. The next probe needs a cap the answers
  finish under (32k-64k) before effort or votes can be compared at equal GPU time.
