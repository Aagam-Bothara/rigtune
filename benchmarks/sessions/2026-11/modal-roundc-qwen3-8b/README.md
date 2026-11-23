# Round C on a GPU: drafters found on the Hub, and draft lengths chosen from their curves

Three runs on a Modal A100 80GB, 22 November 2026: vLLM on Dolly and on synthetic prompts, then SGLang on Dolly. Both
engines picked DeepSeek's DSpark drafter, found by the Hub search: 2.58× on vLLM, 2.37× on SGLang.

The two vLLM runs, 22 November 2026, with [modal_calibrate.py](../../../modal_calibrate.py) on a Modal A100 80GB: vLLM 0.33.0
(its own Docker image), Qwen/Qwen3-8B, rigtune at 3d2500f (round C through C10; the fix in 0f7cd9d came after). One
measurement per trial; the pick re-measured once. The commands' last lines are in [results-dolly.json](results-dolly.json)
and [results-synthetic.json](results-synthetic.json); the logs and rigtune's homes are in the Modal volume
`rigtune-triton-cache` under `runs/qwen3-8b-roundc-dolly/` and `runs/qwen3-8b-roundc-a100/`.

## Dolly prompts: DSpark at 2.6x

`rigtune bench Qwen/Qwen3-8B --speculative all --budget 40m --workload chat --workload-file dolly.jsonl --save`: 32
Dolly prompts of up to 1,536 tokens, answers up to 512 (natural stop), 1, 4 and 8 users.

| Stage | Configuration | tok/s, 8 users | TTFT p95 | J/token |
|---|---|---|---|---|
| quant | bf16, ctx 8192, 64 seqs, no speculation | 704.7 | 85 ms | 0.469 |
| spec | + n-gram, 4 tokens | 1,027.8 | 59 ms | 0.347 |
| spec | + Qwen3-0.6B draft model, 6 tokens | 359.1 | 482 ms | 0.439 |
| spec | + EAGLE3 head (AngelSlim), 6 tokens | 1,083.4 | 61 ms | 0.331 |
| spec | + EAGLE3 head, 3 tokens (the length its curve predicted best) | 1,066.7 | 59 ms | 0.323 |
| spec | + DFlash drafter (z-lab), 15 tokens, `--max-num-batched-tokens 2048` | 1,514.3 | 222 ms | 0.254 |
| spec | + DSpark drafter (DeepSeek), 7 tokens, 2048 | 1,735.2 | 236 ms | 0.204 |
| prefill | DFlash, 8192 | 1,547.2 | 83 ms | — |
| prefill | DSpark, 16384 | **1,894.5** | 65 ms | 0.196 |
| confirm | DSpark, 16384, re-measured | 1,821.7 | 71 ms | 0.204 |

- The pick is DSpark at 7 tokens with a 16,384-token budget: 1,822 tok/s re-measured against 705 without speculation,
  2.58x, at 57% less energy a token and a lower p95 TTFT (71 ms against 85). Before round C none of the three Hub
  drafters was known to rigtune; the best it could reach was n-gram lookup, 1.46x.
- The budget the drafter needed at first (2,048, vLLM's default) left TTFT at 222-236 ms; the prefill stage's larger
  budgets brought it under the speed without speculation.
- DSpark's curve (0.80, 0.65, 0.53, 0.44, 0.39, 0.34, 0.31) holds up deep into its block; DFlash's falls faster (0.66,
  0.44, 0.31, ... 0.17 at 7 of 15).
- C10: the EAGLE3 head's curve (0.57, 0.37, 0.25, 0.15, 0.11, 0.07) and a drafted token costing 0.11 of a step
  predicted 3 tokens at 1.67x; 3 measured 1,067 against 1,083 at 6, so 6 was kept. The draft model (a token costing
  1.23 of a step) was predicted 0.80x at its best length and not tried again.
- The budget ran out after 19 trials (2,324 s): 5 were skipped, among them the EAGLE3 head's prefill trials and two
  confirmations. Three of the 19 were the repeats 0f7cd9d removes (below).

## Synthetic prompts: DSpark at 1.3x

`rigtune bench Qwen/Qwen3-8B --speculative all --budget 40m --save`, rigtune's default workload (16 synthetic prompts,
256 in, 128 out). Acceptance is low on synthetic text (n-gram 0.08 at the first position, the EAGLE3 head 0.30):

| Configuration | tok/s, 8 users |
|---|---|
| no speculation | 646.7 |
| n-gram, 4 | 629.6 |
| draft model, 6 | 216.2 |
| EAGLE3, 6 | 608.4 |
| EAGLE3, 1 (C10: predicted 1.21x, measured 1.12x) | 726.7 |
| DFlash, 15 | 722.6 |
| DSpark, 7 (pick; 833.0 re-measured) | 832.8 |

## SGLang: the same drafters, the same winner

`rigtune bench Qwen/Qwen3-8B --backend sglang --speculative all --budget 40m --workload chat --workload-file
dolly.jsonl --save`, with `modal_calibrate.py --engine sglang` (SGLang 0.5.23 from PyPI), rigtune at 20815df (C7 in);
the same prompts, card and budget as the Dolly run above. Results in [results-sglang-dolly.json](results-sglang-dolly.json).

| Stage | Configuration | tok/s, 8 users | TTFT p95 | mean accepted |
|---|---|---|---|---|
| quant | bf16, ctx 8192, 64 seqs, `--mem-fraction-static` 0.93 | 710.3 | 151 ms | — |
| spec | + n-gram, 12 tokens | 828.1 | 63 ms | 2.29 |
| spec | + EAGLE3 head (AngelSlim), 3 tokens | 933.4 | 176 ms | 2.15 |
| spec | + Qwen3-0.6B draft model, 3 tokens | 956.9 | 88 ms | 2.98 |
| spec | + DFlash (z-lab), 15 tokens, after an out-of-memory retry at 0.88 | 1,343.9 | 65 ms | 3.72 |
| spec | + DSpark (DeepSeek), 7 tokens | **1,684.4** | 65 ms | 4.34 |
| confirm | DSpark, re-measured | 1,681.9 | 62 ms | 4.28 |

- SGLang's pick is DSpark too: 1,682 tok/s re-measured against 710 without speculation, 2.37×, at a lower p95 time to
  first token (62 against 151 ms). vLLM's pick ran 1,822 on the same prompts.
- C1 and C2 held: the EAGLE3 head started at the 8,192-token context and unquantized. C7 held: both block drafters
  started with rigtune's flags, at vLLM's k.
- The draft model behaves very differently on the two engines: 957 tok/s on SGLang, 359 on vLLM, from the same
  checkpoint at a similar acceptance.
- **Four starts ran out of GPU memory at 0.93 and started at 0.88**, the search's out-of-memory retry; every trial was
  rescued, at about 2–3 minutes each. Each is memory SGLang takes outside the fraction that rigtune's plan for a block
  drafter leaves out: DFlash's sampling prewarm (`dflash_worker_v2.py` `prewarm_sampling`: full-vocabulary probability
  tensors, a 594 MiB softmax with 368 MiB free), a 16,384-token prefill chunk's activations with DSpark (768 MiB with
  426 free), and CUDA-graph capture at batch 256 with DSpark, whose requests take 8 tokens a step. To fix.
- The budget ran out after 24 trials (2,383 s); 14 were skipped.

## What the runs found

- Fixed in 0f7cd9d: once DSpark led, the speculation stage's second pass offered n-gram, the draft model and the
  EAGLE3 head again at DSpark's stated budget (2,048), three trials in each run that measured what the first pass had.
- C10's prediction was optimistic in both runs (1.21x predicted, 1.12x measured; 1.67x, 1.51x): the measurement
  decided both times, as designed.
- The EAGLE3 head tried was AngelSlim's (rigtune's own list ranks first); RedHatAI's speculators-format head was found
  but not measured.
