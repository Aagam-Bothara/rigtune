# Qwen3.5-35B-A3B on an H200: rigtune's pick against stock vLLM and the expert rows

Run on 20–21 November 2026 with [modal_calibrate.py](../../../modal_calibrate.py) on a Modal H200: vLLM 0.33.0
(its own Docker image), Qwen/Qwen3.5-35B-A3B at revision 2508b24 (an MoE model: 35B parameters, about 3B used a
token), rigtune at 7ea5e20 installed into vLLM's Python. It took 2 h 11 min, about $10.

```
rigtune compare Qwen/Qwen3.5-35B-A3B --workload chat --workload-file dolly.jsonl --objective balanced \
  --ttft-ceiling 500 --tpot-ceiling 50 --speculative all --drafts verified --budget 60m \
  --eval-workload-file dolly-heldout.jsonl --repeats 3 --expert --bare-stock --verdict \
  --check-answers gsm8k --check-items 300
```

The verdict is in the `*.verdict.md` file beside this one; its data and the scored comparison are the `.json`
files. Calibration took 3,524 s over 18 trials and picked fp8 weights, an fp8 KV cache and the checkpoint's MTP
head at 3 tokens (`vllm/fp8/ctx8192/b64/gmu0.95/kvfp8/sd=mtp:3`).

| Held-out prompts, 8 users | Median tok/s | rigtune's gain | Verdict |
|---|---|---|---|
| **rigtune** | **2,058.5** (runs 2,018–2,076) | | |
| stock vLLM (bf16) | 912.0 | +125.7% | win |
| bare stock (engine defaults) | 922.7 | +123.1% | win |
| stock vLLM, fp8 weights | 1,232.7 | +67.0% | win |
| expert: fp8 KV cache | 933.5 | +120.5% | win |
| expert: `--max-num-batched-tokens 16384` | 919.2 | +123.9% | win |
| expert: n-gram speculation | 720.1 | +185.8% | win |

GSM8K, 300 items, greedy, thinking off: rigtune 290 right, stock vLLM 290 (3 lost, 3 gained); the 95% lower
bound is −2.0 points against the −3 allowed, so the answers hold. Every rigtune run was above every run of every
row.

Limits:
- vLLM only: the image has no SGLang, so the four SGLang expert rows are "not applicable", and the overall verdict
  is a win on the 6 rows that could be compared, not on every row. vLLM's `auto_tune`, llm-optimizer and
  AIConfigurator were not run this time.
- 1 to 8 users, as in the other November sessions: the batch settings the expert rows change barely act there
  (the verdict says so for `--max-num-batched-tokens`).
- Suffix decoding was not tried: arctic-inference is not in the image.
- The calibration's confirm note says "re-measured the best 3 configurations"; two were re-measured before the
  budget ran out (the wording is fixed in 3f5d4d6).
- One session, one card.
