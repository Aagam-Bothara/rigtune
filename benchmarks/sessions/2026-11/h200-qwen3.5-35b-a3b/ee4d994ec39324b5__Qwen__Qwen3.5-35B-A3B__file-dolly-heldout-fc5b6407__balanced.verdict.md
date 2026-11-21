## Verdict: Qwen/Qwen3.5-35B-A3B on NVIDIA H200

measured on `file-dolly-heldout-fc5b6407`, calibrated on `file-dolly-d03d5896`; objective `balanced` (TTFT <= 500 ms, TPOT <= 50 ms).

Ran on NVIDIA H200 (driver 620.70.06, CUDA 13.5), Linux 4.19.0-gvisor, Python 3.12.3; engines: vllm 0.33.0 (flashinfer 0.7.2.post1, tokenizers 0.23.3, torch 2.14.0+cu130, transformers 5.20.2); rigtune 0.3.0 (not run from a checkout).
Prompts: calibration_prompts `dolly.jsonl` sha256 d03d589625eea015d475c8526a27ffc900517fa443c5930b3093f4d6ca7b69eb; measured_prompts `dolly-heldout.jsonl` sha256 fc5b6407bbd865dbb38f5415f4de5a8a7f4130c6776fc7249007c3deb6ca3d17.
Weights: `Qwen/Qwen3.5-35B-A3B` at revision 2508b2437e382c605eb0fc52b83f5545960f6b4a, cached snapshot 2508b2437e382c605eb0fc52b83f5545960f6b4a.
Workload `file-dolly-heldout-fc5b6407` from `file:dolly-heldout.jsonl`, seed 0, temperature 0; to get the prompts again: a copy of the file is beside the results.
Prompt files copied to `ee4d994ec39324b5__Qwen__Qwen3.5-35B-A3B__file-dolly-heldout-fc5b6407__balanced.inputs/dolly.jsonl`, `ee4d994ec39324b5__Qwen__Qwen3.5-35B-A3B__file-dolly-heldout-fc5b6407__balanced.inputs/dolly-heldout.jsonl`.
Added rows from `expert_rows.json` sha256 30ebee03b0f19e484afbfa04a83e16bed467ff8d479a4d8a535050112b8ce896 (their content is in the verdict's JSON).
Command: `rigtune compare Qwen/Qwen3.5-35B-A3B --workload chat --workload-file dolly.jsonl --objective balanced --ttft-ceiling 500 --tpot-ceiling 50 --speculative all --drafts verified --budget 60m --eval-workload-file dolly-heldout.jsonl --repeats 3 --expert --bare-stock --verdict --check-answers gsm8k --check-items 300 --out results`.
Options: objective balanced; workload file-dolly-d03d5896; measured_on file-dolly-heldout-fc5b6407; repeats 3; ttft_ceiling_ms 500.0; ttft_percentile 95; tpot_ceiling_ms 50.0; expert True; bare_stock True; check_answers gsm8k; check_items 300; max_accuracy_loss_points 3.0; verdict_thresholds {'stock': 0.2, 'tuner': 0.1, 'expert': 0.1, 'loss_margin': 0.0}.
Engine logs: Qwen__Qwen3.5-35B-A3B/compare-file-dolly-heldout-fc5b6407, under rigtune's logs folder (rigtune.cache.logs_dir()); each run's log names are in its run record.
Scored `ee4d994ec39324b5__Qwen__Qwen3.5-35B-A3B__file-dolly-heldout-fc5b6407__balanced.json`, sha256 8eb6210b06348e07440897e1a9b098fa63bc27ee16ed614d003c31bb74e58cfd.

Rule: only runs that met the limits count. A win needs rigtune's median and its worst paired round at least +20% above a stock row (+10% above a tuner row, +10% above an expert row) and every rigtune run above every run of that row; a loss is a row ahead in its median and in every round, with every one of its runs above every rigtune run. Either needs 3 runs on each side and 3 paired rounds, or it is provisional. A win also needs rigtune's answers checked and unchanged; ahead on speed without that is "ahead on speed, answers unchecked", never a win. Rigtune behind otherwise is behind when its median and most paired rounds trail by more than the row's own spread, else behind within noise; anything else is a tie. A row that could not start, never met the limits or was not run is not applicable.

rigtune `vllm/fp8/ctx8192/b64/gmu0.95/kvfp8/sd=mtp:3`: median 2058.5 tok/s at 8 users, runs 2018.1-2075.5 (3 of 3 met the limits)

| Row | Kind | Needs | Median tok/s | Runs | Gain | Worst round | Ranges | Verdict |
|---|---|---|---|---|---|---|---|---|
| vllm-default (`vllm/bf16/ctx262144/b256/gmu0.90`) | stock | +20% | 912.0 | 903.0-931.8 (3 of 3) | +125.7% | +121.3% | apart | **win**: +125.7%, worst paired round +121.3%, both at least +20%, ranges apart |
| vllm-fp8-default (`vllm/fp8/ctx262144/b256/gmu0.90`) | stock | +20% | 1232.7 | 1221.7-1247.2 (3 of 3) | +67.0% | +65.2% | apart | **win**: +67.0%, worst paired round +65.2%, both at least +20%, ranges apart |
| vllm-bare (`engine defaults`) | stock | +20% | 922.7 | 913.6-926.2 (3 of 3) | +123.1% | +118.7% | apart | **win**: +123.1%, worst paired round +118.7%, both at least +20%, ranges apart |
| vllm-expert-batched-tokens (`vllm/bf16/ctx262144/b256/gmu0.90/pb16384`) | expert | +10% | 919.2 | 916.9-930.0 (3 of 3) | +123.9% | +120.1% | apart | **win**: +123.9%, worst paired round +120.1%, both at least +10%, ranges apart |
| vllm-expert-ngram (`vllm/bf16/ctx262144/b256/gmu0.90/speculative_config={"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}`) | expert | +10% | 720.1 | 699.4-748.5 (3 of 3) | +185.8% | +177.3% | apart | **win**: +185.8%, worst paired round +177.3%, both at least +10%, ranges apart |
| vllm-expert-kv-fp8 (`vllm/bf16/ctx262144/b256/gmu0.90/kvfp8`) | expert | +10% | 933.5 | 929.7-934.3 (3 of 3) | +120.5% | +116.0% | apart | **win**: +120.5%, worst paired round +116.0%, both at least +10%, ranges apart |
| sglang-expert-ngram | expert | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: sglang is not one of this comparison's engines |
| sglang-expert-eagle3 | expert | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: sglang is not one of this comparison's engines |
| sglang-expert-kv-fp8-e4m3 | expert | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: sglang is not one of this comparison's engines |
| sglang-expert-torch-compile | expert | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: sglang is not one of this comparison's engines |

Answers at 8 users, rigtune's scored level, over every run that answered there (every row is sent the same prompts at a level). Changed from its engine's stock row means a mean length off by more than 25%, a share cut at max tokens off by more than 10 points, or a share that loops above the stock row's by more than 5 points:

| Row | Answers | Mean tokens | Ended on their own | Cut at max tokens | Repeated 4-grams | Looping | Against its stock row |
|---|---|---|---|---|---|---|---|
| rigtune | 96 | 422 | 34 | 62 | 9% | 4 of 96 | same |
| vllm-default | 96 | 427 | 36 | 60 | 9% | 3 of 96 | same |
| vllm-fp8-default | 96 | 392 | 41 | 55 | 9% | 5 of 96 | same |
| vllm-bare | 96 | 425 | 35 | 61 | 8% | 3 of 96 | same |
| vllm-expert-batched-tokens | 96 | 426 | 34 | 62 | 8% | 3 of 96 | same |
| vllm-expert-ngram | 96 | 419 | 35 | 61 | 7% | 1 of 96 | same |
| vllm-expert-kv-fp8 | 96 | 416 | 38 | 58 | 5% | 0 of 96 | same |

Answers graded on `gsm8k` (300 items, greedy), each against its engine's stock row, allowed to lose at most 3 points. The stock rows answer twice; the floor is how many items their two passes disagreed on. Cut off counts the answers that stopped at the token limit, graded wrong whatever they would have said (- where the engine did not say how each answer ended):

| Row | Right | Cut off | Against | Lost | Gained | Lower bound | Floor | Check |
|---|---|---|---|---|---|---|---|---|
| rigtune | 290 of 300 | 7 | vllm-default | 3 | 3 | -2.0 pts | 3 of 300 | holds |
| vllm-default | 290 of 300 | 6 | - | - | - | - | 3 of 300 | holds |
| vllm-fp8-default | 291 of 300 | 4 | vllm-default | 4 | 5 | -1.9 pts | 3 of 300 | holds |
| vllm-bare | 291 of 300 | 3 | vllm-default | 3 | 4 | -1.7 pts | 3 of 300 | holds |
| vllm-expert-batched-tokens | 295 of 300 | 5 | vllm-default | 0 | 5 | +0.4 pts | 3 of 300 | holds |
| vllm-expert-ngram | 288 of 300 | 7 | vllm-default | 5 | 3 | -3.0 pts | 3 of 300 | holds |
| vllm-expert-kv-fp8 | 291 of 300 | 6 | vllm-default | 1 | 2 | -1.3 pts | 3 of 300 | holds |

**Overall: win on the 6 rows that could be compared; 4 not applicable, so not a win against every row**
_vllm-expert-batched-tokens: at up to 8 users, a batched-token budget of 16384 (a step here holds at most 8 x 135 + 8 = 1088 tokens) cannot act, so on these settings it measures like its engine's stock row; measure at more users (compare --levels) to see them act_
_vllm-expert-batched-tokens is an expert row: vllm-default plus `--max-num-batched-tokens 16384`: "For optimal throughput, we recommend setting `max_num_batched_tokens > 8192` especially for smaller models on large GPUs." (docs/configuration/optimization.md, Performance Tuning with Chunked Prefill, read at v0.29.0 and v0.33.0)_
_vllm-expert-ngram is an expert row: vllm-default plus `--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'`: "This document shows how to use [Speculative Decoding](https://arxiv.org/pdf/2302.01318) with vLLM to reduce inter-token latency under medium-to-low QPS (queries per second), memory-bound workloads." (docs/features/speculative_decoding/README.md, introduction, read at v0.29.0 and v0.33.0); "| N-gram | Low to medium gain | Medium gain | Lightweight and easy to enable. |" (docs/features/speculative_decoding/README.md, Method Selection at a Glance, read at v0.29.0 and v0.33.0)_
_vllm-expert-kv-fp8 is an expert row: vllm-default plus `--kv-cache-dtype fp8`: "Quantizing the KV (Key-Value) cache to FP8 format can significantly reduce its memory footprint. This optimization enables you to store more tokens in memory, leading to improved throughput and support for longer context windows." (docs/features/quantization/quantized_kvcache.md, FP8 KV Cache Overview, read at v0.29.0 and v0.33.0)_
_sglang-expert-ngram (expert) was not measured: sglang is not one of this comparison's engines_
_sglang-expert-eagle3 (expert) was not measured: sglang is not one of this comparison's engines_
_sglang-expert-kv-fp8-e4m3 (expert) was not measured: sglang is not one of this comparison's engines_
_sglang-expert-torch-compile (expert) was not measured: sglang is not one of this comparison's engines_
_rigtune's pick was calibrated on `file-dolly-d03d5896`; every row here was measured on `file-dolly-heldout-fc5b6407`, which calibration never saw_
_every row was measured 3 times, interleaved; tok/s is the median run's, and the spread is (max - min) / median_
