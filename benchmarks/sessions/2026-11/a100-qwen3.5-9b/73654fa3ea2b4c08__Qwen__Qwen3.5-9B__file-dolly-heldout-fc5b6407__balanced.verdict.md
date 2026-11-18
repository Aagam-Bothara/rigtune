## Verdict: Qwen/Qwen3.5-9B on NVIDIA A100-SXM4-80GB

measured on `file-dolly-heldout-fc5b6407`, calibrated on `file-dolly-d03d5896`; objective `balanced` (TTFT <= 500 ms, TPOT <= 50 ms).

Ran on NVIDIA A100-SXM4-80GB (driver 595.112.07, CUDA 13.2), Linux 6.8.0-138-generic, Python 3.12.3; engines: sglang 0.5.23 (flashinfer 0.7.2, tokenizers 0.23.2, torch 2.14.1, transformers 5.19.0); vllm 0.33.0 (arctic-inference 0.1.1, flashinfer 0.7.2.post1, tokenizers 0.23.3, torch 2.14.0, transformers 5.20.2); rigtune 0.2.1 (not run from a checkout).
Prompts: calibration_prompts `dolly.jsonl` sha256 d03d589625eea015d475c8526a27ffc900517fa443c5930b3093f4d6ca7b69eb; measured_prompts `dolly-heldout.jsonl` sha256 fc5b6407bbd865dbb38f5415f4de5a8a7f4130c6776fc7249007c3deb6ca3d17.
Weights: `Qwen/Qwen3.5-9B` at revision c202236235762e1c871ad0ccb60c8ee5ba337b9a, cached snapshot c202236235762e1c871ad0ccb60c8ee5ba337b9a.
Workload `file-dolly-heldout-fc5b6407` from `file:dolly-heldout.jsonl`, seed 0, temperature 0; to get the prompts again: a copy of the file is beside the results.
Prompt files copied to `73654fa3ea2b4c08__Qwen__Qwen3.5-9B__file-dolly-heldout-fc5b6407__balanced.inputs/dolly.jsonl`, `73654fa3ea2b4c08__Qwen__Qwen3.5-9B__file-dolly-heldout-fc5b6407__balanced.inputs/dolly-heldout.jsonl`.
Added rows from `rows.json` sha256 a405aace27781e7ac8ef989ad20b6b63e68ec6fce12c62ce2fe810ebad471d82, `llmopt-vllm-rows.json` sha256 bc447e2a4109d2300429107c1cdfdf060de732cd47a428a06e329a02e13697b4, `rows.json` sha256 c0978cfb660df6ad98dc1d9d56a77fb0b03d76da27017b4a11d81017df06a169, `rows.json` sha256 b088cb45485de8f50d4f15b1c7c17251c7604f8942f45a5df8a08f465e893660, `rows.json` sha256 f2bec3d29e7acb172d5bd471f38b5ddec2be5e96675a7f1aa7bc4cdded8a87c1, `expert_rows.json` sha256 30ebee03b0f19e484afbfa04a83e16bed467ff8d479a4d8a535050112b8ce896 (their content is in the verdict's JSON).
Command: `rigtune compare Qwen/Qwen3.5-9B --workload chat --workload-file dolly.jsonl --objective balanced --ttft-ceiling 500 --tpot-ceiling 50 --speculative all --drafts verified --budget 60m --eval-workload-file dolly-heldout.jsonl --repeats 3 --expert --bare-stock --rows rows.json --rows llmopt-vllm-rows.json --rows rows.json --rows rows.json --rows rows.json --verdict --check-answers gsm8k --check-items 0 --out results`.
Options: objective balanced; workload file-dolly-d03d5896; measured_on file-dolly-heldout-fc5b6407; repeats 3; ttft_ceiling_ms 500.0; ttft_percentile 95; tpot_ceiling_ms 50.0; expert True; bare_stock True; rows ['rows.json', 'llmopt-vllm-rows.json', 'rows.json', 'rows.json', 'rows.json']; check_answers gsm8k; max_accuracy_loss_points 3.0; verdict_thresholds {'stock': 0.2, 'tuner': 0.1, 'expert': 0.1, 'loss_margin': 0.0}.
Engine logs: Qwen__Qwen3.5-9B/compare-file-dolly-heldout-fc5b6407, under rigtune's logs folder (rigtune.cache.logs_dir()); each run's log names are in its run record.
Scored `73654fa3ea2b4c08__Qwen__Qwen3.5-9B__file-dolly-heldout-fc5b6407__balanced.json`, sha256 d0846a6ff0696659d36ed398e3f7ea84c80c28391d79160fa1b33f0b4cbc31d7.

Rule: only runs that met the limits count. A win needs rigtune's median and its worst paired round at least +20% above a stock row (+10% above a tuner row, +10% above an expert row) and every rigtune run above every run of that row; a loss is a row ahead in its median and in every round, with every one of its runs above every rigtune run. Either needs 3 runs on each side and 3 paired rounds, or it is provisional. A win also needs rigtune's answers checked and unchanged; ahead on speed without that is "ahead on speed, answers unchecked", never a win. Rigtune behind otherwise is behind when its median and most paired rounds trail by more than the row's own spread, else behind within noise; anything else is a tie. A row that could not start, never met the limits or was not run is not applicable.

rigtune `vllm/bf16/ctx8192/b64/gmu0.95/sd=mtp:3`: median 1069.6 tok/s at 8 users, runs 1067.3-1070.7 (3 of 3 met the limits)

| Row | Kind | Needs | Median tok/s | Runs | Gain | Worst round | Ranges | Verdict |
|---|---|---|---|---|---|---|---|---|
| vllm-default (`vllm/bf16/ctx262144/b256/gmu0.90`) | stock | +20% | 564.6 | 553.6-567.9 (3 of 3) | +89.4% | +88.3% | apart | **win**: +89.4%, worst paired round +88.3%, both at least +20%, ranges apart |
| sglang-default (`sglang/bf16/ctx262144/b256/gmu0.88`) | stock | +20% | 580.4 | 578.5-585.7 (3 of 3) | +84.3% | +82.6% | apart | **win**: +84.3%, worst paired round +82.6%, both at least +20%, ranges apart |
| vllm-bare (`engine defaults`) | stock | +20% | 567.2 | 565.0-574.8 (3 of 3) | +88.6% | +86.1% | apart | **win**: +88.6%, worst paired round +86.1%, both at least +20%, ranges apart |
| sglang-bare (`engine defaults`) | stock | +20% | 582.6 | 581.1-585.1 (3 of 3) | +83.6% | +82.8% | apart | **win**: +83.6%, worst paired round +82.8%, both at least +20%, ranges apart |
| vllm-expert-ngram (`vllm/bf16/ctx262144/b256/gmu0.90/speculative_config={"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}`) | expert | +10% | 578.0 | 558.9-585.1 (3 of 3) | +85.1% | +82.4% | apart | **win**: +85.1%, worst paired round +82.4%, both at least +10%, ranges apart |
| vllm-expert-kv-fp8 (`vllm/bf16/ctx262144/b256/gmu0.90/kvfp8_e5m2`) | expert | +10% | 566.0 | 565.8-577.7 (3 of 3) | +89.0% | +85.3% | apart | **win**: +89.0%, worst paired round +85.3%, both at least +10%, ranges apart |
| sglang-expert-ngram (`sglang/bf16/ctx262144/b256/gmu0.88/speculative_algorithm=NGRAM`) | expert | +10% | 512.9 | 510.6-522.6 (3 of 3) | +108.5% | +104.3% | apart | **win**: +108.5%, worst paired round +104.3%, both at least +10%, ranges apart |
| sglang-expert-kv-fp8-e4m3 (`sglang/bf16/ctx262144/b256/gmu0.88/kvfp8_e4m3`) | expert | +10% | 580.5 | 576.5-583.0 (3 of 3) | +84.3% | +83.5% | apart | **win**: +84.3%, worst paired round +83.5%, both at least +10%, ranges apart |
| sglang-expert-torch-compile (`sglang/bf16/ctx262144/b256/gmu0.88/enable_torch_compile=True`) | expert | +10% | - | - (0 of 3) | - | - | - | **not applicable**: could not start or failed: out of memory at start-up; failed to start (rc=-9) |
| autotune-vllm (`vllm/bf16/ctx4096/b256/gmu0.98/pb4096/pc`) | tuner | +10% | 565.3 | 564.2-565.4 (3 of 3) | +89.2% | +88.8% | apart | **win**: +89.2%, worst paired round +88.8%, both at least +10%, ranges apart |
| llm-optimizer-vllm (`vllm/bf16/ctx4096/b64/pb16384`) | tuner | +10% | 567.8 | 564.6-570.8 (3 of 3) | +88.4% | +87.4% | apart | **win**: +88.4%, worst paired round +87.4%, both at least +10%, ranges apart |
| llm-optimizer-sglang (`sglang/bf16/ctx4096/b128/pb4096`) | tuner | +10% | 581.1 | 579.9-582.0 (3 of 3) | +84.1% | +83.9% | apart | **win**: +84.1%, worst paired round +83.9%, both at least +10%, ranges apart |
| vllm-expert-batched-tokens | expert | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: not on an A100: vLLM's own source: "Setting large `max_num_batched_tokens` for A100 reduces throughput, see PR #17885 for more details." (vllm/engine/arg_utils.py EngineArgs.get_batch_defaults, NOTE(Kuntai), v0.29.0 and v0.33.0), so the row would not be what the page recommends |
| sglang-expert-eagle3 | expert | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: rigtune lists no EAGLE-3 head for Qwen/Qwen3.5-9B (rigtune/speculative.py EAGLE3_HEADS) |
| aiconfigurator-vllm | tuner | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: AIConfigurator exited 1; see output.log |
| aiconfigurator-sglang | tuner | +10% | - | - (0 of 0) | - | - | - | **not applicable**: not run: AIConfigurator exited 1; see output.log |

Answers at 8 users, rigtune's scored level, over every run that answered there (every row is sent the same prompts at a level). Changed from its engine's stock row means a mean length off by more than 25%, a share cut at max tokens off by more than 10 points, or a share that loops above the stock row's by more than 5 points:

| Row | Answers | Mean tokens | Ended on their own | Cut at max tokens | Repeated 4-grams | Looping | Against its stock row |
|---|---|---|---|---|---|---|---|
| rigtune | 96 | 372 | 41 | 55 | 8% | 6 of 96 | same |
| vllm-default | 96 | 381 | 38 | 58 | 9% | 6 of 96 | same |
| sglang-default | 96 | 375 | 38 | 58 | 9% | 5 of 96 | same |
| vllm-bare | 96 | 382 | 37 | 59 | 9% | 6 of 96 | same |
| sglang-bare | 96 | 376 | 37 | 59 | 9% | 5 of 96 | same |
| vllm-expert-ngram | 96 | 380 | 37 | 59 | 8% | 5 of 96 | same |
| vllm-expert-kv-fp8 | 96 | 355 | 42 | 54 | 9% | 6 of 96 | same |
| sglang-expert-ngram | 96 | 375 | 38 | 58 | 9% | 6 of 96 | same |
| sglang-expert-kv-fp8-e4m3 | 96 | 386 | 35 | 61 | 9% | 5 of 96 | same |
| sglang-expert-torch-compile | not recorded | | | | | | unknown: sglang-expert-torch-compile recorded no answers at 8 users |
| autotune-vllm | 96 | 378 | 37 | 59 | 9% | 6 of 96 | same |
| llm-optimizer-vllm | 96 | 380 | 37 | 59 | 9% | 6 of 96 | same |
| llm-optimizer-sglang | 96 | 376 | 38 | 58 | 9% | 5 of 96 | same |

Answers graded on `gsm8k` (1319 items, greedy), each against its engine's stock row, allowed to lose at most 3 points. The stock rows answer twice; the floor is how many items their two passes disagreed on:

| Row | Right | Against | Lost | Gained | Lower bound | Floor | Check |
|---|---|---|---|---|---|---|---|
| rigtune | 1249 of 1319 | vllm-default | 11 | 9 | -0.9 pts | 15 of 1319 | holds |
| vllm-default | 1251 of 1319 | - | - | - | - | 15 of 1319 | holds |
| sglang-default | 1253 of 1319 | - | - | - | - | 8 of 1319 | holds |
| vllm-bare | 1253 of 1319 | vllm-default | 6 | 8 | -0.5 pts | 15 of 1319 | holds |
| sglang-bare | 1246 of 1319 | sglang-default | 12 | 5 | -1.2 pts | 8 of 1319 | holds |
| vllm-expert-ngram | 1249 of 1319 | vllm-default | 11 | 9 | -0.9 pts | 15 of 1319 | holds |
| vllm-expert-kv-fp8 | 1248 of 1319 | vllm-default | 11 | 8 | -1.0 pts | 15 of 1319 | holds |
| sglang-expert-ngram | 1249 of 1319 | sglang-default | 10 | 6 | -1.0 pts | 8 of 1319 | holds |
| sglang-expert-kv-fp8-e4m3 | 1250 of 1319 | sglang-default | 11 | 8 | -1.0 pts | 8 of 1319 | holds |
| sglang-expert-torch-compile | 0 of 0 | - | - | - | - | - | not checked |
| autotune-vllm | 1256 of 1319 | vllm-default | 5 | 10 | -0.2 pts | 15 of 1319 | holds |
| llm-optimizer-vllm | 1249 of 1319 | vllm-default | 10 | 8 | -0.9 pts | 15 of 1319 | holds |
| llm-optimizer-sglang | 1256 of 1319 | sglang-default | 3 | 6 | -0.3 pts | 8 of 1319 | holds |

**Overall: win on the 11 rows that could be compared; 5 not applicable, so not a win against every row**
_autotune-vllm: at up to 8 users, a batched-token budget of 4096 (a step here holds at most 8 x 135 + 8 = 1088 tokens); a context of 4096 (prompts here average 135 tokens, answers at most 512) cannot act, so on these settings it measures like its engine's stock row; measure at more users (compare --levels) to see them act_
_llm-optimizer-vllm: at up to 8 users, a batch cap of 64 (it binds above 64 users); a batched-token budget of 16384 (a step here holds at most 8 x 135 + 8 = 1088 tokens); a context of 4096 (prompts here average 135 tokens, answers at most 512) cannot act, so on these settings it measures like its engine's stock row; measure at more users (compare --levels) to see them act_
_llm-optimizer-sglang: at up to 8 users, a batch cap of 128 (it binds above 128 users); a batched-token budget of 4096 (a step here holds at most 8 x 135 + 8 = 1088 tokens); a context of 4096 (prompts here average 135 tokens, answers at most 512) cannot act, so on these settings it measures like its engine's stock row; measure at more users (compare --levels) to see them act_
_sglang-expert-torch-compile's answers were not checked: not graded: its engine never answered the items_
_vllm-expert-ngram is an expert row: vllm-default plus `--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'`: "This document shows how to use [Speculative Decoding](https://arxiv.org/pdf/2302.01318) with vLLM to reduce inter-token latency under medium-to-low QPS (queries per second), memory-bound workloads." (docs/features/speculative_decoding/README.md, introduction, read at v0.29.0 and v0.33.0); "| N-gram | Low to medium gain | Medium gain | Lightweight and easy to enable. |" (docs/features/speculative_decoding/README.md, Method Selection at a Glance, read at v0.29.0 and v0.33.0)_
_vllm-expert-kv-fp8 is an expert row: vllm-default plus `--kv-cache-dtype fp8`: "Quantizing the KV (Key-Value) cache to FP8 format can significantly reduce its memory footprint. This optimization enables you to store more tokens in memory, leading to improved throughput and support for longer context windows." (docs/features/quantization/quantized_kvcache.md, FP8 KV Cache Overview, read at v0.29.0 and v0.33.0)_
_vllm-expert-kv-fp8: this card does not have vllm's fp8 KV cache as rigtune runs it, so the row uses fp8_e5m2: `--kv-cache-dtype fp8_e5m2`: "`kv_cache_dtype="fp8_e5m2"`: Supported on CUDA 11.8+" (docs/features/quantization/quantized_kvcache.md, Additional kv_cache_dtype Options, read at v0.29.0 and v0.33.0); on a card where rigtune does not run vLLM's fp8 (e4m3) cache, Ampere among them, the row uses the page's other fp8 format, which rigtune runs there through FlashInfer (rigtune/backends/vllm.py AMPERE_FP8_KV)_
_sglang-expert-ngram is an expert row: sglang-default plus `--speculative-algorithm NGRAM`: "**No extra model available**: Use **NGRAM** (`--speculative-algorithm NGRAM`, CUDA-only)." (docs/docs/advanced_features/speculative_decoding.mdx, Quick guidance, read at v0.5.19 and v0.5.23)_
_sglang-expert-kv-fp8-e4m3 is an expert row: sglang-default plus `--kv-cache-dtype fp8_e4m3`: "Quantized KV cache is a memory optimization technique that primarily benefits throughput by allowing more tokens to be cached, but may introduce minimal accuracy degradation depending on the quantization format used." (docs/docs/advanced_features/quantized_kv_cache.mdx, introduction, read at v0.5.19 and v0.5.23); "**Choose the right format**: Use `fp8_e4m3` for better accuracy (recommended), `fp8_e5m2` for larger dynamic range, or `nvfp4` / `fp4_mx_block16` for maximum memory savings (experimental)" (docs/docs/advanced_features/quantized_kv_cache.mdx, best practices, read at v0.5.19 and v0.5.23)_
_sglang-expert-torch-compile is an expert row: sglang-default plus `--enable-torch-compile`: "`torch.compile` accelerates small models on small batch sizes. You can enable it with `--enable-torch-compile`." (docs/docs/advanced_features/hyperparameter_tuning.mdx, Try other options, read at v0.5.19 and v0.5.23)_
_autotune-vllm is a tuner row: vLLM benchmarks/auto_tune/auto_tune.sh at 4ba787ec9959, patched to judge p95 time to first token <= 500 ms, rigtune's ceiling (the patch is auto_tune.diff beside its output); INPUT_LEN=63, OUTPUT_LEN=379, MAX_MODEL_LEN=4096; its search ran with --load-format dummy (random weights), random fixed-length prompts with --ignore-eos, and open-loop load; this row is re-measured with the real weights_
_llm-optimizer-vllm is a tuner row: BentoML llm-optimizer at 7be87d7, --framework vllm: its highest output throughput among 22 measured runs that met [{"op": "<", "value": 500.0, "name": "p95_ttft_ms"}, {"op": "<", "value": 50.0, "name": "mean_itl_ms"}], on its own client with random prompts of 63 tokens and answers of 379; its card 'A100'; this row is re-measured on the comparison's prompts; its search was cut off at the calibration's budget of 3600 s, so this is its pick among the 22 runs it finished, made by its own end-of-run code; its own figure: 664.2 output tok/s at max_concurrency 8; its server command: vllm serve Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000 --max-num-seqs=64 --max-num-batched-tokens=16384 --max-model-len=4096_
_llm-optimizer-sglang is a tuner row: BentoML llm-optimizer at 7be87d7, --framework sglang: its highest output throughput among 27 measured runs that met [{"op": "<", "value": 500.0, "name": "p95_ttft_ms"}, {"op": "<", "value": 50.0, "name": "mean_itl_ms"}], on its own client with random prompts of 63 tokens and answers of 379; its card 'A100'; this row is re-measured on the comparison's prompts; its own figure: 672.7 output tok/s at max_concurrency 8; its server command: /workspace/venvs/sglang/bin/python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --host 127.0.0.1 --port 30000 --max-running-requests=128 --chunked-prefill-size=4096 --context-length=4096_
_vllm-expert-batched-tokens (expert) was not measured: not on an A100: vLLM's own source: "Setting large `max_num_batched_tokens` for A100 reduces throughput, see PR #17885 for more details." (vllm/engine/arg_utils.py EngineArgs.get_batch_defaults, NOTE(Kuntai), v0.29.0 and v0.33.0), so the row would not be what the page recommends_
_sglang-expert-eagle3 (expert) was not measured: rigtune lists no EAGLE-3 head for Qwen/Qwen3.5-9B (rigtune/speculative.py EAGLE3_HEADS)_
_aiconfigurator-vllm (tuner) was not measured: AIConfigurator exited 1; see output.log_
_aiconfigurator-sglang (tuner) was not measured: AIConfigurator exited 1; see output.log_
_rigtune's pick was calibrated on `file-dolly-d03d5896`; every row here was measured on `file-dolly-heldout-fc5b6407`, which calibration never saw_
_every row was measured 3 times, interleaved; tok/s is the median run's, and the spread is (max - min) / median_
