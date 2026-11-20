# The 20-minute calibration again, with vLLM's source found: MTP offered and picked

Run on 20 November 2026 with [modal_calibrate.py](../../../modal_calibrate.py) on a Modal A100 80GB: vLLM 0.33.0
(its own Docker image), Qwen/Qwen3.5-9B, rigtune at 7ea5e20, `rigtune bench Qwen/Qwen3.5-9B --speculative all
--budget 20m --save`. The command's last lines are in [results.json](results.json); the full log and rigtune's
home are in the Modal volume `rigtune-triton-cache` under `runs/qwen9b-a100-bench/`. One measurement per trial.

The same command in [modal-bundle-e2e](../modal-bundle-e2e/README.md), at 3a41abf on the same image, picked plain
bf16: rigtune looked for vLLM's source only in `site-packages`, the image installs it in `dist-packages`, and MTP
was never offered. With aaaa9ad it is:

| Stage | Configuration | tok/s, 8 users | TTFT p95 | J/token |
|---|---|---|---|---|
| quant | bf16, ctx 4096, 64 seqs | 610.6 | 175 ms | 0.517 |
| kv | + fp8_e5m2 KV cache | 608.2 | 177 ms | 0.515 |
| spec | + n-gram, 4 tokens | 538.0 | 177 ms | 0.610 |
| spec | + MTP, 1 token | 843.1 | 173 ms | 0.391 |
| spec | + MTP, 3 tokens | **887.2** | 175 ms | 0.356 |
| memory | ctx 8192 | 610.2 | 177 ms | 0.512 |
| batch | 16 seqs / 256 seqs | 610.7 / 613.0 | 174 / 169 ms | 0.532 / 0.526 |
| prefill | MTP 3, `--max-num-batched-tokens 2048` | 874.1 | 177 ms | 0.373 |
| confirm | MTP 3, re-measured | 866.8 | 179 ms | 0.382 |

- The pick is MTP at 3 tokens: 867 tok/s re-measured against 611 for the best configuration without speculation,
  1.42x on rigtune's default workload (16 prompts, 256 in, 128 out), at a third less energy per token.
- The budget ran out after 10 trials (1,094 s); 14 prefill trials were skipped. The pick was re-measured once.
- The confirm note says "re-measured the best 3 configurations" although one was: it counts the shortlist, not
  the re-measurements. To be fixed.
