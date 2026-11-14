**NVIDIA A100-SXM4-80GB** · meta-llama/Llama-3.2-3B-Instruct · workload `default` · objective `balanced` (TTFT ≤ 500 ms, TPOT ≤ 100 ms)

| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |
|---|---|---|---|---|---|---|---|---|---|
| **rigtune** → vllm/fp8/ctx4096/b64/gmu0.90 | 270 @c4 (median of 3, spread 20.0%) | 293 | - | 1.9 | 9.8 GB | 250 | 0.002 | yes | 1234s / 22 |
| vllm-default (vllm/bf16/ctx131072/b256/gmu0.90) | 62 @c1 (median of 3, spread 20.0%) | 535 | - | 2.0 | 9.8 GB | 250 | 0.002 | no | - |
| vllm-fp8-default (vllm/fp8/ctx131072/b256/gmu0.90) | 84 @c1 (median of 3, spread 20.0%) | 535 | - | 1.5 | 9.8 GB | 250 | 0.001 | no | - |

_every row was measured 3 times, interleaved; tok/s is the median run's, and the spread is (max - min) / median_
