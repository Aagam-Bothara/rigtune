# rigtune's pick against stock vLLM, checked by Giraffe

Run on 20 November 2026 with [modal_giraffe.py](../../../modal_giraffe.py) on one Modal A100 80GB: vLLM 0.33.0,
Qwen/Qwen3.5-9B at revision c202236, and [Giraffe](https://github.com/0dust/giraffe) 0.1.0a2, an endpoint test
suite by another developer. Giraffe ran its default checks (access, serving, correctness, first_output: 14
requests, answers up to 2,048 tokens) on stock vLLM, saved that run as a baseline, and ran the same checks on
rigtune's A100 pick (bf16, MTP at 3 tokens, `--max-num-seqs 64`, `--max-model-len 8192`) against it. Raw output in
[results.json](results.json); Giraffe's report.json and report.html are in the Modal volume `rigtune-triton-cache`
under `giraffe/`.

| | Stock vLLM | rigtune's pick |
|---|---|---|
| Checks | 4 of 4 passed | 5 of 5 passed, the baseline comparison included |
| Correctness | pass | pass, 6 of 6 |
| Request throughput | 1.15 req/s | 2.57 req/s |
| Response latency | 2.78 s | 1.18 s |
| Time to first token, client / server | 131 / 94.1 ms | 123 / 82.4 ms |
| Against the stock baseline | | no material regression in 37 comparable metrics; 23 not comparable |

A smoke test, not a benchmark: 14 requests, and Giraffe's default checks do not measure output-token throughput or
time per output token ("not measured", "insufficient evidence"); its `load` check would. It agrees with rigtune's
own comparison (1.89x stock vLLM at 8 users on held-out prompts) in direction and size, measured by a tool rigtune
did not write. For the pick, Giraffe noted the server "answered one at a time, so concurrent correctness was not
tested".
