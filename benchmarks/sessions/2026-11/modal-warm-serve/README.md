# Starting the calibrated winner warm: same machine, and a fresh one given only its caches

Run on 20 November 2026 with [modal_warm_serve.py](../../../modal_warm_serve.py) on Modal A100 40GB: vLLM 0.33.0,
Qwen/Qwen3.5-9B at revision c202236, the A100 calibration's pick (bf16, MTP at 3 tokens, `--max-num-seqs 64`,
`--max-model-len 8192`, `--gpu-memory-utilization 0.95`). Every cache the engine writes was under one HOME
(vLLM's compile and AOT caches and startup plan, FlashInfer's, Inductor's via `TORCHINDUCTOR_CACHE_DIR`, and
Triton's via `TRITON_CACHE_DIR`), so the bundle is that folder as a tarball. Raw output in
[results.json](results.json); logs and the bundle are in the Modal volume `rigtune-triton-cache` under
`warm-serve/`. One measurement per run.

| Run | Ready | AOT load | Startup plan | Warm-up | torch.compile | CUDA graphs | KV tokens | tok/s, 8 users |
|---|---|---|---|---|---|---|---|---|
| 1 cold, empty HOME | 280.2 s | no | | 74.55 s | 21.58 s | 12 s | 207,971 | 1040.1 |
| 2 warm, same machine (saves the plan) | 106.1 s | yes | | 0.38 s | 0.62 s | 10 s | 239,153 | 1038.1 |
| 3 warm, plan applied | 96.1 s | yes | applied | 0.26 s | 0.52 s | 12 s | 231,490 | 1040.6 |
| 4 fresh machine, bundle only, `VLLM_FORCE_AOT_LOAD=1` | 140.1 s | yes | applied | 3.28 s | 1.79 s | 14 s | 231,490 | 1035.3 |

The bundle is 140 MB.

- A warm start on the same machine takes 106 s against 280 s cold, at the same decode speed, and with 15% more KV
  cache: a cold start loses it to its first-time autotuning, as in
  [modal-triton-cache](../modal-triton-cache/README.md).
- vLLM's startup plan (`VLLM_ENABLE_STARTUP_PLAN=1`) saves 10 s more by skipping memory profiling, but its KV
  value is 3.2% below the warm start's measurement (231,490 against 239,153 tokens): it stores the value that fits
  after the CUDA graphs' measured memory, where profiling estimates them. The plan was saved by the warm start
  (run 2); one saved by a cold start would keep the cold start's 15% smaller cache on every later start.
- A fresh machine given only the 140 MB bundle starts in 140 s, half the cold start, loads its compiled model
  from the copied AOT cache (strict mode on, so a failed load would have been an error), applies the copied plan
  and gets exactly run 3's KV cache, at the same decode speed.
- It is still 44 s slower than a warm restart on the same machine, although compile and warm-up take 5 s. Not
  measured here: likely the weights read from the volume for the first time on that host, and cold imports.
