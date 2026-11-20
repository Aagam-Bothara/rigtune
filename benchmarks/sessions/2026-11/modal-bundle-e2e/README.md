# rigtune's own commands end to end: calibrate, bundle, serve warm on a fresh machine

Run on 20 November 2026 with [modal_bundle_e2e.py](../../../modal_bundle_e2e.py) on three Modal A100 80GB
containers: vLLM 0.33.0 (its own Docker image), Qwen/Qwen3.5-9B, rigtune at 3a41abf installed into vLLM's Python
(pip added only rigtune). Raw output in [results.json](results.json); logs, the profile and the bundle are in the
Modal volume `rigtune-triton-cache` under `bundle-e2e/`. One measurement per run.

1. Machine 1: `rigtune bench Qwen/Qwen3.5-9B --speculative all --budget 20m --save` (exit 0, 1,158 s, 9 trials),
   `rigtune bundle export`, then `rigtune serve --profile` on the same machine.
2. Machine 2, fresh: `rigtune bundle import`, then `rigtune serve --profile`.
3. Machine 3, fresh, no bundle: `rigtune serve --profile` (the control).

Ready is the time to the proxy's first 200 from `/health`; the rest is the start record `rigtune serve` logs.

| Run | Ready | Engine start | Compile cache | Kernel warm-up | CUDA graphs |
|---|---|---|---|---|---|
| Same machine, warm | 111.1 s | 107 s | hit | 1.0 s | 17 s |
| Fresh machine, bundle imported | 135.1 s | 129 s | hit | 3.6 s | 16 s |
| Fresh machine, cold | 297.2 s | 290 s | compiled (20 s) | 91.4 s | 17 s |

- The bundle is 55.5 MB: the profile and 2,724 kernel-cache files, all added on import. A fresh machine given it
  starts the pick in 135 s against 297 s cold, close to the 140 s of the hand-made 140 MB bundle in
  [modal-warm-serve](../modal-warm-serve/README.md), with the compiled model loaded from the cache and Triton's
  warm-up at 3.6 s instead of 91 s.
- `--profile` warned and went on where the machines differed in what does not change the pick: the GPU driver
  (610.57 with CUDA 13.3 against 620.70 with CUDA 13.5) and the host's RAM (847 against 1,024 GiB).
- Each serve answered "The capital of France is" with " Paris."

The pick is wrong in one way: plain bf16, `--max-num-seqs 256`, 613-616 tok/s at 8 users, with no MTP. rigtune
looked for vLLM's source only in `site-packages`, and this image installs it in `dist-packages`, so the model's
MTP head (1.7x on this model in the A100 session) was never offered. Fixed in aaaa9ad. The start-up times above
do not depend on it, but a calibration on this image needs re-running for its pick.
