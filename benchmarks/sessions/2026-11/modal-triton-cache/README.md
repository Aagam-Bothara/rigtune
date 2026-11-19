# A fixed TRITON_CACHE_DIR and Qwen3.5's GDN warm-up (19 Nov 2026)

`benchmarks/modal_triton_cache.py` on one Modal A100 40 GB (sm80), with the `vllm/vllm-openai:v0.33.0` image and
Qwen/Qwen3.5-9B: `--max-model-len 4096`, `--gpu-memory-utilization 0.90`, and `--max-num-seqs` as below. Every cache
was emptied before runs 1 and 4. `results.json` is the script's output. The timings come from vLLM's own log.

| Run | max_num_seqs | Triton cache | GDN warm-up (s) | torch.compile (s) | Engine init (s) | Ready (s) | Autotune lines | KV-cache tokens |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | fixed, empty | 73.37 | 26.93 | 135.11 | 314.2 † | 6 | 248,685 |
| 2 | 128 | fixed, from run 1 | **0.34** | 22.84 | **57.13** | **122.1** | 0 | 284,964 |
| 3 | 64 | fixed, identical relaunch | 0.36 | 0.59 | 25.94 | 88.1 | 0 | 288,768 |
| 4 | 128 | vLLM's default, empty | 73.98 | 25.98 | 141.62 | 224.1 | 6 | 244,882 |
| 5 | 96 | vLLM's default, from run 4 | **74.22** | 26.39 | **136.89** | **200.1** | 6 | 246,345 |

† Includes the one-time download of the weights into the Modal volume.

What it shows:
- **Today's behaviour (runs 4 and 5):** vLLM 0.33 keeps Triton's cache inside each configuration's torch.compile
  directory, whose hash includes `max_num_seqs`. So each new configuration compiles and autotunes the FLA GDN
  kernels again, about 74 s.
- **With `TRITON_CACHE_DIR` fixed (run 2),** a new configuration reuses them. The warm-up falls to 0.3 s, Triton
  autotunes nothing, and engine init falls from about 137 s to 57 s. torch.compile still runs for each new
  configuration, as its cache is keyed by design; only an identical relaunch (run 3) skips it.
- **Warm starts get about 15% more KV cache:** about 285-289k tokens against about 245-249k cold. A configuration
  measured with cold caches, as every calibration trial was before this, runs with less KV cache than it will
  have when served warm.
- **No side effect seen:** no "Cubin file saved by TritonBundler not found" warning, which one forum report
  links to setting the variable.

The script's `aot_loaded` field is false in every run, run 3 included, although run 3 clearly loaded the compiled
model (0.59 s). The text it looked for is not what vLLM 0.33 logs, so ignore that field.
