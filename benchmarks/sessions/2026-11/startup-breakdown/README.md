# Where calibration time goes (21 Nov 2026)

`benchmarks/startup_breakdown.py` over the trial ledgers and engine logs of four sessions, no GPU:

| Session | Card | Model | Engines | Calibrations | Engine logs |
|---|---|---|---|---|---|
| a40-* (16 Nov) | A40 48 GB | Qwen3.5-9B, Gemma 4 12B, Qwen2.5-7B, Llama 3.1 8B | vLLM 0.33.0, SGLang 0.5.23 | 7 | 48 |
| a100-qwen3.5-9b (18 Nov) | A100 SXM 80 GB, RunPod | Qwen3.5-9B | vLLM 0.33.0, SGLang 0.5.23 | 1 | 25 |
| modal-mtp-rerun (19 Nov) | A100 80 GB, Modal | Qwen3.5-9B | vLLM 0.33.0 | 1 | 10 |
| h200-qwen3.5-35b-a3b (20 Nov) | H200 141 GB, Modal | Qwen3.5-35B-A3B | vLLM 0.33.0 | 1 | 18 |

The ledgers and logs are each session's rigtune home (`ledger/`, `logs/`), kept with the session's raw results, not in
this repository. The RunPod sessions ran source older than the warm-up line (`warmup_s`) and the Triton cache fix.

```
python benchmarks/startup_breakdown.py ledger <home>/ledger
python benchmarks/startup_breakdown.py logs <home>/logs
```

## Engine start-up is most of a calibration

| Calibration | Total | Start-up | Measuring | Rest of each trial | Outside trials |
|---|---|---|---|---|---|
| H200, Qwen3.5-35B-A3B, 18 launches | 59 min | **92.6%** | 6.3% | 1.1% | 0% |
| Modal A100, Qwen3.5-9B, 10 launches | 18 min | **95.8%** | 2.7% | 1.5% | 0% |
| RunPod A100, Qwen3.5-9B, 25 launches | 58 min | **82.0%** | 14.0% | 4.0% | 0% |
| A40, Qwen3.5-9B, 21 launches | 58 min | **72.8%** | 23.4% | 3.8% | 0% |
| A40, Gemma 4 12B, 18 launches | 58 min | 55.3% | 24.2% | 4.2% | 16.2% |
| A40, Llama 3.1 8B (SGLang only), 16 launches | 38 min | 54.8% | 27.5% | 6.0% | 11.6% |
| A40, Qwen2.5-7B, 4 calibrations | 18-20 min each | 38-65% | 23-56% | 3-6% | 0-10% |

The Modal A100 rerun measured a short workload (3 s per trial), so its measuring share is lower than a normal run's.

## One start-up, phase by phase

From the engine logs' own timestamps; median seconds per launch.

**vLLM 0.33.0:**

| Phase | H200 | A100, Triton cache warm (Modal) | A100, Triton cache cold (RunPod) | A40 |
|---|---|---|---|---|
| Python + API server imports | 13 | 11 | 11 | 10 |
| Engine process spawn + imports | 26 | 22 | 24 | 39 |
| Distributed + model init | 2 | 4 | 3 | 3 |
| Weights load | 30 | 8 | 6 | 6 |
| KV layout + image-encoder profiling | 10 | 11 | 13 | 16 |
| torch.compile | 34 | 15 | 22 | 21 |
| Profiling/warm-up run (Triton) | 22 | **0** | **82** | 0 |
| CUDA graphs, memory-sizing pass | 17 | 11 | 14 | 14 |
| Kernel warm-up + CUDA graph capture | 18 | 10 | 18 | 20 |
| API server up | 1 | 1 | 1 | 3 |
| **Launch to ready** | **168** | **100** | **196** | **150** |

**SGLang 0.5.23:**

| Phase | A100 (RunPod) | A40 |
|---|---|---|
| Python + imports, args | 12 | 11 |
| Tokenizer/template + scheduler spawn | 11 | 11 |
| Distributed + model init | 3 | 3 |
| Weights load | 4 | 3 |
| KV/mamba pool allocation | 1 | 1 |
| CUDA graph capture (all sets) | 26 | 20 |
| Scheduler finish, HTTP server, warm-up request | 4 | 4 |
| **Launch to ready** | **60** | **55** |

## What it shows

- **Python imports are the largest fixed cost nobody removes:** 33-39 s of every vLLM launch, which imports torch
  and vLLM twice (the API server, then the engine process), and 22 s of every SGLang launch.
- **The Triton cache fix holds:** the GDN warm-up is 82 s on every RunPod launch (cache not kept) and 0 s on every
  Modal launch after the first (`benchmarks/sessions/2026-11/modal-triton-cache`).
- **The H200's warm-up stays at 21-25 s on every launch, cold or warm.** Not explained yet. The FlashInfer GDN
  prefill kernel (JIT, chosen on sm90 only) and the fused-MoE path are the suspects.
- **vLLM captures CUDA graphs twice per launch,** once to size their memory before the KV cache is allocated and
  once for real: the sizing pass alone is a median 11-17 s.
- **torch.compile rarely comes from its cache in a calibration:** 4 of 18 H200 launches, 2 of 10 Modal ones, since
  most settings change its cache key.
- **SGLang's teardown sleeps 5 s** before its crash diagnostics on every stop (part of "rest of each trial").
- **Unexplained:** the Gemma and Llama calibrations spent 567 s and 263 s outside trials.

For a faster calibration this means a warm vLLM launch is still about 100 s on an A100, so eight minutes hold about
four launches. Minutes are only reachable by starting from a matching profile, measuring several configurations per
launch, or cutting the fixed costs above; the order of trials alone cannot get there.

## What rigtune recorded, and the fixes

The ledgers' own start-up fields under-read three things, fixed in 770bc6c, 4358c18 and 80bbf3a:
- `compile_s` read only Inductor's compile-range lines: 18.88 s on an H200 launch where vLLM says 40.95 s in total.
- `capture_s` kept one of vLLM's two capture passes.
- SGLang's start-up lines were not read at all.

The ledger tables above are as recorded, before these fixes; the phase tables come from the logs and are not affected.
