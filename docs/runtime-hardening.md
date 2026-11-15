# Calibration and serving safeguards

Controls added after an outside review. Every one is opt-in or changes nothing that worked before; none has run on a GPU yet, only against fake engines in the test suite.

## Calibration

```bash
rigtune bench Qwen/Qwen2.5-3B-Instruct --workload chat \
  --strict-constraints --confirm-rounds 3 --min-requests 72
```

- `--strict-constraints` fails the calibration when the final pick misses the TTFT and TPOT ceilings or the throughput floor, rather than serving the least-violating configuration. Only the final choice is strict, so the stages still try the variants of the least-violating leader, which is where a configuration that meets the targets is usually found. It does not certify an SLO: the ceilings are judged on the measured percentile.
- `--confirm-rounds N` re-measures each shortlisted configuration N times and judges each on its median run. One engine on one GPU only; with a multi-GPU `--layout` or `--phases` it is refused before calibrating.
- `--min-requests N` sends at least N requests at every measured level, multi-GPU layouts included. At p95, 32 requests leave the second-slowest deciding; 72 is the fewest for which the order statistics give a two-sided 95% interval for the percentile. Selection still uses the measured percentile and the close-call screen, not that interval. A prompt source must hold fresh prompts for every level and a possible repeat (three levels at 72 need 432); session files are refused.

Each adds calibration time, and each is part of the profile's cache key, so a profile calibrated with one is never served where it was not asked for. The search remains heuristic and can miss interactions between settings. Details: [usage](usage.md#search-options).
