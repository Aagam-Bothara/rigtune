# Calibration and serving safeguards

Controls added after an outside review. Every one is opt-in or changes nothing that worked before; none has run on a GPU yet, only against fake engines in the test suite.

## Serving

```bash
export RIGTUNE_API_KEY='replace-with-a-random-secret'
rigtune serve Qwen/Qwen2.5-3B-Instruct --require-api-key \
  --max-in-flight measured --idle-timeout 60 --max-body-mb 64 --body-timeout 30
curl localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $RIGTUNE_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

Without any of these options `rigtune serve` installs no limit at all and serves exactly as before: no request is refused, queued, timed out or capped by rigtune.

- **API key.** Read from `RIGTUNE_API_KEY`, or the variable `--api-key-env` names. When set, every route but `/health` requires `Authorization: Bearer <key>` (the scheme in any case), and `/health` answers only `status`, `alive` and `ready`. The key is not passed on to the engines, which rigtune launches without one. `--require-api-key` refuses to start without it.
- **Admission.** `--max-in-flight N` serves at most N inference requests at once. `measured` derives N from the profile: the busiest concurrency at which the pick met its constraints in calibration, capped by each engine's batch times the number of engines. Up to `--queue` more (default 2 × N) wait up to `--queue-timeout` seconds (default 30); beyond that, 429 with `Retry-After: 1`. A limit protects the engine; it does not promise the calibrated latency under real traffic.
- **Idle streams.** `--idle-timeout S` closes a stream that has sent nothing for S seconds since its last chunk, with a final SSE event carrying an `idle_timeout` error. The clock starts at the stream's first chunk of body, not its headers: engines send headers at once and the first token after queueing and prefill, so the time to first token is never counted as idle. A chunk counts once it has been sent, so a client that reads slowly does not trip it. There is no whole-request deadline: a long answer is not a stalled one.
- **Bodies.** `--max-body-mb M` refuses larger bodies with 413, chunked uploads included; `--body-timeout S` refuses with 408 a body that has not arrived in S seconds.

These are process-wide bounds, not per-client quotas: put TLS termination and per-client rate limits in a proxy in front. 

Two things change whatever options are given. `/health` reports liveness and readiness apart: it answers 503 while the engine is alive but not ready, that is while it restarts and once it has missed three health checks in a row (the number the supervisor waits for before restarting it), where it answered 200 for any live process before; a load balancer that probes `/health` takes such an engine out of rotation. And the single-engine proxy answers an upstream timeout with 504 instead of 502, which only a caller of `create_app` that sets `request_timeout` can see.

## Calibration

```bash
rigtune bench Qwen/Qwen2.5-3B-Instruct --workload chat \
  --strict-constraints --confirm-rounds 3 --min-requests 72
```

- `--strict-constraints` fails the calibration when the final pick misses the TTFT and TPOT ceilings or the throughput floor, rather than serving the least-violating configuration. Only the final choice is strict, so the stages still try the variants of the least-violating leader, which is where a configuration that meets the targets is usually found. It does not certify an SLO: the ceilings are judged on the measured percentile.
- `--confirm-rounds N` re-measures each shortlisted configuration N times and judges each on its median run. One engine on one GPU only; with a multi-GPU `--layout` or `--phases` it is refused before calibrating.
- `--min-requests N` sends at least N requests at every measured level, multi-GPU layouts included. At p95, 32 requests leave the second-slowest deciding; 72 is the fewest for which the order statistics give a two-sided 95% interval for the percentile. Selection still uses the measured percentile and the close-call screen, not that interval. A prompt source must hold fresh prompts for every level and a possible repeat (three levels at 72 need 432); session files are refused.

Each adds calibration time, and each is part of the profile's cache key, so a profile calibrated with one is never served where it was not asked for. The search remains heuristic and can miss interactions between settings. Details: [usage](usage.md#search-options).

## Answer quality of the configuration served

With `--eval-set`, the exact configuration served must pass the accuracy gate, its KV cache and speculation included, not only its weight precision; `--confirm off` does not skip the check. A configuration without grades of its own is graded on a run of its own. If the best one is not certified, the next best is tried, up to `--eval-attempts` (default 3) graded configurations, skipping ones that share a refused configuration's quantized KV cache or speculation method; the reference's own stage-1 configuration is always certified and costs no attempt. A fallback's speed is measured again and the notes say it is a fallback. If nothing can be certified within the attempts and the budget, calibration fails rather than serve an ungraded configuration. An unanswered item counts as lost in this check where the reference was right. Gated profiles cached before this change were checked to a weaker contract and are recalibrated. An eval set cannot be combined with a multi-GPU `--layout` or `--phases`, since those arrangements are never graded; the combination is refused before calibrating. Details: [usage](usage.md#search-options), [why](decisions.md).
