# Changelog

## Unreleased

- The proxy reads streamed replies event by event, so token counts split across network chunks are no longer
  lost, and times TTFT at the first event with generated output rather than the first chunk (usually a role-only
  delta). Generated output includes `reasoning`, `reasoning_content` and `tool_calls`, in the proxy and in the
  calibration client alike. Calibration TTFT changes only where an engine returns those fields: session replay of
  a reasoning model whose engine separates its reasoning (vLLM with gpt-oss, llama.cpp's default reasoning format)
  now times the first reasoning token, where it timed the first answer token.
- The single-engine proxy answers an upstream timeout with 504 (it answered 502); other upstream errors still
  answer 502. `rigtune serve` gives it no upstream timeout, so this appears only for a caller of `create_app` that
  sets `request_timeout`.
- `/health` now answers 503 while the engine is alive but not ready (restarting, or after three missed health
  checks in a row); it answered 200 for any live process before. Its body gains `backend.ready` beside
  `backend.alive`. Start-up and health checks count only a 2xx answer from the engine as ready.
- Cached profiles are reused only where they answer the request: each records the constraints it was scored with
  and the Hub commit of the model it measured, and is checked against both, and against the GPU driver, CUDA and
  engine library versions, when reused. Profiles written by earlier releases carry no constraints, so each one is
  recalibrated once, on first use; the log names the reason whenever a cached profile is not reused.
- New opt-in calibration controls: `--strict-constraints`, `--confirm-rounds N` and `--min-requests N`
  ([safeguards](docs/runtime-hardening.md)). Without them calibration runs as before.
- With `--eval-set`, the configuration served is certified, KV cache and speculation included, or a certified
  fallback is served (`--eval-attempts`, default 3), or calibration fails. Gated profiles cached by earlier releases
  are recalibrated. `--eval-set` with a multi-GPU `--layout` or `--phases` is refused before calibrating.
