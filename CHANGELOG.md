# Changelog

## Unreleased

- `--kernel K=V` on `bench`, `recalibrate`, `compare` and `serve` tries one named engine kernel setting (an
  attention backend, a CUDA-graph mode or cap, a compile level) on each tuned engine's leader and keeps it only on
  a measured win ([kernel settings](docs/usage.md#kernel-settings---kernel)). Opt-in and unmeasured: without it
  nothing changes.
- The trial ledger records each launched trial's start-up cost beside the trial: seconds to the first healthy
  answer, and for vLLM the CUDA-graph capture time and size, compile seconds, compile-cache hit, and any
  CUDA-graph mode the engine said it changed. Profiles are written exactly as before.
- New failure kind `kernel_refused`, for an engine that exits at start-up naming a kernel setting.
- `rigtune probe --engines` asks vLLM one more question (its graph and compile settings), so each install is
  asked again once.
- Every calibration that launches engines appends each of its trials to a trial ledger,
  `~/.rigtune/ledger/<hardware hash>/<model>.jsonl`, and the profile records its `calibration_id` there
  ([the trial ledger](docs/usage.md#the-trial-ledger)). On by default; `RIGTUNE_LEDGER=off` turns it off. It
  changes no trial and no pick, and a profile calibrated without it is written exactly as before.
- `rigtune ledger show`, `rigtune ledger verify <profile>` and `rigtune ledger rebuild <id> --out FILE` list
  the ledger's calibrations and trials, check a profile against its calibration's records (the pick re-derived
  from its table included), and write a lost profile back from the ledger, byte for byte.
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
- `rigtune serve` has an optional bearer API key and opt-in limits: `--max-in-flight N|measured` with a bounded
  queue, `--idle-timeout`, `--max-body-mb` and `--body-timeout`. All are off by default, so serving is unchanged.
- `starlette` and `anyio` are declared as direct dependencies (fastapi already installed them).
- CI pins the direct runtime dependencies (`constraints/core.txt`), keeps an unpinned job beside it, and
  type-checks the two new serving modules with mypy.
