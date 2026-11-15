# Changelog

## Unreleased

- `/health` now answers 503 while the engine is alive but not ready (restarting, or after three missed health
  checks in a row); it answered 200 for any live process before. Its body gains `backend.ready` beside
  `backend.alive`. Start-up and health checks count only a 2xx answer from the engine as ready.
- Cached profiles are reused only where they answer the request: each records the constraints it was scored with
  and the Hub commit of the model it measured, and is checked against both, and against the GPU driver, CUDA and
  engine library versions, when reused. Profiles written by earlier releases carry no constraints, so each one is
  recalibrated once, on first use; the log names the reason whenever a cached profile is not reused.
