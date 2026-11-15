# Changelog

## Unreleased

- Cached profiles are reused only where they answer the request: each records the constraints it was scored with
  and the Hub commit of the model it measured, and is checked against both, and against the GPU driver, CUDA and
  engine library versions, when reused. Profiles written by earlier releases carry no constraints, so each one is
  recalibrated once, on first use; the log names the reason whenever a cached profile is not reused.
