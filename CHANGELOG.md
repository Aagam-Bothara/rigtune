# Changelog

## Unreleased

- Every vLLM launch keeps Triton's kernels and autotune results in `~/.rigtune/kernel-cache/triton`
  (`TRITON_CACHE_DIR`), so a calibration no longer compiles and autotunes them again for each configuration. On
  Qwen3.5-9B on an A100, a new `max_num_seqs` started its engine in 57 s instead of 137 s, and with about 15% more KV
  cache ([the kernel cache](docs/usage.md#the-kernel-cache)). A `TRITON_CACHE_DIR` you set wins;
  `RIGTUNE_KERNEL_CACHE` moves the cache or turns this off. Profiles, the ledger and verdicts do not record it.
- The trial ledger records a vLLM 0.33 start that loaded the whole model from its AOT compile cache as a compile
  cache hit. It recorded no answer before, because 0.33 then prints no compile-range line: the A40 session's three
  Gemma 4 relaunches that compiled in 0.5 s were counted as unknown.
- The trial ledger's start-up record gains `warmup_s`: vLLM's first-forward profiling and warm-up run, where Triton
  compiles and autotunes on a cold cache. Read back from the A100 session's logs, it is 81-82 s on every
  Qwen3.5-9B launch and under 0.3 s on Gemma 4 12B.
- A trial with speculative decoding records its acceptance (`metrics.spec_acceptance`): the share of drafted
  tokens the model kept and the tokens per verification step. On vLLM it comes from the change in its `/metrics`
  counters over the workload; on SGLang, and on vLLM when those cannot be read, from the log lines written while the
  workload ran. The answers a gate or quality probe grades afterwards are never counted. Read from the A100
  session's logs, Qwen3.5-9B's MTP head kept 67% of 3 drafted tokens (3.0 tokens per step) and n-gram 5-10%. Gemma
  4 12B's assistant kept 50% with 1 drafted token, which is why it gained little. Trials without speculation are
  written exactly as before.
- A verdict's answer table gains a "Cut off" column: the answers that stopped at the token limit, graded wrong
  whatever they would have said. It was in verdict.json only; on the A100 comparison it is 19 to 27 per row. A
  verdict of a result from before the count shows "-", not 0.
- Under `--speculative all`, the leading engine is tuned first. Engines offering more strategies went first, which
  was meant for speculation on one engine only, but by then every engine has speculated. On the A100 calibration
  of Qwen3.5-9B, SGLang (937 tok/s with MTP) still offered an fp8 KV cache that vLLM refuses with MTP. So it was
  tuned first, and the hour ran out before vLLM's leader (1023) got a single tuning trial or its confirm round. A
  calibration with time for everything can now run fewer trials: an engine the tuned leader leaves more than 10%
  behind is skipped by the rule that already skipped an engine tuned second.
- `rigtune serve` logs how each engine start went, restarts included, from the engine's log as calibration
  records each trial: for vLLM, `vllm ready in 88 s: compile cache hit, kernel warm-up 0.3 s, CUDA graphs 23 s`,
  or what it compiled when its caches did not hold. The record is under `startup` in `/health`'s `backend` too
  (not in the minimal `/health` an API key leaves). A warm start was silent before, and so was a cold one.
- A `--budget` now holds back the time the confirm stage needs: a typical trial for each configuration it would
  re-measure (at most 3), for each `--confirm-rounds`. On the A100 calibration of Qwen3.5-9B the hour ran out
  in the tuning stages, so the pick was chosen on search runs alone and never re-measured, and the first of those
  had started with a cold kernel cache, which costs about 15% of the KV cache. A budgeted calibration now runs
  fewer tuning trials and always re-measures its pick. Without `--budget`, or with `--confirm off`, nothing
  changes.

## 0.3.0 (2026-11-18)

The release behind the November 2026 results: on Qwen3.5-9B and Gemma 4 12B, with vLLM 0.33.0 and SGLang 0.5.23,
rigtune's pick beat stock vLLM and SGLang, vLLM's `auto_tune`, BentoML's llm-optimizer and expert settings, with
answers checked on all 1,319 GSM8K items ([results](docs/benchmarks.md#2026-models-on-current-engines)).

Upgrading: profiles cached by 0.2.x are recalibrated once, on first use (see "Caching" below).

### Comparing against other setups

- `rigtune compare --verdict` scores a comparison as a win, tie or loss against every row, records every run as it
  is measured, and writes a verdict in Markdown and JSON with what it ran on. Each round starts one row later.
  `rigtune verdict` scores a comparison again, at other thresholds or from the runs of one that stopped early.
- `--expert` adds rows with the settings each engine's own tuning guide recommends (`rigtune/data/expert_rows.json`,
  each quoted with its page); `--bare-stock` adds vLLM and SGLang launched with the model and port alone; `--rows`
  adds other tools' picks (vLLM `auto_tune`, llm-optimizer, AIConfigurator), or records why one does not apply.
- `--check-answers gsm8k` grades every row's answers while its engine is up, every item unless `--check-items`
  says fewer, and no row wins on answers that got worse or went unchecked. A model whose chat template thinks is
  graded with thinking off, answers get 1,024 tokens, and the verdict says how many stopped at that limit.
- `--levels` measures every row at other numbers of users, and the verdict names the settings that cannot act at
  the levels measured.

### Accuracy

- `--eval-set` gates weight precisions and the final configuration on labelled accuracy, with a paired
  non-inferiority bound. GSM8K's test set is bundled as the default, with exact, contains, regex and choice
  graders. `--quant auto` includes pre-quantized checkpoints when an eval set gates them.
- The served configuration is certified, KV cache and speculation included, or a certified fallback is served
  (`--eval-attempts`, default 3), or calibration fails. The gate is graded with thinking off when the model's
  template thinks.

### Speculative decoding

- `--speculative all` offers a checkpoint's own MTP head (read from its files, fp8 heads where the engine takes
  them), n-gram, EAGLE-3 heads verified against their config, Gemma 4 assistants, and draft models on vLLM and
  SGLang, with room planned for the draft. Engines are chosen after the speculation stage.
- `--drafts verified` and `--draft` check a second model against its target from their files and pin the pair;
  verdicts and evidence are kept atomically and read from the Hub, local directories or the cache (`--hub off`).
- Suffix decoding is offered only where arctic-inference's suffix tree loads in vLLM's own Python, and a failed
  start names the cause.

### Models, workloads and goals

- 2026 hybrid-attention models (Qwen3.5, Gemma 4 and others) are planned with KV memory per layer type, even when
  the engine's registry cannot be imported. Image, audio and video input is refused when a text-only pick serves.
- Session replay: `--sessions`, `--session-pool`, `--session-turns`, `--history` and `--session-system` replay
  multi-turn conversations turn by turn, and files may give each request's full message list. Prefix reuse is
  measured and reported by kind of request.
- `--kv-tier` tries a CPU KV-cache tier (vLLM's native one, SGLang's hierarchical cache) only when the working set
  outgrows the GPU pool, after checking host memory.
- Goals: `--optimise` one metric within `--limit` bounds, for goals no preset expresses. `cost_per_mtok` prices a
  goal at `--gpu-hourly-usd` and `--kwh-usd`. `rigtune tradeoffs` shows what a goal's limits cost from the recorded
  trials.
- `--arrival-rate` scores efficiency per request at a stated rate, from measured idle power, wind-down and the
  NVML energy counter. `rigtune serve` reports the live rate and says when it differs.

### Engines

- Checked with vLLM 0.11.0, 0.29.0 and 0.33.0, and SGLang 0.5.19 and 0.5.23. Other releases get a warning, not a
  refusal. `rigtune conformance` checks an engine release on a GPU and writes the record.
- Feature checks ask each engine's own Python for its flags, architectures and speculative methods
  (`rigtune probe --engines`), falling back to its source text.
- `rigtune selfcheck` exercises the load client against a fake engine at 64 and 128 users; above 100 users each
  user gets its own connection.
- An engine's model-code failure at start-up is recorded as a load failure and not retried; vLLM's multimodal
  profiling failure is retried text-only.

### Kernel settings

- `--kernel K=V` on `bench`, `recalibrate`, `compare` and `serve` tries one named engine kernel setting (an
  attention backend, a CUDA-graph mode or cap, a compile level) on each tuned engine's leader and keeps it only on
  a measured win ([kernel settings](docs/usage.md#kernel-settings---kernel)). Opt-in and unmeasured: without it
  nothing changes.
- New failure kind `kernel_refused`, for an engine that exits at start-up naming a kernel setting.
- `rigtune probe --engines` asks vLLM one more question (its graph and compile settings), so each install is
  asked again once.

### Trial ledger

- Every calibration that launches engines appends each of its trials to a trial ledger,
  `~/.rigtune/ledger/<hardware hash>/<model>.jsonl`, and the profile records its `calibration_id` there
  ([the trial ledger](docs/usage.md#the-trial-ledger)). On by default; `RIGTUNE_LEDGER=off` turns it off. It
  changes no trial and no pick, and a profile calibrated without it is written exactly as before.
- The ledger records each launched trial's start-up cost beside the trial: seconds to the first healthy answer,
  and for vLLM the CUDA-graph capture time and size, compile seconds, compile-cache hit, and any CUDA-graph mode
  the engine said it changed. Profiles are written exactly as before.
- `rigtune ledger show`, `rigtune ledger verify <profile>` and `rigtune ledger rebuild <id> --out FILE` list
  the ledger's calibrations and trials, check a profile against its calibration's records (the pick re-derived
  from its table included), and write a lost profile back from the ledger, byte for byte.

### Caching

- Cached profiles are reused only where they answer the request: each records the constraints it was scored with
  and the Hub commit of the model it measured, and is checked against both, and against the GPU driver, CUDA and
  engine library versions, when reused. Profiles written by earlier releases carry no constraints, so each one is
  recalibrated once, on first use; the log names the reason whenever a cached profile is not reused.

### Serving

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
- `rigtune serve` has an optional bearer API key and opt-in limits: `--max-in-flight N|measured` with a bounded
  queue, `--idle-timeout`, `--max-body-mb` and `--body-timeout`. All are off by default, so serving is unchanged.

### Calibration controls

- New opt-in calibration controls: `--strict-constraints`, `--confirm-rounds N` and `--min-requests N`
  ([safeguards](docs/runtime-hardening.md)). Without them calibration runs as before.
- `--eval-set` with a multi-GPU `--layout` or `--phases` is refused before calibrating.

### Packaging and CI

- `starlette` and `anyio` are declared as direct dependencies (fastapi already installed them).
- CI pins the direct runtime dependencies (`constraints/core.txt`), keeps an unpinned job beside it, and
  type-checks the two new serving modules with mypy. The release workflow also checks that the bundled GSM8K set
  is in the wheel.
