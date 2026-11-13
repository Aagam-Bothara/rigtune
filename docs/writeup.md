# rigtune: a benchmark-driven configuration planner for LLM serving

*Technical notes — memory planner, staged search, energy objective.*

## 1. Problem

Serving one model on one machine means choosing a backend (vLLM, SGLang, llama.cpp, …), a precision or quant, a memory budget, a context length and a batch/concurrency limit. The space is small enough to enumerate (~144 points for one model on one box) and large enough that nobody does; people copy defaults, and defaults are tuned for the wrong GPU. rigtune turns this into a measured decision that runs once per (machine, model, objective) and is cached.

The design has three parts that are reusable beyond this project: a **memory planner** that removes configurations that cannot run, a **staged search** that measures a few dozen configurations instead of hundreds, and a set of **constrained-argmax objectives** including an energy one, all measured by the same tracer (llmtrace). Of the three, the search order is the one the evidence supported least: given the same time, random sampling of the same configurations did as well on two GPUs and better on a third, until the search tried speculative decoding first; on that GPU it then matched random search's best in two-thirds of the time, and on the other two it held the tie and calibrated a third faster (§3, §8). Most of the results come from which configurations exist to be tried and how carefully each is measured.

## 2. Memory planner

Every configuration `cfg` is scored *before launch* with an additive model:

```
estimated(cfg) = weights(quant)
               + kv_cache(ctx, batch, kv_dtype)
               + runtime_workspace(backend)
               + safety_margin
keep cfg  iff  estimated(cfg) <= 0.95 * available
```

- `weights`: the GGUF file size reported by the Hub (or the converted file), or `params × bytes(dtype)` for HF weights. Parameter counts come from safetensors metadata when published, otherwise from a dense-transformer estimate (within 3% for Llama-3.2-3B).
- `kv_cache`: `2 × layers × kv_heads × head_dim × kv_tokens × bytes(kv_dtype)`. For multi-head attention this equals the textbook `2 × layers × hidden × …`; for GQA models it is the (smaller) correct number. Hybrid-attention models are planned per layer type (below).
- `runtime_workspace`: one empirical constant per backend (vLLM 1.5 GiB, SGLang 2 GiB, llama.cpp 768 MiB / 512 MiB CPU). It covers CUDA context, CUDA graphs and activation buffers; it is the only fitted quantity in the model.
- `safety_margin`: `max(5% of available, 512 MB)`.
- `available`: *free* VRAM (or RAM) at probe time, not total. For vLLM/SGLang the budget is additionally capped at `gpu_memory_utilization × total`, because that is what the engine will refuse to exceed.

**KV tokens is backend-specific.** llama.cpp allocates the whole `-c` context up front, and `-c` is shared across `-np` slots, so `kv_tokens = ctx × n_parallel` exactly. Paged-KV runtimes (vLLM, SGLang) allocate a pool and preempt requests when it fills; requiring `ctx × max_num_seqs` tokens would reject every realistic batch on a 24 GB card. rigtune budgets `0.25 × ctx × max_num_seqs` for them (`PAGED_KV_FRACTION`). This keeps the planner honest about what must fit (weights + workspace + a useful pool) without pretending the engine cannot run below worst-case occupancy.

**Hybrid attention layouts.** Qwen3.5 interleaves 3 Gated DeltaNet (linear-attention) layers with every full-attention layer, and Gemma 4 and Gemma 3 interleave 5 sliding-window layers (window 1024) with every full one. Gemma 4's full layers also use a different shape: one KV head of 512 against eight of 256. The formula above counts every layer as full. That is too high at long contexts and too low at large batches, because each running sequence also holds a recurrent state of fixed size (49 MiB for Qwen3.5-9B: per layer, a bf16 conv state of 8192 × 3 and a float32 state of 32 × 128 × 128). rigtune reads `layer_types` (or `full_attention_interval`, or Gemma 3's `sliding_window_pattern`, derived the way transformers derives them). With `tokens` the paged budget above and `seqs` = `max_num_seqs`, it plans:

```
kv_cache = full_bytes/token    × tokens
         + sliding_bytes/token × min(tokens / seqs, window) × seqs
         + state_bytes         × state_slots
```

The state term gets no paged fraction, because a sequence holds all of its state however short it is. Then each engine's pools, read from source (file and function cited in `rigtune/backends/vllm.py` and `sglang.py`):

- **vLLM 0.29** keeps every layer type in one shared block pool (the hybrid KV cache manager, on by default on CUDA), frees sliding blocks that leave the window, and gives each running request one state block per linear layer. Prefix caching stays on for hybrid models. `state_slots = max_num_seqs`, times `1 + k` with a speculation method proposing k tokens, since each speculative token gets its own state block, and the conv state grows by one column per token (for suffix decoding's 24, 25 states of 61 MB a sequence on Qwen3.5-9B). Allocation is in blocks: with linear layers the attention block grows until an attention page holds a state (528 tokens for Qwen3.5-9B, 1056 with an fp8 cache), and layer types are grouped in sets as large as the smallest type, padded (gemma-3-27b's 52 sliding layers take 60 pages). Sliding layers also keep the tokens scheduled but not yet settled: 2 × `max_num_batched_tokens` (2048 by default on a 24 GB card), counted once. vLLM refuses to start unless one request at `max_model_len` fits, so `kv_cache` is never planned below that.
- **SGLang 0.5.19**, with its radix cache (the default), gives the recurrent state its own pool: 0.9 / 1.9 of the cache budget, and five slots per running request for Qwen3.5 (three, plus the extra buffer it picks under its default overlap scheduler), so a planned batch is one SGLang will run. Its sliding pool is 0.8 × the full pool's tokens whatever the context, but only for the architectures it gives that pool to (Gemma 4, gpt-oss and Granite SWA): it keeps every Gemma 2 and Gemma 3 layer full, and so does rigtune's plan for it. When the state pool is short, SGLang lowers `max_running_requests` rather than failing. With `--disable-radix-cache` it holds `max_running_requests + 1` states, and `window + 130` sliding tokens per request plus two prefill chunks. Speculation adds an intermediate state per draft token for every request slot (12 for n-gram, 4 for an EAGLE3 head or draft model).

Neither engine needs a flag for these models, and both list `Qwen3_5ForConditionalGeneration` and `Gemma4UnifiedForConditionalGeneration` in their registries. Dense models take the old path byte for byte (a test compares every vLLM and SGLang grid with estimates captured before this change). A layout rigtune does not recognise (Mamba-2 layers, KV-shared layers) keeps the every-layer-full estimate, with a warning and a note in `rigtune plan`, as do llama.cpp and vLLM on CPU, whose rules were not checked. Gemma 2 configs carry no layer pattern, so Gemma 2 stays every-layer-full everywhere, which overstates vLLM's need. The vLLM rules hold for its defaults only: a KV connector without hybrid support turns the hybrid manager off (every layer full), and `VLLM_KV_CACHE_LAYOUT` set to a block-outer layout packs mixed page sizes differently; neither is modelled. When a speculation setting does not fit at the pick's batch because of state, the profile says so; the search does not lower the batch for it. **This has not run on hardware.** The planned sizes change a lot (computed, not measured). For Qwen3.5-9B on vLLM at 4k context and batch 64, `kv_cache` falls from 8 GiB to 5.2 GiB (2.1 GiB of KV and 3.1 GiB of state); at 32k and batch 16 it falls from 16 GiB to 4.9 GiB. Batch 256 and 512 need 12 and 25 GiB of state alone, so on a 24 GB card with fp8 weights they drop out even at short contexts. For Gemma 4 12B at 32k and batch 16 it falls from 48 GiB to 8.3 GiB. *Predictions* for one RTX 4090 with vLLM 0.29: no admitted configuration of either model is refused by vLLM's own start-up check (one request at `max_model_len`), as long as vLLM's non-KV memory stays within the planned workspace and margin. vLLM logs `Setting attention block size to 528 tokens` for Qwen3.5-9B with a bf16 cache. With SGLang at its defaults, Qwen3.5-9B at the planned batch does not log `max_running_requests is capped to N by the mamba state cache`, and its `Mamba Cache is allocated` line gives conv and ssm state sizes whose sum, divided by `max_mamba_cache_size + 1`, is 49 MiB. Refuted if an admitted configuration fails vLLM's start-up check, if SGLang caps an admitted batch, or if that per-slot state is more than 10% away from 49 MiB.

**Partial offload.** For llama.cpp with `n_gpu_layers < layers`, the GPU estimate is scaled by the offloaded fraction and the remainder is checked against host RAM with its own margin. Both must fit. Free host RAM is what the process can use: inside a container, `/proc/meminfo` shows the host's memory, so the probe caps it by the cgroup's limit (v2 `memory.max`, v1 `memory.limit_in_bytes`) less what the container uses beyond its file cache (active and inactive page cache are reclaimed before the limit is enforced, and a fresh weight download is mostly that), as it caps CPU threads by the cgroup's quota.

**KV-cache tiers.** A host-RAM or disk tier below the GPU pool (`--kv-tier auto`) is budgeted the same way: host RAM at 0.95 × free RAM less the margin and 8 GiB left to everything else, disk at 0.95 × the free space under rigtune's own cache dir less the margin, and never on a network filesystem (NFS, SMB, Ceph, GlusterFS, 9p or FUSE, read from `/proc/mounts`; RunPod's `/workspace` by path where the mount table cannot be read). What the tier must hold, the working set, is converted to bytes with the same per-layer-type rules as the GPU pool it is compared with: full-attention layers hold every token, sliding layers one window per conversation, linear-attention layers one state per conversation. For linear-attention layers that is a lower bound: vLLM caches a state at block boundaries for prefix reuse, and how many of them reach its tier was not read from its source. Speculation's extra states are not counted, since a cached conversation does not hold them.

Effect, computed by `rigtune plan` from the HF config alone (feasible / enumerated):

| machine | model | vLLM | SGLang | llama.cpp (CUDA) |
|---|---|---|---|---|
| A100 80 GB | Llama-3.2-3B | 52 / 54 | 26 / 27 | 108 / 108 |
| RTX 4090 24 GB | Llama-3.2-3B | 34 / 54 | 34 / 54 | 108 / 108 |
| RTX 4090 24 GB | Llama-3.1-8B | 22 / 54 | 20 / 54 | 108 / 108 |
| GTX 1080 8 GB | Llama-3.2-3B | not selected (cc 6.1) | not selected | 94 / 108 |
| GTX 1080 8 GB | Llama-3.1-8B | not selected | not selected | 46 / 108 |

A 3B model fits almost everywhere; the planner earns its keep as the model grows and the card shrinks. The llama.cpp grid on an 8 GB card for an 8B model loses every full-offload Q6/Q8 config and every 8-slot high-context one. The planner never launches a process, so this costs milliseconds; the staged search then decides how many of the survivors get measured.

## 3. Staged search

A full grid would be 144 launches × ~30 s (engine startup dominates, not the 10 s workload) ≈ 70 minutes, and the variant stages below multiply it into the hundreds. The staged search runs 8–14 trials on the synthetic presets and 16–35 on the real-text calibrations since. Its order matters where settings interact: at the same budget, random sampling of the same space matched it on an A40 and an A100 and beat it by 29% on an RTX 4090, and an explore stage of random draws after the stages did not change that, but trying speculative decoding first did, matching random search's best there in 43 minutes against 66, and holding the tie on the A40 and A100 in two-thirds of their calibration time ([benchmarks](benchmarks.md#llama-31-8b-on-four-gpus)). What the stages buy is the same answer on every run, every strategy tried at least once, and a profile that says what each change was worth:

1. **Quant / precision.** Group feasible configs by `(backend, quant)`. For each group run *one* baseline: median ctx, median batch, and the *largest* memory setting the planner accepted (full GPU offload, highest `gpu_memory_utilization`); if that launch fails, step down once or twice. Rank by the objective; keep the top two groups. This is where backends get eliminated: a backend whose best quant is not in the top two is never launched again. (An earlier draft used the median memory setting; on an RTX 3090 that ran llama.cpp with 27 of 36 layers on the GPU and measured 35 tok/s instead of 545. Quants must be compared at the memory setting they would actually be served at.)
2. **Memory.** For each kept group, hold batch at the baseline and walk memory settings from largest to smallest — `gpu_memory_utilization` 0.95 → 0.90 → 0.80, or `n_gpu_layers` all → ¾ → ½, with ctx descending within each. Stop at the first config that launches *and* finishes the workload. That is "largest safe": the planner's estimate got it into the list, the measurement confirms it. At most three launches per group.
3. **Batch / concurrency.** With quant and memory fixed, sweep `max_num_seqs` (16 / 64 / 256) or `n_parallel` (1 / 4 / 8) and, on CPU, `n_batch`.

The final choice is the objective's constrained argmax over **every** successful trial from all three stages, not only stage 3, so a stage-1 baseline that happens to be best is not lost. Trials are keyed by config and never repeated. A trial that fails to launch (OOM, unsupported flag) is recorded in the table with its log tail; it is data, not an exception.

Later stages vary the leader along one dimension at a time: speculative decoding first, then the prefill budget, the KV-cache type and prefix-cache flags (on workloads whose prompts share a prefix), and optionally power. [Search options](usage.md#search-options) lists them. Speculation goes first because the other settings pay differently with it on: on an RTX 4090 with Llama 3.1 8B the int8 cache and a 16k prefill budget did nothing alone, yet with the draft model they were part of a configuration 28% faster than a pick that had tuned them before choosing suffix decoding. So the other stages run once for each of the engine's two best speculation settings (no speculation counts as one), on its best configuration with that setting, and the combinations share their trials between the two. The stages run for the leader and for every other engine whose best configuration after the batch stage is within 10% of it, or further behind when it offers a strategy the leader's engine lacks, each on its own configuration: on Dolly-15k prompts SGLang was 2% behind vLLM at that point, and tuning only the leader never tried it with the fp8 cache. Engines offering more strategies are tuned first, and one that an engine tuned before it has left more than 10% behind, with nothing more to offer, is skipped: on that 4090 SGLang spent 11 of 66 minutes on stages that ended 15% behind vLLM with speculation. With `--budget` they run straight after stage 1, before the memory and batch sweeps, since on real text that is where the gains were. Changing one dimension at a time misses strategies that only pay together, and keeps changes that get in a later one's way, so a final stage first measures the leader with each adopted change undone, then takes every change that came within 5% of the leader it was measured against, forms the combinations (one value per dimension, applied to the leader before the variant stages), estimates each by adding its members' gains, and measures the best, eight trials in all. The undo step came from an ablation: on `extract` on an A40 the search adopted the fp8 KV cache (+19% on its own) and then a draft model on top of it, but the draft model without the fp8 cache ran 19% faster than that pick, and summed gains ranked that configuration below every pair. With the undo step, a rerun found it on its second undo trial (726 tok/s against 604 for the adopted pair) and served it at 731 tok/s, 66.5% ahead of stock vLLM. A last stage measures the best three configurations again and chooses on those runs alone, because the best of many noisy trials is flattered by the choosing: on Dolly-15k prompts one pick measured 603 tok/s in calibration and 547 when re-measured. With `--kv-tier auto` and a session file, one more stage runs once per tuned engine, after its variant stages and before its combinations: a host-RAM (and perhaps disk) KV-cache tier, tried only when that engine's leader measured a working set above 0.8× its GPU pool, at one size derived from the working set (two on SGLang when host RAM cannot hold it: host, then host plus a capped disk tier) ([why](decisions.md#why-try-a-kv-cache-tier-only-when-the-working-set-outgrows-the-gpu-pool)). It has not run on hardware, and off it changes no trial (a test compares plans, launch arguments, cache keys and a calibration's trials with those captured before it existed).

Each trial is measured at concurrency 1, 4 and 8, and the objective scores it at whichever level best satisfies the constraint. For `balanced` that is the highest-throughput level whose TTFT is still under the ceiling, so the winner is a (config, load) pair the server can actually be run at, not a throughput number achieved with a TTFT the constraint forbids. Scores within 2% are treated as ties. A tie goes first to the configuration that switches on fewer optional strategies (a strategy is adopted only when it measurably wins), then to the larger context window, then the larger batch, then the lower energy: a 1% tok/s edge is measurement noise, a doubled context is a capability.

On an RTX 3090 with Qwen2.5-3B this produced two decisions a default would not make: vLLM fp8 over bf16 (1065 vs 694 tok/s, 0.94 vs 1.37 J/token), and llama.cpp with 8 parallel slots instead of 4 (645 vs 545 tok/s, TTFT 72 ms vs 952 ms under the same 8-client load, because with 4 slots half the clients queue). Those numbers came from the harness that replayed prompts across concurrency levels (see below), so their absolute values are inflated. Re-measured with fresh prompts on an A40, fp8 still beat bf16 on `chat` (592 vs 464 tok/s) but lost to it on the prefill-heavy `high-concurrency` workload (1358 vs 1724), where fp8's weight-only kernels on Ampere cost more in compute-bound prefill than they save in bandwidth. The llama.cpp slot decision has not been re-measured.

Every trial uses the same fixed workload — 16 prompts × ~256-token prefill × 128-token decode, `temperature 0`, `ignore_eos`, at concurrency 1, 4 and 8 — so numbers are comparable across backends and across machines. Each concurrency level gets fresh prompts. Replaying one level's prompts at the next lets the engine's prefix cache make every later prefill nearly free: an early version of the harness did that and read one configuration at 3638 tok/s that measures 1814 with fresh prompts. A workload's deliberately shared prefix (the `chat-system` and `rag-shared` presets) is kept across levels, since caching it is the behaviour being measured. Summary metrics come from the concurrency level with the highest throughput (the load the server would actually be run at); energy is total joules over total tokens across the levels measured. Under `throughput` and `balanced` the levels are measured busiest first and the trial stops at the first that meets the objective's limits, since throughput rises with load and a quieter level cannot then beat it; replayed over six recorded calibrations (115 trials), that changed no score or pick and skipped 35% of calibration time. The replay could not show one effect: a level measured first pays for kernels compiled on first use, so the warm-up now fills every slot of the busiest level before measuring (see [benchmarks](benchmarks.md#measuring-the-busiest-level-first)). A file of conversations is replayed turn by turn through the chat endpoint instead, each level a fixed budget of turns over sessions of its own, so the reuse a later turn gets from the history it resends is measured and recorded beside what the conversations allowed; that has run only in tests ([usage](usage.md#conversations-session-files)). An agent rewrites its prompt instead of appending to it, so a file can also give every request's whole message list; those are sent as written, chained tool-call follow-ups go ahead of the queue, and each level records how much of its prefill sat behind an edit. That has also run only in tests ([usage](usage.md#agent-traces-explicit-requests)).

## 4. Measurement and the energy objective

Request timing (TTFT, TPOT, tok/s) is measured at the HTTP client, the only vantage point shared by all backends. Device telemetry comes from llmtrace's `GPUSampler` (NVML at 100 ms: power, utilisation, memory), falling back to bare `pynvml`, and on CPU to process RSS plus RAPL package energy when readable. Energy is the trapezoidal integral of sampled power over the trial window; joules/token divides it by output tokens. Beside the samples, each window also reads NVML's total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, millijoules, Volta and newer, where the driver reports it) at its start and end, and records the difference as `counter_energy_j`. On an A100 the sampled power is itself a board average over about a second, so its integral is right over a long window and smears anything shorter; the counter is exact over any window. Joules/token stays the trapezoid, so no recorded number moves, and the two side by side are a free check of every J/tok measured so far. A counter that is missing or did not advance (a driver reload resets it) is recorded as absent.

The four objectives are all constrained argmax:

| objective | minimise / maximise | subject to |
|---|---|---|
| throughput | max tok/s | — |
| latency | min request latency (TTFT p50 + TPOT × output tokens) | tok/s ≥ floor |
| balanced | max tok/s | TTFT p50 ≤ ceiling |
| efficiency | min J/token | tok/s ≥ floor |

The floor defaults to 50% of the best tok/s observed in the same calibration, so it is relative to the machine, not an absolute number that would be meaningless across an A100 and a laptop. The ceiling defaults to 500 ms. If no configuration satisfies the constraint, the least-violating one wins and the profile records that the constraint was relaxed. There is deliberately no weighted score: a weight is a hidden preference, a constraint is a stated one, and the calibration table is saved in full so the choice can be audited.

Why efficiency is its own objective rather than a tie-breaker: on the same GPU, J/token varies by 2–3× across batch sizes because idle power is amortised over more tokens at higher concurrency, while TTFT gets worse. `efficiency` picks the point on that curve that still meets a throughput floor. Whether that is the same point `balanced` picks is an empirical question the benchmark table answers per machine.

## 5. Calibrating the planner against itself

The memory planner is a formula; the question is how wrong it is. Every trial now records the
planner's estimate (weights + KV + workspace, before the safety margin) next to two measurements:
the NVML peak during the trial minus what the device held before launch, and the backend's own
accounting parsed from its startup log (vLLM prints weights, non-KV total and available KV cache;
llama-server prints model, KV and compute buffer sizes per device; SGLang prints weight and KV
pool sizes). `rigtune memory-report` reduces these to, per backend: mean absolute % error,
signed bias, the worst under-prediction, component-wise error for weights and KV, and the number of
OOMs among planner-feasible configs. That last number is the one the planner is judged on: a
config the planner admitted that then OOMed is a planner failure, whatever the average error says.

Two constants are then fitted rather than assumed. `runtime_workspace`, the only hand-set term,
becomes the 95th percentile of measured workspace on this machine. The safety margin becomes the
worst observed under-prediction plus 2%, clamped to [2%, 15%], with the 512 MB floor kept.
`--apply` writes both to `~/.rigtune/memory-model.json` keyed by hardware hash and the planner
uses them from then on. The claim the matrix is meant to support is therefore checkable:
"predicts peak device memory within X% across N GPUs, with zero OOMs among the configs it admitted".

## 6. A performance predictor for the search

Measuring is the ground truth, but a search that must launch an engine to learn anything is slow
(vLLM start-up dominates at two to three minutes a trial). A predictor that ranks configs before
launch lets the search skip what cannot win. The model is a roofline, not a learned black box:

```
decode step, n sequences in flight:
    t_mem  = (device_weights + n · kv_bytes/token · (L_p + L_d/2)) / (α · mem_bw)
    t_comp = n · 2 · params / (β · peak_flops)
    t_step = max(t_mem, t_comp) + overhead
prefill (compute bound):  t_pre = n · L_p · 2 · params / (β · peak_flops) + device_weights / (α · mem_bw)
wave:   t_wave = t_pre + L_d · t_step        tok/s = n · L_d / t_wave
own prefill (chunked prefill interleaves requests): t_own = L_p · 2 · params / (β · peak_flops)
TTFT p50 with c clients and B slots (n = min(c, B)):  t_own + ⌊(c/2)/n⌋ · t_wave
```

Device peak bandwidth and FLOPS come from a small table keyed on the NVML name; the three
parameters per backend (α bandwidth efficiency, β compute efficiency, per-step overhead) are fitted
by grid search on log error from the machine's own calibration trials, one observation per
(config, concurrency level) measured; a trial that stopped at its busiest level contributes only that
level, which is also the load the pick is scored at. `rigtune fit` reports leave-one-trial-out MAPE for tok/s and TTFT
and the Spearman rank correlation between predicted and measured tok/s, which is the number that
matters: pruning is safe when the ranking is right even if the magnitudes are off.

The search uses it conservatively. With fitted parameters, stage 1 skips a (backend, quant) group
whose predicted best is below 40% of the best predicted group, and stage 3 tries batch settings in
predicted order so an interrupted calibration already holds the likely winner. With priors only,
nothing is pruned; every prediction carries a fitted/prior flag. Measurements prune as well: once even a
backend's best raw throughput is below half of what another backend reached within the objective's
constraints, its remaining quants are skipped. Only a feasible result can lead, and a backend's raw
throughput bounds any score it could reach, so the rule cannot drop a backend that would have won. On ShareGPT prompts on an A40 that saved three llama.cpp trials of several minutes each. The queueing term is what made
the llama.cpp result on the RTX 3090 explicable before it was measured: with 8 clients on 4 slots
the median request waits a full wave, which is the 952 ms TTFT the trial recorded, and 8 slots
remove the wait.

## 7. Caching and reproducibility

A profile is keyed by `(hardware_hash, model, objective)`. The hash covers the *shape* of the hardware (GPU model, compute capability, total VRAM, CPU model, core counts, SIMD flags, total RAM) and not free memory, so a busy machine does not invalidate its own cache. The profile stores the backend version; a version change invalidates it. It also stores the full calibration table, workload description and llmtrace version, so a decision can be re-derived offline.

## 8. Limitations

- One GPU. Tensor parallel is a stage the search does not have yet.
- One change at a time. The stages adopt a change only when it pays on its own, and the combination stage joins only changes that came close to the leader alone, so an interaction neither change shows alone is missed. Given the same time, random sampling of the same settings matched the staged search on an A40 and an A100 and found such an interaction on an RTX 4090 (a draft model with an int8 KV cache and a 16k prefill budget, 29% faster than the pick; [benchmarks](benchmarks.md#llama-31-8b-on-four-gpus)). Spending part of the budget on random draws around the leader (`--explore on`) did not fix it; trying speculative decoding first and tuning the other settings on each of the two best speculation methods did, on that card (1041 tok/s against random search's 1041, in 43 minutes against 66). On the A40 and A100 it kept the ties already there (506 against 505, and 1155 against 1160) with a third fewer trials. It catches settings that pay only with a speculation method; other interactions would still be missed.
- The synthetic workload is a proxy. A deployment with 4k-token prompts or 2k-token outputs sits elsewhere on the throughput/latency curve. Live re-tuning under real traffic is on the roadmap.
- `runtime_workspace` is a constant per backend, not a function of model size; large models with many CUDA graphs will exceed it, which the stage-2 "largest safe" walk absorbs at the cost of one failed launch.
- Energy on CPU needs RAPL read permission; without it `efficiency` degrades to throughput ordering and says so.

## 9. Non-goals and roadmap

rigtune sits above the inference engines (vLLM, SGLang and llama.cpp) and launches them; it is not a replacement, a compiler or a kernel library, and it supports only the hardware [measured so far](benchmarks.md#hardware-and-engines-measured). Planned: AMD ROCm and Apple Silicon backends, Windows, and re-tuning under live traffic instead of a one-time calibration.
