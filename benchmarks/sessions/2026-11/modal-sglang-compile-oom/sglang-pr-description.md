Title: [Fix] gdn: Stop torch.compile from cloning the state and KV pools

## Motivation

Fixes #51651. With `--enable-torch-compile`, Qwen3.5-9B never starts: decode CUDA graph capture runs out of memory
at bs=32. Inductor clones SGLang's caches before writing them:

- **GDN state.** `mamba2_layer_cache(layer_id)` slices the all-layers pool inside the compiled graph, and the
  packed decode kernel (a user Triton kernel) writes that slice. Inductor then clones the pool from its start
  through that slice: k + 1 layers' worth for GDN layer k, on every call. On an A100 80GB the 7th layer's clone
  (6.96 GiB) does not fit. The conv state gets the same clone from the fused conv update.
- **KV cache.** `k_buffer[i]` and `v_buffer[i]` are views of one shared allocation, made once at start-up.
  `store_cache` writes both, and Inductor clones the whole allocation (3.39 GiB per layer here) before the write.

Both reduce to a few lines with no SGLang, on torch 2.14.1 / Triton 3.8.0. Largest buffer in the generated code
for a `(24, 509, 64)` tensor:

| Case | Largest buffer |
|---|---|
| user Triton kernel writes `cache[5]`, slice taken inside the compiled fn | 6 slices (a clone of `cache[:6]`) |
| same, slice made outside and passed in | none |
| same launch wrapped in a custom op with `mutates_args` | none |
| custom op writes two views of one tensor made outside (like K and V) | the whole tensor |
| same op, both views sliced inside the compiled fn | none |

Reproduction scripts and logs: https://github.com/Aagam-Bothara/rigtune/tree/main/benchmarks/sessions/2026-11/modal-sglang-compile-oom

## Modifications

- `triton_gdn_fused_proj.py`: `fused_qkvzba_causal_conv1d_update_contiguous` is registered with
  `register_custom_op(mutates_args=["conv_state"])`, with a fake impl for its four outputs.
- `gdn_triton.py`: the packed decode goes through a new `gdn_packed_decode` custom op
  (`mutates_args=["initial_state", "out"]`). It is a wrapper because
  `fused_recurrent_gated_delta_rule_packed_decode` returns its inputs, which a custom op may not.
- `memory_pool.py`: `MHATokenToKVPool` keeps the shared K/V allocations, and `_store_kv_layer` slices K and V
  from them while compiling. `_assign_post_capture_tensors` clears them, so remapped buffers fall back to the
  existing path.

- `test_gdn_torch_compile_pool_writes.py` (new): each of the three writes, on layer 2 of a 4-layer pool under
  torch.compile, must add less than one layer of GPU memory at peak (measured on the second compiled call) and
  match eager exactly. On main at 03fde9cc (nightly dev image, one L4) all three fail without the fix (3 layers,
  3 layers, and K plus V added) and pass with it.

No kernel changes. Without torch.compile, `register_custom_op` calls the functions directly, and the KV change is
skipped (`torch.compiler.is_compiling()`), so eager behaviour is unchanged. Not covered: the non-fused conv
update, the non-packed decode, and other KV layouts, which our configuration does not reach.

One limit of the KV change: the slicing and the `store_cache` call must land in the same compiled graph. The
first `store_cache` call builds its JIT kernel and logs (`kvcache.py:39`), which Dynamo cannot trace, so a first
call under torch.compile splits the graph between them and the clone comes back. Serving builds the kernel
eagerly before capture, so it does not happen there; the test warms up eagerly for the same reason. Passing the
shared allocation into `_set_kv_buffer_impl` would remove the dependency, at the cost of a wider change.

## Accuracy Tests

Qwen3.5-9B (revision c202236), one A100 80GB, `--context-length 262144 --max-running-requests 256
--mem-fraction-static 0.88`, four runs in one container, on v0.5.23 and on main (nightly dev image at 03fde9cc,
whose touched files equal this branch's base). Four prompts, greedy, 64 tokens each:

| Run | Answers equal A, v0.5.23 | Answers equal A, main |
|---|---|---|
| A: unpatched, no torch.compile | (reference) | (reference) |
| B: patched, no torch.compile | 4 of 4 | 4 of 4 |
| C: patched, `--enable-torch-compile` | 4 of 4 | 2 of 4 |
| D: unpatched, `--enable-torch-compile` | did not start (OOM, 6.96 GiB) | did not start (OOM, 6.96 GiB) |

On main, C's other two answers match A's for their first 16 and 173 characters, the answer included, then
continue differently but fluently: greedy decoding diverging once torch.compile reorders the arithmetic.
Unpatched torch.compile cannot be compared, since it does not start.

## Speed Tests and Profiling

Same runs, one measurement each, so these are indicative. A ran first in each fresh container, so its start-up
and 8-user number include one-time warm-ups.

| Run | Ready | Buffers >= 1 GiB in generated code | tok/s, 1 user | tok/s, 8 users |
|---|---|---|---|---|
| v0.5.23 A: unpatched, eager | 163 s | 0 | 89.1 | 593.7 |
| v0.5.23 B: patched, eager | 73 s | 0 | 88.9 | 668.9 |
| v0.5.23 C: patched, torch.compile | 412 s | 0 | 96.7 | 691.9 |
| v0.5.23 D: unpatched, torch.compile | OOM | 7 | - | - |
| main A: unpatched, eager | 136 s | 0 | 85.5 | 577.2 |
| main B: patched, eager | 65 s | 0 | 85.6 | 644.6 |
| main C: patched, torch.compile | 352 s | 0 | 93.4 | 657.0 |
| main D: unpatched, torch.compile | OOM | 6 | - | - |

## Checklist

- [x] Format your code according to the Format code with pre-commit guide (all hooks pass).
- [x] Add unit tests: `test/registered/kernels/ops/attention/test_gdn_torch_compile_pool_writes.py`.
- [x] Update documentation: not needed.
- [x] Provide accuracy and speed benchmark results (above).
- [x] Follow the SGLang code style guidance.
