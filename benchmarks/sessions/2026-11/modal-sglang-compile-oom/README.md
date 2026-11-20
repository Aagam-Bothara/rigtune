# SGLang 0.5.23 with `--enable-torch-compile` on Qwen3.5-9B: why start-up runs out of memory

Run on 20 November 2026 with [modal_sglang_compile_oom.py](../../../modal_sglang_compile_oom.py) on one Modal
A100 80GB: SGLang 0.5.23, torch 2.14.1+cu130, CUDA 13.0, Qwen/Qwen3.5-9B at revision c202236. Raw output in
[results.json](results.json); logs and generated-code excerpts are in the Modal volume `rigtune-triton-cache`
under `sglang-oom/`.

In the November GPU sessions SGLang's expert row with `--enable-torch-compile` never started for Qwen3.5-9B, on
an A100 80GB or an A40 48GB. Without the flag the same settings start and serve on both cards.

## Results

| Run | Change | State cache | Failed allocation | At batch |
|---|---|---|---|---|
| 1 | the GPU session's command | 508 slots, fp32, 23.86 GB | 6.96 GiB fp32 | 32 |
| 2 | `--mamba-ssm-dtype bfloat16` | 994 slots, bf16, 23.32 GB | 6.80 GiB bf16 | 32 |
| 3 | `--max-mamba-cache-size 128` | 128 slots, fp32, 6.05 GB | 6.05 GiB fp32 | 25 |
| 4 | `--torch-compile-max-bs 8` | 508 slots, fp32, 23.86 GB | 6.96 GiB fp32 | 8 |

## What the generated code does

SGLang keeps every GDN layer's recurrent state in one tensor, `(24 layers, slots + 1, 32, 128, 128)`. In the
graph Inductor generates for GDN layer k (counting from 0), before the decode kernel runs, the tensor is cloned
from its start through the end of layer k's slice: k + 1 layers' worth. The kernel writes layer k's slice of the
clone, and that slice is copied back. Each layer gets its own graph, so the clone grows by one layer per layer:

    assert_size_stride(arg1_1, (24, 509, 32, 128, 128), (266862592, 524288, 16384, 128, 1), 'input')
    buf12 = empty_strided_cuda((533725184, ), (1, ), torch.float32)          # layer 1: 2 layers' worth
    triton_poi_fused_as_strided_clone_select_2.run(arg1_1, buf12, 533725184, ...)
    fused_recurrent_gated_delta_rule_packed_decode_kernel_1.run(..., reinterpret_tensor(buf12, (509, 32, 128, 128), ..., 266862592), ...)
    triton_poi_fused_as_strided_3.run(buf12, arg1_1, 266862592, ...)          # layer 1's slice copied back

- In run 1 the clones were 2, 3, 4, 5, 6 and 7 layers' worth (509 states each); the 7th, 6.96 GiB, did not fit.
  The A40's smaller cache let it reach the 9th.
- Run 2: the clone turned bf16 with the state, but SGLang filled the memory bf16 freed with twice the slots, so
  the bytes barely changed.
- Run 3: the clones reached the last layer, a clone of the whole tensor (24 x 129 states, 6.05 GiB), which did not
  fit either: SGLang gave the memory freed to the KV cache.
- Run 4: the same clone at batch 8, so it does not depend on the batch.
- The conv state is cloned the same way (small), and so is the KV cache write in the full-attention layers:
  `torch.ops.sglang.store_cache` writes into a view of a clone of 1,820,942,336 bf16 values (3.39 GiB), which is
  then copied back. That is a custom op, not a Triton kernel, so the pattern is not tied to one kernel: each
  write into a view of a large cache tensor at an offset is preceded by a clone of the tensor up to that view.

These clones run on every decode step in the compiled region, not only at start-up. That they make decoding slow
when they fit (SGLang issue #35777 saw torch.compile with a decode graph about 6x slower) is not measured here.
