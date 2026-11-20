"""Which ways of writing into a slice of a cache tensor make torch.compile clone the tensor?

SGLang 0.5.23 under --enable-torch-compile clones its caches before writing them (see
benchmarks/sessions/2026-11/modal-sglang-compile-oom): the GDN state through a user Triton kernel, the KV cache
through a custom op given two slices of one tensor. On a CPU with a custom op, only the second pattern clones.
This repeats the patterns with a user Triton kernel on a GPU, compiled the way SGLang compiles
(mode="max-autotune-no-cudagraphs"), and reports the largest buffer in each generated graph. No model, a T4.

Run:  modal run benchmarks/modal_triton_view_clone.py                   torch 2.14.1
      modal run benchmarks/modal_triton_view_clone.py --torch nightly   today's PyTorch nightly (CUDA 13.0)
"""

# No `from __future__ import annotations`: torch.library.custom_op reads its function's type hints, and with
# that import they are strings naming `torch`, which is imported inside _probe(), not at module scope.
import modal

app = modal.App("rigtune-triton-view-clone")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.14.1")
nightly_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu130")


@app.function(gpu="T4", image=image, timeout=1200)
def probe() -> list:
    return _probe()


@app.function(gpu="T4", image=nightly_image, timeout=1200)
def probe_nightly() -> list:
    return _probe()


def _probe() -> list:
    import re

    import torch
    import triton
    import triton.language as tl
    from torch._inductor.utils import run_and_get_code

    LAYERS, SLOTS, D = 24, 509, 64
    LAYER = SLOTS * D

    @triton.jit
    def add_one_kernel(ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(ptr + offs, tl.load(ptr + offs, mask=mask) + 1, mask=mask)

    @triton.jit
    def add_one_layer_kernel(ptr, layer, layer_numel, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < layer_numel
        p = ptr + layer * layer_numel + offs
        tl.store(p, tl.load(p, mask=mask) + 1, mask=mask)

    def add_one(t):  # the state update: writes t in place
        add_one_kernel[(triton.cdiv(t.numel(), 1024),)](t, t.numel(), BLOCK=1024)

    def add_one_layer(cache, layer):  # the workaround: the whole cache and a layer number
        add_one_layer_kernel[(triton.cdiv(LAYER, 1024),)](cache, layer, LAYER, BLOCK=1024)

    @torch.library.custom_op("probe::add_one", mutates_args=("t",))
    def add_one_op(t: torch.Tensor) -> None:  # the same launch, wrapped as a custom op
        add_one(t)

    @add_one_op.register_fake
    def _(t):
        return None

    @torch.library.custom_op("probe::add_one_kv", mutates_args=("k", "v"))
    def add_one_kv(k: torch.Tensor, v: torch.Tensor) -> None:  # SGLang's store_cache writes K and V together
        add_one(k)
        add_one(v)

    @add_one_kv.register_fake
    def _(k, v):
        return None

    class Pool:  # SGLang reaches its caches through a pool object
        def __init__(self):
            self.temporal = torch.zeros(LAYERS, SLOTS, D, device="cuda")
            self.fused = torch.zeros(2, SLOTS, D, device="cuda")
            self.k, self.v = self.fused[0], self.fused[1]  # made once, as SGLang's KV pool does

    pool = Pool()

    def slice_inside(cache, x):
        add_one(cache[5])
        return x * 2

    def slice_passed_in(view, x):
        add_one(view)
        return x * 2

    def whole_plus_index(cache, x):
        add_one_layer(cache, 5)
        return x * 2

    def slice_inside_then_read(cache, x):  # the state read again after the write, as SGLang's tracking may
        add_one(cache[5])
        return x * 2 + cache[5].sum()

    def pool_slice_inside(x):
        add_one(pool.temporal[5])
        return x * 2

    def pool_whole_plus_index(x):
        add_one_layer(pool.temporal, 5)
        return x * 2

    def op_slice_inside(cache, x):
        add_one_op(cache[5])
        return x * 2

    def pool_op_slice_inside(x):
        add_one_op(pool.temporal[5])
        return x * 2

    def pool_kv_premade(x):
        add_one_kv(pool.k, pool.v)
        return x * 2

    def pool_kv_sliced_inside(x):
        add_one_kv(pool.fused[0], pool.fused[1])
        return x * 2

    # (name, function, arguments from a fresh cache, or the pool tensor the case writes; layer-sized slices)
    cases = [
        ("slice taken inside", slice_inside, lambda c: (c,)),
        ("slice passed in", slice_passed_in, lambda c: (c[5],)),
        ("whole tensor + index", whole_plus_index, lambda c: (c,)),
        ("slice inside, read after", slice_inside_then_read, lambda c: (c,)),
        ("pool: slice inside", pool_slice_inside, "temporal"),
        ("pool: whole + index", pool_whole_plus_index, "temporal"),
        ("custom op: slice inside", op_slice_inside, lambda c: (c,)),
        ("pool, custom op: slice inside", pool_op_slice_inside, "temporal"),
        ("pool, kv: slices made once", pool_kv_premade, "fused"),
        ("pool, kv: sliced inside", pool_kv_sliced_inside, "fused"),
    ]
    out = []
    for name, fn, args in cases:
        torch._dynamo.reset()
        x = torch.ones(8, device="cuda")
        if isinstance(args, str):
            target = getattr(pool, args)
            target.zero_()
            call = (x,)
        else:
            target = torch.zeros(LAYERS, SLOTS, D, device="cuda")
            call = args(target) + (x,)
        _, codes = run_and_get_code(torch.compile(fn, mode="max-autotune-no-cudagraphs", fullgraph=True), *call)
        code = "\n".join(codes)
        sizes = []
        for m in re.finditer(r"empty_strided_cuda\(\(([\d, ]+)\)", code):
            n = 1
            for d in m.group(1).replace(" ", "").rstrip(",").split(","):
                n *= int(d)
            sizes.append(n)
        big = max(sizes, default=0)
        if target.shape[0] == 2:  # K and V: both slices written once
            correct = bool((target == 1).all())
        else:
            correct = bool((target[5] == 1).all() and (target.sum() == LAYER))
        out.append({"case": name, "correct": correct, "biggest_buffer_slices": round(big / LAYER, 2)})
    out.append({"torch": torch.__version__, "triton": triton.__version__})
    return out


@app.local_entrypoint()
def main(torch: str = "2.14.1") -> None:
    if torch not in ("2.14.1", "nightly"):
        raise SystemExit("--torch is 2.14.1 or nightly")
    for row in (probe_nightly if torch == "nightly" else probe).remote():
        print(row)
