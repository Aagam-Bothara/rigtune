"""Why does the patched KV store still clone K and V together in the unit test? Prints the generated code.

Run:  modal run benchmarks/modal_sglang_kv_debug.py
"""

import pathlib

import modal

SESSION = pathlib.Path(__file__).parent / "sessions/2026-11/modal-sglang-compile-oom"
MAIN_IMAGE = "lmsysorg/sglang:nightly-dev-cu13-20261120-03fde9cc"

app = modal.App("rigtune-sglang-kv-debug")
image = (modal.Image.from_registry(MAIN_IMAGE, add_python="3.12")
         .entrypoint([])
         .apt_install("git")
         .add_local_file(SESSION / "sglang-main-fix.diff", "/patch/fix.diff"))

PROBE = r'''
import re
import torch
import torch._dynamo
from torch._inductor.utils import run_and_get_code
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


def test_like():  # the unit test's exact shape: the pool and tensors are locals the step closes over
    torch.manual_seed(0)
    kv_pool = MHATokenToKVPool(size=64, page_size=1, head_num=2, head_dim=64, dtype=torch.bfloat16,
                               layer_num=4, device="cuda", enable_memory_saver=False)
    loc = torch.arange(1, 5, device="cuda")
    k = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)

    def step(k, v):
        kv_pool.set_kv_buffer(None, loc, k, v, layer_id_override=2)

    torch._dynamo.reset()
    _, codes = run_and_get_code(torch.compile(step), k, v)
    print("compiled pieces:", len(codes))
    for i, c in enumerate(codes):
        hits = re.findall(r"empty_strided_cuda\(\(([\d, ]+)\)[^\n]*", c)
        print(f"=== piece {i}: allocations {hits}")
        body = c[c.find("def call"):]
        print(body[:3000])


test_like()
'''


@app.function(gpu="L4", image=image, timeout=1200)
def debug() -> None:
    import os
    import subprocess

    py = next(p for p in ("/usr/bin/python3", "/usr/local/bin/python3", "python3", "python")
              if subprocess.run([p, "-c", "import sglang"], capture_output=True).returncode == 0)
    site = subprocess.check_output(
        [py, "-c", "import os, sglang; print(os.path.dirname(os.path.dirname(sglang.__file__)))"], text=True).strip()
    subprocess.run(["git", "apply", "-p2", "/patch/fix.diff"], cwd=site, check=True,
                   env={**os.environ, "GIT_DIR": "/nonexistent"})
    # graph_breaks names the file and line of each break
    done = subprocess.run([py, "-c", PROBE], capture_output=True, text=True,
                          env={**os.environ, "TORCH_LOGS": "graph_breaks"})
    out = "\n".join(line for line in (done.stdout + done.stderr).splitlines()
                    if "Warning" not in line and "warn(" not in line and not line.startswith("../"))
    print(out[-12000:])


@app.local_entrypoint()
def main() -> None:
    debug.remote()
