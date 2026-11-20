"""Why does SGLang 0.5.23 run out of memory starting Qwen3.5-9B with --enable-torch-compile?

On an A100 80GB and an A40 48GB (Nov 2026 GPU sessions) every attempt failed while capturing the decode CUDA
graph at bs=32, the first size under torch_compile_max_bs, on one allocation in Inductor's generated code:

    buf12 = empty_strided_cuda((1864368128, ), (1, ), torch.float32)      # 6.95 GiB on the A100

1,864,368,128 is exactly 3,556 x 524,288, and 524,288 = 32 x 128 x 128 is one GDN layer's recurrent state for one
request. So the buffer looks like a copy of state-cache memory, not activations for 32 requests. This runs four
start-ups on one Modal A100 80GB, each with its own empty Inductor cache, and from each it keeps the log, the
failed allocation, every buffer of 1 GiB or more in the generated code, and the code around it:

  1. repro          the failing command from the GPU session
  2. ssm-bf16       + --mamba-ssm-dtype bfloat16: if it copies the state, the buffer halves or turns bf16
  3. small-pool     + --max-mamba-cache-size 128: if it follows the cache, the buffer shrinks about 4x
  4. compile-bs8    + --torch-compile-max-bs 8: if it does not depend on the batch, the buffer is unchanged

Run from a machine with the Modal CLI set up:  modal run benchmarks/modal_sglang_compile_oom.py
It writes modal_sglang_compile_oom.json beside where it is run, and keeps every log and generated-code excerpt
in the Modal volume "rigtune-triton-cache" under sglang-oom/ (the Qwen3.5-9B weights are already there).
"""

from __future__ import annotations

import json

import modal

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # the GPU sessions' revision
GPU = "A100-80GB"  # the card of the A100 session, so the state cache comes out the same size (507 slots)

app = modal.App("rigtune-sglang-compile-oom")
# The GPU sessions installed sglang==0.5.23 from PyPI into a fresh venv on a CUDA 13.0 image; this does the same.
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .apt_install("libnuma1")
         .pip_install("sglang==0.5.23", "nvidia-ml-py"))
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)

BASE = ["python", "-m", "sglang.launch_server", "--model-path", MODEL, "--revision", REVISION,
        "--dtype", "bfloat16", "--host", "127.0.0.1", "--port", "30000",
        "--context-length", "262144", "--max-running-requests", "256", "--mem-fraction-static", "0.88",
        "--enable-torch-compile"]
RUNS = [
    ("1-repro", []),
    ("2-ssm-bf16", ["--mamba-ssm-dtype", "bfloat16"]),
    ("3-small-pool", ["--max-mamba-cache-size", "128"]),
    ("4-compile-bs8", ["--torch-compile-max-bs", "8"]),
]
GIB = 2 ** 30
ITEMSIZE = {"torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2, "torch.int64": 8, "torch.int32": 4,
            "torch.uint8": 1, "torch.int8": 1, "torch.bool": 1}


def big_buffers(path: str, floor: int = GIB) -> list:
    """Every empty_strided_cuda in one generated file whose literal shape comes to `floor` bytes or more, with the
    code around it and every later line that names the buffer."""
    import re

    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    out = []
    for i, line in enumerate(lines):
        m = re.search(r"(buf\d+) = empty_strided_cuda\(\(([^)]*)\), \([^)]*\), (torch\.\w+)\)", line)
        if not m:
            continue
        dims = [d.strip() for d in m.group(2).split(",") if d.strip()]
        if not all(d.isdigit() for d in dims):
            continue  # a symbolic size: not a literal allocation
        numel = 1
        for d in dims:
            numel *= int(d)
        nbytes = numel * ITEMSIZE.get(m.group(3), 4)
        if nbytes < floor:
            continue
        name = m.group(1)
        uses = [f"{j + 1}: {lines[j].strip()}" for j in range(i + 1, len(lines))
                if re.search(rf"\b{name}\b", lines[j])][:12]
        out.append({"file": path, "line": i + 1, "code": line.strip(), "dtype": m.group(3), "numel": numel,
                    "gib": round(nbytes / GIB, 3), "per_gdn_state": numel / (32 * 128 * 128),
                    "before": lines[max(0, i - 40):i], "uses": uses})
    return out


@app.function(gpu=GPU, image=image, volumes={"/vol": volume}, timeout=2 * 3600)
def experiment() -> list:
    import glob
    import os
    import re
    import subprocess
    import time
    import urllib.request

    os.makedirs("/vol/sglang-oom", exist_ok=True)
    import sglang
    import torch
    versions = {"sglang": sglang.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}
    print(json.dumps(versions), flush=True)
    results = []
    for name, extra in RUNS:
        inductor = f"/tmp/inductor/{name}"
        env = {**os.environ, "HF_HOME": "/vol/hf", "TORCHINDUCTOR_CACHE_DIR": inductor}
        log_path = f"/vol/sglang-oom/{name}.log"
        t0, ready = time.time(), None
        with open(log_path, "w") as log:
            proc = subprocess.Popen(BASE + extra, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            while time.time() - t0 < 1500:
                if proc.poll() is not None:
                    break
                try:
                    urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2)
                    ready = time.time() - t0
                    break
                except Exception:
                    time.sleep(3)
            try:
                os.killpg(proc.pid, 15)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                proc.wait()
        subprocess.run("pkill -9 -f sglang.launch_server; sleep 5", shell=True)  # nothing left on the GPU
        text = open(log_path, encoding="utf-8", errors="replace").read()

        def first(pattern):
            m = re.search(pattern, text)
            return m.group(1) if m else None

        captured = re.findall(r"Capturing batches \(bs=(\d+)", text)
        row = {"run": name, "extra_args": extra, "ready_s": round(ready, 1) if ready is not None else None,
               "max_mamba_cache_size": first(r"max_mamba_cache_size: (\d+)"),
               "ssm_state": first(r"ssm_state size: ([\d.]+GB)"),
               "max_running_requests_cap": first(r"max_running_requests is capped to (\d+)"),
               "failed_allocation": first(r"Tried to allocate ([\d.]+ [GM]iB)"),
               "failed_at_bs": captured[-1] if captured and ready is None else None,
               "failed_line": first(r"\.py\", line \d+, in call\n\s+(buf\d+ = empty_strided_cuda[^\n]*)"),
               "decode_capture_end": first(r"(Capture target decode CUDA graph end[^\n]*)")}
        buffers = []
        for path in glob.glob(f"{inductor}/**/*.py", recursive=True):
            try:
                buffers += big_buffers(path)
            except Exception as exc:
                buffers.append({"file": path, "error": str(exc)})
        row["big_buffers"] = [{k: b[k] for k in ("file", "line", "code", "gib", "per_gdn_state") if k in b}
                              for b in buffers]
        with open(f"/vol/sglang-oom/{name}.buffers.json", "w") as fh:
            json.dump(buffers, fh, indent=1)
        if ready is None:
            row["tail"] = "\n".join(text.strip().splitlines()[-4:])
        print(json.dumps({k: v for k, v in row.items() if k != "big_buffers"}), flush=True)
        results.append(row)
        volume.commit()
    return [versions] + results


@app.local_entrypoint()
def main() -> None:
    rows = experiment.remote()
    with open("modal_sglang_compile_oom.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, indent=2)
    print("\n", rows[0])
    for r in rows[1:]:
        print(f"{r['run']}: ready={r['ready_s']} cache={r['max_mamba_cache_size']} ssm={r['ssm_state']} "
              f"failed={r['failed_allocation']} at bs={r['failed_at_bs']} | {len(r['big_buffers'])} buffers >= 1 GiB")
        for b in r["big_buffers"][:4]:
            print(f"    {b.get('gib')} GiB = {b.get('per_gdn_state')} GDN states: {b.get('code')}")
