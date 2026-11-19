"""Does a fixed TRITON_CACHE_DIR stop Qwen3.5's GDN warm-up from repeating on every vLLM launch?

Reading vLLM 0.33's source suggests that, unless TRITON_CACHE_DIR is set, vLLM points Triton's cache inside the
torch.compile directory of each configuration, whose hash includes max_num_seqs, so every new configuration
re-compiles and re-autotunes the FLA kernels (about 82 s on an A100, 66 s on an A40). This runs five start-ups on
one Modal GPU and reads each one's timings from vLLM's own log:

  pinned world (TRITON_CACHE_DIR fixed, all caches empty at the start)
    1. A  max_num_seqs=64    cold: expected to pay the full warm-up
    2. B  max_num_seqs=128   new hash, same Triton cache: the hypothesis says the warm-up is gone
    3. A  max_num_seqs=64    identical relaunch: the torch.compile cache should hit as well
  default world (TRITON_CACHE_DIR unset, all caches empty again)
    4. B  max_num_seqs=128   cold control
    5. C  max_num_seqs=96    new hash: today's behaviour says the warm-up is paid again

Run from a machine with the Modal CLI set up:  modal run benchmarks/modal_triton_cache.py
It writes modal_triton_cache.json beside where it is run, and keeps every log in the Modal volume
"rigtune-triton-cache" under logs/.
"""

from __future__ import annotations

import json

import modal

MODEL = "Qwen/Qwen3.5-9B"
VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"
GPU = "A100"  # 40 GB, sm80: the A100 of the GPU sessions; the 9B's 18 GiB of weights leave room for the KV cache

app = modal.App("rigtune-triton-cache")
# Modal could not find the image's own Python (ConflictError), so it adds one to run this file's stdlib-only loop;
# `vllm` keeps the image's Python, the one the GPU sessions ran.
image = modal.Image.from_registry(VLLM_IMAGE, add_python="3.12").entrypoint([])
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)

RUNS = [
    # (name, world, max_num_seqs, pin the Triton cache)
    ("1-A-pinned-cold", "pinned", 64, True),
    ("2-B-pinned-newhash", "pinned", 128, True),
    ("3-A-pinned-again", "pinned", 64, True),
    ("4-B-default-cold", "default", 128, False),
    ("5-C-default-newhash", "default", 96, False),
]

PATTERNS = {
    "gdn_warmup_s": r"\[backbone\] Initial profiling/warmup run took ([\d.]+) s",
    "torch_compile_s": r"\[backbone\] torch\.compile took ([\d.]+) s",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
}


@app.function(gpu=GPU, image=image, volumes={"/vol": volume}, timeout=3 * 3600)
def experiment() -> list:
    import glob
    import os
    import re
    import shutil
    import subprocess
    import time
    import urllib.request

    os.makedirs("/vol/logs", exist_ok=True)

    def reset_caches() -> None:
        """Every compile and kernel cache empty, the downloaded weights kept."""
        for path in ["/root/.cache/vllm", "/root/.cache/flashinfer", "/root/.triton", "/tmp/worlds"]:
            shutil.rmtree(path, ignore_errors=True)
        for path in glob.glob("/tmp/torchinductor_*"):
            shutil.rmtree(path, ignore_errors=True)

    results, world_seen = [], None
    for name, world, seqs, pinned in RUNS:
        if world != world_seen:
            reset_caches()
            world_seen = world
        root = f"/tmp/worlds/{world}"
        env = {**os.environ, "HF_HOME": "/vol/hf", "VLLM_CACHE_ROOT": f"{root}/vllm",
               "TORCHINDUCTOR_CACHE_DIR": f"{root}/inductor", "TRITON_PRINT_AUTOTUNING": "1"}
        env.pop("TRITON_CACHE_DIR", None)
        if pinned:
            env["TRITON_CACHE_DIR"] = f"{root}/triton"
        log_path = f"/vol/logs/{name}.log"
        argv = ["vllm", "serve", MODEL, "--port", "8000", "--max-model-len", "4096",
                "--max-num-seqs", str(seqs), "--gpu-memory-utilization", "0.90"]
        t0 = time.time()
        with open(log_path, "w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
            ready = None
            while time.time() - t0 < 1800:
                if proc.poll() is not None:
                    break
                try:
                    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                    ready = time.time() - t0
                    break
                except Exception:
                    time.sleep(2)
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        text = open(log_path, encoding="utf-8", errors="replace").read()
        row = {"run": name, "max_num_seqs": seqs, "triton_cache_pinned": pinned,
               "ready_s": round(ready, 1) if ready is not None else None,
               "autotune_lines": text.count("Triton autotuning for function"),
               "bundler_cubin_missing": text.count("Cubin file saved by TritonBundler not found"),
               "aot_loaded": "loaded AOT compiled function" in text or "Loading AOT compiled" in text}
        for key, pattern in PATTERNS.items():
            m = re.search(pattern, text)
            row[key] = (int(m.group(1).replace(",", "")) if key == "kv_cache_tokens" else float(m.group(1))) if m else None
        if ready is None:
            row["error"] = "\n".join(text.strip().splitlines()[-5:])
        print(json.dumps(row), flush=True)
        results.append(row)
        volume.commit()
    return results


@app.local_entrypoint()
def main() -> None:
    rows = experiment.remote()
    with open("modal_triton_cache.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, indent=2)
    cols = ["run", "ready_s", "gdn_warmup_s", "torch_compile_s", "init_engine_s", "autotune_lines", "kv_cache_tokens"]
    print("\n" + " | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c)) for c in cols))
