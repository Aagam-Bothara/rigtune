"""How warm can the calibrated winner start, on the same machine and on a fresh one given only its caches?

The A100 calibration's pick for Qwen3.5-9B (vLLM 0.33 bf16, MTP at 3 tokens, batch 64, context 8192), started
four ways on Modal A100s. Every cache the engine writes lives under one HOME (vLLM's compile and AOT caches,
FlashInfer's, Inductor's, and Triton's, pinned there as rigtune pins it), so "the bundle" is that folder:

  machine 1  1. cold         empty HOME
             2. warm         same HOME; VLLM_ENABLE_STARTUP_PLAN=1 saves the memory plan from this warm start
             3. warm + plan  same HOME; the plan is applied, so memory profiling is skipped
  machine 2  4. bundle       a fresh container given only machine 1's HOME as a tarball, with the plan on and
                             VLLM_FORCE_AOT_LOAD=1, so a compile cache that fails to load is an error, not a rebuild

Each run records seconds to /health, the start-up lines (compile, AOT load, warm-up, CUDA graphs, startup plan),
the KV-cache tokens, and decode tok/s with 8 requests of 256 tokens in flight.

Run:  modal run benchmarks/modal_warm_serve.py
It writes modal_warm_serve.json beside where it is run, and keeps logs and the bundle in the Modal volume
"rigtune-triton-cache" under warm-serve/ (the Qwen3.5-9B weights are already there).
"""

import json

import modal

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
GPU = "A100"  # 40 GB: the 9B in bf16 and its MTP head fit at gpu_memory_utilization 0.95

app = modal.App("rigtune-warm-serve")
image = modal.Image.from_registry("vllm/vllm-openai:v0.33.0", add_python="3.12").entrypoint([])
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)

ARGV = ["vllm", "serve", MODEL, "--revision", REVISION, "--port", "8000", "--dtype", "bfloat16",
        "--max-model-len", "8192", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.95",
        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3})]
PATTERNS = {
    "warmup_s": r"\[backbone\] Initial profiling/warmup run took ([\d.]+) s",
    "torch_compile_s": r"\[backbone\] torch\.compile took ([\d.]+) s",
    "capture_s": r"Graph capturing finished in (\d+) secs",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
}
BUNDLE = "/vol/warm-serve/home.tar"


def _serve(name: str, home: str, extra_env: dict) -> dict:
    """Start the winner with HOME at `home`, wait for /health, probe decode, stop it, and read its log."""
    import os
    import re
    import subprocess
    import threading
    import time
    import urllib.request

    os.makedirs("/vol/warm-serve", exist_ok=True)
    # Inductor's default is /tmp/torchinductor_<user>, outside HOME, so it is pinned inside it too.
    env = {**os.environ, "HOME": home, "HF_HOME": "/vol/hf", "TRITON_CACHE_DIR": f"{home}/.cache/rigtune-triton",
           "TORCHINDUCTOR_CACHE_DIR": f"{home}/.cache/inductor", **extra_env}
    log_path = f"/vol/warm-serve/{name}.log"
    t0, ready = time.time(), None
    row = {"run": name, "env": extra_env}
    with open(log_path, "w") as log:
        proc = subprocess.Popen(ARGV, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        while time.time() - t0 < 1800 and proc.poll() is None:
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                ready = time.time() - t0
                break
            except Exception:
                time.sleep(2)
        if ready is not None:
            done = []

            def one():
                body = json.dumps({"model": MODEL, "prompt": "Tell me a long story about a lighthouse keeper.",
                                   "max_tokens": 256, "temperature": 0, "ignore_eos": True}).encode()
                req = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=body,
                                             headers={"Content-Type": "application/json"})
                done.append(json.loads(urllib.request.urlopen(req, timeout=600).read()))

            one()  # warm-up request, not timed
            done.clear()
            threads = [threading.Thread(target=one) for _ in range(8)]
            t1 = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            row["tok_s_8_users"] = round(sum(r["usage"]["completion_tokens"] for r in done) / (time.time() - t1), 1)
        try:
            os.killpg(proc.pid, 15)
            proc.wait(timeout=120)
        except Exception:
            subprocess.run("pkill -9 -f 'vllm serve'", shell=True)
    subprocess.run("pkill -9 -f 'vllm serve'; sleep 5", shell=True)
    text = open(log_path, encoding="utf-8", errors="replace").read()
    row["ready_s"] = round(ready, 1) if ready is not None else None
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text)
        row[key] = (int(m.group(1).replace(",", "")) if key == "kv_cache_tokens" else float(m.group(1))) if m else None
    row["aot_loaded"] = "Directly load AOT compilation" in text
    row["startup_plan_applied"] = "Applying persisted startup plan" in text
    row["autotune_lines"] = text.count("Triton autotuning for function")
    if ready is None:
        row["tail"] = "\n".join(text.strip().splitlines()[-6:])
    print(json.dumps(row), flush=True)
    volume.commit()
    return row


@app.function(gpu=GPU, image=image, volumes={"/vol": volume}, timeout=3600)
def machine_one() -> list:
    import shutil
    import subprocess

    home = "/tmp/home1"
    shutil.rmtree(home, ignore_errors=True)
    rows = [_serve("1-cold", home, {}),
            _serve("2-warm", home, {"VLLM_ENABLE_STARTUP_PLAN": "1"}),
            _serve("3-warm-plan", home, {"VLLM_ENABLE_STARTUP_PLAN": "1"})]
    subprocess.run(["tar", "-cf", BUNDLE, "-C", home, "."], check=True)
    size = subprocess.check_output(["du", "-sh", BUNDLE], text=True).split()[0]
    print(json.dumps({"bundle": BUNDLE, "size": size}), flush=True)
    volume.commit()
    return rows + [{"bundle_size": size}]


@app.function(gpu=GPU, image=image, volumes={"/vol": volume}, timeout=3600)
def machine_two() -> list:
    import os
    import shutil
    import subprocess

    volume.reload()
    home = "/tmp/home2"
    shutil.rmtree(home, ignore_errors=True)
    os.makedirs(home)
    subprocess.run(["tar", "-xf", BUNDLE, "-C", home], check=True)
    return [_serve("4-bundle-fresh-machine", home, {"VLLM_ENABLE_STARTUP_PLAN": "1", "VLLM_FORCE_AOT_LOAD": "1"})]


@app.local_entrypoint()
def main() -> None:
    rows = machine_one.remote() + machine_two.remote()
    with open("modal_warm_serve.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, indent=2)
    cols = ["run", "ready_s", "aot_loaded", "startup_plan_applied", "warmup_s", "torch_compile_s", "capture_s",
            "kv_cache_tokens", "tok_s_8_users"]
    print("\n" + " | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c)) for c in cols) if "run" in r else f"bundle size: {r.get('bundle_size')}")
