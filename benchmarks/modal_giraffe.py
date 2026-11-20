"""Check rigtune's pick against stock vLLM with Giraffe, an outside endpoint test suite (github.com/0dust/giraffe).

On one Modal A100 80GB (the A100 session's card), vLLM 0.33 serves Qwen3.5-9B twice: stock (the model and port
alone) and rigtune's A100 pick (bf16, MTP at 3 tokens, --max-num-seqs 64, --max-model-len 8192,
--gpu-memory-utilization 0.95). Giraffe 0.1.0a2 runs its default checks (access, serving, correctness,
first_output) on stock, saves that run as the baseline, then runs on the pick against it. Qwen3.5 thinks by
default, so answers get 2,048 tokens, as Giraffe's docs advise for reasoning models.

Run:  modal run benchmarks/modal_giraffe.py
It writes modal_giraffe.json beside where it is run, and keeps Giraffe's reports (report.json, report.html) and
the engine logs in the Modal volume "rigtune-triton-cache" under giraffe/.
"""

import json

import modal

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

app = modal.App("rigtune-giraffe")
# Giraffe goes into the Python Modal adds; `vllm` keeps the image's own.
image = (modal.Image.from_registry("vllm/vllm-openai:v0.33.0", add_python="3.12")
         .entrypoint([])
         .pip_install("giraffe-llm==0.1.0a2"))
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)

STOCK = ["vllm", "serve", MODEL, "--revision", REVISION, "--port", "8000"]
PICK = STOCK + ["--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "64",
                "--gpu-memory-utilization", "0.95",
                "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3})]
GIRAFFE_ARGS = ["--url", "http://127.0.0.1:8000/v1", "--model", MODEL, "--max-output-tokens", "2048", "--html",
                "--fail-on", "never"]


@app.function(gpu="A100-80GB", image=image, volumes={"/vol": volume}, timeout=3 * 3600)
def check() -> dict:
    import os
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request

    giraffe = os.path.join(os.path.dirname(sys.executable), "giraffe")
    out = "/vol/giraffe"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    env = {**os.environ, "HF_HOME": "/vol/hf", "HOME": "/tmp/home",
           "TRITON_CACHE_DIR": "/tmp/home/.cache/rigtune-triton"}

    def serve_and_check(name: str, argv: list, extra: list) -> dict:
        row = {"run": name}
        t0 = time.time()
        with open(f"{out}/{name}-engine.log", "w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            while time.time() - t0 < 1800 and proc.poll() is None:
                try:
                    urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                    row["ready_s"] = round(time.time() - t0, 1)
                    break
                except Exception:
                    time.sleep(3)
            if "ready_s" in row:
                t1 = time.time()
                done = subprocess.run([giraffe, "run", *GIRAFFE_ARGS, "--output", f"{out}/{name}", *extra],
                                      capture_output=True, text=True, timeout=3600)
                row["giraffe_s"] = round(time.time() - t1, 1)
                row["giraffe_exit"] = done.returncode
                with open(f"{out}/{name}-giraffe.txt", "w") as fh:
                    fh.write(done.stdout + "\n--- stderr ---\n" + done.stderr)
                row["giraffe_tail"] = (done.stdout + done.stderr).strip().splitlines()[-40:]
            try:
                os.killpg(proc.pid, 15)
                proc.wait(timeout=120)
            except Exception:
                subprocess.run("pkill -9 -f 'vllm serve'", shell=True)
        subprocess.run("pkill -9 -f 'vllm serve'; sleep 5", shell=True)
        report = f"{out}/{name}/report.json"
        if os.path.exists(report):
            data = json.load(open(report))
            row["report"] = {k: data.get(k) for k in ("overall", "checks", "performance", "conditions")}
        print(json.dumps({k: v for k, v in row.items() if k != "report"}), flush=True)
        volume.commit()
        return row

    stock = serve_and_check("stock", STOCK, [])
    saved = subprocess.run([giraffe, "baseline", "save", f"{out}/stock", f"{out}/stock-baseline.json"],
                           capture_output=True, text=True)
    pick = serve_and_check("rigtune-pick", PICK, ["--baseline", f"{out}/stock-baseline.json"]
                           if saved.returncode == 0 else [])
    import importlib.metadata
    return {"giraffe": importlib.metadata.version("giraffe-llm"),
            "baseline_save": {"exit": saved.returncode, "tail": (saved.stdout + saved.stderr).strip()[-2000:]},
            "runs": [stock, pick]}


@app.local_entrypoint()
def main() -> None:
    result = check.remote()
    with open("modal_giraffe.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2, default=str)
    print("\ngiraffe", result["giraffe"], "| baseline save exit", result["baseline_save"]["exit"])
    for r in result["runs"]:
        print(f"\n=== {r['run']}: ready {r.get('ready_s')} s, giraffe {r.get('giraffe_s')} s, "
              f"exit {r.get('giraffe_exit')}")
        print("\n".join(r.get("giraffe_tail", [])))
