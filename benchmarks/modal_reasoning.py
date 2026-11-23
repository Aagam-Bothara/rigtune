"""The reasoning experiment (rigtune.reasoning) on a Modal GPU: vLLM 0.33 serves the model in the same container, and
`python -m rigtune.reasoning` runs every method against it.

  modal run benchmarks/modal_reasoning.py --name qwen3-4b-dev --model Qwen/Qwen3-4B --gpu L4 \\
      --args "--split dev --methods direct,self-consistency:3,self-consistency:5,self-correction,adaptive:8 --thinking off"

rigtune with its `reason` extra (pyarrow, math-verify) is installed into vLLM's own Python, as modal_calibrate.py does.
The server is stock vLLM with --max-num-seqs at the experiment's sequence limit and a context long enough for a
critique turn; its launch line is recorded as the runs' server, so only runs served alike are compared. Outputs go to
the Modal volume "rigtune-triton-cache" under runs/<name>/ (manifest.json, runs.jsonl, report.md, server.log); MATH's
problems are kept there too (math/), so later runs read them without the Hub.
"""

import base64
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("rigtune-reasoning")
INSTALL = r'''
set -e
PY=""
for p in /usr/bin/python3 /usr/bin/python3.12 /usr/local/bin/python3 /opt/venv/bin/python3; do
  if [ -x "$p" ] && "$p" -c "import vllm" 2>/dev/null; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "no Python here imports vllm"; exit 1; }
"$PY" -m pip install --no-cache-dir "/src/rigtune[reason]" \
  || "$PY" -m pip install --no-cache-dir --break-system-packages "/src/rigtune[reason]" \
  || uv pip install --system --python "$PY" "/src/rigtune[reason]"
cat > /usr/local/bin/vllm-python <<EOF
#!/bin/sh
exec "$PY" "\$@"
EOF
chmod +x /usr/local/bin/vllm-python
vllm-python -c "import torch, vllm, rigtune.reasoning.experiment, math_verify, pyarrow"
'''
image = (modal.Image.from_registry("vllm/vllm-openai:v0.33.0", add_python="3.12")
         .entrypoint([])
         .add_local_dir(REPO, "/src/rigtune", copy=True,
                        ignore=[".git", "**/__pycache__", "benchmarks", "tests", "research_notes", "reports",
                                "*.egg-info", "build", "dist"])
         .run_commands(f"echo {base64.b64encode(INSTALL.encode()).decode()} | base64 -d > /tmp/install.sh "
                       "&& bash /tmp/install.sh"))
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", timeout=3 * 3600)
def run(name: str, model: str, args: str, max_model_len: int, sequences: int, gpu_memory: float) -> dict:
    if "--sequences" in args or "--context" in args:
        raise ValueError("pass --sequences and --max-model-len to this runner, not in --args: the server must match")
    import os
    import shlex
    import shutil
    import subprocess
    import time
    import urllib.request

    out = pathlib.Path("/vol/runs") / name
    if (out / "runs.jsonl").exists():
        raise ValueError(f"runs/{name} already holds runs; choose another --name")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    env = {**os.environ, "HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1"}  # progress lines reach run.log as they come
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    serve = ["vllm", "serve", model, "--port", "8000", "--max-model-len", str(max_model_len),
             "--max-num-seqs", str(sequences), "--gpu-memory-utilization", str(gpu_memory),
             "--enable-prompt-tokens-details"]  # so each reply says how much of its prompt came from the cache
    server_log = open(out / "server.log", "w")
    server = None
    try:
        server = subprocess.Popen(serve, stdout=server_log, stderr=subprocess.STDOUT, env=env)
        deadline = time.time() + 1800
        while True:
            if server.poll() is not None:
                tail = (out / "server.log").read_text(errors="replace").splitlines()[-15:]
                raise RuntimeError(f"vLLM exited with {server.returncode}:\n" + "\n".join(tail))
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("vLLM did not become ready in 30 minutes")
                time.sleep(5)
        label = f"vllm 0.33.0 stock on {gpu}: " + shlex.join(serve[2:])
        cmd = ["vllm-python", "-m", "rigtune.reasoning", "--base-url", "http://127.0.0.1:8000", "--model", model,
               "--server", label, "--context", str(max_model_len), "--sequences", str(sequences),
               "--cache-dir", "/vol/math", "--out", str(out), *shlex.split(args)]
        with open(out / "run.log", "w") as log:
            log.write("$ " + shlex.join(cmd) + "\n")
            log.flush()
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=60)
            except Exception:
                server.kill()
        server_log.close()
        volume.commit()
    report = out / "report.md"
    return {"name": name, "exit": code, "report": report.read_text() if report.exists() else None,
            "tail": (out / "run.log").read_text(errors="replace").splitlines()[-30:]}


@app.local_entrypoint()
def main(name: str, model: str, args: str = "", gpu: str = "L4", hours: float = 2.0, max_model_len: int = 20480,
         sequences: int = 32, gpu_memory: float = 0.9) -> None:
    """--sequences 32 suits the 98-problem dev set (a batch must be several times the limit to keep it full); the
    held-out set of 2,440 can take the engine's own maximum."""
    import json

    result = run.with_options(gpu=gpu, timeout=int(hours * 3600)).remote(name, model, args, max_model_len, sequences,
                                                                         gpu_memory)
    with open(f"modal_{name}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
    print(f"{name}: exit {result['exit']}")
    print(result["report"] or "\n".join(result["tail"]))
