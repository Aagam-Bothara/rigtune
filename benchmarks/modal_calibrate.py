"""One rigtune command on a Modal GPU, rigtune installed from this repo into vLLM 0.33's own Python, or with
--engine sglang into a Python with SGLang 0.5.23 from PyPI (as the November GPU sessions installed it).

  modal run benchmarks/modal_calibrate.py --name qwen9b-a100-bench --gpu A100-80GB --hours 1 \
      --command "bench Qwen/Qwen3.5-9B --speculative all --budget 20m --save"
  modal run benchmarks/modal_calibrate.py --engine sglang --name qwen3-8b-sglang --gpu A100-80GB --hours 1.25 \
      --command "bench Qwen/Qwen3-8B --backend sglang --speculative all --budget 40m --save"
  modal run benchmarks/modal_calibrate.py --name qwen35b-h200-compare --gpu H200 --hours 4 --hf-local \
      --command "compare Qwen/Qwen3.5-35B-A3B --workload-file {prompts}/dolly.jsonl ... --out {out}/results"

In the command, {prompts} is the folder holding dolly.jsonl and dolly-heldout.jsonl (made here with
make_dolly_prompts.py, the 300 + 300 prompts of the November sessions, and sent along) and {out} the run's folder.
The run's folder is runs/<name>/ in the Modal volume "rigtune-triton-cache": the command's output (run.log) and
rigtune's home without its kernel cache (profiles, ledger), committed every 5 minutes so that a run cut off at
--hours keeps what it had. Weights go to the volume's hf/ (where Qwen3.5-9B already is), or with --hf-local to the
container's own disk, which a 67 GiB checkpoint fills faster than a volume and leaves nothing to pay for after.
"""

import base64
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent

app = modal.App("rigtune-calibrate")
# As in modal_bundle_e2e.py: rigtune goes into the Python that imports vllm, not the one Modal adds first on PATH.
INSTALL = r'''
set -e
PY=""
for p in /usr/bin/python3 /usr/bin/python3.12 /usr/local/bin/python3 /opt/venv/bin/python3; do
  if [ -x "$p" ] && "$p" -c "import vllm" 2>/dev/null; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "no Python here imports vllm"; exit 1; }
"$PY" -m pip install --no-cache-dir /src/rigtune \
  || "$PY" -m pip install --no-cache-dir --break-system-packages /src/rigtune \
  || uv pip install --system --python "$PY" /src/rigtune
cat > /usr/local/bin/rigtune-vllm <<EOF
#!/bin/sh
exec "$PY" -c "import sys; sys.argv[0] = 'rigtune'; from rigtune.cli import app; app()" "\$@"
EOF
chmod +x /usr/local/bin/rigtune-vllm
rigtune-vllm --help > /dev/null
'''
image = (modal.Image.from_registry("vllm/vllm-openai:v0.33.0", add_python="3.12")
         .entrypoint([])
         .add_local_dir(REPO, "/src/rigtune", copy=True,
                        ignore=[".git", "**/__pycache__", "benchmarks", "tests", "research_notes", "reports",
                                "*.egg-info", "build", "dist"])
         .run_commands(f"echo {base64.b64encode(INSTALL.encode()).decode()} | base64 -d > /tmp/install.sh "
                       "&& bash /tmp/install.sh"))
# SGLang 0.5.23 from PyPI on a CUDA 13.0 image, as benchmarks/modal_sglang_compile_oom.py and the GPU sessions, with
# rigtune from this repo in the same Python; `rigtune` is then on PATH.
sglang_image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
                .entrypoint([])
                .apt_install("libnuma1")
                .pip_install("sglang==0.5.23", "nvidia-ml-py")
                .add_local_dir(REPO, "/src/rigtune", copy=True,
                               ignore=[".git", "**/__pycache__", "benchmarks", "tests", "research_notes", "reports",
                                       "*.egg-info", "build", "dist"])
                .run_commands("pip install --no-cache-dir /src/rigtune && rigtune --help > /dev/null"))
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-80GB", timeout=3600)
def run(name: str, command: str, prompts: dict, hf_local: bool) -> dict:
    return _run(name, command, prompts, hf_local, "rigtune-vllm")


@app.function(image=sglang_image, volumes={"/vol": volume}, gpu="A100-80GB", timeout=3600)
def run_sglang(name: str, command: str, prompts: dict, hf_local: bool) -> dict:
    return _run(name, command, prompts, hf_local, "rigtune")


def _run(name: str, command: str, prompts: dict, hf_local: bool, cli: str) -> dict:
    import os
    import shlex
    import shutil
    import subprocess
    import threading
    import time

    out = pathlib.Path("/vol/runs") / name
    shutil.rmtree(out, ignore_errors=True)
    (out / "prompts").mkdir(parents=True)
    for file, text in prompts.items():
        (out / "prompts" / file).write_text(text)
    home = pathlib.Path("/root/rigtune-home")
    env = {**os.environ, "RIGTUNE_HOME": str(home), "HF_HOME": "/root/hf" if hf_local else "/vol/hf",
           "COLUMNS": "200"}
    argv = [cli, *shlex.split(command.format(prompts=out / "prompts", out=out))]

    def keep() -> None:
        if home.exists():
            shutil.copytree(home, out / "rigtune-home", dirs_exist_ok=True, ignore=shutil.ignore_patterns(
                "kernel-cache"))
        volume.commit()

    stop = threading.Event()

    def every_five_minutes() -> None:
        while not stop.wait(300):
            try:
                keep()
            except Exception as exc:  # a failed commit must not end the run
                print("commit failed:", exc, flush=True)

    threading.Thread(target=every_five_minutes, daemon=True).start()
    t0 = time.time()
    with open(out / "run.log", "w") as log:
        log.write("$ " + shlex.join(argv) + "\n")
        log.flush()
        code = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    stop.set()
    keep()
    lines = (out / "run.log").read_text(errors="replace").splitlines()
    return {"name": name, "exit": code, "seconds": round(time.time() - t0, 1), "tail": lines[-40:],
            "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
                            and "rigtune-home" not in p.parts)}


@app.local_entrypoint()
def main(name: str, command: str, gpu: str = "A100-80GB", hours: float = 1.0, hf_local: bool = False,
         engine: str = "vllm") -> None:
    import json
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for file, offset in (("dolly.jsonl", 0), ("dolly-heldout.jsonl", 300)):
            subprocess.run([sys.executable, str(REPO / "benchmarks" / "make_dolly_prompts.py"), "--offset",
                            str(offset), "--out", f"{tmp}/{file}"], check=True)
        prompts = {f: pathlib.Path(tmp, f).read_text(encoding="utf-8")
                   for f in ("dolly.jsonl", "dolly-heldout.jsonl")}
    fn = run_sglang if engine == "sglang" else run
    result = fn.with_options(gpu=gpu, timeout=int(hours * 3600)).remote(name, command, prompts, hf_local)
    with open(f"modal_{name}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
    print(f"{name}: exit {result['exit']} in {result['seconds']} s")
    print("\n".join(result["tail"]))
