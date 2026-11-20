"""rigtune's own commands, end to end: calibrate, bundle, and serve the pick warm on a fresh machine.

Three Modal A100 80GB containers, rigtune installed from this repo into vLLM 0.33's own Python:

  machine 1  rigtune bench Qwen/Qwen3.5-9B --speculative all --budget 20m; rigtune bundle export;
             rigtune serve --profile (a warm start on the machine that calibrated)
  machine 2  rigtune bundle import; rigtune serve --profile   (a fresh machine given the bundle)
  machine 3  rigtune serve --profile, no bundle                (a fresh machine, cold: the control)

Each serve is timed to the proxy's first 200 from /health, answers one request, and reports the start record
rigtune serve keeps under /health's backend.startup (seconds, compile-cache hit, warm-up).

Run:  modal run benchmarks/modal_bundle_e2e.py
It writes modal_bundle_e2e.json beside where it is run, and keeps logs, the profile and the bundle in the Modal volume
"rigtune-triton-cache" under bundle-e2e/ (the Qwen3.5-9B weights are already there).
"""

import base64
import json
import pathlib

import modal

MODEL = "Qwen/Qwen3.5-9B"
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = "/vol/bundle-e2e"

app = modal.App("rigtune-bundle-e2e")
# rigtune goes into vLLM's own Python, which its probe needs (it imports vllm), not the Python Modal adds: that
# one comes first on PATH as python3, and the first run installed rigtune there, found no engine, and stopped.
# pip keeps every package already installed that meets rigtune's bounds; the freeze before and after shows what
# it added. `rigtune-vllm` runs rigtune with that Python.
INSTALL = r'''
set -e
PY=""
for p in /usr/bin/python3 /usr/bin/python3.12 /usr/local/bin/python3 /opt/venv/bin/python3; do
  if [ -x "$p" ] && "$p" -c "import vllm" 2>/dev/null; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "no Python here imports vllm"; exit 1; }
echo "vLLM's Python: $PY"
("$PY" -m pip freeze || uv pip freeze --python "$PY") > /tmp/before.txt
"$PY" -m pip install --no-cache-dir /src/rigtune \
  || "$PY" -m pip install --no-cache-dir --break-system-packages /src/rigtune \
  || uv pip install --system --python "$PY" /src/rigtune
("$PY" -m pip freeze || uv pip freeze --python "$PY") > /tmp/after.txt
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
         # Each command becomes one Dockerfile RUN line, so the script travels as one base64 line.
         .run_commands(f"echo {base64.b64encode(INSTALL.encode()).decode()} | base64 -d > /tmp/install.sh "
                       "&& bash /tmp/install.sh"))
RIGTUNE = "rigtune-vllm"
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)


def _env(home: str) -> dict:
    import os

    return {**os.environ, "RIGTUNE_HOME": home, "HF_HOME": "/vol/hf", "COLUMNS": "200"}


def _run(name: str, argv: list, home: str, timeout: float) -> dict:
    import subprocess
    import time

    t0 = time.time()
    done = subprocess.run(argv, capture_output=True, text=True, env=_env(home), timeout=timeout)
    text = done.stdout + "\n--- stderr ---\n" + done.stderr
    pathlib.Path(f"{OUT}/{name}.log").write_text(text)
    return {"step": name, "exit": done.returncode, "seconds": round(time.time() - t0, 1),
            "tail": text.strip().splitlines()[-15:]}


def _serve(name: str, profile: str, home: str) -> dict:
    """rigtune serve --profile, timed to the proxy's first 200 from /health, one request, then stopped."""
    import os
    import subprocess
    import time
    import urllib.request

    row = {"step": name}
    t0 = time.time()
    with open(f"{OUT}/{name}.log", "w") as log:
        proc = subprocess.Popen([RIGTUNE, "serve", MODEL, "--profile", profile, "--port", "8100"], stdout=log,
                                stderr=subprocess.STDOUT, env=_env(home), start_new_session=True)
        while time.time() - t0 < 1800 and proc.poll() is None:
            try:
                health = json.loads(urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=3).read())
                row["ready_s"] = round(time.time() - t0, 1)
                row["startup"] = (health.get("backend") or {}).get("startup")
                break
            except Exception:
                time.sleep(3)
        if "ready_s" in row:
            body = json.dumps({"model": MODEL, "prompt": "The capital of France is", "max_tokens": 16,
                               "temperature": 0}).encode()
            req = urllib.request.Request("http://127.0.0.1:8100/v1/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                row["answer"] = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["text"]
            except Exception as exc:
                row["answer_error"] = repr(exc)
        try:
            os.killpg(proc.pid, 15)
            proc.wait(timeout=120)
        except Exception:
            subprocess.run("pkill -9 -f 'rigtune serve'; pkill -9 -f 'vllm serve'", shell=True)
    subprocess.run("pkill -9 -f 'vllm serve'; sleep 5", shell=True)
    text = pathlib.Path(f"{OUT}/{name}.log").read_text(errors="replace")
    row["log_lines"] = [line for line in text.splitlines() if "ready in" in line or "--profile" in line][-8:]
    if "ready_s" not in row:
        row["tail"] = text.strip().splitlines()[-15:]
    print(json.dumps(row), flush=True)
    volume.commit()
    return row


@app.function(gpu="A100-80GB", image=image, volumes={"/vol": volume}, timeout=3 * 3600)
def machine_one() -> dict:
    import os
    import shutil

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    home = "/tmp/rt1"
    added = sorted(set(open("/tmp/after.txt").read().splitlines()) - set(open("/tmp/before.txt").read().splitlines()))
    rows = {"pip_added": added}
    rows["bench"] = _run("1-bench", [RIGTUNE, "bench", MODEL, "--speculative", "all", "--budget", "20m", "--save"],
                         home, timeout=2 * 3600)
    found = sorted(pathlib.Path(home, "profiles").rglob("*.json"), key=lambda p: p.stat().st_mtime)
    if not found:
        return {**rows, "error": "bench saved no profile"}
    profile = str(found[-1])
    shutil.copy(profile, f"{OUT}/profile.json")
    pick = json.loads(pathlib.Path(profile).read_text())
    rows["pick"] = {"config": pick["config"], "backend_version": pick["backend_version"]}
    rows["export"] = _run("2-export", [RIGTUNE, "bundle", "export", profile, "--out", f"{OUT}/winner.tar.gz"],
                          home, timeout=1800)
    rows["bundle_mb"] = round(os.path.getsize(f"{OUT}/winner.tar.gz") / 2**20, 1)
    volume.commit()
    rows["serve_same_machine"] = _serve("3-serve-same-machine", profile, home)
    return rows


@app.function(gpu="A100-80GB", image=image, volumes={"/vol": volume}, timeout=3600)
def machine_two() -> dict:
    volume.reload()
    home = "/tmp/rt2"
    rows = {"import": _run("4-import", [RIGTUNE, "bundle", "import", f"{OUT}/winner.tar.gz", "--profile-out",
                                        "/tmp/p.json"], home, timeout=1800)}
    rows["serve_fresh_with_bundle"] = _serve("5-serve-fresh-with-bundle", "/tmp/p.json", home)
    return rows


@app.function(gpu="A100-80GB", image=image, volumes={"/vol": volume}, timeout=3600)
def machine_three() -> dict:
    import shutil

    volume.reload()
    shutil.copy(f"{OUT}/profile.json", "/tmp/p.json")
    return {"serve_fresh_cold": _serve("6-serve-fresh-cold", "/tmp/p.json", "/tmp/rt3")}


@app.local_entrypoint()
def main() -> None:
    one = machine_one.remote()
    result = {"machine_one": one}
    if "error" not in one:
        result["machine_two"] = machine_two.remote()
        result["machine_three"] = machine_three.remote()
    with open("modal_bundle_e2e.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
    print("\npip added:", one.get("pip_added"))
    print("bench exit", one.get("bench", {}).get("exit"), "in", one.get("bench", {}).get("seconds"), "s; pick",
          one.get("pick"), "; bundle", one.get("bundle_mb"), "MB")
    for machine, key in (("machine_one", "serve_same_machine"), ("machine_two", "serve_fresh_with_bundle"),
                         ("machine_three", "serve_fresh_cold")):
        r = result.get(machine, {}).get(key, {})
        print(f"{key}: ready {r.get('ready_s')} s, startup {r.get('startup')}, answer {r.get('answer')!r}")
        for line in r.get("log_lines", []):
            print("   ", line)
    if "machine_two" in result:
        print("import:", "\n  ".join(result["machine_two"]["import"]["tail"]))
