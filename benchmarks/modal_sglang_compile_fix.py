"""Does the fix in sessions/2026-11/modal-sglang-compile-oom fix SGLang's torch.compile start-up on Qwen3.5-9B,
and leave everything else as it was?

Four start-ups in one Modal A100 80GB container, with the GPU session's settings:

  A  unpatched, eager          the reference: answers and decode speed
  B  patched,   eager          answers must equal A's exactly: the patch changes nothing without torch.compile
  C  patched,   torch.compile  must start, keep no buffer of 1 GiB or more, and answer like A
  D  unpatched, torch.compile  the control: still fails to start in this same container

Each started run answers four prompts greedily (64 tokens) and times decode at 1 and 8 users (256 tokens each).

Run:  modal run benchmarks/modal_sglang_compile_fix.py                 SGLang 0.5.23 from PyPI, sglang-0.5.23-fix.diff
      modal run benchmarks/modal_sglang_compile_fix.py --sglang main   SGLang's nightly dev image (main at 03fde9cc,
                                                                       20 Nov 2026), sglang-main-fix.diff
It writes modal_sglang_compile_fix[-main].json beside where it is run, and keeps logs in the Modal volume
"rigtune-triton-cache" under sglang-fix/ (the Qwen3.5-9B weights are already there).
"""

import json
import pathlib

import modal

MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SESSION = pathlib.Path(__file__).parent / "sessions/2026-11/modal-sglang-compile-oom"
# Main at 03fde9cc: the three patched files, and the three they call, are identical to the branch's base 08f10090c7.
MAIN_IMAGE = "lmsysorg/sglang:nightly-dev-cu13-20261120-03fde9cc"

app = modal.App("rigtune-sglang-compile-fix")
release_image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
                 .entrypoint([])
                 .apt_install("libnuma1", "git")
                 .pip_install("sglang==0.5.23", "nvidia-ml-py")
                 .add_local_file(SESSION / "sglang-0.5.23-fix.diff", "/patch/fix.diff"))
# SGLang's own image keeps SGLang in its own Python; the one Modal adds runs only this file's stdlib-only loop.
main_image = (modal.Image.from_registry(MAIN_IMAGE, add_python="3.12")
              .entrypoint([])
              .apt_install("git")
              .add_local_file(SESSION / "sglang-main-fix.diff", "/patch/fix.diff"))
volume = modal.Volume.from_name("rigtune-triton-cache", create_if_missing=True)

SERVER_ARGS = ["-m", "sglang.launch_server", "--model-path", MODEL, "--revision", REVISION,
               "--dtype", "bfloat16", "--host", "127.0.0.1", "--port", "30000",
               "--context-length", "262144", "--max-running-requests", "256", "--mem-fraction-static", "0.88"]
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Q: A train travels 120 km in 2 hours. What is its average speed?\nA:",
    "Write three short tips for staying focused while studying:\n1.",
]
GIB = 2 ** 30
ITEMSIZE = {"torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2, "torch.int64": 8, "torch.int32": 4}


def big_buffers(root: str, floor: int = GIB) -> list:
    """Every literal-size empty_strided_cuda of `floor` bytes or more in Inductor's generated code under `root`."""
    import glob
    import re

    out = []
    for path in glob.glob(f"{root}/**/*.py", recursive=True):
        for i, line in enumerate(open(path, encoding="utf-8", errors="replace")):
            m = re.search(r"(buf\d+) = empty_strided_cuda\(\(([\d, ]+)\), \([^)]*\), (torch\.\w+)\)", line)
            if not m:
                continue
            numel = 1
            for d in m.group(2).replace(" ", "").rstrip(",").split(","):
                numel *= int(d)
            nbytes = numel * ITEMSIZE.get(m.group(3), 4)
            if nbytes >= floor:
                out.append({"file": path.rsplit("/", 1)[-1], "line": i + 1, "gib": round(nbytes / GIB, 3),
                            "code": line.strip()})
    return out


def _experiment(label: str) -> dict:
    import os
    import re
    import subprocess
    import threading
    import time
    import urllib.request

    # The interpreter that has SGLang: the image's own, not the one Modal adds.
    py = next(p for p in ("/usr/bin/python3", "/usr/local/bin/python3", "python3", "python")
              if subprocess.run([p, "-c", "import sglang"], capture_output=True).returncode == 0)
    probe = ("import os, sglang, torch; "
             "print(os.path.dirname(os.path.dirname(sglang.__file__)), sglang.__version__, torch.__version__)")
    site, sglang_version, torch_version = subprocess.check_output([py, "-c", probe], text=True).split()
    os.makedirs("/vol/sglang-fix", exist_ok=True)

    def patch(reverse: bool) -> None:
        argv = ["git", "apply", "-v", "-p2"] + (["-R"] if reverse else []) + ["/patch/fix.diff"]
        subprocess.run(argv, cwd=site, check=True, env={**os.environ, "GIT_DIR": "/nonexistent"})
        src = open(f"{site}/sglang/srt/layers/attention/linear/kernels/gdn_triton.py").read()
        assert ("gdn_packed_decode" in src) is not reverse, "the patch did not apply as expected"

    def post(path: str, body: dict, timeout: float = 600) -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:30000{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def complete(prompt: str, tokens: int, ignore_eos: bool = False) -> dict:
        return post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": tokens,
                                        "temperature": 0, "ignore_eos": ignore_eos})

    def decode_speed(users: int) -> float:
        """Generated tokens per second with `users` requests of 256 tokens in flight at once."""
        done = []

        def one():
            done.append(complete("Tell me a long story about a lighthouse keeper.", 256, ignore_eos=True))

        threads = [threading.Thread(target=one) for _ in range(users)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return round(sum(r["usage"]["completion_tokens"] for r in done) / (time.time() - t0), 1)

    def run(name: str, compile_: bool) -> dict:
        name = f"{label}-{name}"
        inductor = f"/tmp/inductor/{name}"
        env = {**os.environ, "HF_HOME": "/vol/hf", "TORCHINDUCTOR_CACHE_DIR": inductor}
        log_path = f"/vol/sglang-fix/{name}.log"
        argv = [py] + SERVER_ARGS + (["--enable-torch-compile"] if compile_ else [])
        t0, row = time.time(), {"run": name, "torch_compile": compile_}
        with open(log_path, "w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            ready = None
            # A fixed start compiles and autotunes every layer at every batch size up to 32: give it an hour.
            while time.time() - t0 < 3600 and proc.poll() is None:
                try:
                    urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2)
                    ready = time.time() - t0
                    break
                except Exception:
                    time.sleep(3)
            row["ready_s"] = round(ready, 1) if ready is not None else None
            if ready is not None:
                complete("Hello", 8)  # warm-up
                row["answers"] = [complete(p, 64)["choices"][0]["text"] for p in PROMPTS]
                row["tok_s_1_user"] = decode_speed(1)
                row["tok_s_8_users"] = decode_speed(8)
            try:
                os.killpg(proc.pid, 15)
                proc.wait(timeout=120)
            except Exception:
                subprocess.run("pkill -9 -f sglang.launch_server", shell=True)
        subprocess.run("pkill -9 -f sglang.launch_server; sleep 5", shell=True)
        text = open(log_path, encoding="utf-8", errors="replace").read()
        m = re.search(r"Tried to allocate ([\d.]+ [GM]iB)", text)
        row["failed_allocation"] = m.group(1) if m else None
        row["big_buffers"] = big_buffers(inductor)
        if ready is None:
            row["tail"] = "\n".join(text.strip().splitlines()[-4:])
        print(json.dumps({k: v for k, v in row.items() if k != "answers"}), flush=True)
        volume.commit()
        return row

    rows = [run("A-unpatched-eager", False)]
    patch(reverse=False)
    rows += [run("B-patched-eager", False), run("C-patched-compile", True)]
    patch(reverse=True)
    rows.append(run("D-unpatched-compile", True))

    ref = rows[0].get("answers")
    for r in rows[1:]:
        if ref and r.get("answers"):
            r["answers_equal_A"] = [a == b for a, b in zip(ref, r["answers"])]
    return {"versions": {"sglang": sglang_version, "torch": torch_version}, "runs": rows}


@app.function(gpu="A100-80GB", image=release_image, volumes={"/vol": volume}, timeout=3 * 3600)
def experiment() -> dict:
    return _experiment("release")


@app.function(gpu="A100-80GB", image=main_image, volumes={"/vol": volume}, timeout=3 * 3600)
def experiment_main() -> dict:
    return _experiment("main")


@app.local_entrypoint()
def main(sglang: str = "0.5.23") -> None:
    if sglang not in ("0.5.23", "main"):
        raise SystemExit("--sglang is 0.5.23 or main")
    result = (experiment_main if sglang == "main" else experiment).remote()
    out = "modal_sglang_compile_fix" + ("-main" if sglang == "main" else "") + ".json"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
    print("\n", result["versions"])
    for r in result["runs"]:
        print(f"{r['run']}: ready={r['ready_s']} failed={r['failed_allocation']} "
              f"big_buffers={len(r['big_buffers'])} tok/s 1 user={r.get('tok_s_1_user')} "
              f"8 users={r.get('tok_s_8_users')} answers equal A={r.get('answers_equal_A')}")
