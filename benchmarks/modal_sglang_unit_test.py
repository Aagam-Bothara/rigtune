"""Does the regression test for SGLang #51651 fail without the fix and pass with it?

Runs test_gdn_torch_compile_pool_writes.py (from the fix branch in the local SGLang clone) on SGLang's nightly
dev image (main at 03fde9cc), first unpatched, then with sessions/2026-11/modal-sglang-compile-oom/
sglang-main-fix.diff applied. No model; one L4.

Run:  modal run benchmarks/modal_sglang_unit_test.py
"""

import pathlib

import modal

SESSION = pathlib.Path(__file__).parent / "sessions/2026-11/modal-sglang-compile-oom"
TEST = pathlib.Path.home() / "sglang/test/registered/kernels/ops/attention/test_gdn_torch_compile_pool_writes.py"
MAIN_IMAGE = "lmsysorg/sglang:nightly-dev-cu13-20261120-03fde9cc"

app = modal.App("rigtune-sglang-unit-test")
image = (modal.Image.from_registry(MAIN_IMAGE, add_python="3.12")
         .entrypoint([])
         .apt_install("git")
         .add_local_file(SESSION / "sglang-main-fix.diff", "/patch/fix.diff")
         .add_local_file(TEST, "/patch/test_gdn_torch_compile_pool_writes.py"))


@app.function(gpu="L4", image=image, timeout=1800)
def run_tests() -> dict:
    import os
    import subprocess

    py = next(p for p in ("/usr/bin/python3", "/usr/local/bin/python3", "python3", "python")
              if subprocess.run([p, "-c", "import sglang"], capture_output=True).returncode == 0)
    site = subprocess.check_output(
        [py, "-c", "import os, sglang; print(os.path.dirname(os.path.dirname(sglang.__file__)))"], text=True).strip()
    if subprocess.run([py, "-c", "import pytest"], capture_output=True).returncode:
        subprocess.run([py, "-m", "pip", "install", "-q", "pytest"], check=True)

    def pytest() -> str:
        done = subprocess.run([py, "-m", "pytest", "/patch/test_gdn_torch_compile_pool_writes.py", "-q", "-p",
                               "no:cacheprovider", "--tb=line"], capture_output=True, text=True)
        out = done.stdout + done.stderr
        print(out[-3000:], flush=True)
        return "\n".join(line for line in out.splitlines() if " passed" in line or " failed" in line
                         or line.startswith(("FAILED", "ERROR", "E ")) or "AssertionError" in line)

    unpatched = pytest()
    subprocess.run(["git", "apply", "-v", "-p2", "/patch/fix.diff"], cwd=site, check=True,
                   env={**os.environ, "GIT_DIR": "/nonexistent"})
    patched = pytest()
    return {"unpatched": unpatched, "patched": patched}


@app.local_entrypoint()
def main() -> None:
    result = run_tests.remote()
    print("\n=== unpatched (should fail) ===\n" + result["unpatched"])
    print("\n=== patched (should pass) ===\n" + result["patched"])
