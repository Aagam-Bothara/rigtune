#!/usr/bin/env python
"""Run vLLM's own tuner, benchmarks/auto_tune/auto_tune.sh, and make its pick a row for `rigtune compare`.

auto_tune.sh (byte-identical at vLLM v0.29.0 and v0.33.0) grids --max-num-seqs by --max-num-batched-tokens,
after stepping --gpu-memory-utilization down from 0.98 until a server starts. Every server it measures runs
with `--load-format dummy` (random weights) and `--enable-prefix-caching`, and is driven by `vllm bench serve`
with random token prompts of one fixed length, `--ignore-eos`, open loop: 1000 prompts at request rate inf,
then 100 at a time at descending integer rates until its latency limit holds. Its pick is the highest request
throughput that met the limit. Those numbers are its own; the row this script writes is re-measured by
`rigtune compare` with the real weights, on the comparison's prompts and closed-loop levels, like every row.

Two rules for the limit, as on the 14 November 2026 pod:
- `--rule e2el99`: auto_tune's own, MAX_LATENCY_ALLOWED_MS on the P99 end-to-end latency.
- `--rule ttft95`: the same search judged on the p95 time to first token, rigtune's `balanced` ceiling. A
  500 ms P99 end-to-end latency with answers of a few hundred tokens is unreachable (decode alone takes
  seconds), so "the same limit as rigtune" only means something on the same metric. `vllm bench serve`
  (vllm/benchmarks/serve.py at both tags) prints `P{p} TTFT (ms):` for each of `--metric-percentiles`
  (default "99") and accepts `--goodput ttft:<ms>`, so the patch adds `--metric-percentiles 95,99`, reads
  `P95 TTFT (ms):` where the script read `P99 E2EL (ms):` (its variable keeps the name `e2el`, so result.txt's
  `e2el:` column then holds p95 TTFT), and sets `--goodput ttft:`. The search itself is untouched. TPOT is not
  limited under either rule.

Both rules also skip the final torch-profiler run of the best configuration (it does not change the pick,
writes large traces and costs minutes; `--final-profile` keeps it), and wait after each of auto_tune's four
`pkill -if "vllm serve"` until nvidia-smi lists no compute process, killing leftovers by PID
(`--no-gpu-hygiene` skips this): pkill matches the API server but not its VLLM::EngineCore child, which can
survive holding the memory and make every later combination fail to start. With `--rule e2el99
--final-profile --no-gpu-hygiene` the copy has the original's text (with LF line endings) and the diff is
empty. The original is never edited: the copy and its diff are written to --out, beside the result.

    python benchmarks/vllm_auto_tune.py lengths --profile PROFILE.json
    python benchmarks/vllm_auto_tune.py run --vllm-src ~/src/vllm --model Qwen/Qwen2.5-7B-Instruct \
        --rule ttft95 --limit-ms 500 --input-len 63 --output-len 395 --max-model-len 4096 --out at-qwen7b
    rigtune compare Qwen/Qwen2.5-7B-Instruct ... --rows at-qwen7b/rows.json

`run` needs bash, nvidia-smi, bc, curl and an installed vLLM; --vllm-src is a checkout of the vLLM repository
whose directory is named `vllm` (auto_tune runs `cd "$BASE/vllm"` and `git rev-parse HEAD` there). A sparse
checkout of benchmarks/auto_tune at the installed release's tag is enough, and keeps an uncompiled `vllm/`
package off the path. `patch` and `row` work on their own for a run made by hand.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

RULES = ("e2el99", "ttft95")
LABEL = "autotune-vllm"

# (old, new, how many times old must occur): a different auto_tune.sh fails loudly, never half-patched.
METRIC = [
    ("--percentile-metrics ttft,tpot,itl,e2el \\",
     "--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 95,99 \\", 3),
    ('grep "P99 E2EL (ms):"', 'grep "P95 TTFT (ms):"', 2),
    ('--goodput e2el:"$MAX_LATENCY_ALLOWED_MS"', '--goodput ttft:"$MAX_LATENCY_ALLOWED_MS"', 3),
]
PROFILE = [('if (( $(echo "$best_throughput > 0" | bc -l) )); then',
            'if [[ "${AT_FINAL_PROFILE:-0}" == 1 ]] && (( $(echo "$best_throughput > 0" | bc -l) )); then',
            1)]
HOST = [("HOSTNAME=$(hostname)", "HOSTNAME=127.0.0.1", 1)]
HYGIENE_FN = """# --- added by rigtune's benchmarks/vllm_auto_tune.py: wait for an empty GPU after pkill ---
at_gpu_wait() {
    local i p
    for i in $(seq 1 30); do
        [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]] && return 0
        sleep 2
    done
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
        echo "at_gpu_wait: killing leftover GPU process $p" >&2
        kill -KILL "$p" 2>/dev/null
    done
    sleep 5
}
# --- end of addition ---
"""
HYGIENE = [('pkill -if "vllm serve" || true', 'pkill -if "vllm serve" || true; at_gpu_wait', 4),
           ('TAG=$(date +"%Y_%m_%d_%H_%M")', HYGIENE_FN + 'TAG=$(date +"%Y_%m_%d_%H_%M")', 1)]


class AutoTuneError(RuntimeError):
    pass


def patch(text: str, rule: str, host_ip: bool = False, gpu_hygiene: bool = True,
          final_profile: bool = False) -> str:
    """auto_tune.sh with the chosen rule and run changes applied (see the module docstring)."""
    if rule not in RULES:
        raise AutoTuneError(f"rule must be one of {', '.join(RULES)}")
    plan = ((METRIC if rule == "ttft95" else []) + ([] if final_profile else PROFILE)
            + (HOST if host_ip else []) + (HYGIENE if gpu_hygiene else []))
    new = text
    for old, repl, n in plan:
        got = new.count(old)
        if got != n:
            raise AutoTuneError(f"auto_tune.sh is not the one this was written for (vLLM v0.29.0 / v0.33.0): "
                                f"{old!r} occurs {got} times, expected {n}")
        new = new.replace(old, repl)
    return new


def write_patched(src: Path, out_dir: Path, rule: str, **kw) -> Tuple[Path, Path]:
    """The patched copy and its unified diff against the original, both in out_dir."""
    text = src.read_text(encoding="utf-8")
    new = patch(text, rule, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy, diff = out_dir / "auto_tune.sh", out_dir / "auto_tune.diff"
    copy.write_bytes(new.encode("utf-8"))  # bytes, so a Windows checkout cannot turn it into CRLF
    copy.chmod(0o755)
    lines = difflib.unified_diff(text.splitlines(True), new.splitlines(True), str(src), "auto_tune.sh (copy)")
    diff.write_bytes("".join(lines).encode("utf-8"))
    return copy, diff


# --------------------------------------------------------------------------- lengths


def lengths_from_profile(profile: dict) -> Optional[Tuple[int, int, int]]:
    """Mean prompt and answer tokens over every measured level of every successful calibration trial, and the
    number of requests behind them: auto_tune sends one fixed length, so the mean is the match. None when the
    profile recorded no token counts."""
    p_sum = o_sum = 0.0
    p_n = o_n = 0
    for trial in profile.get("calibration_table") or []:
        m = trial.get("metrics") or {}
        for lv in list((m.get("by_concurrency") or {}).values()) or [m]:
            ok = int(lv.get("requests") or 0) - int(lv.get("failed") or 0)
            if ok <= 0 or not lv.get("output_tokens"):
                continue
            o_sum += int(lv["output_tokens"])
            o_n += ok
            if lv.get("prompt_tokens"):
                p_sum += float(lv["prompt_tokens"]) * ok
                p_n += ok
    if not o_n or not p_n:
        return None
    return max(1, round(p_sum / p_n)), max(1, round(o_sum / o_n)), o_n


# --------------------------------------------------------------------------- reading its output

BEST = re.compile(r"best_max_num_seqs: (\d+), best_num_batched_tokens: (\d+), best_throughput: ([0-9.]+)")
MEASURED = re.compile(r"max_num_seqs: (\d+), max_num_batched_tokens: (\d+), request_rate: (\S+), "
                      r"e2el: (\S+), throughput: (\S+), goodput: (\S+)")
FAILED = re.compile(r"max_num_seqs: (\d+), max_num_batched_tokens: (\d+) does not meet latency requirement")
# bc prints 0.97 as ".97": `echo "0.98 - 0.01" | bc`.
GMU = re.compile(r"Using gpu_memory_utilization=([0-9.]+) to serve model")
PARAM = re.compile(r"^(MODEL|TP|INPUT_LEN|OUTPUT_LEN|MAX_MODEL_LEN|MAX_LATENCY_ALLOWED_MS|MIN_CACHE_HIT_PCT|"
                   r"NUM_SEQS_LIST|NUM_BATCHED_TOKENS_LIST|RESULT_FILE)=(.*)$", re.M)


def read_output(stdout: str, result: str = "") -> dict:
    """What auto_tune printed (stdout) and wrote (result.txt): its parameters, every combination, its pick.
    result.txt gets the final best_* line only at the very end; stdout gets one after every combination, so a
    run cut short still has its best so far there."""
    params = dict(PARAM.findall(stdout))
    commit = re.search(r"current_hash: (\S+)", stdout) or re.search(r"^hash:(\S+)", result, re.M)
    gmu = GMU.findall(stdout)
    best = BEST.findall(result) or BEST.findall(stdout)
    text = result or stdout
    measured = [dict(zip(("max_num_seqs", "max_num_batched_tokens", "request_rate", "latency_ms", "req_per_s",
                          "goodput"), m)) for m in MEASURED.findall(text)]
    return {
        "params": params,
        "vllm_commit": commit.group(1) if commit else None,
        "finished": "finish permutations" in stdout,
        "gpu_memory_utilization": float(gmu[-1]) if gmu else None,
        "measured": measured,
        "failed_combinations": [{"max_num_seqs": int(s), "max_num_batched_tokens": int(t)}
                                for s, t in FAILED.findall(text)],
        "best": ({"max_num_seqs": int(best[-1][0]), "max_num_batched_tokens": int(best[-1][1]),
                  "req_per_s": float(best[-1][2])} if best else None),
    }


def rule_text(rule: str, limit_ms: Optional[str]) -> str:
    if rule == "e2el99":
        return f"its own rule, P99 end-to-end latency <= {limit_ms} ms"
    return (f"patched to judge p95 time to first token <= {limit_ms} ms, rigtune's ceiling "
            "(the patch is auto_tune.diff beside its output)")


def to_row(out: dict, rule: str, quant: str = "bf16", label: str = LABEL,
           exit_code: Optional[int] = None) -> dict:
    """One entry of a `--rows` file: auto_tune's pick as a rigtune Config, or a skipped entry saying why there
    is none. The Config launches what auto_tune launched (`vllm serve MODEL --gpu-memory-utilization G
    --max-num-seqs S --max-num-batched-tokens T --enable-prefix-caching --max-model-len L`), with real
    weights; rigtune's vLLM backend also passes `--dtype` for `quant`, which should be what auto_tune's dtype
    auto resolved to (bf16 for a bf16 checkpoint)."""
    p = out["params"]
    commit = (out["vllm_commit"] or "an unknown commit")[:12]
    source = (f"vLLM benchmarks/auto_tune/auto_tune.sh at {commit}, "
              f"{rule_text(rule, p.get('MAX_LATENCY_ALLOWED_MS'))}; INPUT_LEN={p.get('INPUT_LEN')}, "
              f"OUTPUT_LEN={p.get('OUTPUT_LEN')}, MAX_MODEL_LEN={p.get('MAX_MODEL_LEN')}; its search ran "
              "with --load-format dummy (random weights), random fixed-length prompts with --ignore-eos, and "
              "open-loop load; this row is re-measured with the real weights")
    if not out["finished"]:
        source += "; auto_tune did not finish its grid, so this is its best so far"
    if exit_code:
        source += f"; auto_tune.sh exited with status {exit_code}"
    row = {"label": label, "kind": "tuner", "source": source}
    best = out["best"]
    if out["gpu_memory_utilization"] is None:
        row["skipped"] = ("auto_tune found no --gpu-memory-utilization from 0.98 down to 0.90 at which vLLM "
                          "started")
    elif best is None or best["req_per_s"] <= 0:
        row["skipped"] = "no combination auto_tune tried met its latency limit"
    else:
        row["config"] = {"backend": "vllm", "quant": quant, "ctx": int(p["MAX_MODEL_LEN"]),
                         "batch": best["max_num_seqs"],
                         "gpu_memory_utilization": round(out["gpu_memory_utilization"], 2),
                         "prefill_budget": best["max_num_batched_tokens"], "prefix_cache": True}
        if int(p.get("TP") or 1) > 1:
            row["config"]["tp"] = int(p["TP"])
    return row


def write_row(stdout_path: Path, result_path: Optional[Path], rule: str, out_dir: Path,
              quant: str = "bf16", exit_code: Optional[int] = None) -> dict:
    """rows.json (for `rigtune compare --rows`) and autotune_summary.json (everything read, and the script's
    exit status when it was run from here) in out_dir."""
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    result = ""
    if result_path is not None and result_path.exists():
        result = result_path.read_text(encoding="utf-8", errors="replace")
    out = read_output(stdout, result)
    if "MAX_MODEL_LEN" not in out["params"]:
        raise AutoTuneError(f"{stdout_path}: no MAX_MODEL_LEN= line; is this auto_tune.sh's output?")
    row = to_row(out, rule, quant, exit_code=exit_code)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.json").write_text(json.dumps({"rows": [row]}, indent=1) + "\n", encoding="utf-8")
    summary = {"rule": rule, "exit_code": exit_code, **out, "row": row}
    (out_dir / "autotune_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return row


# --------------------------------------------------------------------------- running it


def _tee(cmd: List[str], env: Dict[str, str], cwd: Path, log: Path) -> int:
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
            fh.flush()
        return proc.wait()


def run(a: argparse.Namespace, launch: Callable[[List[str], Dict[str, str], Path, Path], int] = _tee) -> int:
    src = Path(a.vllm_src).expanduser().resolve()
    script = src / "benchmarks" / "auto_tune" / "auto_tune.sh"
    if src.name != "vllm" or not script.is_file():
        raise AutoTuneError(f"--vllm-src must be a vLLM checkout in a directory named `vllm` (auto_tune.sh "
                            f"runs `cd \"$BASE/vllm\"`), with benchmarks/auto_tune/auto_tune.sh; got {src}")
    if a.input_len + a.output_len > a.max_model_len:
        raise AutoTuneError(f"--input-len + --output-len = {a.input_len + a.output_len} is more than "
                            f"--max-model-len {a.max_model_len}; auto_tune.sh would exit")
    out = Path(a.out).resolve()
    copy, diff = write_patched(script, out, a.rule, host_ip=a.host_ip, gpu_hygiene=not a.no_gpu_hygiene,
                               final_profile=a.final_profile)
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    env = {"BASE": str(src.parent), "MODEL": a.model, "SYSTEM": "GPU", "TP": str(a.tp),
           "DOWNLOAD_DIR": a.download_dir or str(Path(hf_home) / "hub"), "INPUT_LEN": str(a.input_len),
           "OUTPUT_LEN": str(a.output_len), "MAX_MODEL_LEN": str(a.max_model_len), "MIN_CACHE_HIT_PCT": "0",
           "MAX_LATENCY_ALLOWED_MS": f"{a.limit_ms:g}"}
    if a.num_seqs:
        env["NUM_SEQS_LIST"] = a.num_seqs
    if a.num_batched_tokens:
        env["NUM_BATCHED_TOKENS_LIST"] = a.num_batched_tokens
    params = {"rule": a.rule, "quant": a.quant, "env": env}
    (out / "params.json").write_text(json.dumps(params, indent=1) + "\n", encoding="utf-8")
    print(f"auto_tune.sh copy: {copy}\nits diff against {script}: {diff}", flush=True)
    log = out / "stdout.log"
    code = launch(["bash", str(copy)], {**os.environ, **env}, out, log)
    found = dict(PARAM.findall(log.read_text(encoding="utf-8", errors="replace"))).get("RESULT_FILE")
    result = Path(found) if found else None
    if result is not None and result.parent.is_dir():
        # Its logs and result.txt, beside the row; the torch profiler traces are left where they are.
        shutil.copytree(result.parent, out / "result", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("profile"))
    row = write_row(log, result, a.rule, out, a.quant, exit_code=code)
    print(f"auto_tune.sh exited {code}; row: {json.dumps(row.get('config') or row.get('skipped'))}\n"
          f"wrote {out / 'rows.json'}", flush=True)
    return code  # the row is written either way; non-zero says auto_tune itself did not finish cleanly


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rule", choices=RULES, required=True)
        p.add_argument("--host-ip", action="store_true",
                       help="serve on 127.0.0.1 when $(hostname) does not resolve")
        p.add_argument("--no-gpu-hygiene", action="store_true")
        p.add_argument("--final-profile", action="store_true", help="keep auto_tune's torch-profiler run")

    p = sub.add_parser("patch", help="write the patched copy of auto_tune.sh and its diff")
    p.add_argument("--src", type=Path, required=True,
                   help="benchmarks/auto_tune/auto_tune.sh in a vLLM checkout")
    p.add_argument("--out", type=Path, required=True, help="directory for auto_tune.sh and auto_tune.diff")
    run_options(p)

    p = sub.add_parser("lengths", help="INPUT_LEN and OUTPUT_LEN matched to what a calibration served")
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--max-model-len", type=int, default=4096)

    p = sub.add_parser("run", help="patch, run auto_tune.sh, and write its pick as a row")
    p.add_argument("--vllm-src", required=True, help="vLLM checkout, in a directory named vllm")
    p.add_argument("--model", required=True)
    p.add_argument("--limit-ms", type=float, required=True,
                   help="MAX_LATENCY_ALLOWED_MS, on the rule's metric")
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, required=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--num-seqs", default=None, help='NUM_SEQS_LIST, e.g. "128 256" (default: the script\'s)')
    p.add_argument("--num-batched-tokens", default=None,
                   help="NUM_BATCHED_TOKENS_LIST (default: the script's)")
    p.add_argument("--download-dir", default=None, help="DOWNLOAD_DIR (default: $HF_HOME/hub)")
    p.add_argument("--quant", default="bf16",
                   help="the row's precision: what dtype auto gives this checkpoint")
    p.add_argument("--out", type=Path, required=True)
    run_options(p)

    p = sub.add_parser("row", help="turn a finished run's output into a --rows file")
    p.add_argument("--stdout", type=Path, required=True, help="what auto_tune.sh printed")
    p.add_argument("--result", type=Path, default=None, help="its result.txt")
    p.add_argument("--rule", choices=RULES, required=True, help="the rule the run used")
    p.add_argument("--quant", default="bf16")
    p.add_argument("--out", type=Path, required=True,
                   help="directory for rows.json and autotune_summary.json")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "patch":
            copy, diff = write_patched(a.src, a.out, a.rule, host_ip=a.host_ip,
                                       gpu_hygiene=not a.no_gpu_hygiene, final_profile=a.final_profile)
            print(f"wrote {copy} and {diff}")
        elif a.cmd == "lengths":
            got = lengths_from_profile(json.loads(a.profile.read_text(encoding="utf-8")))
            if got is None:
                print(f"{a.profile}: no token counts in its calibration table; pass lengths by hand",
                      file=sys.stderr)
                return 2
            inp, outp, n = got
            if inp + outp > a.max_model_len:  # auto_tune.sh exits when INPUT_LEN + OUTPUT_LEN > MAX_MODEL_LEN
                outp = max(a.max_model_len - inp, 16)
            print(f"INPUT_LEN={inp} OUTPUT_LEN={outp}  # mean over {n} calibration requests")
        elif a.cmd == "run":
            return run(a)
        else:
            row = write_row(a.stdout, a.result, a.rule, a.out, a.quant)
            print(json.dumps(row, indent=1))
    except AutoTuneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
