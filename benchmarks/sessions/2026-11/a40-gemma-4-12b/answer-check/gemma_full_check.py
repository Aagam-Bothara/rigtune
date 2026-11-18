"""Gemma 4 12B on the A40: cmp's answer check (first 300 GSM8K items) had rigtune's pick lose 4 and gain 0 against
stock vLLM, lower bound -3.4 points against -3, while stock vLLM's own two passes differed on 1. 300 items cannot
tell a small real loss from numerical drift. Grade the whole test set (1319 items) the way cmp does
(rigtune.accuracy_gate.chat_answers, greedy, max_tokens 512, concurrency min(64, max(8, batch))): stock twice,
rigtune's pick once, and report lost/gained/bound on all 1319 and on the first 300.

    python gemma_full_check.py OUT_DIR
Runs on the pod with HF_HOME set; needs the GPU to itself."""
import json
import os
import subprocess
import sys
import time

import httpx

from rigtune.accuracy_gate import _counts, chat_answers, pass_wall_seconds
from rigtune.calibrate.paired import certifies, items_needed, score_bounds
from rigtune.evalset import load_eval_set

OUT = sys.argv[1]
MODEL, REV = "google/gemma-4-12B-it", "43c9d014b231474fa7212dbb48a762566c2b255d"
BASE = ["/usr/local/bin/vllm", "serve", MODEL, "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", "8124",
        "--revision", REV, "--dtype", "bfloat16"]
ROWS = [  # (label, argv, concurrency as RowGrader picks it, passes)
    ("vllm-default", BASE + ["--max-model-len", "262144", "--max-num-seqs", "256", "--gpu-memory-utilization", "0.90"],
     64, 2),
    ("rigtune", BASE + ["--max-model-len", "8192", "--max-num-seqs", "16", "--gpu-memory-utilization", "0.95",
                        "--speculative-config", json.dumps({"method": "suffix", "num_speculative_tokens": 24})], 16, 1),
]
os.makedirs(OUT, exist_ok=True)
items = load_eval_set("gsm8k").items
grades = {}

for label, argv, conc, passes in ROWS:
    log = open(f"{OUT}/{label}.server.log", "w")
    srv = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(240):
            try:
                if httpx.get("http://127.0.0.1:8124/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            raise SystemExit(f"{label}: server never became ready")
        for p in range(1, passes + 1):
            t0 = time.time()
            said = chat_answers("http://127.0.0.1:8124", MODEL, items, conc, 512, pass_wall_seconds(len(items)))
            out = "".join("-" if r is None else ("1" if it.grade(r) else "0") for it, r in zip(items, said))
            grades[f"{label}.{p}"] = out
            with open(f"{OUT}/{label}.pass{p}.jsonl", "w") as f:
                for it, r, g in zip(items, said, out):
                    f.write(json.dumps({"id": it.id, "label": it.answer, "grade": g, "response": r}) + "\n")
            print(f"{label} pass {p}: {out.count('1')}/{len(out)} right, {out.count('-')} unanswered, "
                  f"{time.time() - t0:.0f} s", flush=True)
    finally:
        os.killpg(srv.pid, 9)
        srv.wait()
        time.sleep(10)


def report(name, ref, cand, n_items):
    lost, gained, n = _counts(ref[:n_items], cand[:n_items])
    lo, hi = score_bounds(lost, gained, n)
    need = items_needed(lost, gained, n, 0.03)
    return (f"{name} on first {n_items}: lost {lost}, gained {gained} of {n}; 95% bound {lo * 100:+.2f} to "
            f"{hi * 100:+.2f} pts; certifies -3: {certifies(lost, gained, n, 0.03)}; items needed at these rates: "
            f"{need}")


ref = grades["vllm-default.1"]
lines = [report("stock vs itself (noise)", ref, grades["vllm-default.2"], len(items)),
         report("rigtune vs stock", ref, grades["rigtune.1"], 300),
         report("rigtune vs stock", ref, grades["rigtune.1"], len(items))]
json.dump({"grades": grades, "report": lines}, open(f"{OUT}/summary.json", "w"), indent=1)
print("\n".join(lines), flush=True)
