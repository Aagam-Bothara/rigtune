"""Qwen3.5-9B on the A40: cmp's answer check graded every row ~18% because the model thinks by default and every
answer hit max_tokens 512 mid-thought (probe: 60/60 finish_reason length). Grade all 1319 GSM8K items with
thinking off (chat_template_kwargs enable_thinking=false), greedy, max_tokens 512, concurrency as cmp's RowGrader
picks it: stock vLLM twice, rigtune's pick (MTP:3) once; report lost/gained/bound as cmp would.

    python qwen_full_check.py OUT_DIR
Runs on the pod with HF_HOME set; needs the GPU to itself."""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from rigtune.accuracy_gate import _counts
from rigtune.calibrate.paired import certifies, items_needed, score_bounds
from rigtune.evalset import load_eval_set

OUT = sys.argv[1]
MODEL, REV = "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
URL = "http://127.0.0.1:8125"
BASE = ["/usr/local/bin/vllm", "serve", MODEL, "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", "8125",
        "--revision", REV, "--dtype", "bfloat16"]
ROWS = [  # (label, argv, concurrency, passes)
    ("vllm-default", BASE + ["--max-model-len", "262144", "--max-num-seqs", "256", "--gpu-memory-utilization", "0.90"],
     64, 2),
    ("rigtune", BASE + ["--max-model-len", "8192", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.95",
                        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 3,
                                                            "revision": REV})], 64, 1),
]
os.makedirs(OUT, exist_ok=True)
items = load_eval_set("gsm8k").items
grades = {}


def ask(client, it):
    body = {"model": MODEL, "messages": it.messages, "temperature": 0.0, "max_tokens": it.max_tokens or 512,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
    try:
        j = client.post("/v1/chat/completions", json=body).json()
        ch = j["choices"][0]
        return (ch.get("message") or {}).get("content"), ch.get("finish_reason")
    except Exception:
        return None, None


for label, argv, conc, passes in ROWS:
    log = open(f"{OUT}/{label}.server.log", "w")
    srv = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(240):
            try:
                if httpx.get(f"{URL}/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            raise SystemExit(f"{label}: server never became ready")
        for p in range(1, passes + 1):
            t0 = time.time()
            with httpx.Client(base_url=URL, timeout=180) as c, ThreadPoolExecutor(conc) as pool:
                got = list(pool.map(lambda it: ask(c, it), items))
            out = "".join("-" if r is None else ("1" if it.grade(r) else "0") for it, (r, _) in zip(items, got))
            grades[f"{label}.{p}"] = out
            fin = {}
            with open(f"{OUT}/{label}.pass{p}.jsonl", "w") as f:
                for it, (r, why), g in zip(items, got, out):
                    fin[why] = fin.get(why, 0) + 1
                    f.write(json.dumps({"id": it.id, "label": it.answer, "grade": g, "finish_reason": why,
                                        "response": r}) + "\n")
            print(f"{label} pass {p}: {out.count('1')}/{len(out)} right, {out.count('-')} unanswered, finish {fin}, "
                  f"{time.time() - t0:.0f} s", flush=True)
    finally:
        os.killpg(srv.pid, 9)
        srv.wait()
        time.sleep(10)


def report(name, ref, cand, n_items):
    lost, gained, n = _counts(ref[:n_items], cand[:n_items])
    lo, hi = score_bounds(lost, gained, n)
    return (f"{name} on first {n_items}: lost {lost}, gained {gained} of {n}; 95% bound {lo * 100:+.2f} to "
            f"{hi * 100:+.2f} pts; certifies -3: {certifies(lost, gained, n, 0.03)}; items needed at these rates: "
            f"{items_needed(lost, gained, n, 0.03)}")


ref = grades["vllm-default.1"]
lines = [report("stock vs itself (noise)", ref, grades["vllm-default.2"], len(items)),
         report("rigtune vs stock", ref, grades["rigtune.1"], 300),
         report("rigtune vs stock", ref, grades["rigtune.1"], len(items))]
json.dump({"grades": grades, "report": lines}, open(f"{OUT}/summary.json", "w"), indent=1)
print("\n".join(lines), flush=True)
