"""Which experts does Qwen3.5-35B-A3B use, and does a few prompts' worth of routing predict the rest?

A decision experiment for placing MoE experts by a user's own prompts (keep the busy ones in VRAM, the rest in
RAM): it is worth building only if the router's picks are skewed, and skewed the same way on prompts not yet
seen. Nothing in rigtune changes.

The model runs in Hugging Face transformers on one Modal A100 80GB (bf16, about 67 GiB), with a forward hook on
every layer's router (Qwen3_5MoeTopKRouter, which returns the picked experts' indices). 50 prompts from each of
three domains, one at a time, greedy, thinking off, 64 new tokens: HumanEval (code), GSM8K (maths) and Dolly
(chat, the 300 calibration prompts of the November sessions). The prompt's own tokens (prefill) and the
generated ones (decode, the bandwidth-bound part placement is for) are counted apart, per prompt.

Run:  modal run benchmarks/modal_expert_skew.py
It writes expert_counts.npz (counts[prompt, phase, layer, expert]) and expert_skew.json (the summary) beside where
it is run. To summarise again:  python benchmarks/modal_expert_skew.py expert_counts.npz
"""

import io
import json
import pathlib
import sys

import modal

MODEL = "Qwen/Qwen3.5-35B-A3B"
REVISION = "2508b2437e382c605eb0fc52b83f5545960f6b4a"
PER_DOMAIN = 50
NEW_TOKENS = 64
# One expert's weights: gate, up and down projections of 2048 x 512, at about 4.8 bits a weight (Q4_K_M).
EXPERT_BYTES = 3 * 2048 * 512 * 4.8 / 8
SPARE_VRAM = 1.5 * 2**30  # a 4 GB card once attention, shared experts, KV cache and the desktop have theirs

app = modal.App("rigtune-expert-skew")
image = (modal.Image.from_registry("vllm/vllm-openai:v0.33.0", add_python="3.12").entrypoint([])
         .run_commands("/usr/bin/python3 -m pip install --no-cache-dir accelerate "
                       "|| /usr/bin/python3 -m pip install --no-cache-dir --break-system-packages accelerate"))


@app.function(image=image, gpu="A100-80GB", timeout=2 * 3600)
def route(prompts: list) -> bytes:
    """counts[prompt, phase (0 prefill, 1 decode), layer, expert] for every prompt, as .npz bytes."""
    import os
    import subprocess

    code = r'''
import io, json, sys, time
import numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
prompts = json.load(sys.stdin)
tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map="cuda")
model.eval()
routers = [m for m in model.modules() if type(m).__name__ == "Qwen3_5MoeTopKRouter"]
E, K = routers[0].weight.shape[0], routers[0].top_k
counts = np.zeros((len(prompts), 2, len(routers), E), dtype=np.int32)
cur = [0]
def hook(layer):
    def record(module, inputs, output):
        idx = output[2]
        assert idx.shape[-1] == K, idx.shape
        phase = 0 if idx.shape[0] > 1 else 1
        counts[cur[0], phase, layer] += np.bincount(idx.reshape(-1).cpu().numpy(), minlength=E).astype(np.int32)
    return record
for i, r in enumerate(routers):
    r.register_forward_hook(hook(i))
print(f"{len(routers)} routers, {E} experts, top {K}", file=sys.stderr, flush=True)
t0 = time.time()
for i, p in enumerate(prompts):
    cur[0] = i
    text = tok.apply_chat_template([{"role": "user", "content": p["text"]}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False)
    if i % 10 == 0:
        print(i, p["domain"], f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
buf = io.BytesIO()
np.savez_compressed(buf, counts=counts, domains=np.array([p["domain"] for p in prompts]))
sys.stdout.buffer.write(b"NPZ" + buf.getvalue())
'''.replace("MODEL", repr(MODEL)).replace("REVISION", repr(REVISION)).replace("NEW_TOKENS", str(NEW_TOKENS))
    # vLLM's Python has torch and a transformers that knows Qwen3.5; its stderr (progress) streams to Modal's log.
    done = subprocess.run(["/usr/bin/python3", "-c", code], input=json.dumps(prompts).encode(),
                          stdout=subprocess.PIPE, env={**os.environ, "HF_HOME": "/root/hf"})
    out = done.stdout
    if done.returncode != 0 or b"NPZ" not in out:
        raise RuntimeError(f"routing run failed (exit {done.returncode}): {out[-2000:].decode(errors='replace')}")
    return out[out.index(b"NPZ") + 3:]


def _prompts() -> list:
    import random

    from datasets import load_dataset

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from make_dolly_prompts import select_rows, to_prompt

    code = [r["prompt"] for r in load_dataset("openai/openai_humaneval", split="test")]
    maths = [r["question"] for r in load_dataset("openai/gsm8k", "main", split="test")]
    chat = [to_prompt(r) for r in select_rows(list(load_dataset("databricks/databricks-dolly-15k", split="train")),
                                              300, 0)]
    pick = random.Random(0)
    return [{"domain": d, "text": t} for d, rows in (("code", code), ("maths", maths), ("chat", chat))
            for t in pick.sample(rows, PER_DOMAIN)]


def _top(c, frac):
    """Per layer, the experts that took the most picks: a boolean mask over (layer, expert), frac of each layer."""
    import numpy as np

    n = max(1, round(frac * c.shape[-1]))
    mask = np.zeros(c.shape, dtype=bool)
    np.put_along_axis(mask, np.argsort(-c, axis=-1)[:, :n], True, axis=-1)
    return mask


def _share(c, mask):
    """The share of picks that fall on the masked experts, averaged over layers."""
    return float(((c * mask).sum(-1) / c.sum(-1)).mean())


def summarise(counts, domains) -> dict:
    import numpy as np

    fit = SPARE_VRAM / EXPERT_BYTES / counts[0, 0].size  # the share of all experts that fits in the spare VRAM
    out = {"experts_per_layer": int(counts.shape[-1]), "layers": int(counts.shape[2]), "prompts_per_domain":
           PER_DOMAIN, "vram_fraction": round(fit, 4), "phases": {}}
    names = list(dict.fromkeys(domains.tolist()))
    for phase, label in ((1, "decode"), (0, "prefill")):
        rows = {}
        for d in names:
            mine = counts[domains == d, phase]  # (prompts, layer, expert)
            total = mine.sum(0)
            a, b = mine[0::2].sum(0), mine[1::2].sum(0)  # two halves of the prompts, alternating
            p = total / total.sum(-1, keepdims=True)
            entropy = -(np.where(p > 0, p * np.log(p), 0)).sum(-1)
            row = {
                # In-sample: what the busiest experts take of the picks they were chosen on (uniform = frac).
                "top10_share": round(_share(total, _top(total, 0.1)), 3),
                "top20_share": round(_share(total, _top(total, 0.2)), 3),
                "effective_experts": round(float(np.exp(entropy).mean()), 1),
                # Held out: choose the VRAM-sized set on half the prompts, count what it takes of the other half.
                "vram_set_held_out": round((_share(b, _top(a, fit)) + _share(a, _top(b, fit))) / 2, 3),
                "top20_held_out": round((_share(b, _top(a, 0.2)) + _share(a, _top(b, 0.2))) / 2, 3),
            }
            row["vram_set_other_domains"] = {o: round(_share(counts[domains == o, phase].sum(0), _top(total, fit)), 3)
                                             for o in names if o != d}
            rows[d] = row
        out["phases"][label] = rows
    return out


def _report(s: dict) -> None:
    f = s["vram_fraction"]
    print(f"\n{s['layers']} layers x {s['experts_per_layer']} experts; the spare VRAM of a 4 GB card holds {f:.1%} "
          f"of them, which would take {f:.1%} of the picks if chosen blindly")
    for phase, rows in s["phases"].items():
        print(f"\n{phase}: top 10% / top 20% share (uniform 10% / 20%), effective experts of "
              f"{s['experts_per_layer']}, VRAM-sized set chosen on half the prompts -> share of the other half, "
              "and of other domains")
        for d, r in rows.items():
            other = ", ".join(f"{o} {v:.1%}" for o, v in r["vram_set_other_domains"].items())
            print(f"  {d:6} {r['top10_share']:.1%} / {r['top20_share']:.1%}   eff {r['effective_experts']:5.1f}   "
                  f"held out {r['vram_set_held_out']:.1%} (top 20%: {r['top20_held_out']:.1%})   others: {other}")


@app.local_entrypoint()
def main() -> None:
    import numpy as np

    prompts = _prompts()
    data = route.remote(prompts)
    pathlib.Path("expert_counts.npz").write_bytes(data)
    z = np.load(io.BytesIO(data))
    s = summarise(z["counts"], z["domains"])
    pathlib.Path("expert_skew.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    _report(s)


if __name__ == "__main__":
    import numpy as np

    z = np.load(sys.argv[1])
    _report(summarise(z["counts"], z["domains"]))
