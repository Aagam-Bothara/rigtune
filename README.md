# rigtune

[![tests](https://github.com/Aagam-Bothara/rigtune/actions/workflows/tests.yml/badge.svg)](https://github.com/Aagam-Bothara/rigtune/actions/workflows/tests.yml)

I got tired of guessing serving flags. Every guide says something different, and what's fastest on one GPU is often slower on another. So I built a tool that just measures. rigtune tries engines, weight precisions, KV-cache types and speculative decoding on your card, with your prompts, goes looking on the Hugging Face Hub for draft models built for your model, and then serves whatever won behind an OpenAI-compatible API.

```bash
pip install rigtune "vllm==0.33.0"
rigtune serve Qwen/Qwen3.5-9B --speculative all     # calibrates once, then serves on :8000
```

In November 2026 I tested it against everything I could find: stock vLLM and SGLang, vLLM's own `auto_tune`, BentoML's llm-optimizer, and the settings each engine's docs recommend. On Qwen3.5-9B it ran at least 1.8× faster than all of them. On Gemma 4 12B it found a completely different trick and beat the best expert setting by 1.5–1.7×. On Qwen3.5-35B-A3B, a mixture-of-experts model, on an H200, it ran 2.26× faster than stock vLLM. The answers held up every time, graded on GSM8K.

| Held-out prompts, 8 users, output tok/s | rigtune | stock vLLM | best other tool | best expert setting | answers vs stock (95% bound, −3 allowed) |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (MoE), H200 141 GB | **2018–2076** | 912 | not run ‡ | 934 (vLLM fp8 KV) | −2.0 points |
| Qwen3.5-9B, A100 SXM 80 GB | **1067–1071** | 565 | 581 (llm-optimizer, SGLang) | 581 (SGLang fp8 KV) | −0.9 points |
| Qwen3.5-9B, A40 48 GB | **478–479** | 252 | 262 (llm-optimizer, vLLM) | 260 (vLLM fp8 KV) | −1.16 points |
| Gemma 4 12B, A40 48 GB | **996–1143** | 175 | 175 (auto_tune, llm-optimizer) | 667 (vLLM n-gram) | −0.83 points |

‡ The H200 run was vLLM only, against stock, bare stock, stock with fp8 weights (1,233 tok/s) and vLLM's expert rows. Its answers were graded on 300 GSM8K items, the others on all 1,319.

**What won.** On Qwen it was the model's own multi-token-prediction (MTP) head, which sits in the checkpoint and is off by default. On the H200 it added fp8 weights and an fp8 KV cache, which the Ampere runs left out. On Gemma it was suffix decoding, which reuses text from the prompt. Different model and card, different winner. That's the whole reason to measure instead of guess.

**New, and not yet a head-to-head.** Most models have no MTP head, so rigtune now searches the Hugging Face Hub for drafters built for your model (EAGLE3 heads, DFlash and DSpark drafters), checks each one's config against what vLLM and SGLang will actually load, and picks how many tokens a drafter guesses from the engine's own acceptance counts. On Qwen3-8B on an A100 it found a DeepSeek DSpark drafter that no list of mine included and served **1,822 tok/s against 705 for stock vLLM (2.58×)** on Dolly prompts, at lower time-to-first-token. That's one calibration run with one measurement per trial and ungraded answers, not the comparison above ([session](benchmarks/sessions/2026-11/modal-roundc-qwen3-8b/)). The drafter search is on `main` and not yet in a PyPI release, so install from GitHub to try it (below). SGLang gets the same drafters, read from its source but not yet run on a GPU.

**Before you get excited, here's what this doesn't show:**
- **On Qwen, most of the gain is MTP.** You could turn it on yourself if you read the model card, and none of the expert rows did. With it on, SGLang reached 937–948 tok/s in the A100 calibration, so a careful engineer would close much of the gap by hand.
- **The Gemma gain depends on your prompts.** Suffix decoding shines when answers copy from their context, so expect less on open-ended chat.
- **I only measured 1–8 users.** At that load the batch settings the other tools search can't really kick in yet.
- **Three models on three cards, one session each.** If you've got different hardware, I'd honestly love to see where it breaks.

**How it was measured.** vLLM 0.33.0 and SGLang 0.5.23. rigtune tuned on 300 Dolly prompts, then every row was measured three times, interleaved, on 300 prompts it never saw. Every other tool got the same card, the same hour and the same latency limits. I tried AIConfigurator too, but it had no data for the A40 or for Qwen3.5's hybrid layers, so it couldn't give an answer. The H200 run is one command on [Modal](benchmarks/modal_calibrate.py), about $10, if you want to check it yourself ([how](benchmarks/sessions/2026-11/h200-qwen3.5-35b-a3b/)).

**Why the answer check matters.** On my first run every row scored about 18% on GSM8K, stock included. Qwen3.5 thinks by default, and every answer ran out of tokens mid-thought. I nearly read that as "the speedup breaks accuracy". rigtune now turns thinking off for grading when the chat template has a switch for it. The A100 answers were graded by the comparison itself. The A40 ones were graded by scripts committed next to the results, because that run happened before the fix.

Raw files are in [benchmarks/sessions/2026-11](benchmarks/sessions/2026-11/). The full method and limits are in [docs/benchmarks.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#2026-models-on-current-engines).

## Earlier results on Llama 3.1 8B

Before the 2026 models came out I ran Llama 3.1 8B on vLLM 0.29 and compared rigtune with a fixed rule of thumb, the kind of thing an informed engineer sets without measuring:

| Llama 3.1 8B, held-out prompts | rigtune | a fixed rule of thumb † | (stock defaults) |
|---|---|---|---|
| RTX 4090 | **1041 tok/s** | 667 (**+56%**) | SGLang 448 |
| A100 | **1155 tok/s** | 674 (**+71%**) | vLLM 726 |
| A40 | **506 tok/s** | 257 (**+97%**) | vLLM 262 |

† The rule: fp8 weights and KV cache, a short context, batch 256. It was **slower than changing nothing** on both Ampere cards, because an fp8 KV cache costs throughput there. A rule that's right on one card can be wrong on the next ([full sessions](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#llama-31-8b-on-four-gpus)).

A few other things I learned along the way:
- **Your traffic matters as much as your card.** On one A40 with Qwen2.5-3B the winner changed with the prompts: suffix decoding on Dolly (+122% over stock), a draft model on news-article extraction (+66.5%), an fp8 KV cache on real chat (+10%).
- **Calibration pays for itself fast.** It takes 22–56 minutes per model and workload, and that time comes back after 0.5–2.6 hours of busy serving.
- **Measuring matters more than search order.** Given the same time, random sampling matched rigtune's staged search on all three 8B cards. On a 14B model in fp8 the staged search won by 4.4% ([the evidence](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#qwen25-14b-where-memory-binds)). I kept the staged search because it gives the same answer every run, tries every strategy at least once, and writes down what each setting was worth.

Every result, with methods, ablations, quality checks and limits: [docs/benchmarks.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md).

## How is this different from auto_tune, llm-optimizer and AIConfigurator?

- **vLLM's `auto_tune` and BentoML's llm-optimizer** sweep one engine's batching knobs, like the batch cap and the batched-token budget, to find the best setting under a latency target. I ran both head to head above. rigtune treats the engine itself as a variable too, alongside precision, KV-cache type and speculative decoding, and those are the dimensions that carried almost all the gain. It also finds draft models the engines don't know about: neither tool looks beyond the settings you give it. On an RTX 4090, SGLang led the early trials and vLLM only overtook it once speculative decoding and a quantized cache were tried. Tuning the early leader alone would have thrown away the eventual winner.
- **AIConfigurator** predicts a deployment from performance data it collected per GPU. It's very cheap because it launches nothing, but it can only answer for cards and model layers it has data for, which on these runs was none. rigtune measures instead. That costs 20–60 minutes per workload, but the answer is specific to your prompts. I also tried fitting my own predictor: it ranks the true winner at a median of 15th out of 25 measured configurations, so prediction is a filter here, never the decision ([why](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/decisions.md)).
- **The honest gap:** every result above is my own run. If you try rigtune, please post your card, model and what it picked in [Discussions](https://github.com/Aagam-Bothara/rigtune/discussions), good or bad, or [open an issue](https://github.com/Aagam-Bothara/rigtune/issues) if something broke.

## Quickstart

```bash
pip install rigtune            # 0.3.0; for the Hub drafter search: pip install "git+https://github.com/Aagam-Bothara/rigtune"
pip install "vllm==0.33.0"     # ran on driver 595; 0.29.0 needs 580+; older drivers: "vllm==0.11.0" "transformers>=4.56,<5"
rigtune serve Qwen/Qwen2.5-3B-Instruct --workload chat
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

The first launch calibrates and caches the result for this machine, model, objective and workload. After that it serves straight away. To tune for your own traffic, pass a sample of it with `--workload-file prompts.jsonl`, one prompt per line. Add `--speculative all` to also try a checkpoint's own MTP head, draft models, and the EAGLE3, DFlash and DSpark drafters the Hub has for it.

To skip the calibration on another machine with the same GPU, carry the result over: `rigtune bundle export <model> --out pick.tar.gz` packs the profile with every compiled kernel, and `rigtune bundle import pick.tar.gz` checks it against the new machine first. On fresh Modal A100s that started the pick in 135 s instead of 297 s cold ([session](benchmarks/sessions/2026-11/modal-bundle-e2e/)).

Optional extras:
- **Suffix decoding** needs arctic-inference built for the Python that runs vLLM: `CMAKE_ARGS="-DPython_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')" python -m pip install --no-cache-dir --force-reinstall arctic-inference==0.1.1` ([why each part](docs/usage.md#suffix-decoding)).
- **SGLang** can live in its own environment: set `SGLANG_PYTHON` to its `python`.
- **llama.cpp** needs `llama-server` built with CUDA, on `PATH` or in `LLAMA_SERVER`.

With Docker (built on the official vLLM image):

```bash
docker build -t rigtune .
docker run --gpus all --ipc=host -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.rigtune:/root/.rigtune rigtune serve Qwen/Qwen2.5-3B-Instruct
```

### Supported hardware

Linux, Python 3.10–3.13.
- **NVIDIA GPUs of compute capability 7.5 or newer** run vLLM, SGLang and llama.cpp. I've measured on an A40, A100, H100 NVL, H200, L4 and RTX 4090.
- **Older NVIDIA GPUs and x86 CPUs** run llama.cpp.
- **Never benchmarked:** vLLM-CPU and pre-Turing GPUs ([details](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#hardware-and-engines-measured)).

Checked with vLLM 0.11.0, 0.29.0 and 0.33.0 and SGLang 0.5.19 and 0.5.23. `rigtune probe` lists them. Any other release gets a warning, not a refusal.

Opt-in calibration and serving controls: [safeguards](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/runtime-hardening.md).

## Learn more

- [docs/benchmarks.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md): every result, at a glance and in full, with methods, ablations, quality, limits and what's still unmeasured.
- [docs/usage.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/usage.md): how calibration works, workloads, objectives, every search option and the CLI.
- [docs/writeup.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/writeup.md): the design of the memory planner, calibration and the predictor, what the evidence does and doesn't support, and the roadmap.
- [docs/decisions.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/decisions.md): why each choice is the way it is, including why not just random search, with the measurement that settled it.
- [benchmarks/strategies/SUMMARY.md](https://github.com/Aagam-Bothara/rigtune/blob/main/benchmarks/strategies/SUMMARY.md): every table, regenerated from the raw JSON.
- [CONTRIBUTING.md](https://github.com/Aagam-Bothara/rigtune/blob/main/CONTRIBUTING.md): development setup, tests and adding a backend.

*rigtune used to be called PolyServe. I renamed it because a [2025 paper](https://arxiv.org/abs/2507.17769) already uses that name. Old profiles, results and settings still work ([details](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/usage.md#renamed-from-polyserve)).*

MIT licensed.
