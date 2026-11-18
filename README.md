# rigtune

[![tests](https://github.com/Aagam-Bothara/rigtune/actions/workflows/tests.yml/badge.svg)](https://github.com/Aagam-Bothara/rigtune/actions/workflows/tests.yml)

*Formerly PolyServe, renamed because a separate [2025 paper](https://arxiv.org/abs/2507.17769) already has that name. Profiles, results and settings from before the rename keep working ([details](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/usage.md#renamed-from-polyserve)).*

**rigtune finds the fastest way to serve an LLM on your GPU and your traffic, then serves it behind an OpenAI-compatible API.** It measures real configurations instead of guessing: engine, weight precision, batch size, KV-cache type and speculative decoding. The settings that won changed on every card tried.

**On this year's models and engines, its pick ran at least 1.8× the throughput of every other row on Qwen3.5-9B. On Gemma 4 12B it ran 1.5–1.7× the best expert setting and about 6.5× the rest.** The other rows were stock vLLM, stock SGLang, vLLM's `auto_tune`, BentoML's llm-optimizer and settings from each engine's own docs. The answers matched stock's on all 1,319 GSM8K items.

| Held-out prompts, 8 users, output tok/s | rigtune | stock vLLM | best competitor tool | best expert setting | answers vs stock (95% bound, −3 allowed) |
|---|---|---|---|---|---|
| Qwen3.5-9B, A100 SXM 80 GB | **1067–1071** | 565 | 581 (llm-optimizer, SGLang) | 581 (SGLang fp8 KV) | −0.9 points |
| Qwen3.5-9B, A40 48 GB | **478–479** | 252 | 262 (llm-optimizer, vLLM) | 260 (vLLM fp8 KV) | −1.16 points |
| Gemma 4 12B, A40 48 GB | **996–1143** | 175 | 175 (auto_tune, llm-optimizer) | 667 (vLLM n-gram) | −0.83 points |

Method:
- **Engines:** vLLM 0.33.0 and SGLang 0.5.23.
- **Calibration:** 300 Dolly prompts; every row was then measured on 300 others, three times, interleaved.
- **Competitor tools:** each ran on the same card, with the same hour and the same latency limits as rigtune. AIConfigurator was run too but produced no row: it has no A40 data, and no data for Qwen3.5's hybrid layers on the A100.
- **Answers:** the A100's were graded by rigtune's comparison itself. The A40's were graded by scripts committed beside the results, because that comparison's built-in check was broken for thinking models at the time.
- **Raw files:** [benchmarks/sessions/2026-11](benchmarks/sessions/2026-11/); method and limits in [docs/benchmarks.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#2026-models-on-current-engines).

**What this does not show:**
- **On Qwen, most of the gain is one setting:** the checkpoint's own multi-token-prediction head, which rigtune finds and tries and the other tools do not search. No expert row turned it on. An engineer who reads the model card and does so by hand would close much of the gap: in calibration, SGLang with the head reached 937–948 tok/s.
- **On Gemma, the gain is suffix decoding** on prompts that copy from their context, so it would be smaller on open-ended chat.
- **Only 1–8 users were measured.** At that load the competitor tools' batch and token-budget settings cannot act yet.
- **Two models, two Ampere cards, and no MoE model.**

## Results on 2025 engines

On Llama 3.1 8B with vLLM 0.29, rigtune also beat a fixed rule of thumb, which is what an informed engineer sets without measuring:

| Llama 3.1 8B, held-out prompts | rigtune | a fixed rule of thumb † | (stock defaults) |
|---|---|---|---|
| RTX 4090 | **1041 tok/s** | 667 (**+56%**) | SGLang 448 |
| A100 | **1155 tok/s** | 674 (**+71%**) | vLLM 726 |
| A40 | **506 tok/s** | 257 (**+97%**) | vLLM 262 |

† The rule: fp8 weights and KV cache, a short context, batch 256. It was **slower than changing nothing** on both Ampere cards, because an fp8 KV cache costs throughput there. A rule that is right on one card is wrong on another ([full sessions](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#llama-31-8b-on-four-gpus)).

- **Your traffic decides what pays, not the card alone.** On one A40 with Qwen2.5-3B the winning setting changed with the prompts: suffix decoding on Dolly prompts (+122% over stock), a draft model on news-article extraction (+66.5%), an fp8 KV cache on real chat (+10%). A tuner that ignores your prompts cannot find these.
- **Calibration pays for itself within hours.** It takes 22–56 minutes per model and workload, repaid by 0.5–2.6 hours of busy serving.
- **The search order is worth less than the measuring.** Given the same time, random sampling of the same space matched this staged search on all three 8B cards, and on a 14B in fp8 the staged search won by 4.4% ([the evidence](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#qwen25-14b-where-memory-binds)). Staging gives the same answer every run, tries every strategy at least once, and writes a profile saying what each setting was worth.

Every result, with methods, ablations, quality checks and limits: [docs/benchmarks.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md).

## How this differs from auto_tune, llm-optimizer and AIConfigurator

- **vLLM's `auto_tune` and BentoML's llm-optimizer** each sweep one engine's batching knobs, such as the batch cap and the batched-token budget, for the best setting that meets a latency target. Both were run head to head above. rigtune treats the engine itself as a variable, alongside weight precision, KV-cache type and speculative decoding, the dimensions that carried almost all the gain. On an RTX 4090, SGLang led the early trials and vLLM only overtook it once speculative decoding and a quantized cache were tried. Tuning the early leader alone would have eliminated the eventual winner.
- **AIConfigurator** predicts deployments from a performance model it collected per GPU. It is very cheap because it launches nothing, but it can only answer for cards and model layers it has data for, which on these runs was none. rigtune measures instead, which costs 20–60 minutes per workload and makes its answers specific to your prompts. Our own fitted predictor ranks the true winner at a median of 15th of 25 measured configurations, so prediction is a filter here, never the decision ([why](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/decisions.md)).
- **The honest gap:** no one outside this project has run rigtune yet.

## Quickstart

```bash
pip install rigtune
pip install "vllm==0.33.0"     # ran on driver 595; 0.29.0 needs 580+; older drivers: "vllm==0.11.0" "transformers>=4.56,<5"
rigtune serve Qwen/Qwen2.5-3B-Instruct --workload chat
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

The first launch calibrates and caches the result per machine, model, objective and workload; later launches serve at once. To tune for your own traffic, pass a sample of it: `--workload-file prompts.jsonl`, one prompt per line. `--speculative all` also tries a checkpoint's own MTP head and draft models.

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
- **NVIDIA GPUs of compute capability 7.5 or newer** run vLLM, SGLang and llama.cpp. Measured on an A40, A100, H100 NVL, L4 and RTX 4090.
- **Older NVIDIA GPUs and x86 CPUs** run llama.cpp.
- **Never benchmarked:** vLLM-CPU and pre-Turing GPUs ([details](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md#hardware-and-engines-measured)).

Checked with vLLM 0.11.0, 0.29.0 and 0.33.0 and SGLang 0.5.19 and 0.5.23. `rigtune probe` lists them, and any other release gets a warning, not a refusal.

Opt-in calibration and serving controls: [safeguards](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/runtime-hardening.md).

## Learn more

- [docs/benchmarks.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/benchmarks.md): every result, at a glance and in full, with methods, ablations, quality, limits and what is still unmeasured.
- [docs/usage.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/usage.md): how calibration works, workloads, objectives, every search option and the CLI.
- [docs/writeup.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/writeup.md): the design of the memory planner, calibration and the predictor, what the evidence does and does not support, and the roadmap.
- [docs/decisions.md](https://github.com/Aagam-Bothara/rigtune/blob/main/docs/decisions.md): why each choice is the way it is, including why not just random search, with the measurement that settled it.
- [benchmarks/strategies/SUMMARY.md](https://github.com/Aagam-Bothara/rigtune/blob/main/benchmarks/strategies/SUMMARY.md): every table, regenerated from the raw JSON.
- [CONTRIBUTING.md](https://github.com/Aagam-Bothara/rigtune/blob/main/CONTRIBUTING.md): development setup, tests and adding a backend.

MIT licensed.
