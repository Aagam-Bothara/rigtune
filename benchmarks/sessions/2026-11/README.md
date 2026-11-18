# GPU sessions, November 2026

Raw results of the three head-to-head comparisons on 2026 models and current engines (vLLM 0.33.0, SGLang 0.5.23). The method, the numbers and the limits are in [docs/benchmarks.md](../../../docs/benchmarks.md#2026-models-on-current-engines).

| Folder | Card | Model | rigtune at commit | Answers graded by |
|---|---|---|---|---|
| [a40-qwen3.5-9b](a40-qwen3.5-9b/) | A40 48 GB | Qwen/Qwen3.5-9B | main during the session of 16–18 Nov, before the answer-check fix | the script in `answer-check/`, outside the comparison |
| [a40-gemma-4-12b](a40-gemma-4-12b/) | A40 48 GB | google/gemma-4-12B-it | same | the script in `answer-check/`, outside the comparison |
| [a100-qwen3.5-9b](a100-qwen3.5-9b/) | A100 SXM4 80 GB | Qwen/Qwen3.5-9B | 7bbab8a (18 Nov) | the comparison itself (`--check-answers gsm8k --check-items 0`) |

Each folder holds what `rigtune compare --verdict` wrote:

- `*.verdict.md` is the verdict as a reader sees it.
- `*.verdict.json` holds the same verdict as data.
- `*__balanced.json` is the scored comparison: every row, every run and the environment.

Left out, for size: the graded answers (25 MB per comparison), the request logs and the copies of the prompts. The prompt files are named by their sha256 in each verdict. `benchmarks/make_dolly_prompts.py` rebuilds them.

## Read the A40 verdicts with this

Both A40 verdict files end in "no row counts". In each one, the comparison's built-in answer check failed rigtune's pick. On both models the check was wrong, not the pick:

- **Qwen3.5-9B.** Its chat template thinks unless told not to. Every answer ran out of the 512-token budget mid-thought, and every row scored about 18%. `answer-check/qwen_full_check.py` graded all 1,319 GSM8K items with thinking off: stock vLLM twice, rigtune's pick once. rigtune lost 19 and gained 16, a 95% lower bound of −1.16 points against the −3 allowed. Stock vLLM differed from itself on 24 items. The result is in `answer-check/summary.json`.
- **Gemma 4 12B.** On the first 300 items rigtune's pick lost 4 and gained 0, a bound of −3.4. Three hundred items cannot tell that from noise. `answer-check/gemma_full_check.py` graded all 1,319 items the same way the comparison grades: rigtune lost 9 and gained 7, a bound of −0.83. Stock differed from itself on 15 items.

The A100 comparison ran after both fixes were built into rigtune: thinking switched off when the template has a switch, every item graded, and a 1,024-token budget. Its verdict is a win on all 11 rows that ran, with the answers certified in the verdict itself.
