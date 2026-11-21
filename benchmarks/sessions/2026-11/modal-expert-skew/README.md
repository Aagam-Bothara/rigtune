# Which experts Qwen3.5-35B-A3B uses, and whether half the prompts predict the other half

Run on 21 November 2026 with [modal_expert_skew.py](../../../modal_expert_skew.py) on a Modal A100 80GB: Hugging
Face transformers 5.20.2 (vLLM 0.33.0's image), Qwen/Qwen3.5-35B-A3B at revision 2508b24 in bf16, a forward hook
on each of the 40 routers (256 experts, 8 picked a token, plus one shared expert not counted). 50 prompts from each
domain, one at a time, greedy, thinking off, 64 new tokens: HumanEval (code), GSM8K test (maths), Dolly (chat).
Raw counts in [expert_counts.npz](expert_counts.npz) (`counts[prompt, phase, layer, expert]`), the summary in
[expert_skew.json](expert_skew.json). One run.

The question: for placing experts by a user's own prompts (the busiest in VRAM, the rest in RAM), are the
router's picks skewed, and skewed the same way on prompts not yet seen? The spare VRAM of a 4 GB card (taken as
1.5 GiB, about 1.9 MB an expert at Q4_K_M) holds 8.3% of the 10,240 experts, which would take 8.3% of the picks
if chosen blindly.

Generated tokens (decode, the part a small card's memory bandwidth limits):

| Domain | Top 10% / top 20% take | Effective experts of 256 | 8.3% set chosen on half the prompts: share of the other half | Of the other domains |
|---|---|---|---|---|
| code | 45.2% / 62.0% | 128.7 | **39.8%** | maths 15.4%, chat 8.9% |
| maths | 46.9% / 65.2% | 119.2 | **40.9%** | code 15.9%, chat 15.0% |
| chat | 34.8% / 53.9% | 159.6 | **26.0%** | code 9.4%, maths 14.9% |

The prompt's own tokens (prefill) are similar and a little more skewed (held out: code 41.9%, maths 45.0%, chat
27.9%).

- The picks are skewed and the skew carries to unseen prompts of the same kind: a VRAM-sized set chosen on 25
  prompts takes 3.1x (chat) to 4.9x (maths) the blind share of the other 25. In-sample shares are only a few points
  higher than held-out ones, so 25 prompts already find most of the busy experts.
- The busy experts differ by domain: a set chosen on code takes 8.9% of chat's picks, about the blind share.
  Placement has to be learned from the user's own prompts; one set for everyone would not do it.
- What it could be worth, by arithmetic only: 8 experts in each of 40 layers is about 605 MB of expert weights a
  token. With 8.3% of them in VRAM about 555 MB come from RAM; with code's 39.8%, about 364 MB. Where reading RAM
  sets the pace, that is about 1.5x faster decoding. Not measured.

Limits:
- Each domain is one dataset with one style (HumanEval prompts all look alike), which flatters the held-out
  split; Dolly, the most varied, is the weakest at 26%. Real traffic mixes kinds.
- Some of the skew may follow the format (chat template, docstrings) rather than the subject.
- 64 generated tokens a prompt, greedy; long answers may route differently.
- llama.cpp keeps each layer's experts in one tensor and places whole tensors, so using this needs an engine that
  places single experts.
