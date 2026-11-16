# Decisions, and the measurement behind each

Every design choice here was settled by a measurement, and several were settled against the first guess. This
page is the short answer to each, with the number that decided it and where the full evidence lives. Where the
evidence is thin or absent, it says so; those are the honest weak points, listed again in
[Limits](benchmarks.md#limits) and [Not yet measured](benchmarks.md#not-yet-measured).

## Why not just random search?

Random sampling of the same space is a strong baseline, and it was measured rather than dismissed: given equal
time it matched the staged search on an A40 and an A100 and beat it by 29% on an RTX 4090
([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

What is worth keeping is not the ordering but the space: random search samples the 720–984 configurations that
remain after the memory planner drops what cannot fit and the capability filter drops what the engine cannot do.
Run blind over the raw product of settings and most draws fail to launch. Random search here *is* rigtune
minus the stage order, and it inherits the expensive parts — the planner, the measurement harness, the objective.

The stages buy three things random draws do not: the same answer on every run, where a seed change moves random's
pick; every strategy tried at least once, so the profile says what each setting was worth instead of only naming
a winner; and a sensible order under `--budget`, where the first handful of trials decide everything.

The case where staging should win outright is a space where good configurations are rare, rather than the roughly
one in four that carry speculative decoding on those cards — and that case has since been measured twice, with
two different answers. On a 14B in fp8 on a 24 GB card, where only 2 of 54 configurations fit, the staged search
measured 385 tok/s against random sampling's 369 at equal time, ranges apart. On the same card in the
788-configuration space that also contains 4-bit checkpoints, the two tied: 1412 against 1416, with random search
given 2524 s to the staged search's 3028 s and sampling 3% of the space
([evidence](benchmarks.md#qwen25-14b-where-memory-binds)).

So the honest score is one win in a small space and one tie in a large one, and the prediction held in only half
the cases it was made for. Nothing here supports "the search order is what pays"; what it supports is the
narrower claim these stages were built for — the same answer every run, every strategy tried at least once, and
a profile that says what each setting was worth.

Two ways of doing better were then measured against the recorded calibrations, and neither pays.

**Reaching the answer sooner.** If staging found its pick in a handful of trials while random sampling needed
dozens, the claim would be speed rather than quality. Across 13 recorded calibrations it reaches 95% of its own
final answer at a median of **13 of 25 trials**, having spent 43% of its wall-clock — against the 24 to 44 draws
random search needed. That is roughly a factor of two, not the order of magnitude that would make "calibration in
five minutes" true, and the spread is wide: two runs got there in 3–4 trials, three needed 16–19.

**Sampling from a prior instead of uniformly.** The obvious way to beat random sampling is to draw configurations
in proportion to how good they are predicted to be, using the performance predictor fitted from previous
calibrations — memory across runs being the one advantage random sampling cannot have. Leave one calibration out,
fit on the rest, and ask where the predictor ranks the configuration that actually won: **median rank 15 of ~25,
and never in the top 5 across 8 folds** — and those 25 are already the configurations the staged search chose to
measure, so they are biased towards good ones. A rank correlation of 0.88–0.94 is enough to prune quantizations
that cannot win and nowhere near enough to order near-optimal configurations against each other, which is exactly
what a sampler needs. Sampling on those scores would be worse than uniform.

Both were computed from data already on disk, before spending anything on a GPU, which is the only reason it was
cheap to learn that the search is at its practical ceiling for this space.

## Why measure at all, instead of a rule of thumb?

Because a defensible rule of thumb lost, including to doing nothing. The rule (fp8 weights and KV cache, a short
context, batch 256) came in 56–97% behind the measured pick on three cards, and on both Ampere cards it was
*slower than stock defaults* — 674 against 724 tok/s on an A100 — because an fp8 KV cache costs throughput there
while helping on Ada ([evidence](benchmarks.md#llama-31-8b-on-four-gpus)). The winning setting also changed with
the prompts on one card: suffix decoding on Dolly-15k, a draft model on extraction, an fp8 cache on real chat
([evidence](benchmarks.md#re-measured-on-an-a40)). A fixed rule cannot follow that.

## Why compare against stock defaults at all?

Stock `vllm serve` is what a machine runs when nobody tunes it, so it is the honest zero point, but it is a weak
opponent and the headline should not rest on it. The rule-of-thumb comparison is the one that answers "could an
informed engineer have guessed this?", which is why the README leads with it. Against someone who already knows
the winning flags for this card and this traffic, rigtune ties by construction — knowing them takes measuring
([limits](benchmarks.md#limits)).

## Why change one setting at a time?

So the profile can attribute the gain: each stage's trials differ from the leader in one dimension, which makes
the table readable as "what each change was worth". The cost is interactions, and that cost is real: on the 4090,
an int8 KV cache and a 16k prefill budget each did nothing alone but were part of a configuration 29% faster with
a draft model. Two mitigations followed, and only the second worked — a stage of random draws around the leader
(`--explore on`) found nothing better in 9 draws, while trying speculation first and tuning everything else per
speculation method closed the gap ([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

## Why try speculative decoding before the other settings?

Because the other settings pay differently depending on it, so their order relative to it decides the result. On
the 4090 the int8 cache and the prefill budget were worthless alone and valuable with a draft model. Speculation
first, then the remaining stages once per each of the two best speculation methods, reached 1041 tok/s in 24
trials and 43 minutes where the previous order reached 813 in 31 trials and 66 minutes. On the A40 and A100 it
reached the same picks as before in about two-thirds of the time ([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

## Why tune engines that are already behind?

Because a lead after three stages is not a lead after all of them. SGLang led vLLM in fp8 by 14% on the 4090, so
tuning only the leader never tried vLLM's speculative decoding, which SGLang's backend does not offer — and that
is where the card's best configuration was. The search now also tunes an engine further behind when a stage has
something to try on it and nothing on the leader's, and drops an engine that a tuned rival has left more than 10%
behind with nothing more to offer ([design](writeup.md#3-staged-search)).

## Why does `--speculative all` decide which engines to keep tuning after speculation?

Because speculation is the strategy most likely to reorder engines, and choosing before it judges an engine
before its best move. On the 4090, SGLang led after the batch stage (776 against 669 tok/s) and vLLM got through
only by a special rule: it offered speculative decoding and SGLang did not. Once SGLang is offered speculation
that rule no longer applies, so under `--speculative all` the search runs the speculation stage on every engine
still in it and only then keeps the engines within 10% of the leader; the rule stays for strategies one engine
lacks. The same option keeps a third speculation setting per engine, so a new method cannot push out the
second-best existing one (the 4090's recorded pick came from the second: a draft model at 878 tok/s behind suffix
decoding at 901). The default search is unchanged.

What the replay shows (`benchmarks/replay_search.py`, no GPU) is limited. No recording has SGLang speculation or
an EAGLE3 head, so both were excluded from it: what it can show is the cost of the option's other changes. Of the
21 replayable calibrations, the default search reproduces 5 exactly, reaches the recorded pick past unmeasured
trials in 13, and differs in 3, as it did before this option existed. Under `--speculative all` every one reaches
the same pick as the default search, but only 1 reproduces exactly: the others ask for 1 to 7 trials that were
never measured, 57 in all, mostly the third speculation setting's prefill and KV trials (7 more on the 4090, whose
calibration ran 24). Those picks are inconclusive, not confirmed, and SGLang's own speculation, which would add
trials of its own, has not been measured anywhere.

## Why re-measure the best three configurations at the end?

Because the best of many noisy trials is flattered by the choosing. Of five picks re-measured before this stage
existed, four landed within 1.4% of their calibration number and one measured 9.3% lower. The confirm stage costs
up to three trials and decides on the fresh runs alone: on the Dolly p95 re-run the pick measured 601 tok/s in the
search and 616 re-measured ([option](usage.md#search-options)).

## Why measure the busiest concurrency level first?

Because most trials die there, and dying early is cheap. Trials run their heaviest level first and stop when it
cannot beat the leader, which cut 29–45% off calibration across the Llama runs and changed no pick
([evidence](benchmarks.md#measuring-the-busiest-level-first)).

## Why one metric within limits, not a weighted score?

Because a weight has no unit anyone can check. "0.6 × tok/s + 0.4 × (1 / TTFT)" mixes tokens per second with
milliseconds, so its pick moves with the weights' scale and nobody can say whether a 3% throughput gain is worth
40 ms of tail latency without the answer already in mind. A limit is a number an operator already has, a TTFT
SLO, a power budget, a price, and a pick within it can be checked against it. So a goal is one metric to optimise
and any number of limits ([usage](usage.md#stating-your-own-goal)): still the constrained argmax the four presets
always were, and the presets are four such goals, built from the same fields, scoring exactly as before (every
recorded calibration table, under each preset and seven sets of constraints, ranks with `==` the same as a frozen
copy of the old code, `tests/test_preset_pins.py`). When nothing meets every limit a custom goal picks what misses
the fewest, then what is least far past them relative to each bound: excesses in milliseconds and watts do not add
up either. The presets keep summing each excess in its own unit, a second convention kept so that no preset pick
moves. A goal a preset says is that preset, with its cache file, so no question is calibrated twice; and what the
limits cost is shown, from the recorded trials, rather than folded into a score
([trade-off table](usage.md#what-the-limits-cost-the-trade-off-table)).

## Why do most custom goals measure every level?

Because only a metric that improves with load lets the busiest level that meets the limits settle a trial. Tokens
per second rises with load, so for a goal optimising `tok_s` (or `cost_per_mtok` at an hourly price alone, which
is that price over tok/s) no quieter level can beat the first one that meets every limit, and stopping there
skipped 35% of calibration time in the replay that checked it, changing no score and no pick
([evidence](benchmarks.md#measuring-the-busiest-level-first)). Time to first token, time per token and request
latency get worse with load, and energy per token and cost with an energy price can move either way, so for those
a quieter level may win and an early stop could miss the winner; measuring every level never does, at about half
as long again per trial. Stopping bottom-up for metrics that worsen with load would be sound on the same argument
mirrored, but it has not been checked on recorded calibrations the way the top-down stop was, so it is not used.

## Why score energy per request at an arrival rate?

Because joules per token measured closed-loop describes a server that is never idle, and most servers are. The
load generator keeps every slot busy, so neither the card's idle draw between requests nor its wind-down after a
burst ever reaches J/token; at a few requests a minute the idle draw alone can be most of what each request costs,
and it is the card's, so no configuration moves it. `--arrival-rate` composes J/request at the stated rate from
the levels a calibration already measures plus a short idle tail after each trial, instead of driving every trial
open-loop at that rate, which at 0.1 req/s would take minutes per trial and still measure one point. The composed
figure is a model, so the pick is re-measured once with Poisson arrivals at the rate and the profile records the
prediction next to the measurement: the model is checked on the same card in every calibration that uses it. The motivation for a wind-down term came from a separate project of the
author's, which saw about 40 J per request of post-request hold on a different setup; that figure is not in this
repository's data, and nothing here depends on it. What the idle draw and the hold are on the cards rigtune has
run on is unmeasured ([gaps](benchmarks.md#not-yet-measured)).

## Why score at the 95th percentile rather than the mean?

Because a mean hides the requests users complain about, and a trial that looks fastest can be breaking its latency
ceiling at its fastest level. Scoring at p95 demotes such a trial to the level it actually meets: a draft-model
combination on the A100 whose p95 reached 770 ms at 8 users was scored at 4 users instead
([evidence](benchmarks.md#llama-31-8b-on-four-gpus)). Each level's p95 rests on 32 requests, so its second-slowest
request can decide a close call — one pick scored at 32 users with a p95 of 999 ms against a 1000 ms ceiling
([evidence](benchmarks.md#the-p95-search-on-hardware-and-the-new-options)) — and a level too close to call is
measured again, a rule that never fired on the Llama runs and is reported rather than quietly dropped
([evidence](benchmarks.md#judged-at-the-95th-percentile)).

## Why plan memory instead of launching and catching the failure?

Because a failed launch costs 30 seconds and tells you nothing, while an estimate costs nothing and removes most
of the grid. The planner predicts weights, KV cache and workspace, and its error against measured peak memory is
tracked as a first-class result rather than assumed ([accuracy](benchmarks.md#memory-planner-accuracy)). The
sharper reason is that a wrong memory setting silently produces a *slow* run rather than a crash: on an RTX 3090
llama.cpp with 27 of 36 layers on the GPU measured 35 tok/s where a full offload measured 545, so quantizations
have to be compared at the memory setting they would actually be served at ([design](writeup.md#2-memory-planner)).

## Why is the explore stage off by default?

Because it did not pay. It spends about 30% more trials on random configurations of the leading engine and
precision; on the 4090 its 9 draws found nothing better than the stages had, and random search given the same 66
minutes still won by 28%. The option stays, off, with the negative result recorded
([evidence](benchmarks.md#llama-31-8b-on-four-gpus)).

## Why is answer quality checked separately from calibration?

Because the obvious way to enforce it during tuning was built, measured, and did not work. `--quality-probe`
answers the calibration prompts greedily at each precision while its engine is up and records how far they drift
from the most faithful precision that runs. Making that a *constraint* — drop a precision that drifts too far —
is what failed, and the numbers say why: identical fp8 weights answering the same prompts twice differed on 10%
of answer openings, while 4-bit and int8 checkpoints differed on 83–100%. There is no threshold between those,
and GSM8K put the same 4-bit weights about 2 points behind, so exact-match drift and answer quality disagree
outright ([measurements](benchmarks.md#qwen25-14b-where-memory-binds)).

So the probe reports and never refuses. It tells an operator that a precision rewrites most of its answers,
which is worth knowing before choosing it; deciding whether those answers are *worse* needs labels, and that
stays a separate, labelled step.

Three things about it are worth stating plainly. Its first run on hardware did nothing at all: the probe was never
built, because the guard tested prompts that a file workload only materialises later, so a 14B calibration kept a
4-bit checkpoint that had been compared against nothing (fixed, with a regression test).

Its first design was also too strict to be useful. Comparing whole 128-token greedy completions across 16 prompts
meant one changed answer was 6% — past any sane tolerance — and greedy decoding diverges as soon as a single
token differs, so nearly every cheaper precision would have been refused, with nothing to distinguish a real
change from the engine's own non-determinism. It now compares the first 32 tokens of 48 answers and measures each
precision twice, so the reference's disagreement with itself is the floor a candidate must clear.

And even working, it is not an accuracy guarantee: instruction-following answers can agree perfectly while
multi-step arithmetic degrades, which is exactly what happened on that card — near-zero drift where GSM8K lost
about 2 points ([evidence](benchmarks.md#qwen25-14b-where-memory-binds)). Labelled grading is what can gate.

The labelled check remains, because calibration measures speed and latency, and quantization can cost accuracy
that no throughput number shows. By default it is still a separate step, `benchmarks/task_quality.py`, and its
narrowness is the weak point: GSM8K only, where quantization cost Qwen2.5-3B 2–5 points and 7B about one, and fp8
with activation quantization on Ada cost 7B nothing. Other tasks, other families and llama.cpp's GGUF formats are
ungraded, which is why 4-bit checkpoints are not in `--quant auto` for larger models
([gaps](benchmarks.md#not-yet-measured)). `--eval-set` brings the same grading into calibration, on labels rather
than drift and on items the operator chooses; the next section says how it decides. It is opt-in and has not run
on a GPU yet.

## Why does the accuracy gate use a non-inferiority bound?

Because the question it answers is "is this precision's loss within the tolerance?", and a significance test
answers a different one, "is there any difference?". The two disagree exactly where the gate matters. On
Qwen2.5-14B at 24 GB, GPTQ against fp8 on 200 GSM8K problems lost 4 and gained none: McNemar's test gives
p = 0.125, not significant, so a gate built on it would admit GPTQ at any tolerance, and with fewer problems it
would admit anything, because a small sample convicts nothing. The lower end of a 95% interval for the
difference is −5.0 points on those counts (computed in `tests/test_paired.py`), so the gate refuses GPTQ at a
3-point tolerance and admits it at 6. Small samples refuse; the note says what tolerance and how many items
would have admitted.

The interval is Tango's score interval for paired proportions, not the textbook Wald interval: when nothing is
discordant the Wald interval has zero width at any sample size, while the score interval on 300 items still
spans ±1.3 points.

What a precision is compared with matters as much. One reference for every engine, the most precise format
anywhere, would gate vLLM's fp8 against llama.cpp's Q8_0 (which ranks above it) and refuse it for differences
that belong to the engine and its chat template. One reference per engine, whatever its format, fails the other
way: on the 14B at 24 GB SGLang's fp8 failed to start and its GPTQ was the fastest first trial (701 tok/s), so
GPTQ would have been SGLang's reference and been served ungated. So each engine's reference is its most precise
*faithful* format (bf16, fp16, fp8, Q8_0: the model's own weights or the 8-bit format the engine makes from
them); fp8 under bf16 on one engine is gated against it; a checkpoint is gated against its own engine's
reference, or against another engine's when its own has none, with the note saying the comparison is
cross-engine; and faithful formats are never gated across engines. Each reference runs first, so neither the
predictor nor a faster engine can skip it, and answers twice: if it cannot certify the tolerance against itself,
the note says the set is too small to certify anything.

Four choices fail closed: a precision that ran but answered nothing is refused; a checkpoint with no faithful
format graded on any engine is refused; a precision the gate may yet refuse does not set the pace that drops
slower engines in stage 1; and refusals go to the same `refused` set as the drift gate's would have, which every
later stage and the final pick respect. The withdrawn drift gate taught that last one: dropping a precision from
stage 1's kept list only stopped it being tuned, and the argmax over every trial served it anyway.

The configuration served is gated as well, its KV cache and speculation included. Speculative decoding is designed
not to change greedy answers, but an fp8 or int8 KV cache can, so stage 1's grade of the weights cannot vouch for
what the later stages built on them. Until 14 November 2026 the pick's own grades were only reported, on the
argument that refusing it would leave nothing tuned to fall back to. That argument was wrong: the reference's
stage-1 configuration is always certified and already measured, and the next-best configurations can be graded
on a run of their own. So the final stage grades the best configuration that has no grade of its own, serves it
only if the same bound certifies it, and otherwise falls back, best first, to the next, re-measuring a fallback's
speed. The fallback search is capped (`--eval-attempts`, default 3 graded configurations) and skips
configurations that share a refused one's quantized cache or speculation method, because each grading pass may take
up to its wall cap, 660 s at 300 items, on top of a trial. In that check an unanswered item counts as lost where the reference was right: a
missing answer must not make a configuration easier to pass, and one missing answer should cost one item of
evidence, not the certificate.

## Why does the agent trace generator write a file?

`benchmarks/make_agent_trace.py` writes a file of explicit requests rather than adding an `agent` preset. A file's
hash already names the profile calibrated on it, so a change to the generator or its parameters can never serve a
profile measured on other traffic, and no new cache-key entry is needed. The file is also what a user can open,
diff against their own agent's requests, or replace with a recording, and every option of explicit replay applies
to it unchanged. The generator is deterministic (one seeded stream per session, sorted keys, ASCII, Unix newlines),
and a test pins the hash of its default output, as `make_oasst_prompts.py` rebuilds the same file every time.

It models MemGPT's published design (Packer et al. 2023), not any framework's code: where an agent framework puts
its memory, and whether it rewrites a status line every step, was not checked against a framework's source, so the
generator exposes those placements as parameters (`retrieval_at`, `meta_refresh`, `layout`) instead of claiming one.
The share of the queue evicted at once, 0.3 by default, is Letta's documented default (`sliding_window_percentage`);
the paper was not checked for a figure of its own, and the generated file's `gen` key says where the number came from.
A search result is treated as an append, a function result at the tail of the queue, as we read the paper; that
makes retrieval bring eviction sooner rather than rewrite the prompt itself. `retrieval_at=head` models the other
placement, the latest result in the system prompt, as frameworks that retrieve into it do.

## Why try a KV-cache tier only when the working set outgrows the GPU pool?

A host-RAM tier can only help a request whose prefix the GPU pool has already evicted. While the live sessions'
prompts and answers fit in the pool, a tier adds a copy of every block to host memory and recovers nothing: both
engines write blocks to the tier as they are produced (vLLM 0.29's offloading connector offers every finished
chunk; SGLang's default write policy is write-through), and vLLM's own guide says a tier smaller than the pool
just mirrors it. So `--kv-tier auto` looks first at what the leader's own session trial measured: its working set
against the pool the engine logged, in the same units. Only a working set above 0.8× the pool earns a trial; the tier
is sized from the working set (1.25×) rather than swept, and a disk tier is tried only when host RAM cannot hold the
working set. That is at most two trials per engine, and none on traffic that fits. The thresholds are starting
points, not measurements; the GPU run in [Not yet measured](benchmarks.md#not-yet-measured) is what would move them.

Reuse short of what the conversations allowed is recorded but is not a gate. An earlier draft also opened the gate
when reuse fell 10 points short, but that arm could never produce a trial: below 0.8× the pool, 1.25× the working set
is under the pool itself, and a tier must be 1.2× the pool to hold more than the pool does. The working set counts a
system prompt the sessions share once, since the cache holds it once; counted per session, 64 sessions sharing a
5.5k-token preamble looked ten times larger than they are.

The pool is the bytes the engine says it allocated ("Available KV cache memory" on vLLM, the pools SGLang logs),
not vLLM's "GPU KV cache size" line: that is maximum concurrency times `max_model_len`, which for a hybrid model
is not a token count of any pool. The working set is converted with the hybrid planner's rules for the same
reason: on Qwen3.5 or Gemma 4 most layers hold a window or a fixed state, not every token.

Only SGLang's disk tier is offered. vLLM 0.29 cascades every block it stores to its filesystem tier, which takes
no size limit and never deletes, so one session trial at a pool of 64 could write the whole replay to disk; SGLang's
file backend takes a cap, and rigtune always sets it.

It is off by default because it has never run on hardware, and because the tier holds host RAM, and perhaps disk
holding the served prompts' KV cache, that the machine may want for something else.

## Why native tiers rather than LMCache?

Both engines now ship a tier of their own: vLLM 0.29's `OffloadingConnector` (CPU, with an unbounded filesystem
tier behind it) and SGLang's hierarchical cache (host memory, with a file backend). LMCache through vLLM needs a separate
`lmcache server` process, the hybrid KV cache manager turned off (which changes how every hybrid model's cache is
laid out, so the A/B would compare two cache layouts as well as a tier), and a size that vLLM does not pass on;
its current PyPI release depends on a CUDA 13 build of cupy. Each of those is a confound or a failure mode the
native tiers do not have, so LMCache is not searched.

## Why ask the engine instead of reading its source?

rigtune's feature checks read an engine's source text for a name, in the files where one release keeps it,
because importing vLLM costs seconds and SGLang usually lives in another environment. That broke on the next
release: vLLM 0.33 moved its server arguments to `entrypoints/common/cli_args.py`, and the check for
`--enable-prompt-tokens-details` answered no on 0.33 until it was told the new path, so session-file
calibrations launched without per-reply cached tokens. SGLang 0.5.23 moved its argument definitions to
`arg_groups/fields/`, and its checks still answer only because a tuple of field names elsewhere happens to
list them. The parser an engine builds for its own server, its model registry and its speculative config say
the same things wherever the release keeps them.

So rigtune asks the engine's own Python, once per install, with no GPU visible and a 120-second limit, and
keeps the answers. The probe comes first and the source checks stay as the fallback, so a probe that cannot
run costs a note, never a crash or a changed pick. What the probe cannot do is say that something works: a
flag can exist and mean something else (SGLang 0.5.23 changed a 24 GB card's default prefill chunk behind an
unchanged flag). Measured failures, version gates for what was checked on which release, the memory formulas
and the tables of second models stay hand-written; a probe can only take a method away, never offer one they
hold back. For architectures it is the other way round: the probe can add one the built-in list lacks
(gpt-oss, on either engine), never remove one, because a registry listed with no GPU visible can miss a model
whose module needs one to import, and refusing a working model would hide it from every calibration. Whether a
release behaves as rigtune plans is for a conformance run on a GPU to say.

## Why not Bayesian optimisation?

At roughly 30 trials over a space this shape, the published results for SLO-aware Bayesian tuners do not separate
from staged or random search, and the machinery costs a dependency and reproducibility. Random search already
matches the staged search here — a smarter optimiser is worth adding only once there is a space where blind
sampling visibly degrades, which is the same missing experiment named above
([roadmap](writeup.md#9-non-goals-and-roadmap)).

## What would change these answers?

- Long-context traffic at high concurrency, where batch and cache sizing decide the result. The memory-bound
  half of this gap has now been measured twice on a 14B at 24 GB, and split: the staged search won by 4.4% in a
  104-configuration space and tied in a 788-configuration one
  ([details](benchmarks.md#qwen25-14b-where-memory-binds)). A third measurement would say which is typical.
- Traffic that drifts after calibration: profiles are cached per machine, model, objective and workload, and
  nothing re-tunes when the prompts change shape.
- Prefix reuse in real traffic. Every pick so far was made on prompts sent once each, while chat traffic resends
  each conversation's history every turn; where a shared prefix was measured, prefix caching was worth +129% and
  +409%. Session files replay conversations turn by turn, but have not yet run on hardware
  ([prediction](benchmarks.md#not-yet-measured)); if the picks they make differ from those made on the same
  conversations flattened, every flat calibration of chat traffic was tuned for the wrong load.
- Agent traffic, which rewrites its prompt rather than appending to it. Explicit-request replay and a MemGPT-like
  trace generator exist but have not run on hardware ([prediction](benchmarks.md#not-yet-measured)); if a rewriting
  layout and an appending one of the same events pick differently, a pick tuned on chat sessions is tuned on the
  wrong tail for agents.
- A second person running a calibration on hardware and traffic the author does not control.


## Why check token ids when the engine checks sizes?

The Gemma 4 E2B and 12B configs both declare 262144 rows but disagree at token id
249788. vLLM's plain-draft check compares row counts. SGLang's STANDALONE worker has
a map comparison, but its draft worker has no tokenizer, so that comparison is not
performed. RigTune therefore needs identical tokenizer hashes or equal complete
maps, including added tokens, before admitting a plain draft. Missing evidence is
unknown. Assistant heads share target embeddings and have their own variant check.
The engine contracts were checked in vLLM 0.29/0.33 and SGLang 0.5.19/0.5.23.
A table proposes; revision-pinned evidence decides. Table mode remains the default
until the planned cross-engine GPU check succeeds.
