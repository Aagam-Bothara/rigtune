"""How many tokens a chain drafter should draft, from the acceptance curve one trial measured.

A chain drafter (an EAGLE3 head, a checkpoint's MTP head, a draft model, a Gemma 4 assistant) drafts its tokens one
after another, each from the one before, so drafting fewer leaves the earlier ones as they were: one trial at k tokens
tells what a step at any k' <= k would yield. vLLM records, for each draft position, the share of drafts accepted at
least that far (models.SpecAcceptance.per_position, rigtune.memlog), so a step drafting k' tokens yields

    E(k') = 1 + a_1 + ... + a_k'   tokens.

What a drafted token costs the step cannot be read from one trial; a model gives it from two that the search runs
anyway: the engine's leader without speculation (throughput r0) and the same configuration drafting k tokens (r1).
With a step's time t(k) = t0 (1 + alpha k), the extra time each drafted token adds as a share of a plain step,

    r1 / r0 = E(k) / (1 + alpha k)   so   alpha = (E(k) r0 / r1 - 1) / k,

and the best length is the k' that maximises E(k') / (1 + alpha k'). One more trial measures it, and calibration keeps
whichever of the two is faster: the model chooses what to measure, never the pick. A DFlash or DSpark drafter drafts a
block at once, trained at one size, so a shorter one is another drafter: none is proposed for them.

The model is linear in k and holds at the load it was measured at: drafting costs little on an idle GPU and much on a
busy one, which a length per number of users (vLLM's num_speculative_tokens_per_batch_size) would meet; not here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rigtune import speculative
from rigtune.models import Config, PreparedModel, TrialResult

CHAIN_KINDS = ("eagle3", "mtp", "draft", "assist")
# Tokens a chain drafter first drafts on vLLM under --speculative all, so its curve covers the lengths worth weighing:
# EAGLE3's authors evaluate chains up to 7 (EAGLE-3, Li et al. 2025), and a longer probe than the length chosen
# costs only its deeper positions' compute, up to the limits probe_tokens keeps it under. A checkpoint's MTP head keeps
# its own settings (speculative.MTP_TOKENS): a one-layer head repeated past its depth accepts less, and vLLM refuses
# some lengths for a deeper one.
PROBE_TOKENS = {"eagle3": 6, "draft": 6, "assist": 6}
# The length each drafted before this round, tried instead when a probe does not start.
BEFORE = {"eagle3": speculative.EAGLE3_TOKENS, "draft": speculative.DRAFT_TOKENS_VLLM, "assist": 1}
# vLLM captures CUDA graphs up to 512 tokens a step by default on cards before Blackwell (config/vllm.py lines
# 2460-2470), and a decoding request takes k + 1 of them: past 512 / (k + 1) running requests a step runs eagerly.
VLLM_GRAPH_TOKENS = 512
MIN_GAIN = 1.02  # a length predicted to gain less over no speculation is not worth a launch


def probe_tokens(kind: str, users: Optional[int], batched_tokens: int) -> int:
    """The tokens a chain drafter of `kind` first drafts for a workload of at most `users` running requests: the
    probe, or less where at that load it would run outside vLLM's CUDA graphs (VLLM_GRAPH_TOKENS) or have its drafts
    trimmed by the token budget (scheduler.py lines 845-848, 1018-1021; a trimmed draft spoils the curve), which a
    shorter draft would not; never below the length it drafted before."""
    probe = PROBE_TOKENS[kind]
    if users and users > 0:
        probe = min(probe, VLLM_GRAPH_TOKENS // users - 1, batched_tokens // users - 1)
    return max(probe, BEFORE[kind])


def expected_tokens(curve: List[float], k: int) -> float:
    """Tokens a step drafting k tokens yields: 1 + the first k per-position acceptance rates."""
    return 1.0 + sum(curve[:k])


def draft_cost(curve: List[float], k: int, plain: float, drafting: float) -> Optional[float]:
    """alpha: the share of a plain step each drafted token adds, from the throughput without speculation (`plain`) and
    drafting k tokens (`drafting`), both at one number of users; None when either is not a positive throughput. Never
    below 0: a measurement whose noise says drafting is free is read as free."""
    if plain <= 0 or drafting <= 0 or k < 1:
        return None
    return max(0.0, (expected_tokens(curve, k) * plain / drafting - 1.0) / k)


def best_length(curve: List[float], alpha: float, k_max: int, allowed=lambda k: True) -> Tuple[int, float]:
    """(k, predicted speed-up over no speculation) maximising E(k) / (1 + alpha k) over the allowed k in 1..k_max; the
    shorter on a tie, within 0.5%, since a shorter draft costs the engine less memory and its prediction is surer."""
    best_k, best = None, 0.0
    for k in range(1, min(k_max, len(curve)) + 1):
        if not allowed(k):
            continue
        gain = expected_tokens(curve, k) / (1.0 + alpha * k)
        if best_k is None or gain > best * 1.005:
            best_k, best = k, gain
    return (best_k or k_max), best


def with_tokens(spec: str, k: int) -> str:
    """The same speculation setting drafting k tokens."""
    kind, model_id, _ = speculative.parse(spec)
    return f"{kind}:{k}" if model_id is None else f"{kind}:{model_id}:{k}"


def _comparable(a: Config, b: Config) -> bool:
    """Whether two configurations differ in their speculation alone (and the memory fraction a start-up retry may
    lower, which moves the KV pool, not a step's time), so their throughputs measure its cost."""
    keep = {"spec_decode": None, "spec_head_quant": None, "gmu_before_spec": None, "gpu_memory_utilization": None}
    return a.model_copy(update=keep) == b.model_copy(update=keep)


def _shared_level(plain: TrialResult, drafting: TrialResult) -> Optional[Tuple[int, float, float]]:
    """(users, plain tok/s, drafting tok/s) at one number of users both trials measured: the drafting trial's best,
    else the busiest they share. A trial's headline throughput is its best level's (calibrate/measure.py), so two
    trials' headlines may be measured at different loads, where a drafted token costs differently."""
    a, b = plain.metrics.by_concurrency, drafting.metrics.by_concurrency
    if not a or not b:
        if plain.metrics.concurrency == drafting.metrics.concurrency:
            return plain.metrics.concurrency, plain.metrics.tok_s, drafting.metrics.tok_s
        return None
    shared = sorted(set(a) & set(b), key=int)
    if not shared:
        return None
    level = str(drafting.metrics.concurrency) if str(drafting.metrics.concurrency) in shared else shared[-1]
    return int(level), a[level].tok_s, b[level].tok_s


def _skipped(trial: TrialResult) -> bool:
    """Whether the search chose not to launch the trial (budget spent, or its method failed to start before)."""
    from rigtune.calibrate.search import failure_kind

    return (failure_kind(trial) or "").startswith("skipped_")


def refine(trials: List[TrialResult], done: Dict[str, TrialResult], models: Dict[str, PreparedModel]
           ) -> List[Tuple[Optional[Config], str]]:
    """What to measure next for each chain drafter the stage tried, with what to say about it: the length its curve
    predicts best, or the length it drafted before when the probe did not start. Each is weighed against a trial
    without speculation at the same settings among those already run (`done`, by key). A drafter with nothing to
    measure gets a note (config None) saying why."""
    out: List[Tuple[Optional[Config], str]] = []
    plain = [r for r in done.values() if r.ok and not r.config.spec_decode]
    for trial in trials:
        cfg = trial.config
        if not cfg.spec_decode or cfg.backend != "vllm":
            continue
        kind, _, k = speculative.parse(cfg.spec_decode)
        if kind not in CHAIN_KINDS:
            continue
        label = f"{cfg.backend} {cfg.spec_decode}"
        if not trial.ok:
            fallback = BEFORE.get(kind)
            if not trial.launched and not _skipped(trial) and fallback and fallback < k:
                again = cfg.model_copy(update={"spec_decode": with_tokens(cfg.spec_decode, fallback)})
                if again.key() not in done:
                    out.append((again, f"{label} did not start, so {fallback} tokens, the length drafted before "
                                       "the curve chose one"))
            continue
        acc = trial.metrics.spec_acceptance
        if acc is None or not acc.per_position or len(acc.per_position) != k or acc.full_drafts is not True:
            why = ("vLLM gave no per-position count" if acc is None or not acc.per_position else
                   "some drafts were cut short, so the curve understates a full draft" if acc.full_drafts is not True
                   else "its curve has another number of positions than tokens drafted")
            if k > 1:
                out.append((None, f"{label}: its draft length not tuned: {why}"))
            continue
        base = next((r for r in plain if r.config.backend == cfg.backend and _comparable(r.config, cfg)), None)
        level = _shared_level(base, trial) if base is not None else None
        if level is None:
            if k > 1:
                out.append((None, f"{label}: its draft length not tuned: no trial without speculation at the same "
                                  "settings and number of users to weigh a drafted token's cost against"))
            continue
        users, r0, r1 = level
        alpha = draft_cost(acc.per_position, k, r0, r1)
        if k == 1:
            continue
        if alpha is None:
            out.append((None, f"{label}: its draft length not tuned: no throughput at {users} users in both trials"))
            continue
        model = models.get(cfg.backend)
        layers = model.arch.mtp_layers if (model is not None and kind == "mtp") else 0
        # vLLM refuses more MTP tokens than the head's layers unless a multiple of them (config/speculative.py).
        allowed = (lambda n: n <= layers or n % layers == 0) if layers > 1 else (lambda n: True)
        best, gain = best_length(acc.per_position, alpha, k, allowed)
        curve = ", ".join(f"{a:.2f}" for a in acc.per_position)
        why = (f"from its acceptance curve ({curve}) and a drafted token costing {alpha:.2f} of a step at {users} "
               f"users ({r0:.0f} tok/s without speculation, {r1:.0f} with)")
        candidate = cfg.model_copy(update={"spec_decode": with_tokens(cfg.spec_decode, best)})
        if best == k:
            out.append((None, f"{label}: kept at {k} tokens: {why}, no shorter draft is predicted faster"))
        elif gain <= MIN_GAIN:
            out.append((None, f"{label}: no length tried: {why}, the best, {best} tokens, is predicted "
                              f"{gain:.2f}x the speed without speculation"))
        elif candidate.key() not in done:
            out.append((candidate, f"{label} at {best} tokens, predicted {gain:.2f}x the speed without "
                                   f"speculation {why}"))
    return out
