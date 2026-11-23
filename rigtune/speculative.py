"""Speculative decoding: guess several tokens cheaply, verify them in one step of the big model.

Decode is memory-bandwidth bound, so verifying k proposed tokens costs about the same as
generating one. When most guesses are accepted, per-token latency falls; when the batch is already
large the GPU has no idle bandwidth to spend on verification and it becomes a loss. rigtune
therefore tries it as a calibration stage and lets the objective decide.

Proposers:
  * n-gram prompt lookup: proposes continuations copied from the prompt. Free: no second model, and
    strong when outputs echo inputs (RAG, extraction, code edits). On vLLM 0.29 and later it runs on
    the GPU (`ngram_gpu`): vLLM 0.29 turns async scheduling off for its CPU n-gram and suffix
    proposers and keeps it for `ngram_gpu` and draft models. The CPU version lost 87% of throughput
    from four users up on an A40; whether losing async scheduling is why has not been measured.
  * suffix decoding (vLLM, needs arctic-inference 0.1.1 in vLLM's own Python: suffix_refusal): matches the prompt and
    the model's earlier outputs in a suffix tree and adapts how many tokens it proposes per request.
  * a draft model: a small model of the same family, sharing the tokenizer.
  * an EAGLE3 head: one decoder layer trained on the target's own hidden states, which proposes from them
    rather than from tokens alone. Heads are published per target model, so they are searched for on the Hub
    (rigtune.headsearch, with EAGLE3_HEADS' entries first), and a head is used only when its config.json shows that
    the engine loads it as one and that it fits the target (rigtune.heads).
  * the checkpoint's own MTP (multi-token-prediction) head: Qwen3.5-family checkpoints ship one beside the model,
    trained with it, so no second repository is needed (rigtune.introspect finds it). Offered only under
    --speculative all, and only where the config declares the head and the weights hold it.

Spec strings kept in Config.spec_decode: "ngram:<k>", "ngram_gpu:<k>", "suffix:<k>", "mtp:<k>", "draft:<hf_id>:<k>"
or "eagle3:<hf_id>:<k>".
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from rigtune.models import ArchInfo, ModelSpec, PreparedModel

logger = logging.getLogger(__name__)

# (target pattern, draft model). The draft must share the target's tokenizer. Llama drafts come from the target's
# own organisation, so unsloth's ungated copies of Meta's gated models draft with an ungated model too.
DRAFTS = [
    # Qwen2.5 splits its vocabulary by size: 0.5B, 1.5B and 3B carry 151936 tokens, 7B and larger 152064 (read
    # from each config.json). vLLM refuses a draft whose vocabulary differs from the target's, so the small draft
    # serves only the small targets. Offering it for 14B produced a configuration that could never launch:
    # "Target and draft model should have the same vocabulary size. Target model vocab_size=152064. Draft model
    # vocab_size=151936" (Qwen2.5-14B-Instruct on an RTX 4090, 2026-09-16). Qwen2.5-Coder splits the same way.
    (r"^Qwen/Qwen2\.5-(?:1\.5|3)B-Instruct$", "Qwen/Qwen2.5-0.5B-Instruct"),
    (r"^Qwen/Qwen2\.5-Coder-(?:1\.5|3)B-Instruct$", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    # Qwen3 keeps one vocabulary (151936) from 0.6B through 32B, so one draft covers every size here.
    (r"^Qwen/Qwen3-(?:1\.7|4|8|14|32)B$", "Qwen/Qwen3-0.6B"),
    (r"^(meta-llama|unsloth)/Llama-3\.2-3B-Instruct$", r"\1/Llama-3.2-1B-Instruct"),
    (r"^(meta-llama|unsloth)/(?:Meta-)?Llama-3\.1-(?:8|70)B-Instruct$", r"\1/Llama-3.2-1B-Instruct"),
    (r"^(meta-llama|unsloth)/Llama-3\.3-70B-Instruct$", r"\1/Llama-3.2-1B-Instruct"),
]

# (target pattern, EAGLE3 heads ranked first in the Hub search, rigtune.headsearch). Each one's config.json was read
# on the Hub; a head is still used only when its config passes rigtune.heads.check_head, so a repository that changes
# or disappears costs a lookup, not a trial.
#   lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B: architectures ["LlamaForCausalLMEagle3"], vocab_size 128256,
#     hidden_size 4096, one layer, draft_vocab_size 32000, about 850 MB.
#   AngelSlim/Qwen3-8B_eagle3: ["LlamaForCausalLMEagle3"], vocab_size 151936, hidden_size 4096.
# Not listed: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B declares plain LlamaForCausalLM, which SGLang 0.5.19 would load
# as a full model rather than a head (its _config_draft_model renames only MTP/NextN architectures).
EAGLE3_HEADS = [
    (r"^(meta-llama|unsloth)/(?:Meta-)?Llama-3\.1-8B-Instruct$", ("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",)),
    (r"^Qwen/Qwen3-8B$", ("AngelSlim/Qwen3-8B_eagle3",)),
]

# Proposals only: every row is checked from immutable files before use. Hub
# config/tokenizer facts read 2026-11-15 (plan-drafts section 2.3).
PAIR_CANDIDATES = [
    # Gemma 4: first-party assistants declare Gemma4[Unified]AssistantForCausalLM.
    (r"^google/gemma-4-(?:E2B|E4B|12B|26B-A4B|31B)-it$", (("assistant", r"\g<0>-assistant"),)),
    # Ministral 3 2512: 131072 rows and matching tokenizer/tekken hashes; cost still checked.
    (r"^mistralai/Ministral-3-(?:8|14)B-(Instruct|Reasoning)-2512$",
     (("draft", r"mistralai/Ministral-3-3B-\1-2512"),)),
    # Existing Qwen and Llama rows: config vocab and tokenizer evidence documented above;
    # gated Meta files must be read, never equated to the unsloth copy by name.
    *((pattern, (("draft", repo),)) for pattern, repo in DRAFTS),
]


def candidates_for(hf_id: str, draft_repo: Optional[str] = None) -> List[Tuple[str, str]]:
    out = [("draft", draft_repo)] if draft_repo else []
    for pattern, pairs in PAIR_CANDIDATES:
        match = re.match(pattern, hf_id)
        if match:
            out.extend((kind, match.expand(repo)) for kind, repo in pairs)
    return list(dict.fromkeys(out))

# vLLM 0.11's V1 engine rejects a separate draft model ("not supported yet"); n-gram works there.
VLLM_DRAFT_MIN = (0, 12)
# Both methods are in vLLM 0.29's SpeculativeConfig; earlier releases were not checked.
VLLM_NGRAM_GPU_MIN = (0, 29)
VLLM_SUFFIX_MIN = (0, 29)
# vLLM 0.29's model registry maps LlamaForCausalLMEagle3 to its Eagle3 Llama model and its SpeculativeConfig takes
# method "eagle3"; earlier releases were not checked.
VLLM_EAGLE3_MIN = (0, 29)
# DFlash and DSpark drafters were read in vLLM v0.33.0 (config/speculative.py SpeculativeMethod, v1/worker/gpu/
# spec_decode/dflash and dspark); earlier releases were not checked.
VLLM_BLOCK_DRAFTER_MIN = (0, 33)

NGRAM_TOKENS = 4
NGRAM_TOKENS_LLAMACPP = 64  # llama.cpp's ngram-mod default
SUFFIX_TOKENS = 24  # vLLM's default for suffix decoding: its maximum tree depth
DRAFT_TOKENS_VLLM = 4
DRAFT_TOKENS_LLAMACPP = 16
# Tokens an EAGLE3 head proposes per step, as a chain (one candidate per position): SGLang's own chain default
# (3 steps, top-k 1) for models it does not tune as Llama-class.
EAGLE3_TOKENS = 3
# SGLang: a draft model proposes a chain of 3, as an EAGLE3 head does, and n-gram lookup keeps SGLang's own 12
# draft tokens (its tree is speculative_ngram_max_bfs_breadth wide, 10 by default).
DRAFT_TOKENS_SGLANG = 3
NGRAM_TOKENS_SGLANG = 12

# The Qwen3.5 family, whose own MTP head ("mtp:<k>") was read in both engines' source and measured:
#   vLLM (v0.29.0 and v0.33.0): config/speculative.py SpeculativeConfig.hf_config_override turns model types
#     qwen3_5, qwen3_5_moe, qwen3_5_text and qwen3_5_moe_text (lines 882-899 / 987-1007) into Qwen3_5MTP or
#     Qwen3_5MoeMTP (model_executor/models/qwen3_5_mtp.py); with no "model" the draft is the target's own
#     checkpoint and quantization (SpeculativeConfig.__post_init__, lines 1096-1111 / 1252-1267). vLLM is now asked
#     for every family (rigtune.backends.own_head); this list is what it is checked against when it cannot be asked,
#     and what the planner's MTP rule and the fp8 output head were read for.
#   SGLang (v0.5.19 and v0.5.23): configs/model_config.py ModelConfig._config_draft_model loads a draft of these
#     four architectures as Qwen3_5ForCausalLMMTP (lines 751-773 / 1159-1181).
MTP_ARCHS = frozenset({"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
                       "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"})
# Tokens the head proposes per step, each tried on both engines so they are compared on one setting, as with EAGLE3:
#   1, what vLLM's docs/features/speculative_decoding/mtp.md advises starting with (both tags) and v0.33.0's
#      recipes/models/Qwen/Qwen3.5-9B.yaml passes; it accepts at most 2 tokens a step.
#   3, SGLang's Qwen3.5 cookbook chain (3 steps, top-k 1, 4 draft tokens: docs/cookbook/autoregressive/Qwen/
#      Qwen3.5.mdx, lines 246-249 at v0.5.19, 262-265 at v0.5.23).
# vLLM takes any count for a one-layer head (more than n_predict must be a multiple of it, config/speculative.py
# lines 1417-1430 / 1584-1613) and warns that more than one reruns the same layer, "which may result in lower
# acceptance rate"; whether it does here is what the trial measures. SGLang with 1 step drafts 2 tokens.
MTP_TOKENS = (1, 3)
MTP_KEY = "mtp"  # PreparedModel.spec_repos / spec_bytes key of the head
MTP_SHARED_KEY = "mtp_shared"  # spec_bytes key of the target's embedding and output head (introspect.MtpHead)

LOOKUP = ("ngram", "ngram_gpu", "suffix")  # proposers that need no second model
# Drafters that read the target's hidden states and draft a block at once (rigtune.heads): DFlash, DSpark. vLLM only
# until SGLang's launch is written (sglang_config refuses them).
BLOCK_DRAFTERS = ("dflash", "dspark")
SECOND_MODEL = ("draft", "eagle3", "assist", *BLOCK_DRAFTERS)  # proposers whose second set of weights the spec names
SELF_HEAD = ("mtp",)  # proposers whose weights ship in the target's own checkpoint


def draft_for(hf_id: str) -> Optional[str]:
    """A smaller model from the same family that can draft for `hf_id`, or None."""
    for pattern, draft in DRAFTS:
        m = re.match(pattern, hf_id)
        if m:
            return m.expand(draft)
    return None


def eagle3_heads_for(hf_id: str) -> Tuple[str, ...]:
    """The EAGLE3 heads EAGLE3_HEADS lists for `hf_id`, ranked first in the Hub search."""
    for pattern, heads in EAGLE3_HEADS:
        m = re.match(pattern, hf_id)
        if m:
            return tuple(m.expand(h) for h in heads)
    return ()


def verified_eagle3_head(cfg: Optional[Dict[str, Any]], target: ArchInfo, engine: str,
                         files: Optional[List[str]] = None) -> bool:
    """Whether a head's config.json shows an EAGLE3 head that `engine` loads as one and that can draft for `target`
    (rigtune.heads.check_head: the engine's own rules for which architectures are heads, and the vocabulary, hidden
    size and layers the head reads from the target). A repository's name is not evidence: yuhuili's EAGLE3 head for
    Llama 3.1 8B declares plain LlamaForCausalLM, which vLLM 0.33 renames into its Eagle3 model and SGLang 0.5.23
    would load as a full Llama."""
    from rigtune.heads import check_head

    if not cfg:
        return False
    verdict = check_head(target, cfg, engine, files=files)
    return verdict.status == "ok" and verdict.evidence.get("kind") == "eagle3"


class HubLookupError(RuntimeError):
    """The Hub could not be asked: offline, rate-limited, or a gated repository without a token."""


def repo_bytes(repo: str, token: Optional[str] = None) -> Optional[int]:
    """Size of the weight files an engine loads from a repository (hfconfig.listed_bytes: its safetensors, else its
    .bin files); None when it has none or one has no size. Raises when the Hub cannot say."""
    from huggingface_hub import HfApi, hf_hub_download

    from rigtune.hfconfig import INDEX, hub_listing, listed_bytes

    info = HfApi(token=token).model_info(repo, files_metadata=True)
    sha = getattr(info, "sha", None)
    return listed_bytes(hub_listing(info), lambda: hf_hub_download(
        repo, INDEX, revision=sha if isinstance(sha, str) else None, token=token))


def _best_head(found: Any, target: ArchInfo, engine: str, kind: str = "eagle3", release: Optional[str] = None,
               notes: Optional[List[str]] = None) -> Any:
    """The best drafter of `kind` in a search for `engine`: the first that fits, one whose size is known first (SGLang
    offers no second model of unknown size, and vLLM's planner counts a block drafter's weights), else None. One the
    engine's `release` refused before is passed over (headsearch.heads_for), said in `notes`."""
    from rigtune import headsearch

    heads = headsearch.heads_for(found, target, engine, kind, release=release, notes=notes)
    return sorted(heads, key=lambda c: c.size_bytes is None)[0] if heads else None


def found_head_notes(engine: str, spec: ModelSpec, target: ArchInfo, repo: str) -> List[str]:
    """The profile's notes for the head `engine` got: where it came from when rigtune does not list it, and the other
    engine's best head when it is another, so a comparison between the engines (when both are calibrated) is read
    knowing they would draft with different heads. Read from the search kept for this target (no request)."""
    from rigtune import headsearch

    notes = []
    if repo not in eagle3_heads_for(spec.hf_id):
        notes.append(f"{engine}: EAGLE3 head {repo} found by a Hub search; its config fits {spec.hf_id} on {engine} "
                     "(rigtune.heads); it has not been run by rigtune before")
    other = "sglang" if engine == "vllm" else "vllm"
    try:
        found, _ = headsearch.kept_or_search(spec.hf_id, spec.revision, hub_off=True)
        theirs = _best_head(found, target, other) if found is not None else None
    except Exception:
        theirs = None
    if theirs is not None and theirs.repo != repo:
        notes.append(f"{engine}: drafts with EAGLE3 head {repo}; {other}'s best head for this model is "
                     f"{theirs.repo}, which {engine} does not load, ranks lower, or refused before")
    return notes


def find_drafter(spec: ModelSpec, target: ArchInfo, *, engine: str, kind: str, hub: Any = None,
                 hub_off: Optional[bool] = None, release: Optional[str] = None,
                 notes: Optional[List[str]] = None) -> Any:
    """The best drafter of `kind` ("eagle3", "dflash", "dspark") for `spec` that `engine` loads and that fits
    `target` (a headsearch.Candidate: repo, size, config), or None when none does. The drafters come from a Hub
    search (rigtune.headsearch: by name and base model, the best of each kind read and judged by
    rigtune.heads.check_head per engine), kept for the day; EAGLE3_HEADS' entries rank first, then downloads, and one
    whose size is known comes before one whose is not. Each engine gets its own best. With the Hub off (`hub_off`, or
    HF_HUB_OFFLINE) what was kept is used. A drafter the engine's `release` ("vllm 0.33.0") refused to start with in
    an earlier calibration is passed over for the next (headsearch.refuse), said in `notes`. Raises HubLookupError when no search could be made or kept, or when an
    incomplete one found none, so the caller can tell "none fits" from "could not ask". Never searches for a local
    directory."""
    from rigtune import headsearch
    from rigtune.pairing import hub_off as offline, is_local

    if is_local(spec.hf_id):
        return None
    try:
        found, said = headsearch.kept_or_search(spec.hf_id, spec.revision, eagle3_heads_for(spec.hf_id),
                                                 hub_off=bool(hub_off) or offline(), hub=hub)
    except Exception as exc:
        raise HubLookupError(f"could not search the Hub ({type(exc).__name__}: {exc})") from None
    if found is None:
        raise HubLookupError("; ".join(said))
    passed_over: List[str] = []
    best = _best_head(found, target, engine, kind, release, passed_over)
    if notes is not None:
        notes += passed_over
    if best is None and not found.complete and not passed_over:  # a refused drafter is no failed lookup
        raise HubLookupError("the search was incomplete and found no head: " + "; ".join(found.notes))
    return best


def record_block_drafters(prepared: PreparedModel, engine: str, kinds: Iterable[str],
                          release: Optional[str] = None) -> List[str]:
    """Record in `prepared` the best drafter of each of `kinds` ("dflash", "dspark") the Hub search found that `engine`
    loads and that fits the model (find_drafter), with what the backends plan and launch it by: its k and lookahead
    (drafter_tokens: from its block size, the same k on both engines), its size, its KV per token, and the commit
    whose config was checked. Returns notes saying where each came from, or why none was recorded."""
    notes: List[str] = []
    for kind in kinds:
        try:
            cand = find_drafter(prepared.spec, prepared.arch, engine=engine, kind=kind, hub_off=prepared.hub_off,
                                release=release, notes=notes)
        except Exception as exc:
            notes.append(f"{engine}: no {kind} drafter tried, the Hub could not be asked ({exc})")
            continue
        if cand is None:
            continue
        tokens = drafter_tokens(cand.config, kind)
        if tokens is None:
            notes.append(f"{engine}: {kind} drafter {cand.repo} not tried: its config states no block size, so the "
                         "number of tokens it drafts is unknown")
            continue
        prepared.spec_repos[kind] = cand.repo
        if cand.size_bytes:
            prepared.spec_bytes[cand.repo] = int(cand.size_bytes)
        prepared.spec_facts[cand.repo] = {"k": tokens[0], "lookahead": tokens[1],
                                          "kv_elements": drafter_kv_elements(cand.config),
                                          "needs_v2": drafter_needs_v2(cand.config)}
        if cand.sha:  # launched at the commit whose config was checked
            prepared.spec_revisions[cand.repo] = cand.sha
        notes.append(f"{engine}: {kind} drafter {cand.repo} found by a Hub search, drafting {tokens[0]} tokens a step "
                     "(its block size); it has not been run by rigtune before")
    return notes


def find_eagle3_head(spec: ModelSpec, target: ArchInfo, *, engine: str, hub: Any = None,
                     hub_off: Optional[bool] = None, release: Optional[str] = None,
                     notes: Optional[List[str]] = None) -> Optional[Tuple[str, Optional[int]]]:
    """(repo, size in bytes or None) of the best EAGLE3 head (find_drafter), or None."""
    best = find_drafter(spec, target, engine=engine, kind="eagle3", hub=hub, hub_off=hub_off, release=release,
                        notes=notes)
    return (best.repo, best.size_bytes) if best is not None else None


def look_for_mtp_head(spec: ModelSpec, engine: str, inspect: Optional[Callable[[ModelSpec], Any]] = None):
    """The checkpoint's own MTP head (rigtune.introspect: its config's layer count and its tensors, from the
    safetensors headers only), as (introspect.ModelFacts or None, notes). The notes say why none was found when the
    config declares one (weights without it, or a Hub that could not be asked)."""
    from rigtune import introspect

    look = inspect or (lambda s: introspect.inspect_model(s, features=("mtp",)))
    try:
        facts = look(spec)
    except Exception as exc:  # inspect_model never raises; an injected one might
        return None, [f"{engine}: no MTP head tried ({exc})"]
    if facts.mtp is None:
        return None, ([f"{engine}: no MTP head tried: " + "; ".join(facts.notes)] if facts.notes else [])
    return facts, []


def record_mtp_head(prepared: PreparedModel, engine: str,
                    inspect: Optional[Callable[[ModelSpec], Any]] = None, found: Any = None) -> List[str]:
    """Record the checkpoint's own MTP head in `prepared` (`found`, from look_for_mtp_head, or looked for now):
    spec_repos["mtp"] is the model itself and spec_bytes["mtp"] the head's size and spec_bytes["mtp_shared"] that of
    the target's embedding and output head, when the listing gave them. Returns notes for the profile."""
    facts = found
    if facts is None:
        facts, notes = look_for_mtp_head(prepared.spec, engine, inspect)
        if facts is None:
            return notes
    prepared.spec_repos[MTP_KEY] = prepared.spec.hf_id
    if facts.mtp.bytes:
        prepared.spec_bytes[MTP_KEY] = int(facts.mtp.bytes)
    if facts.mtp.shared_bytes is not None:  # 0: the head brings its own embedding and output head
        prepared.spec_bytes[MTP_SHARED_KEY] = int(facts.mtp.shared_bytes)
    # The planner's count (hfconfig.arch_from_config) comes from the config alone; a head counted from its tensors
    # (Qwen3-Next declares none) must reach it too.
    if prepared.arch.mtp_layers != facts.mtp.layers:
        prepared.arch.mtp_layers = facts.mtp.layers
    return [f"{engine}: {n}" for n in facts.notes]


def _vllm_version() -> Optional[str]:
    try:
        from importlib.metadata import version as installed

        return installed("vllm")
    except Exception:
        return None


def _vllm_at_least(minimum: Tuple[int, int], version: Optional[str] = None) -> bool:
    m = re.match(r"(\d+)\.(\d+)", (version if version is not None else _vllm_version()) or "")
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= minimum


def _vllm_lists(method: str, version: Optional[str]) -> bool:
    """For the installed vLLM (no `version` given): False only when it answered the engine probe and its
    SpeculativeMethod does not list `method`. The version gates above stay: the probe can take a method away, never
    offer one a gate holds back, since a gate records which releases were checked or measured, which a Literal cannot
    say."""
    if version is not None:
        return True
    from rigtune.backends.vllm import _probe_lists

    return _probe_lists("spec_methods", method)


def vllm_supports_draft(version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM accepts a draft model for speculative decoding."""
    return _vllm_at_least(VLLM_DRAFT_MIN, version) and _vllm_lists("draft_model", version)


def vllm_supports_ngram_gpu(version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM runs n-gram lookup on the GPU."""
    return _vllm_at_least(VLLM_NGRAM_GPU_MIN, version) and _vllm_lists("ngram_gpu", version)


def vllm_supports_block_drafter(kind: str, version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM is one checked to run a DFlash or DSpark drafter of `kind`."""
    return _vllm_at_least(VLLM_BLOCK_DRAFTER_MIN, version) and _vllm_lists(kind, version)


def vllm_supports_eagle3(version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM is one checked to run an EAGLE3 head."""
    return _vllm_at_least(VLLM_EAGLE3_MIN, version) and _vllm_lists("eagle3", version)


def vllm_suffix_release(version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM is a release with suffix decoding."""
    return _vllm_at_least(VLLM_SUFFIX_MIN, version) and _vllm_lists("suffix", version)


def vllm_suffix_refusal() -> Optional[str]:
    """Why Arctic Inference cannot run suffix decoding in the Python the vLLM launch command runs, from a positive
    finding there (suffix_refusal); None when it can, or when that could not be asked (rigtune.backends.vllm
    _suffix_caveat then says so)."""
    from rigtune.backends.vllm import _suffix_refusal

    return _suffix_refusal()


def vllm_supports_suffix(version: Optional[str] = None) -> bool:
    """Whether vLLM can use suffix decoding here: a release with it, and nothing found wrong with Arctic Inference,
    which holds its code, in vLLM's own Python. Until 14 Nov 2026 the second half was find_spec("arctic_inference")
    in rigtune's own process, which passed on a pod where every suffix trial then failed to start."""
    return vllm_suffix_release(version) and vllm_suffix_refusal() is None


# Suffix decoding's code is Arctic Inference's (pip arctic-inference), not vLLM's. What vLLM v0.29.0 and v0.33.0 need
# of it (both read):
#   config/speculative.py SpeculativeConfig._validate_suffix_decoding raises ImportError("Arctic Inference is required
#     for suffix decoding. Install via `pip install arctic-inference==0.1.1`.") unless
#     utils/import_utils.has_arctic_inference(), which imports the top-level `arctic_inference` package only
#     (_has_module). It checks no version.
#   v1/spec_decode/suffix_decoding.py SuffixDecodingProposer.__init__ imports SuffixDecodingCache from
#     arctic_inference.suffix_decoding, whose cache.py imports the native suffix tree, arctic_inference.suffix_decoding
#     ._C. That module is built from the sdist on the user's machine (every release from 0.0.6 is an sdist only, PyPI),
#     so a broken build passes the first check and fails in the engine core, after the server has started loading.
#   arctic-inference 0.1.0 to 0.3.0 ship byte-identical suffix_decoding/ and csrc/suffix_decoding/ (read from each
#     sdist); 0.0.9's tree has another native API (Candidate where 0.1.x has Draft), which vLLM was not written
#     against. So any release from 0.1.0 is accepted, and 0.1.1, the one vLLM's message names and the one measured,
#     is the one advised.
#   Its vLLM plugin (entry point vllm.general_plugins) is loaded in every vLLM process and run outside
#     load_plugins_by_group's try (plugins/__init__.py load_general_plugins), whatever the speculative method. It does
#     nothing unless ARCTIC_INFERENCE_ENABLED=1; with it, every release raises RuntimeError unless vLLM is the one
#     release it patches (0.1.1: vllm==0.10.1, its "vllm" extra; 0.1.2: 0.11.0; 0.2.0: 0.18.0; 0.3.0:
#     utils.VLLM_PATCH_VERSION "0.26.0"), unless ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1. So it stops every vLLM launch,
#     not only suffix trials ("Arctic Inference plugin requires vllm==0.10.1 but found vllm==0.29.0!", captured).
#     vLLM's suffix decoding does not need the plugin.
ARCTIC_ADVISED = "0.1.1"
ARCTIC_MIN = (0, 1, 0)
SUFFIX_DOCS = "docs/usage.md#suffix-decoding"
# Run in vLLM's Python, never in rigtune's: it imports the package and the suffix tree as vLLM does, builds one, and
# says which vLLM an enabled plugin would demand. It imports numpy and the extension, never vLLM.
SUFFIX_PROBE = (
    "import importlib.metadata as m, json, os, sys\n"
    "def v(n):\n"
    "    try:\n"
    "        return m.version(n)\n"
    "    except Exception:\n"
    "        return None\n"
    "def err(exc):\n"
    "    return (type(exc).__name__ + ': ' + str(exc))[:300]\n"
    "out = {'python': sys.version.split()[0], 'arctic_inference': v('arctic_inference'), 'vllm': v('vllm')}\n"
    "try:\n"
    "    import arctic_inference\n"
    "    out['toplevel'] = 'ok'\n"
    "except BaseException as exc:\n"
    "    out['toplevel'] = err(exc)\n"
    "try:\n"
    "    from arctic_inference.suffix_decoding import SuffixDecodingCache\n"
    "    SuffixDecodingCache(max_tree_depth=24, max_cached_requests=1)\n"
    "    out['suffix'] = 'ok'\n"
    "except BaseException as exc:\n"
    "    out['suffix'] = err(exc)\n"
    "if os.environ.get('ARCTIC_INFERENCE_ENABLED') == '1' "
    "and os.environ.get('ARCTIC_INFERENCE_SKIP_VERSION_CHECK') != '1' and out['toplevel'] == 'ok':\n"
    "    try:\n"
    "        from arctic_inference.utils import get_compatible_vllm_version\n"
    "        out['plugin_vllm'] = get_compatible_vllm_version()\n"
    "    except BaseException:\n"
    "        out['plugin_vllm'] = None\n"
    "print(json.dumps(out))\n")


def arctic_install_line(python: str) -> str:
    """The install line advised for the Python vLLM runs from. arctic-inference's setup.py hands CMake
    -DPYTHON_EXECUTABLE, which the FindPython its csrc/suffix_decoding/CMakeLists.txt calls does not read, so CMake
    takes the first Python with headers on PATH. Rerunning the 14 Nov pod's install step in its image (Python 3.9-3.13
    from deadsnakes) with /usr/bin first on PATH, as the pod's scripts put it, CMake took python3.13 and built
    _C.cpython-313-x86_64-linux-gnu.so for a vLLM on 3.12, while `import arctic_inference` still worked. CMAKE_ARGS
    (which setup.py passes on) names the interpreter; --no-cache-dir because pip otherwise reinstalls the wheel it
    cached from the broken build; --force-reinstall to replace an installed one. Each was checked in that image
    (docs/benchmarks.md, Limits)."""
    return f'CMAKE_ARGS="-DPython_EXECUTABLE={python}" {python} -m pip install --no-cache-dir ' \
           f"--force-reinstall arctic-inference=={ARCTIC_ADVISED}"


def run_suffix_probe(python: str, timeout: float = 60.0) -> Dict[str, Any]:
    """SUFFIX_PROBE's report from `python`, in rigtune's environment (the engine inherits it), run from an empty
    directory so that nothing in the current one (a directory named arctic_inference) is imported in its place:
    `python -c` puts the working directory first on sys.path, and -P, which would not, needs Python 3.11. Raises when
    it cannot be run (OSError, subprocess.TimeoutExpired) or says nothing readable (RuntimeError, ValueError)."""
    import json
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rigtune-probe-") as empty:
        out = subprocess.run([python, "-c", SUFFIX_PROBE], capture_output=True, text=True, timeout=timeout,
                             cwd=empty)
    lines = (out.stdout or "").strip().splitlines()
    if out.returncode != 0 or not lines:
        tail = ((out.stderr or "").strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"exited {out.returncode}: {tail[:200]}")
    report = json.loads(lines[-1])
    if not isinstance(report, dict):
        raise ValueError("the probe printed no report")
    return report


def _no_paths(text: str) -> str:
    """`text` with directories dropped from any path in it, so a note in a committed profile names no user's home."""
    return re.sub(r"(?:[A-Za-z]:)?(?:[\\/][^\s'\"\\/:]+)+[\\/]", "", text)


def suffix_refusal(report: Dict[str, Any]) -> Optional[str]:
    """Why vLLM's suffix decoding would fail to start with the Arctic Inference that `report` (SUFFIX_PROBE's) shows,
    or None when nothing is wrong. Only a positive finding refuses; what could not be asked is suffix_caveat's.
    The text names no interpreter path: it goes into the profile, which may be committed."""
    version, suffix = report.get("arctic_inference"), str(report.get("suffix") or "")
    toplevel = str(report.get("toplevel") or "")
    recalibrate = f"see {SUFFIX_DOCS}, then run `rigtune recalibrate`"
    if suffix == "ok":
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", version or "")
        if m and tuple(int(g) for g in m.groups()) < ARCTIC_MIN:
            return (f"arctic-inference {version} predates 0.1.0, whose suffix tree vLLM's proposer was written "
                    f"against; {recalibrate}")
    elif not version and toplevel.startswith("ModuleNotFoundError") and "'arctic_inference'" in toplevel:
        return f"arctic-inference is not installed in vLLM's Python; {recalibrate}"
    elif toplevel != "ok":
        return (f"arctic-inference {version or '(version unknown)'} fails to import in vLLM's Python "
                f"({_no_paths(toplevel)}); {recalibrate}")
    else:
        return (f"arctic-inference {version or '(version unknown)'}'s suffix tree does not load in vLLM's Python "
                f"({_no_paths(suffix)}), so vLLM would fail in its engine core; rebuild it, {recalibrate}")
    plugin = plugin_refusal(report)
    return f"{plugin} (vLLM's suffix decoding does not need the plugin)" if plugin else None


def plugin_refusal(report: Dict[str, Any]) -> Optional[str]:
    """What an enabled arctic-inference plugin does to every vLLM launch, when `report` shows it would refuse this
    vLLM; None otherwise."""
    if "plugin_vllm" not in report or report.get("plugin_vllm") == report.get("vllm"):
        return None
    version = report.get("arctic_inference")
    return (f"ARCTIC_INFERENCE_ENABLED=1 turns on arctic-inference{' ' + version if version else ''}'s vLLM plugin, "
            f"which refuses vLLM {report.get('vllm') or '(version unknown)'} at start-up (it patches "
            f"vllm=={report.get('plugin_vllm') or '(unknown)'} only), so every vLLM launch would fail; unset it")


def suffix_caveat(error: str) -> str:
    """The note for a check that could not run: the method is offered as before, since a launch that fails costs
    one trial, while removing a working method could cost rigtune's largest recorded gain."""
    return f"could not check that arctic-inference loads in vLLM's Python ({error}); suffix decoding offered anyway"


def ngram(k: int = NGRAM_TOKENS) -> str:
    return f"ngram:{k}"


def ngram_gpu(k: int = NGRAM_TOKENS) -> str:
    return f"ngram_gpu:{k}"


def suffix(k: int = SUFFIX_TOKENS) -> str:
    return f"suffix:{k}"


def draft(model_id: str, k: int) -> str:
    return f"draft:{model_id}:{k}"


def eagle3(head: str, k: int = EAGLE3_TOKENS) -> str:
    return f"eagle3:{head}:{k}"


def assist(repo: str, k: int = 1) -> str:
    return f"assist:{repo}:{k}"


def block_drafter(kind: str, repo: str, k: int) -> str:
    """A DFlash or DSpark drafter proposing k tokens a step (vLLM's num_speculative_tokens)."""
    if kind not in BLOCK_DRAFTERS:
        raise ValueError(f"not a block drafter: {kind}")
    return f"{kind}:{repo}:{k}"


def drafter_tokens(config: Dict[str, Any], kind: str) -> Optional[Tuple[int, int]]:
    """(k, lookahead) a DFlash or DSpark drafter is trained for, from its config.json; None when it states no block.

    DFlash drafts a block of block_size: an anchor token and block_size - 1 masks, so it proposes block_size - 1
    (vLLM's recipes: 7 for RedHatAI's block-8 drafters, "block_size 8 minus one", recipes/models/Qwen/Qwen3.8-27B.yaml
    line 130); vLLM requires k for it (config/speculative.py lines 1614-1618) and checks it against no block size: its
    query is 1 + k tokens whatever the block (v1/worker/gpu/spec_decode/dflash/speculator.py line 65), and its own
    acceptance test runs z-lab/Qwen3-8B-DFlash-b16 at 16, the anchor's position past the trained block. The block less
    one is the trained choice, and what C10's acceptance curve is to revisit. A
    DSpark drafter that samples from its anchor (DeepSeek's dense drafters, sample_from_anchor unset) proposes the whole
    block (vLLM defaults k to block_size, lines 1592-1599; its test runs deepseek-ai/dspark_qwen3_8b_block7 at 7); the
    speculators-format ones set sample_from_anchor false and propose block_size - 1. The lookahead is the query tokens
    a request takes per step, k + 1 except for an anchor-sampling DSpark (k): vLLM caps the running requests at
    max_num_batched_tokens // lookahead (config/vllm.py lines 2378-2396, with only a warning)."""
    if kind not in BLOCK_DRAFTERS or not isinstance(config, dict):
        return None
    inner = config.get("dflash_config") if isinstance(config.get("dflash_config"), dict) else {}
    block = inner.get("block_size") or config.get("block_size") or config.get("dspark_block_size")
    if not isinstance(block, int) or isinstance(block, bool) or block < 2:
        return None
    if kind == "dspark":
        from rigtune.heads import is_speculators

        anchor = config.get("sample_from_anchor", not is_speculators(config))
        return (block, block) if anchor else (block - 1, block)
    return block - 1, block


def drafter_kv_elements(config: Dict[str, Any]) -> Optional[int]:
    """KV elements a drafter holds per token across its attention layers (2 x KV heads x head size x layers), from its
    config.json (the speculators format keeps them under transformer_layer_config); None when it does not say. vLLM
    gives every drafter attention layer a KV spec of its own in the target's pool (v1/worker/gpu/attn_utils.py lines
    136-152, is_draft)."""
    from rigtune.heads import is_speculators

    if not isinstance(config, dict):
        return None
    layer = config.get("transformer_layer_config") if is_speculators(config) else config
    if not isinstance(layer, dict):
        return None
    layers, heads = layer.get("num_hidden_layers"), layer.get("num_attention_heads")
    kv_heads = layer.get("num_key_value_heads") or heads
    head_dim = layer.get("head_dim") or ((layer.get("hidden_size") // heads)
                                         if isinstance(layer.get("hidden_size"), int) and heads else None)
    if not all(isinstance(v, int) and v > 0 for v in (layers, kv_heads, head_dim)):
        return None
    # Gemma 4's full layers have heads of their own (transformers configuration_gemma4.py: global_head_dim, and with
    # attention_k_eq_v num_global_key_value_heads), which vLLM builds each DSpark layer from (gemma4_dspark.py lines
    # 50-52, per_layer_config): deepseek-ai/dspark_gemma4_12b_block7's five full layers hold 1 x 512, not 8 x 256.
    types = layer.get("layer_types")
    full_dim = layer.get("global_head_dim")
    if isinstance(types, list) and len(types) == layers and isinstance(full_dim, int) and full_dim > 0:
        full_heads = layer.get("num_global_key_value_heads") if layer.get("attention_k_eq_v") else None
        full = 2 * (full_heads if isinstance(full_heads, int) and full_heads > 0 else kv_heads) * full_dim
        return sum(full if t == "full_attention" else 2 * kv_heads * head_dim for t in types)
    return 2 * kv_heads * head_dim * layers


# DFlash drafters vLLM v0.33.0 runs on Model Runner V2 only (config/vllm.py _is_dflash_candidate_draft, lines
# 769-784, refused at 3133-3137).
VLLM_V2_ONLY_DRAFTERS = frozenset({"DFlash2DraftModel", "DFlash2BeamDraftModel", "LiLiCorrDraftModel"})


def drafter_needs_v2(config: Dict[str, Any]) -> bool:
    """Whether vLLM runs this DFlash drafter on Model Runner V2 only: one of VLLM_V2_ONLY_DRAFTERS, or one whose layers
    mix sliding and full attention (config/vllm.py lines 3129-3131), as z-lab's Qwen3.5-9B one (five sliding, one
    full). vLLM reads layer_types after converting the config, which for the speculators format keeps them under
    transformer_layer_config."""
    from rigtune.heads import is_speculators

    if not isinstance(config, dict):
        return False
    if any(a in VLLM_V2_ONLY_DRAFTERS for a in config.get("architectures") or []):
        return True
    layer = config.get("transformer_layer_config") if is_speculators(config) else config
    types = layer.get("layer_types") if isinstance(layer, dict) else None
    return isinstance(types, list) and "full_attention" in types and any(t != "full_attention" for t in types)


def mtp(k: int) -> str:
    """The checkpoint's own MTP head proposing k tokens (vLLM's num_speculative_tokens, SGLang's steps)."""
    return f"mtp:{k}"


def own_block_drafter(k: int) -> str:
    """The checkpoint's own DSpark drafter proposing k tokens (Inkling 2's: introspect.own_block_drafter)."""
    return f"dspark:{k}"


def own_drafter(spec: Optional[str]) -> bool:
    """Whether a spec uses the checkpoint's own block drafter: a block drafter named without a repository."""
    if not spec:
        return False
    kind, model_id, _ = parse(spec)
    return kind in BLOCK_DRAFTERS and model_id is None


def parse(spec: str) -> Tuple[str, Optional[str], int]:
    """(kind, None, k) for the lookup proposers ("ngram", "ngram_gpu", "suffix"), the checkpoint's own head ("mtp")
    and its own block drafter ("dspark:7"), or (kind, model_id, k) for "draft", "eagle3" and a drafter's repository."""
    kind, _, rest = spec.partition(":")
    if kind in LOOKUP:
        return kind, None, int(rest or (SUFFIX_TOKENS if kind == "suffix" else NGRAM_TOKENS))
    if kind in SELF_HEAD:
        k = int(rest or 0)
        if k < 1:
            raise ValueError(f"bad speculative spec {spec!r}: an MTP head proposes at least one token")
        return kind, None, k
    if kind == "dspark" and rest.isdigit():  # no repository: the checkpoint's own (own_block_drafter)
        if int(rest) < 1:
            raise ValueError(f"bad speculative spec {spec!r}: a drafter proposes at least one token")
        return kind, None, int(rest)
    if kind in SECOND_MODEL:
        model_id, _, k = rest.rpartition(":")
        if not model_id:
            raise ValueError(f"bad speculative spec {spec!r}")
        return kind, model_id, int(k)
    raise ValueError(f"unknown speculative method in {spec!r}")


def vllm_config(spec: str, revision: Optional[str] = None, head_quant: Optional[str] = None,
                sample_method: Optional[str] = None) -> Dict[str, object]:
    """The value for vLLM's --speculative-config. `revision` is the target's for the checkpoint's own head or drafter,
    the drafter's for a second model. `sample_method` is a block drafter's draft_sample_method (vLLM's default is
    "greedy", config/speculative.py line 701 on main), given only where the installed vLLM has the field."""
    kind, model_id, k = parse(spec)
    if head_quant is not None and (kind != "mtp" or head_quant != "fp8"):
        raise ValueError("fp8 output-head quantization is offered only for the checkpoint's own MTP head")
    if kind in ("ngram", "ngram_gpu"):  # one code path in vLLM, with the same lookup settings
        return {"method": kind, "num_speculative_tokens": k, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    if kind == "suffix":
        return {"method": "suffix", "num_speculative_tokens": k}
    if kind == "mtp":
        # No "model": the draft is the target's own checkpoint. Its revision is SpeculativeConfig.revision, not the
        # target's (the draft ModelConfig takes revision=self.revision: v0.29.0 config/speculative.py line 1252,
        # v0.33.0 line 1420), so a target pinned to a revision passes it here too, or the head would come from the
        # default branch.
        out: Dict[str, object] = {"method": "mtp", "num_speculative_tokens": k}
        if head_quant:
            out["draft_lm_head_quantization"] = head_quant
        if revision:
            out["revision"] = revision
        return out
    if kind == "eagle3":  # vLLM 0.29 reads the head's architecture from its config
        return {"method": "eagle3", "model": model_id, "num_speculative_tokens": k}
    if kind in BLOCK_DRAFTERS:  # the method is named, never left to vLLM's guess from the repository's name
        out = {"method": kind, "num_speculative_tokens": k}
        if model_id is not None:  # else the checkpoint's own: vLLM takes the target as the draft's model
            out["model"] = model_id
        if revision:  # the commit whose config was checked (rigtune.heads), not the default branch
            out["revision"] = revision
        if sample_method:
            out["draft_sample_method"] = sample_method
        return out
    out = {"method": "mtp" if kind == "assist" else "draft_model", "model": model_id, "num_speculative_tokens": k}
    if revision:
        out["revision"] = revision
    return out


def sglang_config(spec: str, model_path: Optional[str] = None, revision: Optional[str] = None, *,
                   draft_revision: Optional[str] = None, draft_quantization: Optional[str] = None,
                   draft_path: Optional[str] = None) -> List[str]:
    """SGLang's launch flags for a spec (flags as in SGLang 0.5.19's server_args.py). `model_path` and `revision`
    are the target's, used only by an MTP head.

    An EAGLE3 head and a draft model (SGLang's STANDALONE) decode a chain: top-k 1, k steps, k + 1 draft tokens
    (with top-k 1 SGLang forces the draft tokens to steps + 1 anyway). SGLang's own default for Llama-class
    models is a tree (5 steps, top-k 4, 8 tokens), but 0.5.19 always runs speculation on its V2 (overlap)
    worker, which its docs support with top-k 1 only, and a chain keeps the comparison with vLLM's k fair.

    An MTP head is NEXTN (resolved to EAGLE, arg_groups/speculative_hook.py _resolve_speculative_algorithm_alias)
    with the cookbook's chain. The draft path is the target's own, passed explicitly although SGLang would fall
    back to it (configs/model_config.py ModelConfig.from_server_args: model_path or the target's), because only a
    passed path makes SGLang size its KV pool for the head's layer (pool_configurator.py DefaultPoolConfigurator
    scales the cell by 1 + draft layers / full-attention layers): model_executor/model_runner_components/
    spec_aux_hidden_state.py _resolve_eagle_aux_hidden_state reads the draft's layer count only when
    speculative_draft_model_path is set at v0.5.19 (lines 80-101), and at v0.5.23 counts a path-less draft from the
    target's config (lines 168-178), where a Qwen3.5 checkpoint has no num_nextn_predict_layers (it declares
    mtp_num_hidden_layers, and SGLang's configs/qwen3_5.py defines no such field). A passed path also sets the
    draft's revision to "main" unless one is given (speculative_hook.py handle_speculative_decoding, lines 81-91 /
    121-131), so the target's revision goes with it."""
    kind, model_id, k = parse(spec)
    if kind == "ngram":
        return ["--speculative-algorithm", "NGRAM", "--speculative-num-draft-tokens", str(k)]
    if kind == "mtp":
        if not model_path:
            raise ValueError("SGLang's MTP draft needs the target's model path")
        out = ["--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", model_path]
        if revision:
            out += ["--speculative-draft-model-revision", revision]
        return out + ["--speculative-num-steps", str(k), "--speculative-eagle-topk", "1",
                      "--speculative-num-draft-tokens", str(k + 1)]
    if kind in BLOCK_DRAFTERS:
        # SGLang 0.5.23 forces one step and top-k 1 for both (arg_groups/speculative_hook.py lines 334-366, 757-789).
        # A DFlash block holds the last accepted token and k drafts, so its draft tokens are k + 1, the checkpoint's
        # block size (lines 368-425; dflash_worker_v2.py 2534-2535); a DSpark block size is the tokens it drafts,
        # gamma, and SGLang adds one (fields/spec.py 78-81; speculative_hook.py 824-839, which refuses a draft-token
        # count other than gamma + 1). Both stated, though SGLang would read them from the checkpoint, so the
        # launch says what was planned. The same k as vLLM's: tokens drafted a step (drafter_tokens).
        if model_id is None and not draft_path:
            # The checkpoint's own (Inkling 2): SGLang serves it only in an unmerged PR (#51926), so it is not offered.
            raise ValueError(f"SGLang is not offered the checkpoint's own {kind} drafter ({spec!r})")
        out = ["--speculative-algorithm", kind.upper(), "--speculative-draft-model-path", draft_path or str(model_id)]
        out += (["--speculative-num-draft-tokens", str(k + 1)] if kind == "dflash"
                else ["--speculative-dspark-block-size", str(k)])
        if draft_revision:
            out += ["--speculative-draft-model-revision", draft_revision]
        if draft_quantization:
            out += ["--speculative-draft-model-quantization", draft_quantization]
        return out
    if kind in SECOND_MODEL:
        out = ["--speculative-algorithm", "NEXTN" if kind == "assist" else "EAGLE3" if kind == "eagle3" else "STANDALONE",
               "--speculative-draft-model-path", draft_path or str(model_id), "--speculative-num-steps", str(k),
               "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", str(k + 1)]
        if draft_revision:
            out += ["--speculative-draft-model-revision", draft_revision]
        if draft_quantization:
            out += ["--speculative-draft-model-quantization", draft_quantization]
        return out
    raise ValueError(f"SGLang has no {kind} speculative decoding")


def own_head(spec: Optional[str]) -> bool:
    """Whether a spec uses the checkpoint's own MTP head."""
    return bool(spec) and str(spec).partition(":")[0] in SELF_HEAD


def draft_bytes(spec: Optional[str], model: PreparedModel) -> Optional[int]:
    """Bytes of the second set of weights a spec loads: 0 for none (no spec, or a lookup proposer), None when it
    loads a second model whose size is not known. Used for SGLang's room outside its reservation (spec_fraction).

    The checkpoint's own MTP head counts as one, and more than its mtp.* tensors: SGLang's Qwen3_5ForCausalLMMTP
    builds its own embed_tokens and (untied) ParallelLMHead, full vocabulary, and swaps in the target's only in
    EAGLEWorkerV2.alloc_memory_pool -> init_lm_head, after the scheduler has sized the target's pools
    (managers/scheduler.py init_model_worker: target, draft, then init_memory_pools; lines 1047-1056 at v0.5.19,
    1128-1137 at v0.5.23). So both copies are resident when the pools are sized: 2 x 248,320 x 4,096 x 2 B =
    4.07 GB for Qwen3.5-9B in bf16, beside its 0.49 GB head. A head that brings its own embedding and output head
    (GLM's, DeepSeek's) has them in its own bytes, and shared bytes of 0. None when either size is unknown."""
    if not spec:
        return 0
    kind, model_id, _ = parse(spec)
    if kind in SELF_HEAD:
        head, shared = model.spec_bytes.get(MTP_KEY), model.spec_bytes.get(MTP_SHARED_KEY)
        return None if not head or shared is None else head + shared
    if kind not in SECOND_MODEL:
        return 0
    return model.spec_bytes.get(model_id or "")
