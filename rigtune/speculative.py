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
    rather than from tokens alone. Heads are published per target model, so EAGLE3_HEADS lists them by
    target, and a head is used only when its config.json shows it is one and fits the target.
  * the checkpoint's own MTP (multi-token-prediction) head: Qwen3.5-family checkpoints ship one beside the model,
    trained with it, so no second repository is needed (rigtune.introspect finds it). Offered only under
    --speculative all, and only where the config declares the head and the weights hold it.

Spec strings kept in Config.spec_decode: "ngram:<k>", "ngram_gpu:<k>", "suffix:<k>", "mtp:<k>", "draft:<hf_id>:<k>"
or "eagle3:<hf_id>:<k>".
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# (target pattern, EAGLE3 heads to try in order). Each one's config.json was read on the Hub; a head is still used
# only when its config passes verified_eagle3_head, so a repository that changes or disappears costs a lookup, not
# a trial.
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

# The checkpoint's own MTP head ("mtp:<k>"), for the architectures both engines rewrite to an MTP draft:
#   vLLM (v0.29.0 and v0.33.0): config/speculative.py SpeculativeConfig.hf_config_override turns model types
#     qwen3_5, qwen3_5_moe, qwen3_5_text and qwen3_5_moe_text (lines 882-899 / 987-1007) into Qwen3_5MTP or
#     Qwen3_5MoeMTP (model_executor/models/qwen3_5_mtp.py); with no "model" the draft is the target's own
#     checkpoint and quantization (SpeculativeConfig.__post_init__, lines 1096-1111 / 1252-1267).
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
SECOND_MODEL = ("draft", "eagle3", "assist")  # proposers whose second set of weights the spec names
SELF_HEAD = ("mtp",)  # proposers whose weights ship in the target's own checkpoint


def draft_for(hf_id: str) -> Optional[str]:
    """A smaller model from the same family that can draft for `hf_id`, or None."""
    for pattern, draft in DRAFTS:
        m = re.match(pattern, hf_id)
        if m:
            return m.expand(draft)
    return None


def eagle3_heads_for(hf_id: str) -> Tuple[str, ...]:
    """The EAGLE3 heads EAGLE3_HEADS lists for `hf_id`, in the order to try them."""
    for pattern, heads in EAGLE3_HEADS:
        m = re.match(pattern, hf_id)
        if m:
            return tuple(m.expand(h) for h in heads)
    return ()


def verified_eagle3_head(cfg: Optional[Dict[str, Any]], target: ArchInfo) -> bool:
    """Whether a head's config.json shows an EAGLE3 head that can draft for `target`.

    A repository's name is not evidence, as with pre-quantized checkpoints: yuhuili's EAGLE3 head for Llama 3.1 8B
    declares plain LlamaForCausalLM, and the speculators format names classes of its own. The engines choose the
    head's code by its architecture (vLLM 0.29 maps LlamaForCausalLMEagle3 to its Eagle3 model), the head reads
    the target's hidden states, and it proposes tokens of the target's vocabulary, so all three must match."""
    if not cfg:
        return False
    archs = cfg.get("architectures") or []
    if not archs or not str(archs[0]).endswith("Eagle3"):
        return False
    return cfg.get("vocab_size") == target.vocab_size and cfg.get("hidden_size") == target.hidden_size


class HubLookupError(RuntimeError):
    """The Hub could not be asked: offline, rate-limited, or a gated repository without a token."""


def _fetch_config(repo: str, token: Optional[str] = None) -> Dict[str, Any]:
    """A repository's config.json; raises when the Hub cannot be reached, unlike quantized._default_fetch_config,
    so an offline calibration can say what it skipped."""
    import json

    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json", token=token), "r", encoding="utf-8") as fh:
        return json.load(fh)


def repo_bytes(repo: str, token: Optional[str] = None) -> Optional[int]:
    """Total size of a repository's safetensors files; None when it has none. Raises when the Hub cannot say."""
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(repo, files_metadata=True)
    sizes = [getattr(s, "size", None) for s in (getattr(info, "siblings", None) or [])
             if getattr(s, "rfilename", "").endswith(".safetensors")]
    sizes = [s for s in sizes if s]
    return int(sum(sizes)) if sizes else None


def find_eagle3_head(
    spec: ModelSpec,
    target: ArchInfo,
    fetch_config: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    sizes: Optional[Callable[[str], Optional[int]]] = None,
    token: Optional[str] = None,
) -> Optional[Tuple[str, Optional[int]]]:
    """(repo, size in bytes or None) of the first listed EAGLE3 head whose config passes the check, or None when
    no listed head fits. Raises HubLookupError when no head was found and a listed one could not be read, so the
    caller can tell "none fits" from "could not ask".

    Reads only the listed repositories' config.json and file sizes; it never searches the Hub, and makes no
    request at all for a model EAGLE3_HEADS does not list."""
    heads = eagle3_heads_for(spec.hf_id)
    if not heads:
        return None
    fetch = fetch_config or (lambda rid: _fetch_config(rid, token))
    size_of = sizes or (lambda rid: repo_bytes(rid, token))
    errors = []
    for repo in heads:
        try:
            cfg = fetch(repo)
        except Exception as exc:
            errors.append(f"{repo}: {exc}")
            continue
        if not verified_eagle3_head(cfg, target):
            logger.info("not using EAGLE3 head %s: its config does not fit %s", repo, spec.hf_id)
            continue
        try:
            size = size_of(repo)
        except Exception as exc:
            logger.warning("could not read the size of EAGLE3 head %s: %s", repo, exc)
            size = None
        return repo, size
    if errors:
        raise HubLookupError("could not read " + "; ".join(errors))
    return None


def record_mtp_head(prepared: PreparedModel, engine: str,
                    inspect: Optional[Callable[[ModelSpec], Any]] = None) -> List[str]:
    """Look for the checkpoint's own MTP head (rigtune.introspect: its config's layer count and its mtp.* tensors,
    from the safetensors headers only) and record it in `prepared`: spec_repos["mtp"] is the model itself and
    spec_bytes["mtp"] the head's size and spec_bytes["mtp_shared"] that of the target's embedding and output head,
    when the listing gave them. Returns a note for the profile when no head was
    recorded although the config declares one (weights without it, or a Hub that could not be asked)."""
    from rigtune import introspect

    look = inspect or (lambda spec: introspect.inspect_model(spec, features=("mtp",)))
    try:
        facts = look(prepared.spec)
    except Exception as exc:  # inspect_model never raises; an injected one might
        return [f"{engine}: no MTP head tried ({exc})"]
    if facts.mtp is None:
        return [f"{engine}: no MTP head tried: " + "; ".join(facts.notes)] if facts.notes else []
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


def mtp(k: int) -> str:
    """The checkpoint's own MTP head proposing k tokens (vLLM's num_speculative_tokens, SGLang's steps)."""
    return f"mtp:{k}"


def parse(spec: str) -> Tuple[str, Optional[str], int]:
    """(kind, None, k) for the lookup proposers ("ngram", "ngram_gpu", "suffix") and the checkpoint's own head
    ("mtp"), or (kind, model_id, k) for "draft" and "eagle3"."""
    kind, _, rest = spec.partition(":")
    if kind in LOOKUP:
        return kind, None, int(rest or (SUFFIX_TOKENS if kind == "suffix" else NGRAM_TOKENS))
    if kind in SELF_HEAD:
        k = int(rest or 0)
        if k < 1:
            raise ValueError(f"bad speculative spec {spec!r}: an MTP head proposes at least one token")
        return kind, None, k
    if kind in SECOND_MODEL:
        model_id, _, k = rest.rpartition(":")
        if not model_id:
            raise ValueError(f"bad speculative spec {spec!r}")
        return kind, model_id, int(k)
    raise ValueError(f"unknown speculative method in {spec!r}")


def vllm_config(spec: str, revision: Optional[str] = None, head_quant: Optional[str] = None) -> Dict[str, object]:
    """The value for vLLM's --speculative-config. `revision` is the target's, used only by an MTP head."""
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
