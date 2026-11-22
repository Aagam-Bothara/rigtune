"""Staged search over the feasible configuration set.

1. quant / precision  - one short run per feasible (backend, quant); keep the top 2.
2. memory config      - largest safe gpu_memory_utilization / offload layers / context.
3. batch / concurrency - sweep max_num_seqs (vLLM/SGLang) or n_parallel/n_batch (llama.cpp): the
                        decode knob.
3b. prefill           - on the leading config, sweep the prefill knob: the chunked-prefill token
                        budget (vLLM, SGLang) or the micro-batch (llama.cpp). Default on.
3c. kv                - quantized KV caches, including the batch step a smaller cache admits.
3d. prefix            - prefix-cache settings, when the workload's prompts share a prefix.
3e. spec              - speculative decoding (n-gram lookup, a small draft model).
3e+. tier             - a host-RAM or disk KV-cache tier, with --kv-tier auto and a session file, only when
                        the leader's working set outgrows the GPU pool.
3f. combine           - combinations of the changes from 3b-3e that were promising on their own.
3g. explore           - random configurations of the leading engine and precision, for settings that
                        only pay together.
4. power (optional)   - on the leading config, sweep GPU power caps and/or locked SM clocks
                        through NVML without relaunching, and keep the setting that saves energy.

The winner is the objective's constrained argmax over *every* successful trial, not just stage 3.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from rigtune import kvtier
from rigtune.backends.base import BaseBackend, free_port
from rigtune.calibrate.measure import MIN_OPEN_LOOP_RATE, run_trial, spec_counters
from rigtune.calibrate.objectives import Constraints, pick, rank, resolve_goal, strict_refusal
from rigtune.calibrate.rate import compose
from rigtune.calibrate.workload import Workload
from rigtune.memlog import (
    acceptance_from_counters,
    device_used_mb,
    merge,
    parse_acceptance,
    STARTUP_LOGS,
    parse_log,
    parse_startup,
)
from rigtune.memory import estimate as memory_estimate
from rigtune.models import (
    Config,
    HardwareDescriptor,
    MemoryObservation,
    PreparedModel,
    SpecAcceptance,
    StartupCost,
    TrialMetrics,
    TrialResult,
)
from rigtune.power import PowerSetting, setting_of, with_power

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, Config, Optional[TrialResult]], None]


class TrialRunner(Protocol):
    def run(self, cfg: Config, stage: str) -> TrialResult: ...


STAGE_NAMES = {"kv": "a quantized KV cache", "prefix": "prefix-cache settings", "spec": "speculative decoding"}
_ANY = object()  # "any speculation setting", for _leader and the stages that can hold one fixed


def mentions_oom(log: str) -> bool:
    """Whether an engine log shows a CUDA out-of-memory failure."""
    text = log.lower()
    return "out of memory" in text or "outofmemoryerror" in text


# How an engine says at start-up that memory was short without the words "out of memory". vLLM: its free-memory check
# ("... is less than desired GPU memory utilization", v1/worker/utils.py line 530 at v0.29.0, 766 at v0.33.0) and its
# KV-cache checks ("No available memory for the cache blocks. Try increasing `gpu_memory_utilization` ...", "... is
# larger than the available KV cache memory ...", v1/core/kv_cache_utils.py 860-886 at v0.29.0,
# v1/core/kv_cache_planning.py 58-82 at v0.33.0). SGLang's pool sizing, model_executor/pool_configurator.py: "Not
# enough memory. Please try to increase --mem-fraction-static." (line 83 at v0.5.19, 96 at v0.5.23), the SWA and
# DSV4 caps ending "... or increase --mem-fraction-static." (1025-1031 and 1679-1684 at v0.5.23), and "SWA pool (N
# tokens) cannot hold even one request" (612-616 at v0.5.19, 211-215 at v0.5.23). Matched lower-case, backticks
# dropped. Not "available kv cache memory" alone: vLLM logs "Available KV cache memory: X GiB" on every start.
MEMORY_SHORT = ("less than desired gpu memory utilization", "no available memory for the cache blocks",
                "than the available kv cache memory", "try increasing gpu_memory_utilization", "not enough memory",
                "increase --mem-fraction-static", "cannot hold even one request")
MEMORY_SHORT_CAUSE = "memory short at start-up"  # an MTP trial's error, when the log said so above its tail


def _normal(text: str) -> str:
    return " ".join(text.lower().replace("`", "").split())


def memory_short_line(log: str) -> Optional[str]:
    """The first log line that says memory was short at start-up (MEMORY_SHORT), or None."""
    for line in log.splitlines():
        if any(phrase in _normal(line) for phrase in MEMORY_SHORT):
            return line.strip()[:200]
    return None


def memory_short(res: "TrialResult") -> bool:
    """Whether a failed start was the memory's, not the configuration's: out of memory, or a shortage in other words
    (MEMORY_SHORT), in its error or found in its log above the tail (MEMORY_SHORT_CAUSE)."""
    if failure_kind(res) == "oom_startup" or (res.error or "").startswith(MEMORY_SHORT_CAUSE):
        return True
    return any(phrase in _normal(res.error or "") for phrase in MEMORY_SHORT)


# How a start-up failure's error begins when the engine log shows its cause (SubprocessTrialRunner.run).
MM_PROFILING_FAILED = "multimodal profiling failed at start-up"
LOAD_FAILED = "model failed to load at start-up"
SUFFIX_FAILED = "suffix decoding failed at start-up"  # Arctic Inference not usable; suffix is not tried again
ARCTIC_PLUGIN_FAILED = "arctic-inference's plugin stopped vLLM at start-up"  # any vLLM launch
SUFFIX_TRIAL_ERROR = "a suffix trial failed to start"  # anything else on a suffix trial: quoted, not blamed
QUANT_REFUSED = "the engine refused the quantization on this GPU at start-up"  # every config of the precision fails
STARTUP_CAUSES = (MM_PROFILING_FAILED, LOAD_FAILED, SUFFIX_FAILED, ARCTIC_PLUGIN_FAILED, SUFFIX_TRIAL_ERROR,
                  QUANT_REFUSED)
# TrialResult.failure_kind: why a trial failed, as a word a program can match on rather than an error's prefix.
#   oom_startup        out of memory at start-up (the error's "out of memory at start-up; ")
#   mm_profiling, load_failure, suffix_unusable, arctic_plugin
#                      a start-up failure whose cause the log showed (STARTUP_CAUSES, by _CAUSE_KINDS)
#   startup_other      any other start-up failure, a suffix trial's quoted error among them
#   power_unavailable  the engine started but the trial's power setting could not be applied
#   request_failures   it started and ran, but more requests failed than TrialMetrics.FAILURE_TOLERANCE allows
#   crashed            the trial itself raised (an exception in rigtune, a full disk for a tier)
#   skipped_budget     not launched: --budget was spent
#   skipped_blocked    not launched: its speculation method failed to start earlier in the calibration
#   kernel_refused     the engine exited at start-up naming a kernel setting of the config (Config.kernels)
#   quant_refused      the engine's own check refused the checkpoint's quantization on this GPU (QUANT_REFUSED)
FAILURE_KINDS = ("oom_startup", "mm_profiling", "load_failure", "suffix_unusable", "arctic_plugin", "startup_other",
                 "power_unavailable", "request_failures", "crashed", "skipped_budget", "skipped_blocked",
                 "draft_refused", "draft_unavailable", "kernel_refused", "quant_refused")
_CAUSE_KINDS = {MM_PROFILING_FAILED: "mm_profiling", LOAD_FAILED: "load_failure", SUFFIX_FAILED: "suffix_unusable",
                ARCTIC_PLUGIN_FAILED: "arctic_plugin", SUFFIX_TRIAL_ERROR: "startup_other",
                QUANT_REFUSED: "quant_refused"}
BUDGET_SKIPPED = "skipped: calibration budget reached"
KERNEL_REFUSED = "the engine refused a kernel setting"

# How an engine says it will not start with a kernel setting. Whole phrases, read in vLLM v0.29.0 / v0.33.0:
#   "Selected backend X is not valid for this configuration. Reason: ..."      platforms/cuda.py:423-429 / 439-445
#   "CUDAGraphMode.X is not supported with ..." raised by the resolver         config/compilation.py:1399-1410, 1473
#                                                                              / 1400-1411, 1488
# The resolver logs the same opening as a warning when it only changes the mode; that line ends
# "; setting cudagraph_mode=NAME" and is no refusal. An argparse error counts when it names a flag this config's
# kernel settings rendered (both engines; SGLang's choices are argparse's). vLLM's free-memory check at start-up and
# an out-of-memory during graph capture name no setting and are not refusals of one.
_BACKEND_INVALID = re.compile(r"Selected backend \S+ is not valid for this configuration\. Reason: [^\n]{0,300}")
_CG_UNSUPPORTED = re.compile(r"CUDAGraphMode\.[A-Z_]+ is not supported with [^\n]{0,300}")
_ARG_ERROR = re.compile(r"error: argument (--[a-z0-9-]+)(?:/\S+)?: [^\n]{0,300}")


def kernel_startup_refusal(cfg: Config, log: str) -> Optional[str]:
    """The line in which the engine refused one of `cfg`'s kernel settings at start-up, or None. Never looked for
    on a config without kernel settings, so every other failure keeps the kind it had."""
    if cfg.kernels is None or not log:
        return None
    from rigtune import kernels
    from rigtune.bench.compare import scrub_text

    settings = kernels.parse(cfg.kernels)
    if cfg.backend.startswith("vllm"):
        # Only the flags these settings rendered: the attention backend a cache type forced is not one of them.
        flags = {flag for flag, on in (("--attention-backend", "attn" in settings),
                                       ("--optimization-level", "O" in settings),
                                       ("--compilation-config", kernels.vllm_compilation_config(settings) is not None))
                 if on}
        match = _BACKEND_INVALID.search(log) if "attn" in settings else None
        if match is None:
            match = next((m for m in _CG_UNSUPPORTED.finditer(log) if "; setting cudagraph_mode=" not in m.group(0)),
                         None)
    else:
        flags = {kernels.SGLANG_FLAGS[key] for key in settings if key in kernels.SGLANG_FLAGS}
        match = None
    if match is None:
        match = next((m for m in _ARG_ERROR.finditer(log) if m.group(1) in flags), None)
    return scrub_text(match.group(0).strip()) if match else None

_REQUESTS_FAILED = re.compile(r"\d+/\d+ requests failed")  # TrialMetrics.failure_summary
# vLLM profiles memory at start-up with dummy multimodal inputs, and a processor that rejects them stops the engine.
# That happens when the engine and transformers mismatch: vLLM 0.29.0 declares only transformers>=5.10.4, and with
# transformers 5.23.0 (whose ProcessorMixin.validate_inputs raises the first line; 5.12.1 has no such check) it failed
# google/gemma-4-12B-it on an A40 pod. Later vLLM releases pin transformers below 5.21 (0.33.0: >=5.16.1,<5.21.0).
#   ValueError: Found 2048 <|video|> tokens in the text but no videos were passed.
#   ValueError: Failed to apply Gemma4UnifiedProcessor on data={'text': '00:00 <|image><|video|>...'} with kwargs=...
_MM_PROFILING = (re.compile(r"tokens in the text but no (?:images|videos|audios) were passed"),
                 re.compile(r"Failed to apply \w*Processor\b"))
# The engine's own model code failing as it builds the model. SGLang 0.5.19 with google/gemma-4-12B-it, a known
# SGLang bug for Gemma 4 unified models fixed in 0.5.21 (pipeline.KNOWN_LOAD_BUGS):
#   AttributeError: 'Gemma4UnifiedForConditionalGeneration' object has no attribute 'lm_head_is_tied'
_LOAD_FAILURE = re.compile(r"AttributeError: .*object has no attribute")


# Arctic Inference's causes of a start-up failure, from what vLLM and arctic-inference print (speculative.py has the
# sources; the check before launch should catch each, so these are for what slips past it). Most specific first:
#   "plugin": arctic-inference's vLLM plugin, enabled by ARCTIC_INFERENCE_ENABLED=1, refused this vLLM. It runs in every
#     vLLM process, so this is read on any vLLM start-up failure, not only a suffix trial ("Arctic Inference plugin
#     requires vllm==0.10.1 but found vllm==0.29.0!" from 0.1.1, captured; "Arctic Inference is enabled
#     (ARCTIC_INFERENCE_ENABLED=1) but the installed vllm==..." from 0.3.0, read).
#   "import": the package is there but fails to import; vLLM's utils/import_utils._has_module logs "Module
#     arctic_inference was found but failed to import" before its own "Arctic Inference is required" error.
#   "build": the package imports but its native suffix tree does not, where SuffixDecodingProposer imports it: "No module
#     named 'arctic_inference.suffix_decoding._C'" for an extension built for another interpreter (as reproduced in the
#     14 Nov pod's image), or a loader error naming arctic_inference (undefined symbol, GLIBCXX, a missing library).
#   "missing": not installed: SpeculativeConfig._validate_suffix_decoding's ImportError ("Arctic Inference is required
#     for suffix decoding. Install via `pip install arctic-inference==0.1.1`.", captured).
ARCTIC_CAUSES = (
    ("plugin", "arctic-inference's vLLM plugin refused this vLLM",
     re.compile(r"Arctic Inference (?:plugin requires vllm==|is enabled \(ARCTIC_INFERENCE_ENABLED=1\) but)")),
    ("import", "arctic-inference fails to import where vLLM runs",
     re.compile(r"Module arctic_inference was found but failed to import")),
    ("build", "arctic-inference's suffix tree did not load (a broken build)",
     re.compile(r"No module named '?arctic_inference\.suffix_decoding\._C|"
                r"(?:undefined symbol|cannot open shared object|GLIBCXX).*arctic_inference|"
                r"arctic_inference.*(?:undefined symbol|cannot open shared object|GLIBCXX)")),
    ("missing", "arctic-inference is not installed where vLLM runs",
     re.compile(r"Arctic Inference is required for suffix decoding|No module named '?arctic_inference'")),
)
# Any other error on a suffix trial is not blamed on suffix decoding: the trial's error quotes the log's last exception
# line before vLLM's engine-core wrapper ("Engine core initialization failed. See root cause above."), which names no
# cause itself; earlier exceptions are often benign (vLLM logs optional imports that fail, with tracebacks). A pydantic
# ValidationError's reason is on the lines after it, so the first of those is quoted too.
_EXCEPTION = re.compile(r"\b[A-Z]\w*(?:Error|Exception): ")
_ENGINE_WRAPPER = re.compile(r"Engine core initialization failed")
_PROCESS = re.compile(r"^\s*\([^)]*pid=\d+\)")  # vLLM's "(APIServer pid=1) " before each line of a process


def _first_line(log: str, patterns) -> Optional[str]:
    for pattern in patterns:
        for line in log.splitlines():
            if pattern.search(line):
                line = line.strip()
                return line if len(line) <= 160 else line[:157] + "..."
    return None


def mm_profiling_failure(log: str) -> Optional[str]:
    """The log line that shows the engine's multimodal profiling failed at start-up, or None."""
    return _first_line(log, _MM_PROFILING)


def load_failure(log: str) -> Optional[str]:
    """The log line that shows the engine's model code failed while building the model, or None."""
    return _first_line(log, (_LOAD_FAILURE,))


# How both engines refuse, at start-up, a quantization this GPU cannot run. Whole phrases, vLLM v0.33.0 / SGLang
# v0.5.23, after each resolved the checkpoint's method (an NVFP4 ModelOpt checkpoint's "modelopt" to modelopt_fp4):
#   "The quantization method X is not supported for the current GPU. Minimum capability: N. Current capability: M."
#       vLLM config/vllm.py VllmConfig._get_quantization_config (930-936), SGLang model_loader/loader.py (259-264),
#       and SGLang's "The quantization method moe_wna16 + awq is not supported for the current GPU. ..."
#   "Quantization scheme is not supported for the current GPU. Min capability: N. Current capability: M."
#       compressed-tensors, checked per layer's scheme (vLLM also says "Required capability"): vLLM
#       compressed_tensors.py (401-414); SGLang (421-425) passes the parts as separate arguments, so its log shows
#       them quoted and comma-separated
#   "<dtype> is not supported for quantization method X. Supported dtypes: [...]"     both, right after the first
_QUANT_GPU = re.compile(r"(?:quantization method (\S+(?: \+ \S+)?)|quantization scheme) is not supported for\W+the "
                        r"current GPU\W+(?:minimum|min|required) capability: (\d+)\W+current capability: (\d+)",
                        re.IGNORECASE)
_QUANT_DTYPE = re.compile(r"\S+ is not supported for quantization method \S+\. Supported dtypes: \[[^\]\n]{0,200}\]")


def quant_refusal(log: str) -> Optional[str]:
    """What the engine said when it would not run the checkpoint's quantization here, or None: "modelopt_fp4 needs
    compute capability 10.0, this GPU has 8.9", or its dtype line."""
    m = _QUANT_GPU.search(log)
    if m:
        need, have = int(m.group(2)), int(m.group(3))
        return (f"{m.group(1) or 'its quantization scheme'} needs compute capability {need // 10}.{need % 10}, "
                f"this GPU has {have // 10}.{have % 10}")
    m = _QUANT_DTYPE.search(log)
    return _short(m.group(0)) if m else None


def _short(line: str, limit: int = 160) -> str:
    line = line.strip()
    return line if len(line) <= limit else line[:limit - 3] + "..."


def arctic_failure(log: str, causes: Tuple[str, ...] = ("plugin", "import", "build", "missing")) -> Optional[str]:
    """"<what>: <log line>" for the first of ARCTIC_CAUSES (among `causes`) the engine log shows, or None. For "import",
    the exception that follows vLLM's warning is quoted, since the warning names no cause."""
    lines = log.splitlines()
    for key, what, pattern in ARCTIC_CAUSES:
        if key not in causes:
            continue
        for i, line in enumerate(lines):
            if pattern.search(line):
                if key == "import":
                    after = next((x for x in lines[i + 1:i + 60] if _EXCEPTION.search(x)), None)
                    line = after or line
                return f"{what}: {_short(line)}"
    return None


def arctic_cause(line: str) -> Optional[str]:
    """The ARCTIC_CAUSES key of a line arctic_failure returned."""
    return next((key for key, what, _ in ARCTIC_CAUSES if line.startswith(what + ": ")), None)


def log_error(log: str) -> Optional[str]:
    """The log's last exception line before vLLM's engine-core wrapper (the last in the log without one), with a
    ValidationError's reason line after it; None when it has none."""
    lines = log.splitlines()
    end = max((i for i, x in enumerate(lines) if _ENGINE_WRAPPER.search(x)), default=len(lines))
    found = [i for i in range(end) if _EXCEPTION.search(lines[i]) and not _ENGINE_WRAPPER.search(lines[i])]
    if not found:
        return None
    i = found[-1]
    line = lines[i].strip()
    if "ValidationError" in line:
        reason = next((_PROCESS.sub("", x).strip() for x in lines[i + 1:i + 4] if "error" in x.lower()), None)
        if reason:
            return _short(f"{line} {reason}", 240)
    return _short(line)


def _acceptance(backend: str, window: Tuple[Optional[Tuple[float, ...]], int], port: int, hooks,
                proc) -> Optional[SpecAcceptance]:
    """Speculation's acceptance over one trial's workload: from the change in vLLM's counters where both readings
    exist, else from the engine's log lines written since the workload began; None when neither says. The per-position
    curve comes from the counters, else from the log. Only whole log lines are read: a line still being written when
    the workload ended is left out. Never fails the trial: a reading that goes wrong costs the figure."""
    try:
        before, offset = window
        found = None
        text = None
        if hasattr(proc, "log_since"):
            text = proc.log_since(offset)
            text = text[:text.rfind("\n") + 1] if text else text
        if before is not None:
            after = spec_counters(f"http://127.0.0.1:{port}", hooks)
            if after is not None:
                deltas = [a - b for a, b in zip(after, before)]
                # Positions past the first three exist on both readings only when the per-position counter does,
                # with the same number of positions (zip stops at the shorter).
                per_position = deltas[3:] if len(after) == len(before) and len(after) > 3 else None
                found = acceptance_from_counters(*deltas[:3], per_position=per_position)
        if found is None and text is not None:
            found = parse_acceptance(backend, text)
        elif found is not None and "per_position" not in found and text:
            logged = parse_acceptance(backend, text) or {}
            # The counters had no positions (an older vLLM): the log's curve, when it describes the same steps (its
            # lines cover vLLM's logging intervals, which may run past the workload's either end).
            if "per_position" in logged and abs(1 + sum(logged["per_position"]) - found["mean_length"]) <= 0.02:
                found["per_position"], found["full_drafts"] = logged["per_position"], logged.get("full_drafts")
        return SpecAcceptance(**found) if found else None
    except Exception as exc:
        logger.warning("speculative acceptance not read: %s", type(exc).__name__)
        return None


def startup_cause(error: Optional[str]) -> Tuple[Optional[str], str]:
    """(the STARTUP_CAUSES kind, the log line) from a start-up failure's error; (None, "") otherwise."""
    for kind in STARTUP_CAUSES:
        if (error or "").startswith(kind + " ("):
            head = (error or "").split("\n", 1)[0]
            return kind, head[len(kind) + 2:].rsplit("); failed to start", 1)[0]
    return None, ""


def failure_kind(res: TrialResult) -> Optional[str]:
    """Why `res` failed (FAILURE_KINDS), or None when it succeeded: its failure_kind where the runner set one, else,
    for a result recorded without one (a profile written before it existed, another runner), what its error says."""
    if res.failure_kind is not None:
        return res.failure_kind
    if res.ok:
        return None
    error = res.error or ""
    if error.startswith("out of memory at start-up"):
        return "oom_startup"
    if error.startswith(KERNEL_REFUSED):
        return "kernel_refused"
    cause = startup_cause(error)[0]
    if cause is not None:
        return _CAUSE_KINDS[cause]
    if error == BUDGET_SKIPPED:
        return "skipped_budget"
    if error.startswith("skipped: "):
        return "skipped_blocked"
    if not res.launched:
        return "startup_other" if "failed to start" in error else "crashed"
    if not error or _REQUESTS_FAILED.match(error):
        return "request_failures"
    return "power_unavailable" if error.startswith("power ") else "crashed"


def startup_finders(cfg: Config, log: str) -> List[Tuple[str, Callable[[str], Optional[str]]]]:
    """(kind, finder) pairs tried in order on a start-up failure that was not out of memory. An enabled arctic-inference
    plugin stops any vLLM launch; on a suffix trial Arctic Inference's other causes come first, and anything else is
    quoted last (SUFFIX_TRIAL_ERROR) unless the log looks like memory, which the memory report must keep counting."""
    finders: List[Tuple[str, Callable[[str], Optional[str]]]] = []
    if cfg.backend.startswith("vllm"):
        finders.append((ARCTIC_PLUGIN_FAILED, lambda text: arctic_failure(text, ("plugin",))))
    suffix = (cfg.spec_decode or "").startswith("suffix:")
    if suffix:
        finders.append((SUFFIX_FAILED, lambda text: arctic_failure(text, ("import", "build", "missing"))))
    finders += [(QUANT_REFUSED, quant_refusal), (MM_PROFILING_FAILED, mm_profiling_failure),
                (LOAD_FAILED, load_failure)]
    lowered = log.lower()
    if suffix and "failed to allocate" not in lowered and "no available memory" not in lowered:
        finders.append((SUFFIX_TRIAL_ERROR, lambda text: (lambda e: f"error in the log: {e}" if e else None)(
            log_error(text))))
    return finders


# Download and access errors as huggingface_hub and transformers word them, which is what both engines print when a
# repository cannot be fetched at start-up. Bare numbers are not enough: a start-up log is full of them (pids,
# "Tried to allocate 404.00 MiB"), and vLLM prints the draft repository in every launch log.
_ACCESS_ERROR = re.compile(
    r"\b(?:401|403|404) (?:client error|unauthorized|forbidden|not found)\b"
    r"|\bhttp(?: error| status)?:? (?:401|403|404)\b"
    r"|\b(?:gatedrepoerror|repositorynotfounderror)\b"
    r"|\bcannot access gated repo\b|\btrying to access a gated repo\b|\brepository not found\b"
    r"|\bcannot access content at\b|\bis not a valid model identifier\b|\baccess to model \S+ is restricted\b")


def _names_repo(text: str, repo: str) -> bool:
    """The repository id as a whole name: not inside a longer id, a file name or a number."""
    return re.search(r"(?<![\w.-])" + re.escape(repo) + r"(?![\w-]|\.\w)", text) is not None


def _draft_access_error(log: str, repo: str) -> bool:
    """An access error that names the second repository: on its own line, or (when that line names no repository,
    as in transformers' two-line gated message) on one of the next two."""
    lines = log.lower().splitlines()
    repo = repo.lower()
    for i, line in enumerate(lines):
        if not _ACCESS_ERROR.search(line):
            continue
        if _names_repo(line, repo):
            return True
        if "huggingface.co/" not in line and any(_names_repo(after, repo) for after in lines[i + 1:i + 3]):
            return True
    return False


def draft_startup_failure(cfg: Config, log: str) -> Optional[str]:
    """A second model's own start-up failure. The engine's vocabulary refusal is named even beside the incidental
    out-of-memory text of a dying worker; a download failure only when the log shows no out-of-memory text."""
    from rigtune.speculative import parse
    if not cfg.spec_decode:
        return None
    kind, repo, _ = parse(cfg.spec_decode)
    if kind not in ("draft", "assist"):
        return None
    text = " ".join(log.lower().split())
    if ("should have the same vocabulary size" in text or
            "requires the draft model to share the same vocabulary" in text):
        return "draft_refused"
    if "is less than desired gpu memory utilization" in text:
        return "startup_other"
    # Attribute download failures to the second repo, not any target access failure.
    if repo and not mentions_oom(log) and _draft_access_error(log, repo):
        return "draft_unavailable"
    return None


def _point_workload(workload: Workload, i: int, points: int) -> Workload:
    """The workload for the i-th of `points` power points of a sweep on one running engine.

    Every point shares the engine, so a session file replayed at each point would find the earlier points'
    conversations still cached. Each point takes sessions past those of every level and re-measure before it, and
    when the file cannot give every point that many, each point draws on fewer sessions rather than wrapping
    around to ones already replayed. Other workloads keep their prompts, as they always have."""
    if not workload.sessions:
        return workload
    from rigtune.calibrate import datasets

    slices = 2 * len(workload.concurrencies)  # each level and its close-call re-measure
    available = len(datasets.session_pool(workload.source[len(datasets.FILE_PREFIX):], workload.force_sessions))
    n = min(workload.n_prompts, available // (points * slices))
    if n < 1:
        if i == 0:
            logger.warning("%d sessions cannot give %d power points sessions of their own; later points replay "
                           "sessions an earlier point cached", available, points)
        n = workload.n_prompts
    elif n < workload.n_prompts and i == 0:
        logger.warning("%d sessions over %d power points: each level draws on %d sessions instead of %d, so none is "
                       "replayed twice on one engine", available, points, n, workload.n_prompts)
    if i == 0 and n == workload.n_prompts:
        return workload
    return dataclasses.replace(workload, n_prompts=n, prompts=[], session_list=[], fitted=False, token_counts=[],
                               sample_offset=workload.sample_offset + i * slices * n)


class SubprocessTrialRunner:
    """Launch the backend for one config, measure, tear down."""

    def __init__(
        self,
        backends: Dict[str, BaseBackend],
        models: Dict[str, PreparedModel],
        hw: HardwareDescriptor,
        workload: Workload,
        log_dir: Optional[Path] = None,
        startup_timeout: float = 900.0,
        request_timeout: float = 180.0,
        power: Optional[object] = None,
        power_settle_s: float = 2.0,
        enough: Optional[Callable[[TrialMetrics], bool]] = None,
        close_call: Optional[Callable[[TrialMetrics], bool]] = None,
        quality: Optional[object] = None,
        gate: Optional[object] = None,
        idle_tail_s: float = 0.0,
        answers: bool = False,
        on_level: Optional[Callable[[int, list], None]] = None,
    ):
        self.backends = backends
        self.models = models
        self.hw = hw
        self.workload = workload
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self._stamps = None  # (started, ready) monotonic stamps of the last launch, for its start-up record
        self.request_timeout = request_timeout
        self.power = power  # rigtune.power.PowerController, or None when energy tuning is off
        self.power_settle_s = power_settle_s
        self.enough = enough  # measure levels highest first and stop early (objectives.enough_level)
        self.close_call = close_call  # measure a level again when its tail is too close to call (close_call_level)
        self.quality = quality  # rigtune.quality.QualityProbe: does this precision still answer the same way?
        self.gate = gate  # rigtune.accuracy_gate.AccuracyGate: grades each precision on labelled items
        self.idle_tail_s = idle_tail_s  # seconds of idle measured after each trial's last level (0 = none)
        self.answers = answers  # record each level's AnswerStats (compare --verdict); off for calibration
        self.on_level = on_level  # given each level's requests as measured (compare --verdict keeps them)

    def _apply_power(self, cfg: Config) -> Optional[str]:
        """Apply cfg's power setting to the GPU. Returns an error message if it cannot be applied."""
        setting = setting_of(cfg)
        if setting.is_default:
            if self.power is not None and getattr(self.power, "applied", None) is not None:
                self.power.restore()  # type: ignore[attr-defined]
            return None
        if self.power is None:
            return "power setting requested but no power controller is configured"
        try:
            self.power.apply(setting)  # type: ignore[attr-defined]
        except Exception as exc:
            return f"power control unavailable: {exc}"
        if self.power_settle_s > 0:
            time.sleep(self.power_settle_s)  # let clocks and board power settle under the new limit
        return None

    def _restore_power(self) -> None:
        if self.power is not None and getattr(self.power, "applied", None) is not None:
            try:
                self.power.restore()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.error("power restore failed: %s", exc)

    def _predict(self, cfg: Config, backend: BaseBackend, model: PreparedModel):
        try:
            return memory_estimate(self.hw, model, cfg, backend.memory_model(self.hw))
        except Exception as exc:
            logger.debug("no memory estimate for %s: %s", cfg.key(), exc)
            return None

    def run_open_loop(self, cfg: Config, rate: float, stage: str = "rate-check") -> TrialResult:
        """Launch cfg and send it Poisson arrivals at `rate` requests per second (run_trial's open-loop run)."""
        return self.run(cfg, stage, arrival_rate=rate)

    def run(self, cfg: Config, stage: str, arrival_rate: Optional[float] = None) -> TrialResult:
        from rigtune.pairing import redact_trial
        self._stamps = None
        result = self._run(cfg, stage, arrival_rate)
        if self._stamps is not None:  # an engine process was started for this trial, ready or not
            started, ready, log = self._stamps
            result.startup = StartupCost(
                seconds=round(ready - started, 3) if started is not None and ready is not None else None,
                **parse_startup(cfg.backend, log))
        model = self.models[cfg.backend]
        return redact_trial(result, getattr(model, "spec_paths", {}))

    def _run(self, cfg: Config, stage: str, arrival_rate: Optional[float] = None) -> TrialResult:
        backend = self.backends[cfg.backend]
        model = self.models[cfg.backend]
        port = free_port()
        log_path = None
        if self.log_dir:
            safe = re.sub(r"[^A-Za-z0-9._=+-]+", "_", cfg.key())
            log_path = self.log_dir / f"{int(time.time())}_{safe}.log"
        hooks = backend.workload_hooks(self.hw, model)
        if cfg.tp > 1 and hooks.gpu_ids:  # one engine over several GPUs: measure all of them
            gpus = [g.index for g in self.hw.gpus if g.vendor == "nvidia"][: cfg.tp]
            hooks = dataclasses.replace(hooks, gpu_ids=gpus or hooks.gpu_ids)
        if cfg.kv_tier is not None:  # where its cached tokens came from (calibrate/measure.py _tier_stats)
            hooks = dataclasses.replace(hooks, tier_report=True,
                                        tier_details_field=backend.tier_details_field())
        observation = MemoryObservation(predicted=self._predict(cfg, backend, model))
        baseline_mb = device_used_mb(hooks.gpu_ids)
        proc = None
        try:
            if cfg.kv_tier is not None:
                kvtier.clear(port)  # starts empty: files another engine left are hits this one never earned
                if cfg.tier_gib[1]:
                    kvtier.prepare(port)  # a 0700 directory of its own; a full disk fails the trial (OSError)
            proc = backend.launch(cfg, model, port, log_path=log_path)
            if not proc.wait_ready(timeout=self.startup_timeout):
                tail, log = proc.tail_log(20), proc.tail_log(400)
                observation.measured = parse_log(cfg.backend, log)
                # vLLM prints its engine's out-of-memory error well above the 20-line tail: say so in
                # the error, so the memory report counts it as the planner failure it is.
                pair_failure = draft_startup_failure(cfg, log)
                cause = (f"{pair_failure}; " if pair_failure else
                         "out of memory at start-up; " if mentions_oom(log) else "")
                why = pair_failure or ("oom_startup" if cause else "startup_other")
                # A kernel setting the engine named in refusing to start is that, whatever else the log holds; an
                # out-of-memory on such a trial stays one, and its error says which setting it ran with.
                refused = kernel_startup_refusal(cfg, log)
                if refused:
                    cause, why = f"{KERNEL_REFUSED} ({refused}); ", "kernel_refused"
                elif cfg.kernels is not None and why == "oom_startup":
                    cause += f"kernel settings {cfg.kernels}; "
                if pair_failure == "draft_refused" and model is not None:
                    from rigtune.pairing import refute
                    if refute(model, cfg.spec_decode, cfg.backend, backend.version(self.hw) if self.hw else None):
                        cause += "rigtune's check was wrong; the pair is blocked on this engine release; "
                # The causes the search acts on or reports, also often above the tail (StagedSearch._startup_retry).
                for kind, find in startup_finders(cfg, log):
                    line = None if cause else find(log)  # never when a cause is already known
                    if line:
                        cause, why = f"{kind} ({line}); ", _CAUSE_KINDS[kind]
                # vLLM raises its memory shortages in the engine core, above the API server's own traceback that ends
                # the log, so for an MTP trial, whose failures can block the setting, the line is found in the log.
                short = memory_short_line(log) if not cause and (cfg.spec_decode or "").startswith("mtp:") else None
                if short:
                    cause = f"{MEMORY_SHORT_CAUSE} ({short}); "
                return TrialResult(
                    config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                    error=f"{cause}failed to start (rc={proc.returncode()})\n{tail}", memory=observation,
                    failure_kind=why,
                )
            perr = self._apply_power(cfg)
            if perr:
                return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), error=perr, memory=observation,
                                   failure_kind="power_unavailable")
            # Speculation's acceptance over the workload alone: counters and log read either side of it, before the
            # gate and the quality probe send their own prompts to the same engine.
            spec_window = None
            if cfg.spec_decode:
                size = getattr(proc, "log_size", None)  # a stand-in process may keep no log
                spec_window = (spec_counters(f"http://127.0.0.1:{port}", hooks) if cfg.backend == "vllm" else None,
                               size() if size else 0)
            metrics = run_trial(
                f"http://127.0.0.1:{port}", hooks, self.workload, pid=proc.pid,
                request_timeout=self.request_timeout, enough=self.enough, close_call=self.close_call,
                idle_tail_s=self.idle_tail_s, arrival_rate=arrival_rate, answers=self.answers,
                **({"on_level": self.on_level} if self.on_level is not None else {}),
            )
            if spec_window is not None and metrics.ok:
                metrics.spec_acceptance = _acceptance(cfg.backend, spec_window, port, hooks, proc)
            if self.quality is not None and metrics.ok and arrival_rate is None:
                # The engine is up and warm: answer the quality prompts here rather than launch it again later.
                self.quality.record(f"http://127.0.0.1:{port}", cfg.backend, cfg.quant)
            observation.measured = merge(
                parse_log(cfg.backend, proc.tail_log(400)), metrics.peak_mem_mb, baseline_mb,
                metrics.telemetry_source,
            )
            if self.gate is not None and metrics.ok:
                # Same warm engine, after the log is read: grading traffic would push its memory lines out of the tail.
                self.gate.record(f"http://127.0.0.1:{port}", hooks.model_name, cfg)
            err = None
            if not metrics.ok:
                err = metrics.failure_summary()
            elif metrics.failed:
                logger.info("%s: tolerated %d/%d failed requests", cfg.key(), metrics.failed, metrics.requests)
            return TrialResult(config=cfg, stage=stage, metrics=metrics, error=err, memory=observation,
                               failure_kind="request_failures" if err else None)
        except Exception as exc:
            if getattr(model, "spec_paths", {}):
                from rigtune.pairing import redact_paths
                logger.error("trial %s crashed: %s", cfg.key(), redact_paths(str(exc), model.spec_paths))
            else:
                logger.exception("trial %s crashed", cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=str(exc),
                               memory=observation, failure_kind="crashed")
        finally:
            self._restore_power()
            if proc is not None:
                proc.stop()  # first, and whatever follows: nothing here may leave an engine holding the GPU
                log = ""
                if cfg.backend in STARTUP_LOGS:  # the engines whose start-up lines are known
                    try:  # its file outlives the process; a log that cannot be read costs the record, not the trial
                        log = proc.tail_log(1_000_000)
                    except Exception as exc:
                        logger.warning("start-up log of %s not read: %s", cfg.key(), type(exc).__name__)
                self._stamps = (getattr(proc, "started_at", None), getattr(proc, "ready_at", None), log)
            if cfg.kv_tier is not None:
                kvtier.clear(port)  # its files are the served prompts' KV cache: never kept (SECURITY.md)

    def sweep(
        self,
        base: Config,
        settings: Sequence[PowerSetting],
        stage: str,
        progress: Optional[ProgressFn] = None,
    ) -> List[TrialResult]:
        """Launch `base` once, then measure it under each power setting in turn.

        Power caps and clock locks take effect on a running server, so one launch serves the whole
        sweep: faster than relaunching per point, and every point shares the same warm engine, so
        differences come from the setting rather than from launch-to-launch variance. Include the
        default setting to get a same-launch baseline.
        """
        backend = self.backends[base.backend]
        model = self.models[base.backend]
        from rigtune.pairing import redact_paths, redact_trial
        private_paths = getattr(model, "spec_paths", {})
        port = free_port()
        log_path = None
        if self.log_dir:
            log_path = self.log_dir / f"{int(time.time())}_{re.sub(r'[^A-Za-z0-9._=+-]+', '_', base.key())}_power.log"
        hooks = backend.workload_hooks(self.hw, model)
        if base.kv_tier is not None:
            hooks = dataclasses.replace(hooks, tier_report=True,
                                        tier_details_field=backend.tier_details_field())
        results: List[TrialResult] = []
        proc = None
        try:
            if base.kv_tier is not None:
                kvtier.clear(port)
                if base.tier_gib[1]:
                    kvtier.prepare(port)
            proc = backend.launch(base.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None}), model,
                                  port, log_path=log_path)
            if not proc.wait_ready(timeout=self.startup_timeout):
                err = redact_paths(f"failed to start (rc={proc.returncode()})\n{proc.tail_log(20)}", private_paths)
                return [TrialResult(config=with_power(base, s), stage=stage, metrics=TrialMetrics(), launched=False,
                                    error=err, failure_kind="startup_other") for s in settings]
            for i, s in enumerate(settings):
                cfg = with_power(base, s)
                if progress:
                    progress(stage, cfg, None)
                perr = self._apply_power(cfg)
                if perr:
                    res = TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), error=perr,
                                      failure_kind="power_unavailable")
                else:
                    try:
                        metrics = run_trial(f"http://127.0.0.1:{port}", hooks,
                                            _point_workload(self.workload, i, len(settings)), pid=proc.pid,
                                            request_timeout=self.request_timeout, idle_tail_s=self.idle_tail_s)
                        err = None if metrics.ok else metrics.failure_summary()
                        res = TrialResult(config=cfg, stage=stage, metrics=metrics, error=err,
                                          failure_kind="request_failures" if err else None)
                    except Exception as exc:
                        if private_paths:
                            logger.error("power point %s crashed: %s", cfg.key(), redact_paths(str(exc), private_paths))
                        else:
                            logger.exception("power point %s crashed", cfg.key())
                        res = TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), error=str(exc),
                                          failure_kind="crashed")
                res = redact_trial(res, private_paths)
                results.append(res)
                if progress:
                    progress(stage, cfg, res)
            return results
        finally:
            self._restore_power()
            if proc is not None:
                proc.stop()
            if base.kv_tier is not None:
                kvtier.clear(port)


# --------------------------------------------------------------------------- staged search


def _memory_knob(cfg: Config) -> float:
    """Single scalar for 'how much memory this config asks for' (higher = more)."""
    if cfg.gpu_memory_utilization is not None:
        return cfg.gpu_memory_utilization
    if cfg.n_gpu_layers is not None:
        return float(cfg.n_gpu_layers)
    return 0.0


def _baseline_candidates(group: Sequence[Config]) -> List[Config]:
    """Stage-1 baselines for a (backend, quant) group: median ctx and batch, memory knob largest first.

    The planner already vetted every config here, so the largest memory setting (full GPU
    offload, highest gpu_memory_utilization) is the fair one to compare quants at; the
    smaller ones are fallbacks if that launch fails.

    The median context and median batch need not co-occur: the planner prunes the grid
    asymmetrically, so a long-context workload on a small card can keep (32k, batch 4) and
    (16k, batch 64) while dropping (32k, batch 64). Fall back to the nearest surviving batch
    at that context, then to the whole group, rather than returning nothing.
    """
    if not group:
        return []
    ctxs = sorted({c.ctx for c in group})
    batches = sorted({c.batch for c in group})
    ctx = ctxs[len(ctxs) // 2]
    batch = batches[len(batches) // 2]
    same = [c for c in group if c.ctx == ctx and c.batch == batch]
    if not same:
        at_ctx = [c for c in group if c.ctx == ctx] or list(group)
        nearest = min({c.batch for c in at_ctx}, key=lambda b: (abs(b - batch), b))
        same = [c for c in at_ctx if c.batch == nearest]
    return sorted(same, key=lambda c: (_memory_knob(c), c.n_batch or 0), reverse=True)


def _baseline(group: Sequence[Config]) -> Config:
    return _baseline_candidates(group)[0]


@dataclass
class StagedSearch:
    objective: str
    runner: TrialRunner
    constraints: Constraints = field(default_factory=Constraints)
    top_quants: int = 2
    max_memory_trials_per_quant: int = 3
    progress: Optional[ProgressFn] = None
    # Optional performance predictor (rigtune.predict.Predictor) + the models it needs.
    predictor: Optional[object] = None
    models: Dict[str, PreparedModel] = field(default_factory=dict)
    workload: Optional[Workload] = None
    prune_below: float = 0.4  # skip a quant whose predicted best tok/s < this fraction of the best predicted
    # Skip a backend's remaining quants once even its best raw tok/s is below this fraction of the
    # throughput a feasible leader reached (throughput and balanced objectives). On an A40 with ShareGPT
    # prompts llama.cpp reached 470 tok/s against vLLM's 1677 and its other three GGUF quants still took
    # 25 minutes to lose.
    drop_backend_below: float = 0.5
    # --budget: stop starting trials once the typical trial would end past this many seconds from the
    # start; the best trial so far wins. Stages run most valuable first, so a budget cuts the least
    # valuable trials. The clock is injectable so tests need not wait.
    budget_s: Optional[float] = None
    clock: Callable[[], float] = time.monotonic
    # Stage 4: power settings to try on the leading config (the default setting first). Empty = off.
    power_points: List[PowerSetting] = field(default_factory=list)
    # Stage 3b: configs differing from the leader only in the prefill knob. None = stage off.
    prefill_variants: Optional[Callable[[Config], List[Config]]] = None
    # Stages 3c-3e: (name, variants of the current leader), run in order after the prefill stage.
    variant_stages: List[Tuple[str, Callable[[Config], List[Config]]]] = field(default_factory=list)
    # Stage 3f: combinations. Stages 3b-3e change one dimension at a time, which misses strategies that
    # only pay together: on an A40, llama.cpp's --kv-unified tied the leader before speculation joined
    # it, and the pair, never tried, measured 12.7% better. Every change that came within combine_band
    # of the leader it was measured against is a candidate; combinations of two or more (one value per
    # dimension) are measured, best estimated first, up to max_combinations.
    combine: bool = True
    combine_band: float = 0.05
    max_combinations: int = 8
    feasible_fn: Optional[Callable[[Config], bool]] = None  # rejects combinations that cannot run
    # Stages 3b-3f run for the leading engine and for every other engine within this fraction of it
    # after the batch stage (0 = the leader only). Tuning only the leader can eliminate the eventual
    # winner: on Dolly-15k prompts on an A40, SGLang led the first trial, fell behind once vLLM got
    # the fp8 cache and a draft model, and was never tried with either.
    contender_band: float = 0.10
    # --quality-probe: a rigtune.quality.QualityProbe. Stage 1 records how far each precision's answers drift
    # from the most faithful one that runs; it refuses nothing, because the comparison saturates.
    quality: Optional[object] = None
    # (backend, quant) pairs that may not be chosen. Nothing populates this today — drift gating was withdrawn —
    # but the invariant it enforces was learned the hard way: dropping a precision from stage 1's `kept` only
    # stops it being tuned, and the winner is the argmax over every successful trial, so a refused precision
    # whose baseline was fastest got served anyway. Any future gate must add to this, not just to `kept`.
    refused: set = field(default_factory=set)
    # --eval-set: a rigtune.accuracy_gate.AccuracyGate. Stage 1 grades each precision on labelled items and refuses,
    # through `refused`, every gated one whose loss against its reference is not certified within a tolerance.
    gate: Optional[object] = None
    # With a gate, configurations the final-quality stage may grade beyond those already graded: each costs one
    # trial and one grading pass (660 s at 300 items). Already-graded configurations, such as the reference's own
    # stage-1 configuration, cost no attempt, so a certified fallback is always within reach.
    final_attempts: int = 3
    # Stage 5: re-measure the leader and up to confirm_top - 1 others within confirm_band of it,
    # confirm_rounds times each in turn, and choose on those runs alone (0 = off).
    confirm_top: int = 0
    confirm_band: float = 0.10
    confirm_rounds: int = 1
    # Stage 3g: explore. The stages change one setting at a time and combine only changes that paid on their
    # own, so settings that pay only together are never measured: on an RTX 4090 with Llama 3.1 8B, random
    # sampling of the same space found a draft model with an int8 KV cache and a 16k prefill budget, 29%
    # faster than the staged pick, whose stages had measured the draft model and the int8 cache one at a time.
    # After the stages, explore_share of the trials run so far (at least explore_min) goes to configurations
    # drawn at random from explore_space that share the leader's engine and precision. Empty = off. Measured on
    # that 4090 with equal time, its 9 draws found nothing better and random search still won by 28%, so the
    # pipeline passes a space only under --explore on.
    explore_space: Sequence[Config] = field(default_factory=list)
    explore_share: float = 0.3
    explore_min: int = 4
    explore_seed: int = 0
    # Speculative decoding first, then the prefill knob, the other variant stages and the combinations on the
    # engine's best configuration for each of its spec_beam best speculation settings (none counts as one). The
    # other settings pay differently with speculation on: on an RTX 4090 with Llama 3.1 8B the int8 cache and a
    # 16k prefill budget did nothing alone, and with the draft model they were part of a configuration 28% faster
    # than the staged pick, which had tuned them before choosing suffix decoding. The pipeline turns it on;
    # callers that pass their own stages keep the plain order unless they ask.
    spec_first: bool = False
    spec_beam: int = 2
    # The lowest fraction an out-of-memory retry may reach for a config, when it is not MIN_RETRY_GMU: SGLang's
    # second-model trials start below it on purpose (pipeline.retry_floor_fn). None: MIN_RETRY_GMU for every config.
    retry_floor: Optional[Callable[[Config], Optional[float]]] = None
    # With spec_first, run the speculation stage on every engine still in the search before choosing which engines
    # to tune further (--speculative all). Choosing after the batch stage judged an engine before its most valuable
    # strategy: on an RTX 4090 with Llama 3.1 8B, SGLang led vLLM by 14% there (776 against 669 tok/s), and vLLM led
    # only once it had speculated; it got that far only because SGLang was offered no speculation.
    decide_after_spec: bool = False
    # Stage 3e+: a KV-cache tier (--kv-tier auto). (leader trial) -> (configs, note), from
    # pipeline.tier_variants_fn, which needs the leader's measured working set and GPU pool, not just its
    # config. Run once per tuned engine, after its variant stages and before its combinations, so a tier can
    # be combined. None = off.
    tier_variants: Optional[Callable[[TrialResult], Tuple[List[Config], Optional[str]]]] = None
    # A start-up failure whose cause the log shows (startup_cause): (failed config, its result) -> (the config to
    # retry, a note), from pipeline.startup_failure_fn. Only multimodal profiling is retried, once, text-only; every
    # later trial of that engine then launches text-only. None = no such failure is acted on or noted.
    startup_failure: Optional[Callable[[Config, TrialResult], Tuple[Optional[Config], Optional[str]]]] = None

    # After the pick under `efficiency` at an arrival rate: measure the winner once with Poisson arrivals at that
    # rate and record the composed J/request beside the measured one. The pick never depends on it.
    rate_check: bool = True
    # Given each trial the search does not launch (budget spent, or a speculation method already failed to start),
    # which never reaches the runner: the trial ledger records them so the order trials ran in can be audited.
    # None = nobody is told, as before.
    on_skip: Optional[Callable[[TrialResult], None]] = None
    _variant_log: List[Tuple[str, Config, Config]] = field(default_factory=list)  # (dimension, leader, variant)
    results: List[TrialResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    _done: Dict[str, TrialResult] = field(default_factory=dict)
    _started: Optional[float] = None
    _confirming: bool = False  # the confirm stage has begun: the budget it held back is its own
    _trial_seconds: List[float] = field(default_factory=list)
    _skipped: List[str] = field(default_factory=list)
    _noted_contenders: set = field(default_factory=set)
    _spec_tried: set = field(default_factory=set)  # engines whose speculation stage has run (decide_after_spec)
    _tier_tried: set = field(default_factory=set)  # engines whose tier stage has run
    # --kernel: (an engine's leader trial) -> (that config with the named kernel setting, or nothing and why).
    # None, the default, adds no stage, no trial and no note.
    kernel_variants: Optional[Callable[[TrialResult], Tuple[List[Config], Optional[str]]]] = None
    _kernel_tried: set = field(default_factory=set)  # engines whose kernel stage has run
    _text_only: set = field(default_factory=set)  # engines whose multimodal profiling failed: trials run text-only
    _startup_notes: List[str] = field(default_factory=list)
    # Speculation kinds not launched again: suffix decoding, after a trial showed Arctic Inference unusable
    # (SUFFIX_FAILED); kind -> the configurations that were not launched for it.
    _spec_blocked: Dict[str, List[str]] = field(default_factory=dict)
    _pair_blocked: set = field(default_factory=set)
    # (engine, kind, second model) that started in this calibration: a later start-up failure of the pair is put
    # down to that trial's other settings, not the second model, and blocks nothing.
    _pair_started: set = field(default_factory=set)
    # The checkpoint's own MTP head, per setting (_head_key: engine, weights, KV cache, tensor parallelism, spec and
    # output-head quantization): one whose start failed for a reason of its own is not launched again at other
    # batches or budgets (_head_failed); one that started in this calibration is never blocked. key -> the config
    # keys skipped for it (a set: a stage may ask for a skipped key again).
    _head_blocked: Dict[tuple, set] = field(default_factory=dict)
    _head_started: set = field(default_factory=set)
    _head_noted: set = field(default_factory=set)
    _headroom_retried: set = field(default_factory=set)  # config keys launched again with less memory reserved
    # (engine, precision) whose quantization the engine refused on this GPU (quant_refused) in a trial without a
    # second model: its check is the method's and the card's, so every other config of the precision would fail too.
    _quant_blocked: set = field(default_factory=set)
    block_failed_pairs: bool = False  # verified mode only: table mode launches what it always launched

    # ---- predictor helpers

    def _predicted_tok_s(self, cfg: Config) -> Optional[float]:
        """Predicted best-level tok/s under the objective's TTFT ceiling, or None if unavailable/unfitted."""
        if self.predictor is None or cfg.backend not in self.models:
            return None
        try:
            if not self.predictor.is_fitted(cfg.backend):  # type: ignore[attr-defined]
                return None
            wl = self.workload or Workload()
            if wl.sessions:
                # A session file's prefill_tokens is its longest possible prompt, mostly history the prefix cache
                # serves; the roofline would price all of it as uncached prefill and prune what may well win.
                return None
            ttft = resolve_goal(self.objective, self.constraints).ttft_limit  # the goal's TTFT ceiling, if any
            ceiling = ttft.bound if ttft is not None else None
            kwargs = {}
            if self.constraints.tpot_ceiling_ms is not None:
                kwargs["tpot_ceiling_ms"] = self.constraints.tpot_ceiling_ms
            p = self.predictor.best_level(  # type: ignore[attr-defined]
                self.models[cfg.backend], cfg, wl.prefill_tokens, wl.decode_tokens, wl.concurrencies, ceiling,
                **kwargs,
            )
            return float(p.tok_s)
        except Exception as exc:
            logger.debug("predictor failed for %s: %s", cfg.key(), exc)
            return None

    # ---- helpers

    # A reservation engine (vLLM, SGLang) that runs out of memory at start-up is retried with this much
    # less of the GPU reserved, down to MIN_RETRY_GMU. Its KV pool fills its reservation, so what ran out
    # is the memory outside it: on an A40, vLLM 0.11 failed twice at batch 512 with the fp8 cache at 0.95,
    # while vLLM 0.29 started that same shape every time, so no fixed buffer size fits every version.
    RETRY_GMU_STEP = 0.05
    MIN_RETRY_GMU = 0.85

    def _headroom_retry(self, cfg: Config, res: TrialResult) -> Optional[Config]:
        """The same config with less GPU memory reserved, after an out-of-memory start-up failure."""
        gmu = cfg.gpu_memory_utilization
        if res.launched or not (res.error or "").startswith("out of memory at start-up") or gmu is None:
            return None
        if cfg.kv_tier is not None:
            # A tier was sized against the GPU pool this fraction gives, and its failure may be host memory
            # (pinned or /dev/shm), which less GPU memory would not fix: recorded, not retried.
            self.notes.append(f"{cfg.key()} ran out of memory at start-up; a KV-cache tier trial is not "
                              "retried with less GPU memory reserved")
            return None
        lower = round(gmu - self.RETRY_GMU_STEP, 2)
        floor = self.MIN_RETRY_GMU
        if self.retry_floor is not None:  # a trial that reserved less on purpose steps down from there
            own = self.retry_floor(cfg)
            if own is not None:
                floor = min(floor, own)
        if lower < floor - 1e-9:
            return None
        retry = cfg.model_copy(update={"gpu_memory_utilization": lower})
        if retry.key() in self._done or (self.feasible_fn is not None and not self.feasible_fn(retry)):
            return None
        return retry

    def _startup_retry(self, cfg: Config, res: TrialResult) -> Optional[Config]:
        """The same config text-only, once, after its multimodal profiling failed at start-up; notes the cause of
        any start-up failure the log showed (startup_failure)."""
        if res.launched or self.startup_failure is None or startup_cause(res.error)[0] is None:
            return None
        retry, note = self.startup_failure(cfg, res)
        if note and note not in self._startup_notes:  # one note per engine and cause, however many trials hit it
            self.notes.append(note)
            self._startup_notes.append(note)  # also said when every stage-1 trial failed
        if retry is None or cfg.text_only or not retry.text_only or retry.key() in self._done:
            return None
        self._text_only.add(cfg.backend)
        return retry

    def _as_run(self, cfg: Config) -> Config:
        """`cfg` as it would launch: text-only once its engine's multimodal profiling has failed at start-up."""
        if cfg.backend in self._text_only and not cfg.text_only:
            return cfg.model_copy(update={"text_only": True})
        return cfg

    def _over_budget(self) -> bool:
        """Whether a typical trial started now would end past the budget. The first always runs.

        Until the confirm stage begins, the time to re-measure the pick is held back, a typical trial for each confirm
        round: on an A100 the hour ran out in the tuning stages, so the pick was never re-measured and was chosen on
        search runs alone, the first of which had started with a cold kernel cache (about 15% less KV cache). Only
        the pick's re-measurement is held, not the whole shortlist's: a cold first start makes the typical trial long,
        and holding three of them can take a short budget's last stages away. The confirm stage re-measures the rest
        of its shortlist while the budget lasts."""
        if self.budget_s is None or not self._trial_seconds:
            return False
        if self._started is None:
            self._started = self.clock()
        typical = sum(self._trial_seconds) / len(self._trial_seconds)
        held = 0.0 if self._confirming or self.confirm_top <= 0 else typical * max(1, self.confirm_rounds)
        return self.clock() + typical + held > self._started + self.budget_s

    def _skip(self, res: TrialResult) -> TrialResult:
        """A trial the search did not launch, passed to on_skip."""
        if self.on_skip is not None:
            self.on_skip(res)
        return res

    def _budget_skip(self, cfg: Config, stage: str) -> None:
        """A re-measurement the budget did not allow (confirm, rate check), for on_skip; it has no result otherwise."""
        if self.on_skip is not None:
            self._skip(TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                                   error=BUDGET_SKIPPED, failure_kind="skipped_budget"))

    def _run(self, cfg: Config, stage: str) -> TrialResult:
        cfg = self._as_run(cfg)  # the planner's grid knows nothing of a text-only retry
        key = cfg.key()
        if key in self._done:
            return self._done[key]
        if self._over_budget():
            self._skipped.append(f"{stage} {key}")
            return self._skip(TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                                          error=BUDGET_SKIPPED, failure_kind="skipped_budget"))
        if (cfg.backend, cfg.quant) in self._quant_blocked:
            return self._skip(TrialResult(
                config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                error=f"skipped: the engine refused {cfg.quant} on this GPU earlier in this calibration",
                failure_kind="skipped_blocked"))
        spec_kind = (cfg.spec_decode or "").partition(":")[0]
        pair_key = None
        if spec_kind in ("draft", "assist"):
            from rigtune.speculative import parse
            pair_key = (cfg.backend, spec_kind, parse(cfg.spec_decode)[1])
        if pair_key in self._pair_blocked:
            return self._skip(TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                                          error="skipped: this second model failed to start on this engine",
                                          failure_kind="skipped_blocked"))
        if spec_kind in self._spec_blocked:  # its start-up failure would only repeat
            self._spec_blocked[spec_kind].append(key)
            return self._skip(TrialResult(
                config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                error=f"skipped: {spec_kind} decoding failed to start earlier in this calibration",
                failure_kind="skipped_blocked"))
        head_key = self._head_key(cfg) if spec_kind == "mtp" else None
        if head_key in self._head_blocked:
            self._head_blocked[head_key].add(key)
            return self._skip(TrialResult(
                config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                error="skipped: the checkpoint's MTP head failed to start with this setting earlier in this calibration",
                failure_kind="skipped_blocked"))
        if self.progress:
            self.progress(stage, cfg, None)
        t0 = self.clock()
        if self._started is None:
            self._started = t0
        graded = self._grading_seconds()
        res = self.runner.run(cfg, stage)
        # Grading runs inside a trial but only once per precision, so --budget's typical trial leaves it out.
        self._trial_seconds.append(self.clock() - t0 - (self._grading_seconds() - graded))
        self._done[key] = res
        if pair_key is not None and res.launched:
            self._pair_started.add(pair_key)
        if head_key is not None and res.launched:
            self._head_started.add(head_key)
        if not res.launched and spec_kind and startup_cause(res.error)[0] == SUFFIX_FAILED:
            self._spec_blocked.setdefault(spec_kind, [])
        # A trial with a second model may have failed on that model's own quantization (a draft, an fp8 MTP head).
        if not res.launched and not cfg.spec_decode and failure_kind(res) == "quant_refused":
            self._quant_blocked.add((cfg.backend, cfg.quant))
            self.notes.append(f"{cfg.backend}: the engine refused {cfg.quant} on this GPU "
                              f"({startup_cause(res.error)[1]}); it is not launched again in this calibration")
        retry = self._headroom_retry(cfg, res)
        if retry is not None:
            self.results.append(res)  # the failed launch stays on record for the memory report
            if self.progress:
                self.progress(stage, cfg, res)
            self.notes.append(f"{key} ran out of memory at start-up; retried at gpu_memory_utilization "
                              f"{retry.gpu_memory_utilization:.2f}")
            self._headroom_retried.add(retry.key())
            return self._run(retry, stage)
        retry = self._startup_retry(cfg, res)
        if retry is not None:
            self.results.append(res)  # the failed launch stays on record
            if self.progress:
                self.progress(stage, cfg, res)
            return self._run(retry, stage)
        # A trial with kernel settings says nothing about its second model: the engine refused the setting, or
        # ran out of memory with it, on a pair its leader had already started. Nor does any failure of a pair that
        # has started in this calibration: on Gemma 4 12B (A40, 16 Nov 2026) its assistant served at 237 tok/s,
        # then failed with a 2,048-token prefill budget (smaller than one image item, 2,496 tokens, exactly as the
        # suffix trial at that budget failed), and the block skipped the four trials at larger budgets.
        if (self.block_failed_pairs and pair_key is not None and not res.launched and cfg.kernels is None and
                (failure_kind(res) in ("draft_refused", "draft_unavailable") or
                 "failed to start" in (res.error or ""))):
            if pair_key in self._pair_started:
                note = (f"{cfg.backend}: {pair_key[2]} failed to start with {key}, but it started earlier in this "
                        "calibration; the failure is put down to that trial's other settings and nothing is blocked")
                if note not in self.notes:
                    self.notes.append(note)
            else:
                self._pair_blocked.add(pair_key)
                self.notes.append(f"{cfg.backend}: {pair_key[2]} failed to start ({failure_kind(res)}); "
                                  "only this candidate is blocked on this engine for this calibration")
        if head_key is not None and key not in self._headroom_retried and self._head_failed(cfg, res):
            setting = self._head_setting(cfg)
            if head_key in self._head_started:
                if head_key not in self._head_noted:
                    self._head_noted.add(head_key)
                    self.notes.append(f"{cfg.backend}: the MTP head setting {setting} failed to start with {key}, "
                                      "but it started earlier in this calibration; the failure is put down to that "
                                      "trial's other settings and nothing is blocked")
            else:
                self._head_blocked[head_key] = set()
                self.notes.append(f"{cfg.backend}: the MTP head setting {setting} failed to start with {key} "
                                  f"({failure_kind(res)}); it is not launched again in this calibration")
        self.results.append(res)
        if self.progress:
            self.progress(stage, cfg, res)
        return res

    @staticmethod
    def _head_key(cfg: Config) -> tuple:
        """What an MTP head's start depends on besides batch and budget: the engine, the target's weights and KV
        cache, tensor parallelism, and the head setting itself. A sample stage_explore draws with every other setting
        free is not skipped for a failure at another of these."""
        return (cfg.backend, cfg.quant, cfg.kv_dtype, cfg.tp, cfg.spec_decode, cfg.spec_head_quant)

    @staticmethod
    def _head_setting(cfg: Config) -> str:
        return cfg.spec_decode + (f" with a {cfg.spec_head_quant} output head" if cfg.spec_head_quant else "")

    @staticmethod
    def _head_failed(cfg: Config, res: TrialResult) -> bool:
        """Whether an MTP trial's failed start says something about the head setting: it did not start, for a reason
        that is neither the memory's (memory_short: a batch or a fraction problem, which another trial can avoid) nor
        another part of the configuration's (a kernel setting, the target's multimodal profiling, Arctic Inference),
        and not because it ran out of time (rc=None: a cold first start can be slow; the next would be warmer). A
        load failure, or a start-up failure the log gave no cause for, is the head's."""
        return (not res.launched and cfg.kernels is None and not memory_short(res)
                and "(rc=None)" not in (res.error or "")
                and failure_kind(res) in ("load_failure", "startup_other"))

    def _grading_seconds(self) -> float:
        return float(getattr(self.gate, "seconds_total", 0.0)) if self.gate is not None else 0.0

    def _eligible(self, results: Sequence[TrialResult]) -> List[TrialResult]:
        """The trials that may still win.

        Dropping a refused precision from stage 1's `kept` only stops it being tuned further; the winner is the
        objective's argmax over every successful trial, so a precision the gate refused could still be served if
        its stage-1 baseline happened to be fastest. That is exactly what happened on a 14B on an RTX 4090: the
        gate refused 4-bit GPTQ on drift and the search picked it anyway.
        """
        if not self.refused:
            return list(results)
        return [r for r in results if (r.config.backend, r.config.quant) not in self.refused]

    def _best_of(self, results: Sequence[TrialResult]) -> Optional[TrialResult]:
        winner, _ = pick(self._eligible(results), self.objective, self.constraints)
        return winner

    # ---- stages

    def _losing_backends(self, results: Sequence[TrialResult], already: Dict[str, str]) -> Dict[str, str]:
        """Backends that cannot catch the leader: even their best raw tok/s is under drop_backend_below of
        the best throughput another backend reached within the objective's constraints.

        Only a feasible result can lead. Under `balanced`, a configuration that is fast but breaks the TTFT
        ceiling must not push out a slower backend that meets it. A backend's raw tok/s bounds any score it
        could reach, so dropping on that bound cannot drop a backend that would have won; that holds only for a
        goal that scores tok/s (`throughput`, `balanced`)."""
        if not self.drop_backend_below or not resolve_goal(self.objective, self.constraints).scores_tok_s:
            return {}
        ok = [r for r in results if r.ok]
        within: Dict[str, float] = {}  # best constrained tok/s per backend (score is -tok/s for both objectives)
        for x in rank(ok, self.objective, self.constraints):
            if x.feasible and (self.gate is None or self.gate.cannot_be_refused(  # type: ignore[attr-defined]
                    (x.result.config.backend, x.result.config.quant))):  # a precision the gate may refuse sets no pace
                b = x.result.config.backend
                within[b] = max(within.get(b, 0.0), -x.score)
        bound: Dict[str, float] = {}  # best raw tok/s per backend
        for r in ok:
            if r.metrics.tok_s:
                bound[r.config.backend] = max(bound.get(r.config.backend, 0.0), r.metrics.tok_s)
        out: Dict[str, str] = {}
        for backend, tok in bound.items():
            rivals = {b: t for b, t in within.items() if b != backend}
            if backend in already or not rivals:
                continue
            leader, lead_tok = max(rivals.items(), key=lambda kv: kv[1])
            if tok < self.drop_backend_below * lead_tok:
                out[backend] = (f"{backend} reached at most {tok:.0f} tok/s, under {self.drop_backend_below:.0%} of "
                                f"{leader}'s {lead_tok:.0f} within the objective's constraints")
        return out

    def stage_quant(self, feasible: Sequence[Config]) -> List[Tuple[str, str]]:
        groups: Dict[Tuple[str, str], List[Config]] = {}
        for c in feasible:
            groups.setdefault((c.backend, c.quant), []).append(c)
        references: set = set()
        if self.gate is not None:
            # Each engine's planned reference runs first, so neither the predictor nor a faster engine can skip the
            # precisions everything else is compared with.
            references = set(self.gate.plan(list(groups)))  # type: ignore[attr-defined]
            groups = dict(sorted(groups.items(), key=lambda kv: kv[0] not in references))
        # Predictor-guided pruning: skip (backend, quant) groups predicted far below the best group.
        predicted: Dict[Tuple[str, str], float] = {}
        for key, group in groups.items():
            p = self._predicted_tok_s(_baseline(group))
            if p is not None:
                predicted[key] = p
        if predicted:
            best_pred = max(predicted.values())
            for key, p in predicted.items():
                if p < self.prune_below * best_pred and key not in references:
                    self.notes.append(
                        f"skipped {key[0]}/{key[1]}: predicted {p:.0f} tok/s < {self.prune_below:.0%} of best "
                        f"predicted {best_pred:.0f}"
                    )
                    groups.pop(key, None)
        stage_results: Dict[Tuple[str, str], TrialResult] = {}
        dropped: Dict[str, str] = {}  # backend -> why its remaining quants were skipped
        for key, group in groups.items():
            if key[0] in dropped:
                self.notes.append(f"skipped {key[0]}/{key[1]}: {dropped[key[0]]}")
                continue
            candidates = _baseline_candidates(group)[: self.max_memory_trials_per_quant]
            if not candidates:
                logger.warning("no baseline config for %s; skipping", key)
                continue
            # Largest memory setting first; step down only if the launch itself fails.
            for cfg in candidates:
                res = self._run(cfg, "quant")
                stage_results[key] = res
                if res.launched or startup_cause(res.error)[0] == LOAD_FAILED:  # less memory cannot fix its code
                    break
            dropped.update(self._losing_backends(list(stage_results.values()), dropped))
        ok = [r for r in stage_results.values() if r.ok]
        from rigtune.calibrate.objectives import rank

        ranked = rank(ok, self.objective, self.constraints)
        # A dropped backend does not go on to the later stages either: its batch sweep would lose the same way.
        kept = [(r.result.config.backend, r.result.config.quant) for r in ranked
                if r.result.config.backend not in dropped]
        if self.quality is not None:
            # Recorded, not enforced. Gating on this drift was tried and withdrawn: identical weights differ
            # from themselves on 10% of answer openings while cheaper precisions differ on 83-100%, so no
            # threshold separates them, and GSM8K put those same weights ~2 points back (quality.py).
            self.notes.extend(self.quality.summary())
        if self.gate is not None:
            refused, notes = self.gate.decide()  # type: ignore[attr-defined]
            self.refused |= refused
            self.notes.extend(notes)
            kept = [k for k in kept if k not in self.refused]
        kept = kept[: self.top_quants]
        logger.info("stage 1 kept: %s", kept)
        return kept

    def stage_memory(self, feasible: Sequence[Config], kept: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], Config]:
        chosen: Dict[Tuple[str, str], Config] = {}
        for key in kept:
            group = [c for c in feasible if (c.backend, c.quant) == key]
            base = next(
                (r.config for r in self.results if r.ok and (r.config.backend, r.config.quant) == key),
                _baseline(group),
            )
            # Fix batch to the baseline's; vary memory knob and ctx, largest first.
            same_batch = [c for c in group if c.batch == base.batch and (c.n_batch == base.n_batch)]
            ordered = sorted(same_batch, key=lambda c: (_memory_knob(c), c.ctx), reverse=True)
            # De-duplicate identical (knob, ctx) pairs.
            seen = set()
            trials: List[Config] = []
            for c in ordered:
                sig = (_memory_knob(c), c.ctx)
                if sig in seen:
                    continue
                seen.add(sig)
                trials.append(c)
            winner: Optional[Config] = None
            for c in trials[: self.max_memory_trials_per_quant]:
                res = self._run(c, "memory")
                if res.ok:
                    winner = c
                    break  # largest safe config found
            if winner is None:
                # Fall back to whatever succeeded in stage 1 for this quant.
                prior = [r for r in self.results if r.ok and (r.config.backend, r.config.quant) == key]
                if prior:
                    winner = prior[0].config
            if winner is not None:
                chosen[key] = winner
        logger.info("stage 2 chose: %s", {k: v.key() for k, v in chosen.items()})
        return chosen

    def stage_batch(self, feasible: Sequence[Config], chosen: Dict[Tuple[str, str], Config]) -> None:
        for key, mem_cfg in chosen.items():
            variants = [
                c
                for c in feasible
                if (c.backend, c.quant) == key
                and c.ctx == mem_cfg.ctx
                and _memory_knob(c) == _memory_knob(mem_cfg)
            ]
            # Try the batch settings the predictor likes best first, so an interrupted run has the winner.
            preds = {c.key(): self._predicted_tok_s(c) for c in variants}
            if all(v is not None for v in preds.values()):
                variants.sort(key=lambda c: -preds[c.key()])
            else:
                variants.sort(key=lambda c: (c.batch, c.n_batch or 0))
            for c in variants:
                self._run(c, "batch")

    def stage_prefill(self, base: Config) -> None:
        """Tune the prefill knob on the leading config; stage 3 already tuned the decode knob."""
        if self.prefill_variants is None:
            return
        try:
            variants = self.prefill_variants(base)
        except Exception as exc:
            logger.warning("no prefill variants for %s: %s", base.key(), exc)
            return
        for v in variants:
            if self.feasible_fn is not None and not self.feasible_fn(v):
                continue  # a known start-up failure, such as SGLang's large prefill budgets on a 24 GB card
            self._run(v, "prefill")
            self._variant_log.append(("prefill", base, v))

    # Config fields a single-dimension stage may change (plus `extra`, compared key by key).
    _DELTA_FIELDS = ("batch", "n_batch", "prefill_budget", "kv_dtype", "prefix_cache", "spec_decode",
                     "kv_tier", "spec_head_quant", "kernels")
    # Changed together with the speculation setting only: SGLang lowers the memory fraction to leave a second
    # model room, and a combination that carries the speculation over must carry the lower fraction with it.
    _SPEC_MEMORY_FIELDS = ("gpu_memory_utilization", "gmu_before_spec")

    @classmethod
    def _delta(cls, ref: Config, var: Config) -> Dict[str, Any]:
        """What a variant changed relative to the leader it was measured against."""
        d: Dict[str, Any] = {f: getattr(var, f) for f in cls._DELTA_FIELDS if getattr(var, f) != getattr(ref, f)}
        if "spec_head_quant" in d:
            d["spec_decode"] = var.spec_decode
        if "spec_decode" in d:
            d.update({f: getattr(var, f) for f in cls._SPEC_MEMORY_FIELDS if getattr(var, f) != getattr(ref, f)})
        extra = {k: v for k, v in var.extra.items() if ref.extra.get(k) != v}
        if extra:
            d["extra"] = extra
        return d

    @staticmethod
    def _apply(cfg: Config, delta: Dict[str, Any]) -> Config:
        update = {k: v for k, v in delta.items() if k != "extra"}
        if "spec_decode" in update and not (update["spec_decode"] or "").startswith("mtp:"):
            update["spec_head_quant"] = None
        if "extra" in delta:
            update["extra"] = {**cfg.extra, **delta["extra"]}
        return cfg.model_copy(update=update)

    def _relative_gain(self, ref_cfg: Config, var_cfg: Config) -> Optional[float]:
        """How much better a variant scored than its leader under the objective (0.05 = 5% better).
        None when either trial failed or the variant breaks the objective's constraints."""
        ref, var = self._done.get(ref_cfg.key()), self._done.get(var_cfg.key())
        if ref is None or var is None or not ref.ok or not var.ok:
            return None
        scored = {r.result.config.key(): r for r in rank([ref, var], self.objective, self.constraints)}
        a, b = scored.get(ref_cfg.key()), scored.get(var_cfg.key())
        if a is None or b is None or not b.feasible:
            return None
        return (a.score - b.score) / abs(a.score) if a.score else 0.0

    def _leader_without_each(self, base: Config, spec: Any = _ANY) -> List[Config]:
        """The leader (with the speculation setting `spec`, when given) with each change the variant stages
        adopted undone, one at a time.

        A change adopted early can hurt one adopted later, and only removing it shows that. The
        batch goes back together with the KV cache type too, since the KV stage raises the batch
        when a smaller cache makes room."""
        leader = self._leader(base.backend, spec)
        if leader is None:
            return []
        lead = leader.config
        changed = self._delta(base, lead)
        # A speculation setting that lowered the memory fraction (SGLang's draft room) is undone with it.
        paired = ("spec_decode", "spec_head_quant") + self._SPEC_MEMORY_FIELDS
        together = "spec_decode" in changed and any(f in changed for f in ("spec_head_quant",) + self._SPEC_MEMORY_FIELDS)
        out: List[Config] = [lead.model_copy(update={f: getattr(base, f)}) for f in changed
                             if f != "extra" and not (together and f in paired)]
        if together:
            out.append(lead.model_copy(update={f: getattr(base, f) for f in paired}))
        for k in changed.get("extra", {}):
            extra = {kk: vv for kk, vv in lead.extra.items() if kk != k}
            if k in base.extra:
                extra[k] = base.extra[k]
            out.append(lead.model_copy(update={"extra": extra}))
        if "kv_dtype" in changed and "batch" in changed:
            out.append(lead.model_copy(update={"kv_dtype": base.kv_dtype, "batch": base.batch}))
        return out

    def stage_combinations(self, base: Config, spec: Any = _ANY, limit: Optional[int] = None) -> None:
        """Stage 3f: measure combinations of the changes that were promising on their own.

        Candidates are changes that came within combine_band of the leader they were measured
        against, or beat it; each dimension keeps its best two. A combination takes at most one value
        per dimension, is applied to `base` (the leader before stages 3b-3e), and is estimated by
        adding its members' gains. Single changes count too: one measured on a later leader may do
        better without an earlier change. On an A40 (`extract`) the fp8 cache won alone, the draft
        model then won on top of it at 610 tok/s, and the draft model without the fp8 cache measured
        728. Summed gains rank that last (the pair sums higher), so the leader with each adopted
        change undone goes first, then the rest by estimate, up to max_combinations new and feasible
        trials in all; the objective and its tie-break then decide as for any trial.
        """
        first = [c for c in self._leader_without_each(base, spec)
                 if c.key() not in self._done and (self.feasible_fn is None or self.feasible_fn(c))]
        options: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        for dim, ref_cfg, var_cfg in self._variant_log:
            if ref_cfg.backend != base.backend:  # another engine's changes do not carry over
                continue
            if spec is not _ANY and dim == "spec":  # the speculation setting is held fixed for this base
                continue
            gain, delta = self._relative_gain(ref_cfg, var_cfg), self._delta(ref_cfg, var_cfg)
            if gain is None or gain < -self.combine_band or not delta:
                continue
            kept = options.setdefault(dim, [])
            if all(d != delta for _, d in kept):
                kept.append((gain, delta))
        choices = [[(0.0, None)] + sorted(kept, key=lambda x: -x[0])[:2] for kept in options.values() if kept]
        candidates: List[Tuple[float, Config]] = []
        for combo in (itertools.product(*choices) if choices else ()):
            members = [(g, d) for g, d in combo if d is not None]
            if not members:
                continue
            cfg = base
            for _, d in members:
                cfg = self._apply(cfg, d)
            if cfg.key() in self._done or (self.feasible_fn is not None and not self.feasible_fn(cfg)):
                continue
            candidates.append((sum(g for g, _ in members), cfg))
        candidates.sort(key=lambda x: -x[0])
        tried: set = set()
        for cfg in first + [c for _, c in candidates]:
            if len(tried) >= (self.max_combinations if limit is None else limit):
                break
            if cfg.key() not in tried:
                tried.add(cfg.key())
                self._run(cfg, "combine")

    def _leader(self, backend: Optional[str] = None, spec: Any = _ANY) -> Optional[TrialResult]:
        """The objective's current winner, overall or among one engine's trials, and among those with one
        speculation setting when `spec` is given (None: no speculation)."""
        pool = [r for r in self._eligible(self.results) if (backend is None or r.config.backend == backend)
                and (spec is _ANY or r.config.spec_decode == spec)]
        leader, _ = pick(pool, self.objective, self.constraints)
        return leader

    def _offers_more(self, cfg: Config, lead: Config) -> Optional[str]:
        """The first variant stage with something to try on `cfg` and nothing on `lead`, if any."""
        for name, fn in self.variant_stages:
            try:
                if fn(cfg) and not fn(lead):
                    return name
            except Exception:
                continue
        return None

    def _offer_count(self, cfg: Config) -> int:
        """How many variant stages have something to try on `cfg`."""
        n = 0
        for _, fn in self.variant_stages:
            try:
                n += bool(fn(cfg))
            except Exception:
                continue
        return n

    def _outpaced(self, base: Config) -> bool:
        """Whether an engine tuned before this one now leads it by more than contender_band while this one
        offers nothing that engine lacks, so its later stages cannot pay: on an RTX 4090 SGLang, with no
        speculative decoding, spent 11 of 66 minutes on stages that ended 15% behind vLLM with it."""
        ranked = rank(self._eligible(self.results), self.objective, self.constraints)
        if not ranked or self.contender_band <= 0:
            return False
        lead = ranked[0]
        lb = lead.result.config.backend
        mine = next((x for x in ranked if x.result.config.backend == base.backend), None)
        if lb == base.backend or mine is None or not (lead.feasible and mine.feasible):
            return False
        gap = (mine.score - lead.score) / abs(lead.score) if lead.score else 0.0
        # What an engine can do, not what is left to try: the leader's first configuration, before its stages
        # adopted a KV cache or a speculation setting, stands for its engine.
        lead_base = next(r.config for r in self.results if r.config.backend == lb)
        if gap <= self.contender_band or self._offers_more(mine.result.config, lead_base):
            return False
        self.notes.append(f"{base.backend} was not tuned further: {lb} led it by {gap:.0%} once tuned, and "
                          f"{base.backend} offers nothing {lb} does not")
        return True

    def _spec_settings(self, backend: str) -> List[Optional[str]]:
        """The engine's spec_beam best speculation settings, best first; no speculation counts as one."""
        out: List[Optional[str]] = []
        for x in rank([r for r in self._eligible(self.results) if r.config.backend == backend],
                      self.objective, self.constraints):
            s = x.result.config.spec_decode
            if s not in out:
                out.append(s)
            if len(out) >= self.spec_beam:
                break
        return out

    def _contenders(self, after: str = "the batch stage") -> List[Config]:
        """The leader's config, then the best config of every other engine within contender_band of it, or
        further behind when a variant stage has something for that engine and nothing for the leader's: on an
        RTX 4090, SGLang in fp8 led vLLM by 14% with Llama 3.1 8B, and only vLLM has speculative decoding,
        worth 57-93% with a draft model on the other cards."""
        ranked = rank(self._eligible(self.results), self.objective, self.constraints)
        if not ranked:
            return []
        lead = ranked[0]
        lb = lead.result.config.backend
        out, seen = [lead.result.config], {lb}
        # What an engine can do, as in _outpaced: the leader's first configuration stands for its engine. Its tuned
        # configuration has already adopted a KV cache or a speculation setting, so judged against that, another
        # engine seemed to offer what the leader's engine lacks, was tuned, and was then dropped by _outpaced for
        # offering nothing it lacks; the notes said both.
        lead_base = next(r.config for r in self.results if r.config.backend == lb)
        for x in ranked[1:]:
            backend = x.result.config.backend
            if backend in seen:
                continue
            seen.add(backend)
            if self.contender_band <= 0 or not (lead.feasible and x.feasible):
                continue
            close = x.score - lead.score <= self.contender_band * abs(lead.score)
            extra = None if close else self._offers_more(x.result.config, lead_base)
            if close or extra:
                out.append(x.result.config)
                if backend not in self._noted_contenders:  # a budgeted run asks twice; say it once
                    self._noted_contenders.add(backend)
                    self.notes.append(
                        f"{backend} came within {self.contender_band:.0%} of the leading {lb} after {after}, so it "
                        "was tuned as well" if close else
                        f"{backend} was more than {self.contender_band:.0%} behind the leading {lb} after {after}, "
                        f"but was tuned as well: it offers {STAGE_NAMES.get(extra, extra)}, which {lb} does not")
        return out

    def _tune(self, base: Config) -> None:
        """Stages 3b-3f for one engine: the prefill knob, the variant stages and the combinations, each on
        that engine's own leader. With spec_first, speculative decoding goes first, and the rest runs once for
        each of the engine's spec_beam best speculation settings, on its best configuration with that setting
        (the combinations share max_combinations between them)."""
        spec_stages = [(n, fn) for n, fn in self.variant_stages if n == "spec"] if self.spec_first else []
        if not spec_stages:
            if self.prefill_variants is not None:
                self.stage_prefill(base)
            for name, fn in self.variant_stages:
                self.stage_variants(name, fn, backend=base.backend)
            self.stage_tier(base.backend)
            self.stage_kernels(base.backend)
            if self.combine:
                self.stage_combinations(base)
            return
        if base.backend not in self._spec_tried:
            for name, fn in spec_stages:
                self.stage_variants(name, fn, backend=base.backend)
        settings = self._spec_settings(base.backend)
        limit = max(2, self.max_combinations // max(1, len(settings)))
        for spec in settings:
            start = self._leader(base.backend, spec)
            if start is None:
                continue
            if self.prefill_variants is not None:
                self.stage_prefill(start.config)
            for name, fn in self.variant_stages:
                if name != "spec":
                    self.stage_variants(name, fn, backend=base.backend, spec=spec)
            self.stage_tier(base.backend)  # once per engine: on its best speculation setting's leader
            self.stage_kernels(base.backend)  # likewise
            if self.combine:
                self.stage_combinations(start.config, spec=spec, limit=limit)

    def stage_spec_each(self, backends: Sequence[str]) -> bool:
        """decide_after_spec: the speculation stage on every engine still in the search, in stage 1's order.
        Returns whether a speculation stage ran."""
        stages = [(n, fn) for n, fn in self.variant_stages if n == "spec"]
        if not (self.spec_first and self.decide_after_spec and stages):
            return False
        for backend in backends:
            for name, fn in stages:
                self.stage_variants(name, fn, backend=backend)
            self._spec_tried.add(backend)
        return True

    def stage_variants(self, name: str, fn: Callable[[Config], List[Config]], backend: Optional[str] = None,
                       spec: Any = _ANY) -> None:
        """Measure the leader's variants along one dimension (one engine's leader when `backend` is given, and
        its leader with one speculation setting when `spec` is); the objective keeps the best."""
        leader = self._leader(backend, spec)
        if leader is None:
            return
        try:
            variants = fn(leader.config)
        except Exception as exc:
            logger.warning("no %s variants for %s: %s", name, leader.config.key(), exc)
            return
        for v in variants:
            self._run(v, name)
            self._variant_log.append((name, leader.config, v))

    def stage_tier(self, backend: str) -> None:
        """Stage 3e+: a KV-cache tier on the engine's leader, once per engine. tier_variants decides from the
        leader's own trial whether its working set outgrows the GPU pool, and says why when it offers nothing;
        the objective then judges the tier like any other change, and a tier that only ties loses
        (objectives._strategies)."""
        if self.tier_variants is None or backend in self._tier_tried:
            return
        self._tier_tried.add(backend)
        leader = self._leader(backend)
        if leader is None:
            return
        try:
            variants, note = self.tier_variants(leader)
        except Exception as exc:
            logger.warning("no KV-cache tier variants for %s: %s", leader.config.key(), exc)
            self.notes.append(f"no KV-cache tier tried on {backend}: {exc}")
            return
        if note:
            self.notes.append(note)
        for v in variants:
            self._run(v, "tier")
            self._variant_log.append(("tier", leader.config, v))

    def stage_kernels(self, backend: str) -> None:
        """One named kernel setting (--kernel) on the engine's leader, once per engine, as a full trial. The
        setting is not launched when the engine would refuse it or it changes nothing, with a note saying which;
        the objective then judges it like any other change, and one that only ties loses (objectives._strategies)."""
        if self.kernel_variants is None or backend in self._kernel_tried:
            return
        self._kernel_tried.add(backend)
        leader = self._leader(backend)
        if leader is None:
            return
        variants, note = self.kernel_variants(leader)  # an error in a rule is raised, not read as "refused"
        if note:
            self.notes.append(note)
        for v in variants:
            self._run(v, "kernels")
            self._variant_log.append(("kernels", leader.config, v))

    def stage_explore(self) -> None:
        """Stage 3g: configurations of the leading engine and precision drawn at random from explore_space,
        with every other setting free, for the interactions one change at a time cannot see."""
        lead = self._leader()
        if lead is None or not self.explore_space:
            return
        lc = lead.config
        pool = [c for c in self.explore_space
                if c.backend == lc.backend and c.quant == lc.quant and self._as_run(c).key() not in self._done]
        n = min(len(pool), max(self.explore_min, math.ceil(self.explore_share * len(self.results))))
        if n <= 0:
            return
        for cfg in random.Random(self.explore_seed).sample(pool, n):
            self._run(cfg, "explore")
        best = self._leader()
        found = best is not None and best.stage == "explore"
        self.notes.append(f"explored {n} random configurations of {lc.backend} {lc.quant} after the stages"
                          + (f"; one took the lead: {best.config.key()}" if found else "; none took the lead"))

    def _remeasure(self, cfg: Config) -> Optional[TrialResult]:
        """A fresh trial of a config already measured, bypassing the cache; None once the budget is spent."""
        if self._over_budget():
            self._skipped.append(f"confirm {cfg.key()}")
            self._budget_skip(cfg, "confirm")
            return None
        if self.progress:
            self.progress("confirm", cfg, None)
        t0, graded = self.clock(), self._grading_seconds()
        res = self.runner.run(cfg, "confirm")
        self._trial_seconds.append(self.clock() - t0 - (self._grading_seconds() - graded))
        self.results.append(res)
        if self.progress:
            self.progress("confirm", cfg, res)
        return res

    def stage_confirm(self) -> Optional[List[TrialResult]]:
        """Stage 5: re-measure the best few configurations and choose among the fresh runs only.

        The search keeps the best of many noisy trials, which flatters the leader: on Dolly-15k prompts
        on an A40 the pick measured 603 tok/s in calibration and 547 when `compare` re-measured it (the
        other four picks re-measured so far came within 1.4%). Judging a shortlist on new runs alone
        takes that bias out of the choice and out of the number the profile reports. Each shortlisted
        config is represented by its median run, the worse of the middle two for an even count. Returns
        those runs, or None when nothing could be re-measured.
        """
        self._confirming = True
        ranked = rank(self._eligible(self.results), self.objective, self.constraints)
        feasible = [x for x in ranked if x.feasible]
        if feasible:
            best = min(x.score for x in feasible)
            candidates = [x for x in feasible if x.score - best <= self.confirm_band * abs(best)]
        else:
            candidates = ranked[:1]
        shortlist: List[Config] = []
        for x in candidates:
            if len(shortlist) >= self.confirm_top:
                break
            if all(c.key() != x.result.config.key() for c in shortlist):
                shortlist.append(x.result.config)
        if self.gate is not None and shortlist:
            self.gate.final_target = shortlist[0].base_key()  # grade the likely pick's own configuration on its run
        runs: Dict[str, List[TrialResult]] = {c.key(): [] for c in shortlist}
        for _ in range(max(1, self.confirm_rounds)):
            for cfg in shortlist:
                res = self._remeasure(cfg)
                if res is None:
                    break
                runs[cfg.key()].append(res)
        reps: List[TrialResult] = []
        for cfg in shortlist:
            order = rank(runs[cfg.key()], self.objective, self.constraints)  # the successful runs, best first
            if order:
                reps.append(order[len(order) // 2].result)
            elif runs[cfg.key()]:
                self.notes.append(f"{cfg.key()} failed when re-measured, so it was not chosen")
        if not reps:
            return None
        winner, _ = pick(reps, self.objective, self.constraints)
        measured = sum(1 for c in shortlist if runs[c.key()])
        note = "re-measured the best " + (f"{measured} configurations" if measured > 1 else "configuration")
        if measured < len(shortlist):
            note += f" of {len(shortlist)} shortlisted (the budget ran out)"
        note += " and chose on the fresh runs"
        orig = self._done.get(winner.config.key()) if winner is not None else None
        if winner is not None and orig is not None and orig.ok:
            then = rank([orig], self.objective, self.constraints)[0].metrics.tok_s
            now = rank([winner], self.objective, self.constraints)[0].metrics.tok_s
            note += f": the pick measured {then:.0f} tok/s in the search and {now:.0f} re-measured"
        self.notes.append(note)
        return reps

    def stage_final_quality(self, pool: Sequence[TrialResult]) -> List[TrialResult]:
        """With an eval set, serve only a configuration whose own grades certify it (AccuracyGate.certified).

        Candidates are taken best first: `pool` (the confirm stage's fresh runs, or every eligible trial), then, if
        none of those can be served, the other eligible trials, each configuration by its best-ranked run. An
        ungraded candidate is re-measured once with grading on (final_attempts of them at most), and the ranking is
        redone on the fresh run. After a refusal, candidates with the refused configuration's quantized KV cache or
        speculation method are skipped, since they most likely lose the same answers. A fallback whose run is a
        search-stage best is re-measured, so the speed the profile reports is a fresh one. Returns the one
        configuration to serve, or nothing."""
        gate = self.gate
        if gate is None:
            return list(pool)

        def best_per_key(results: Sequence[TrialResult], skip: Set[str]) -> Dict[str, TrialResult]:
            out: Dict[str, TrialResult] = {}
            for x in rank(results, self.objective, self.constraints):  # best first: keep the first per key
                out.setdefault(x.result.config.key(), x.result)
            return {k: v for k, v in out.items() if k not in skip}

        pending = best_per_key(pool, set())
        dropped: Set[str] = set()
        fresh: Set[str] = {k for k, r in pending.items() if r.stage == "confirm"}
        refused_kv: Set[str] = set()
        refused_spec: Set[str] = set()
        skipped: List[str] = []
        attempts, expanded, first = 0, False, None
        while True:
            ordered = rank(list(pending.values()), self.objective, self.constraints)
            if not expanded and (not ordered or not ordered[0].feasible):
                expanded = True
                for k, r in best_per_key(self._eligible(self.results), dropped | set(pending)).items():
                    pending[k] = r
                continue
            if not ordered:
                break
            top = ordered[0].result
            cfg, key = top.config, top.config.key()
            first = first or key

            def drop(why: Optional[str] = None) -> None:
                pending.pop(key)
                dropped.add(key)
                gate.fallback_from.append(key)  # ranked above whatever is served, and not certified
                if why:
                    skipped.append(f"{key} ({why})")

            spec = (cfg.spec_decode or "").split(":")[0]
            if not gate.graded(cfg):
                if (cfg.kv_dtype != "auto" and cfg.kv_dtype in refused_kv) or (spec and spec in refused_spec):
                    drop("shares a refused KV cache or speculation method")
                    continue
                if attempts >= self.final_attempts:
                    drop(f"over the {self.final_attempts} final grading attempts")
                    continue
                attempts += 1
                gate.final_target = cfg.base_key()
                measured = self._remeasure(cfg)
                gate.final_target = None
                if measured is None:
                    drop("the calibration budget was spent")
                    continue
                if measured.ok and gate.graded(cfg):
                    pending[key] = measured
                    fresh.add(key)
                    continue  # the fresh run may change which candidate leads
                drop("its re-measurement failed" if not measured.ok else "it could not be graded")
                continue
            if not gate.certified(cfg):
                if cfg.kv_dtype != "auto":
                    refused_kv.add(cfg.kv_dtype)
                if spec:
                    refused_spec.add(spec)
                self.notes.append(f"accuracy gate: final configuration {key} refused: "
                                  f"{gate.config_records[key].reason}")
                drop()
                continue
            if key != first and key not in fresh:  # a fallback: never serve a search-stage best-of-noisy number
                measured = self._remeasure(cfg)
                fresh.add(key)
                if measured is not None and measured.ok:
                    pending[key] = measured
                    continue  # ranked again on the fresh run; it is certified, so it is taken unless overtaken
                self.notes.append(f"accuracy gate: the fallback {key} could not be re-measured, so the profile "
                                  "reports its speed from the search")
            if skipped:
                self.notes.append("accuracy gate: not graded for the final choice: " + "; ".join(skipped))
            return [pending[key]]
        if skipped:
            self.notes.append("accuracy gate: not graded for the final choice: " + "; ".join(skipped))
        return []

    def stage_power(self, base: Config) -> None:
        """Measure the leading config under each power setting and let the objective choose."""
        settings = list(self.power_points)
        if not any(not s.is_default for s in settings):
            return
        sweep = getattr(self.runner, "sweep", None)
        if sweep is not None:
            # One launch; the default setting is re-measured in it so every point shares a baseline.
            for res in sweep(base, settings, "power", progress=self.progress):
                self.results.append(res)
                self._done.setdefault(res.config.key(), res)
        else:
            for s in settings:
                if not s.is_default:
                    self._run(with_power(base, s), "power")

    def _power_note(self, winner: TrialResult, pool: Optional[Sequence[TrialResult]] = None) -> Optional[str]:
        s = setting_of(winner.config)
        base_key = winner.config.base_key()
        pool = self.results if pool is None else pool
        uncapped = [r for r in pool if r.ok and r.config.base_key() == base_key and setting_of(r.config).is_default]
        tried = [r for r in self.results if r.config.base_key() == base_key and not setting_of(r.config).is_default]
        if not tried:
            return None
        if s.is_default:
            return f"power stage tried {len(tried)} settings; none saved energy within the allowed throughput loss"
        if not uncapped:
            return f"power setting chosen: {s.label()}"
        ref = max(uncapped, key=lambda r: r.metrics.tok_s)
        w, r = winner.metrics, ref.metrics
        tok = (w.tok_s - r.tok_s) / r.tok_s * 100 if r.tok_s else 0.0
        note = f"power setting chosen: {s.label()} ({tok:+.1f}% tok/s"
        if w.joules_per_token and r.joules_per_token:
            note += f", {(w.joules_per_token - r.joules_per_token) / r.joules_per_token * 100:+.1f}% J/token"
        if w.power_w and r.power_w:
            note += f", {w.power_w:.0f} W vs {r.power_w:.0f} W"
        return note + " against the same config at default power)"

    def stage_rate_check(self, winner: TrialResult) -> Optional[str]:
        """Measure the pick with Poisson arrivals at the arrival rate it was scored for, and say how the composed
        J/request compared. The run is appended to the results and never ranked (objectives._ok_results)."""
        rate = self.constraints.arrival_rate
        run_open_loop = getattr(self.runner, "run_open_loop", None)
        if not self.rate_check or self.objective != "efficiency" or rate is None or run_open_loop is None:
            return None
        if rate < MIN_OPEN_LOOP_RATE * (1 - 1e-9):
            return (f"rate check skipped: at {rate:g} req/s even one arrival needs more than "
                    f"{1 / MIN_OPEN_LOOP_RATE:.0f} s, the longest window it measures")
        if self._over_budget():
            self._skipped.append(f"rate-check {winner.config.key()}")
            self._budget_skip(winner.config, "rate-check")
            return None
        # Scored as in the pick: among every trial, so in the request size the pick was counted in.
        ranked = [x for x in rank(self.results, self.objective, self.constraints) if x.result is winner] \
            or rank([winner], self.objective, self.constraints)
        comp = ranked[0].rate if ranked else None
        if self.progress:
            self.progress("rate-check", winner.config, None)
        t0 = self.clock()
        res = run_open_loop(winner.config, rate, "rate-check")
        self._trial_seconds.append(self.clock() - t0)
        self.results.append(res)
        if self.progress:
            self.progress("rate-check", winner.config, res)
        m = res.metrics
        if not res.ok or m.joules_per_request is None:
            why = res.error or "no energy telemetry"
            return f"rate check at {rate:g} req/s did not measure J/request: {why.splitlines()[0][:120]}"
        # The prediction is recomputed in requests as long as the check's own answers: the idle term P_idle / λ is
        # per request whatever its length, so scaling the pick's figure per token would mis-state it.
        served = m.requests - m.failed
        measured_tpr = m.output_tokens / served if served else 0.0
        note = f"rate check at {rate:g} req/s over {m.requests} Poisson arrivals ({m.duration_s:.0f} s): "
        measured = f"measured {m.joules_per_request:.1f} J/request at {measured_tpr:.0f} tokens/request"
        again = None
        if comp is not None and comp.total_j and measured_tpr:
            again = compose(winner.metrics, rate, self.constraints.max_utilisation, tokens_per_request=measured_tpr,
                            idle_w=comp.idle_w)
        if again is not None and again.total_j:
            note += (f"predicted {again.total_j:.1f} J/request at those {measured_tpr:.0f} tokens/request, {measured} "
                     f"({(m.joules_per_request - again.total_j) / again.total_j * 100:+.0f}%)")
        else:
            note += measured
        if m.ttft_p95_ms is not None:
            note += (f"; p95 TTFT {m.ttft_p95_ms:.0f} ms, from {served} requests, so close to their slowest "
                     f"rather than a 95th percentile")
        return note

    # ---- entry

    def run(self, feasible: Sequence[Config]) -> Tuple[Optional[TrialResult], List[str]]:
        if not feasible:
            return None, ["memory planner left no feasible configuration"]
        kept = self.stage_quant(feasible)
        if not kept and self.gate is not None and any(r.ok for r in self.results):
            return None, self.notes + ["the accuracy gate refused every precision that ran"]
        if not kept:
            return None, ["every stage-1 trial failed"] + self._startup_notes + [
                f"{r.config.key()}: {r.error}" for r in self.results if r.error
            ]
        if self.budget_s is not None:
            # With a budget, try the variations before the memory and batch sweeps: on Dolly-15k prompts a
            # 10-minute budget spent all its trials on those sweeps and served at stock speed, while the gain
            # (vLLM's draft model and fp8 cache) sat in the variation stages. Every engine within the band
            # gets them, since SGLang led that first stage.
            for base in self._contenders("the precision stage"):
                for name, fn in self.variant_stages:
                    self.stage_variants(name, fn, backend=base.backend)
        chosen = self.stage_memory(feasible, kept)
        self.stage_batch(feasible, chosen)
        engines = list(dict.fromkeys(b for b, _ in chosen))
        speculated = self.stage_spec_each(engines)
        contenders = self._contenders("the speculation stage" if speculated else "the batch stage")
        if speculated:  # an engine left out now gets the note it would have got when skipped below
            tuned = {c.backend for c in contenders}
            for b in engines:
                mine = None if b in tuned else self._leader(b)
                if mine is not None:
                    self._outpaced(mine.config)
        # Engines offering more strategies go first (a stable sort, so the leader first among equals), and one
        # that an engine tuned before it has left clearly behind, with nothing more to offer, is skipped. Once every
        # engine has speculated, the strategy that ordering was for (speculation on one engine only) is measured,
        # and what is left to offer is a minor stage, so the leader goes first: on an A100, SGLang (937 tok/s
        # with MTP) offered an fp8 KV cache vLLM refuses with MTP, was tuned first, and the hour ran out before
        # vLLM's leader (1023) was tuned or confirmed.
        order = contenders if speculated else sorted(contenders, key=lambda c: -self._offer_count(c))
        for base in order:
            if not self._outpaced(base):
                self._tune(base)
        self.stage_explore()
        pool = self._eligible(self.results)
        if self.confirm_top > 0:
            pool = self.stage_confirm() or pool  # the choice is made on the fresh runs alone
        if self.power_points:
            leader, _ = pick(pool, self.objective, self.constraints)
            if leader is not None:
                before = len(self.results)
                self.stage_power(leader.config.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None}))
                if pool is not self.results:
                    pool = list(pool) + self.results[before:]
        compared = pool  # what the power note compares the pick with, before the gate narrows the choice
        if self.gate is not None:
            if self.gate.decided:  # type: ignore[attr-defined]
                pool = self.stage_final_quality(pool)
            else:
                pool = []
                self.notes.append("accuracy gate: stage 1 never finished, so no configuration could be certified")
        winner, notes = pick(pool, self.objective, self.constraints)
        # Judged among the measured pool, as without the gate: the gate narrows the choice to one configuration, and
        # a throughput floor relative to the best measured would always pass against that one alone.
        judged = list(compared) + ([winner] if winner is not None and all(r is not winner for r in compared)
                                   else [])
        refusal = strict_refusal(winner, judged, self.objective, self.constraints)
        if refusal:  # only here: every stage before chose its leader leniently, so none was left untried
            self.notes.append(refusal)
            winner, notes = None, []
        if winner is not None and self.power_points:
            pn = self._power_note(winner, compared)
            if pn:
                notes.append(pn)
        if winner is not None:
            rn = self.stage_rate_check(winner)
            if rn:
                notes.append(rn)
        if self.gate is not None:
            notes.extend(self.gate.final_notes(winner, self.results, self.objective, self.constraints))  # type: ignore
        for kind, skipped in self._spec_blocked.items():
            self.notes.append(f"{kind} decoding failed to start (its cause is noted above) and was not launched again "
                              f"in this calibration" + (f": {len(skipped)} configurations skipped" if skipped else ""))
        for (engine, quant, kv, tp, spec, head_quant), skipped in self._head_blocked.items():
            if skipped:
                setting = spec + (f" with a {head_quant} output head" if head_quant else "")
                n = len(skipped)
                self.notes.append(f"{engine}: {n} configuration{'s' if n != 1 else ''} with the MTP head setting "
                                  f"{setting} ({quant} weights, {kv} KV cache, tp {tp}) skipped after it failed to "
                                  "start (noted above)")
        if self._skipped:
            shown = "; ".join(self._skipped[:5]) + (f"; and {len(self._skipped) - 5} more" if len(self._skipped) > 5 else "")
            self.notes.append(f"calibration budget of {self.budget_s:.0f}s reached after {len(self.results)} trials; "
                              f"skipped {len(self._skipped)}: {shown}")
        return winner, self.notes + notes
