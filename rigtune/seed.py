"""A seed: the command a user (or a model's published recipe) would run, read into rigtune's settings.

`rigtune calibrate --seed "vllm serve ..."` measures that command first and improves from it, so rigtune's pick is
never measured below it (.git/wip/plan-recipe-seed.md). This module only reads the command: which engine, which
model, the settings rigtune tunes (Config fields), the flags it passes through untouched, and what it dropped.

Read for vLLM (`vllm serve MODEL ...`, `python -m vllm.entrypoints.openai.api_server --model MODEL ...`) and SGLang
(`python -m sglang.launch_server --model-path MODEL ...`, `sglang serve ...`). Flags take `--flag value` or
`--flag=value`, with dashes or underscores (both engines accept either); vLLM's dotted form
(`--speculative-config.method dspark`) is read as the JSON object it stands for.

The rule: a setting is either read into what rigtune launches, refused with the reason, or noted as not kept; never
silently changed. A secret (an API key, a Hub token, an environment variable's value) is never kept or echoed.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rigtune import speculative as sp

ENGINES = ("vllm", "sglang")


@dataclass
class Seed:
    """A command read into rigtune's terms. `fields` are Config fields (no quant when the command leaves the
    checkpoint's own: the model decides it); `extra` are flags passed through unchanged (Config.extra, a JSON object's
    value kept as JSON text); `revision` is the model's (for its ModelSpec); `notes` say what was not kept, and why."""

    engine: str
    model: str
    revision: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# Never kept, never echoed (the flag's name is noted, not its value).
SECRET = ("api-key", "hf-token", "token")
# Flags rigtune sets itself on every launch, or a stage of its own owns (the KV tier, the kernel stage): not kept.
OWNED = {
    "vllm": ("host", "port", "served-model-name", "kv-transfer-config", "kv-offloading-size", "kv-offloading-backend",
             "compilation-config", "optimization-level", "attention-backend"),
    "sglang": ("host", "port", "served-model-name", "enable-hierarchical-cache", "hicache-ratio", "hicache-size",
               "hicache-storage-backend", "enable-torch-compile", "torch-compile-max-bs", "attention-backend"),
}
# Flags that change the GPUs used or the memory the plan sizes, which rigtune's plan would not see: refused.
REFUSED = {
    "vllm": ("pipeline-parallel-size", "data-parallel-size", "kv-cache-memory-bytes", "cpu-offload-gb",
             "num-gpu-blocks-override", "swap-space"),
    "sglang": ("pp-size", "dp-size", "data-parallel-size", "max-total-tokens", "cpu-offload-gb"),
}
# Flags that take no value (every other flag takes one).
BARE = {
    "vllm": ("enable-prefix-caching", "no-enable-prefix-caching", "trust-remote-code", "enforce-eager",
             "enable-chunked-prefill", "no-enable-chunked-prefill", "disable-log-requests", "disable-log-stats",
             "enable-auto-tool-choice", "language-model-only", "disable-custom-all-reduce",
             "enable-prompt-tokens-details", "async-scheduling", "no-async-scheduling", "enable-expert-parallel"),
    "sglang": ("disable-radix-cache", "trust-remote-code", "enable-torch-compile", "disable-cuda-graph",
               "disable-overlap-schedule", "enable-mixed-chunk", "enable-dp-attention", "enable-metrics",
               "skip-server-warmup", "enable-cache-report"),
}
# Single-dash short forms, as the engines spell them (vLLM's -tp, -pp and -q).
SHORT = {"vllm": {"-tp": "tensor-parallel-size", "-pp": "pipeline-parallel-size", "-q": "quantization"}, "sglang": {}}
# --dtype values rigtune launches, by its labels; "auto" leaves the checkpoint's own.
DTYPES = {"bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "half": "fp16", "fp16": "fp16"}
# --quantization values: fp8 is online quantization rigtune launches as such; these name a checkpoint's own
# quantization, which both engines read from its config without the flag (rigtune serves it as stored).
CHECKPOINT_QUANT = ("awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors", "modelopt", "modelopt_fp4",
                    "mxfp4", "w8a8_int8", "w8a8_fp8", "fbgemm_fp8")
ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _engine_and_model(tokens: List[str], notes: List[str]) -> Tuple[str, Optional[str], List[str]]:
    """The engine a command starts, its positional model if any, and the flags after them. Never echoes a token: a
    command may carry a secret."""
    envs = 0
    while envs < len(tokens) and ENV.match(tokens[envs]):
        envs += 1
    if envs:
        names = ", ".join(t.split("=", 1)[0] for t in tokens[:envs])
        notes.append(f"environment variables before the command are not kept ({names}); set them where rigtune runs")
        tokens = tokens[envs:]
    if tokens[:2] == ["vllm", "serve"]:
        rest = tokens[2:]
        if rest and not rest[0].startswith("-"):
            return "vllm", rest[0], rest[1:]
        return "vllm", None, rest
    for i, tok in enumerate(tokens[:4]):
        if tok.startswith("vllm.entrypoints"):
            return "vllm", None, tokens[i + 1:]
        if tok == "sglang.launch_server":
            return "sglang", None, tokens[i + 1:]
    if tokens[:2] == ["sglang", "serve"]:
        return "sglang", None, tokens[2:]
    raise ValueError("not a vLLM or SGLang server command (expected `vllm serve MODEL ...` or "
                     "`python -m sglang.launch_server --model-path MODEL ...`)")


def _set_dotted(obj: Dict[str, Any], key: str, value: Any) -> None:
    *path, last = key.split(".")
    for part in path:
        obj = obj.setdefault(part, {})
        if not isinstance(obj, dict):
            raise ValueError(f"dotted flag {key!r} sets a key inside a value that is not an object")
    obj[last] = value


def _flags(engine: str, rest: List[str],
           spans: Optional[List[Tuple[str, Any, int, int]]] = None) -> List[Tuple[str, Any]]:
    """(name, value) in order, names with dashes; a bare flag's value is True. vLLM's dotted flags are gathered into
    one object per flag, merged into the same flag's JSON value when both are given. `spans`, when given, gets
    (name, value, start, end) per flag as written: the tokens rest[start:end] it took."""
    out: List[Tuple[str, Any]] = []
    dotted: Dict[str, Dict[str, Any]] = {}
    i = 0
    while i < len(rest):
        start = i
        tok = rest[i]
        if tok in SHORT[engine]:
            tok = "--" + SHORT[engine][tok]
        if not tok.startswith("--"):
            raise ValueError(f"unexpected argument at position {i + 1} of the seed's flags (not a --flag)")
        name, eq, value = tok[2:].partition("=")
        base, dot, key = name.partition(".")
        base = base.replace("_", "-")  # both engines accept --max_model_len for --max-model-len
        name = base + dot + key
        if base in BARE[engine]:
            if eq:
                raise ValueError(f"--{base} takes no value (give it or leave it out)")
            value, i = True, i + 1
        elif eq:
            i += 1
        elif i + 1 < len(rest) and (not rest[i + 1].startswith("-") or _number(rest[i + 1])):
            value, i = rest[i + 1], i + 2
        elif base in SECRET:
            value, i = "", i + 1
        else:
            raise ValueError(f"--{base} needs a value")
        if spans is not None:
            spans.append((name, value, start, i))
        if engine == "vllm" and dot:
            _set_dotted(dotted.setdefault(base, {}), key, _scalar(value))
            continue
        out.append((name, value))
    for base, obj in dotted.items():
        given = next((j for j, (n, _) in enumerate(out) if n == base), None)
        if given is None:
            out.append((base, obj))
        else:  # --speculative-config '{...}' with --speculative-config.k v: the dotted keys win, as in vLLM
            out[given] = (base, {**_json_value(base, out[given][1]), **obj})
    return out


def _number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _scalar(value: Any) -> Any:
    """A dotted flag's value as JSON would give it (numbers and booleans), else the string."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _json_value(name: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        obj = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--{name} is not JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"--{name} is not a JSON object")
    return obj


def _int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"--{name} takes a whole number")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--{name} takes a whole number, not {value!r}") from exc
    if out < minimum:
        raise ValueError(f"--{name} is {out}; rigtune reads {minimum} or more")
    return out


def _fraction(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--{name} takes a number, not {value!r}") from exc
    if not 0 < out <= 1:
        raise ValueError(f"--{name} is {out}; a fraction of the GPU's memory is above 0 and at most 1")
    return out


def vllm_spec(conf: Dict[str, Any], notes: List[str]) -> str:
    """rigtune's spec string for vLLM's --speculative-config, with a note for each key rigtune sets its own way."""
    method = conf.get("method")
    k = conf.get("num_speculative_tokens")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("--speculative-config needs num_speculative_tokens, a whole number of at least 1")
    if method is None:
        raise ValueError("--speculative-config names no method: rigtune does not guess it from the drafter's name, "
                         "as vLLM does; add \"method\"")
    model = conf.get("model")
    if method in ("ngram", "ngram_gpu"):
        spec = f"{method}:{k}"
        lookup = (conf.get("prompt_lookup_min"), conf.get("prompt_lookup_max"))
        if lookup != (2, 4):
            given = "vLLM's own default" if lookup == (None, None) else f"{lookup[0]}-{lookup[1]}"
            notes.append(f"n-gram lookup ({given}) is not kept: rigtune launches n-gram with lookup 2-4")
    elif method == "suffix":
        spec = sp.suffix(k)
    elif method == "mtp":
        spec = sp.mtp(k) if model is None else sp.assist(model, k)
    elif method == "eagle3" and model:
        spec = sp.eagle3(model, k)
    elif method in sp.BLOCK_DRAFTERS:
        if model is None and method != "dspark":
            raise ValueError(f"--speculative-config: a {method} drafter names its model")
        spec = sp.own_block_drafter(k) if model is None else sp.block_drafter(method, model, k)
    elif method == "draft_model" and model:
        spec = sp.draft(model, k)
    else:
        raise ValueError(f"--speculative-config method {method!r} is not one rigtune launches")
    known = {"method", "num_speculative_tokens", "model", "prompt_lookup_min", "prompt_lookup_max"}
    for key in sorted(set(conf) - known):
        notes.append(f"speculative {key} is not kept: rigtune sets it for the engine and drafter it runs (a block "
                     "drafter's sampling, a pinned revision)")
    return spec


SGLANG_SPEC_READ = ("speculative-algorithm", "speculative-draft-model-path", "speculative-num-steps",
                    "speculative-num-draft-tokens", "speculative-eagle-topk", "speculative-dspark-block-size")


def sglang_spec(flags: Dict[str, Any], model: str, notes: List[str]) -> str:
    """rigtune's spec string for SGLang's --speculative-* flags. A count SGLang would work out itself is asked for:
    rigtune launches the count it planned."""
    algo = str(flags["speculative-algorithm"]).upper()
    path = flags.get("speculative-draft-model-path")
    topk = flags.get("speculative-eagle-topk")

    def given(name: str) -> int:
        if name not in flags:
            raise ValueError(f"--speculative-algorithm {algo}: give --{name} (rigtune launches the count it planned, "
                             "not SGLang's own choice)")
        return _int(name, flags[name])

    if topk is not None and _int("speculative-eagle-topk", topk) != 1:
        notes.append(f"a draft tree (top-k {topk}) is not kept: rigtune drafts a chain (top-k 1)")
    if algo == "NGRAM":
        return sp.ngram(given("speculative-num-draft-tokens"))
    if algo in ("DFLASH", "DSPARK"):
        if not path:
            raise ValueError(f"SGLang's {algo} needs --speculative-draft-model-path")
        k = given("speculative-num-draft-tokens") - 1 if algo == "DFLASH" else given("speculative-dspark-block-size")
        return sp.block_drafter(algo.lower(), str(path), k)
    if algo in ("NEXTN", "EAGLE") and (path is None or path == model):
        return sp.mtp(given("speculative-num-steps"))
    if algo == "EAGLE3" and path:
        return sp.eagle3(str(path), given("speculative-num-steps"))
    if algo == "STANDALONE" and path:
        return sp.draft(str(path), given("speculative-num-steps"))
    raise ValueError(f"--speculative-algorithm {algo} is not one rigtune launches as given")


def _quant(seed: Seed, name: str, value: Any) -> None:
    q = str(value).lower()
    if q == "fp8":
        seed.fields["quant"] = "fp8"
    elif q in CHECKPOINT_QUANT:
        seed.notes.append(f"--{name} {q} is not passed: the checkpoint's own quantization config selects it, as rigtune "
                          "serves a quantized checkpoint")
    else:
        raise ValueError(f"--{name} {q} is not one rigtune launches (fp8, or a checkpoint's own quantization)")


def _dtype(seed: Seed, name: str, value: Any) -> None:
    d = str(value).lower()
    if d == "auto":
        return
    if d not in DTYPES:
        raise ValueError(f"--{name} {d} is not one rigtune launches (bfloat16 or float16)")
    seed.fields.setdefault("dtype_given", DTYPES[d])


def _ctx(seed: Seed, name: str, value: Any) -> None:
    if str(value).strip() in ("-1", "auto"):
        seed.notes.append(f"--{name} {value} (the engine's automatic length) is not kept: rigtune sets the context "
                          "from the model and the workload")
        return
    seed.fields["ctx"] = _int(name, value)


def _common(seed: Seed, name: str, value: Any) -> bool:
    """Flags both engines spell alike or that this module treats alike; True when handled."""
    if name in SECRET:
        seed.notes.append(f"--{name} is not kept (a secret): set it where rigtune runs")
    elif name in REFUSED[seed.engine]:
        raise ValueError(f"--{name} changes the GPUs or the memory rigtune's plan sizes, which it does not model; "
                         "remove it to seed with this command")
    elif name in OWNED[seed.engine]:
        seed.notes.append(f"--{name} is not kept: rigtune sets it itself (its address, KV tier or kernel stage)")
    elif name == "revision":
        seed.revision = str(value)
    elif name == "quantization":
        _quant(seed, name, value)
    elif name == "dtype":
        _dtype(seed, name, value)
    elif name == "kv-cache-dtype":
        seed.fields["kv_dtype"] = str(value)
    elif name == "language-model-only":
        seed.fields["text_only"] = True
    else:
        return False
    return True


def _read_vllm(seed: Seed, flags: List[Tuple[str, Any]]) -> None:
    f = seed.fields
    for name, value in flags:
        if _common(seed, name, value):
            continue
        if name == "model":
            seed.model = seed.model or str(value)
        elif name == "max-model-len":
            _ctx(seed, name, value)
        elif name == "max-num-seqs":
            f["batch"] = _int(name, value)
        elif name == "gpu-memory-utilization":
            f["gpu_memory_utilization"] = _fraction(name, value)
        elif name == "max-num-batched-tokens":
            f["prefill_budget"] = _int(name, value)
        elif name == "tensor-parallel-size":
            f["tp"] = _int(name, value)
        elif name in ("enable-prefix-caching", "no-enable-prefix-caching"):
            f["prefix_cache"] = name == "enable-prefix-caching"
        elif name == "speculative-config":
            f["spec_decode"] = vllm_spec(_json_value(name, value), seed.notes)
        else:
            _pass_through(seed, name, value)


def _read_sglang(seed: Seed, flags: List[Tuple[str, Any]]) -> None:
    f = seed.fields
    by_name = dict(flags)
    for name, value in flags:
        if name.startswith("speculative-"):
            continue  # read together below
        if _common(seed, name, value):
            continue
        if name in ("model-path", "model"):
            seed.model = seed.model or str(value)
        elif name == "context-length":
            _ctx(seed, name, value)
        elif name == "max-running-requests":
            f["batch"] = _int(name, value)
        elif name == "mem-fraction-static":
            f["gpu_memory_utilization"] = _fraction(name, value)
        elif name == "chunked-prefill-size":
            f["prefill_budget"] = _int(name, value)
        elif name in ("tp-size", "tp", "tensor-parallel-size"):
            f["tp"] = _int(name, value)
        elif name == "disable-radix-cache":
            f["prefix_cache"] = False
        else:
            _pass_through(seed, name, value)
    spec_flags = sorted(n for n in by_name if n.startswith("speculative-"))
    if "speculative-algorithm" in by_name:
        f["spec_decode"] = sglang_spec(by_name, seed.model, seed.notes)
        spec_flags = [n for n in spec_flags if n not in SGLANG_SPEC_READ]
    for name in spec_flags:
        seed.notes.append(f"--{name} is not kept: rigtune sets it for the drafter it runs (or none is named)")


def _pass_through(seed: Seed, name: str, value: Any) -> None:
    seed.extra[name.replace("-", "_")] = json.dumps(value) if isinstance(value, dict) else value
    seed.notes.append(f"--{name} is passed through unchanged, not tuned")


RECIPE = "recipe"  # --seed recipe: the shipped recipe for this model, engine and GPU (data/recipes.json)


def recipe_for(model: str, engines: List[str], gpu_name: str) -> Optional[Tuple[str, str]]:
    """(command, where it was read) of the shipped recipe for `model` on one of `engines`, in the file's order,
    the first whose `gpus` (if any) names part of this GPU's name; None when none applies."""
    from importlib import resources

    data = json.loads(resources.files("rigtune").joinpath("data", "recipes.json").read_text(encoding="utf-8"))
    name = gpu_name.lower()
    mig = " mig " in f" {name} "  # a MIG slice is named after its card but has a fraction of its memory
    for r in data["recipes"]:
        if r["model"] != model or r["engine"] not in engines:
            continue
        if r.get("gpus") and (mig or not any(g in name for g in r["gpus"])):
            continue
        return r["command"], r["source"]
    return None


def seed_config(seed: Seed, hw: Any, model: Any, precisions: List[str]) -> Tuple[Any, List[str]]:
    """The Config rigtune launches for a seed on this machine: what the command says, and for what it leaves to the
    engine, the engine's own default where rigtune must state one (it always passes the context, batch and memory
    fraction), each noted. `model` is the PreparedModel, `precisions` the backend's on this GPU (its first is what an
    unquantized checkpoint loads as with --dtype auto)."""
    from rigtune.backends.vllm import vllm_default_max_num_seqs
    from rigtune.models import Config
    from rigtune.quantized import stored_quant

    f = dict(seed.fields)
    notes: List[str] = []
    if "quant" not in f:
        f["quant"] = stored_quant(model.weights_bytes) or precisions[0]
    if "ctx" not in f:
        f["ctx"] = model.arch.max_position_embeddings
        notes.append(f"seed: no context given; the model's own, {f['ctx']:,} tokens, as the engine takes")
    if "batch" not in f:
        if seed.engine == "vllm":
            f["batch"] = vllm_default_max_num_seqs(hw)
            notes.append(f"seed: no --max-num-seqs; vLLM's default on this GPU, {f['batch']:,}")
        else:  # SGLang sizes it from its pool at start-up; rigtune states the batch its stock row launches
            f["batch"] = 256
            notes.append("seed: no --max-running-requests; SGLang would size it from its pool, rigtune launches 256 "
                         "as its stock row does")
    if "gpu_memory_utilization" not in f:
        f["gpu_memory_utilization"] = 0.90 if seed.engine == "vllm" else 0.88
        if seed.engine == "sglang":
            notes.append("seed: no --mem-fraction-static; rigtune launches 0.88, as its stock row does (SGLang picks "
                         "its own per GPU)")
    if seed.extra:
        f["extra"] = dict(seed.extra)
    return Config(backend=seed.engine, **f), notes


# The flags a drop-in (`rigtune vllm serve ...`) takes for rigtune's own process rather than the engine's launch: where
# it listens, the key its clients send, the Hub token its downloads use, the name clients ask for. They never enter
# the seed (its cache key), so a new key, token or port is not a new calibration.
SERVER_FLAGS = ("host", "port", "api-key", "hf-token", "token", "served-model-name")
DEFAULT_PORT = {"vllm": 8000, "sglang": 30000}
# Where each engine listens with no --host: vLLM on every interface (uvicorn's host None), SGLang on this machine.
DEFAULT_HOST = {"vllm": "0.0.0.0", "sglang": "127.0.0.1"}


def server_flags(tokens: List[str]) -> Tuple[str, List[str], Dict[str, Any]]:
    """A server command as argv: its engine, the tokens without SERVER_FLAGS, and those flags' values (the last given
    wins, as in both engines). Raises ValueError as parse_command does, never echoing a value."""
    notes: List[str] = []
    if not tokens:
        raise ValueError("the command is empty")
    engine, _, rest = _engine_and_model(list(tokens), notes)
    head = len(tokens) - len(rest)
    spans: List[Tuple[str, Any, int, int]] = []
    _flags(engine, rest, spans)
    found: Dict[str, Any] = {}
    drop = set()
    for name, value, start, end in spans:
        if name in SERVER_FLAGS:
            found[name] = value
            drop.update(range(head + start, head + end))
    return engine, [t for j, t in enumerate(tokens) if j not in drop], found


def parse_command(command: str) -> Seed:
    """Read a vLLM or SGLang server command into a Seed. Raises ValueError, with the reason, on a command rigtune
    cannot read or launch as given; the message never repeats the command's values."""
    try:
        tokens = shlex.split(command.replace("\\\n", " "))
    except ValueError as exc:
        raise ValueError(f"the seed command does not parse as a shell command ({exc})") from None
    notes: List[str] = []
    if not tokens:
        raise ValueError("the seed command is empty")
    engine, model, rest = _engine_and_model(tokens, notes)
    seed = Seed(engine=engine, model=model or "", notes=notes)
    flags = _flags(engine, rest)
    (_read_vllm if engine == "vllm" else _read_sglang)(seed, flags)
    dtype = seed.fields.pop("dtype_given", None)
    if dtype is not None:
        if "quant" in seed.fields:
            seed.notes.append(f"--dtype is not kept beside --quantization {seed.fields['quant']}: rigtune launches "
                              "that with the checkpoint's own dtype")
        else:
            seed.fields["quant"] = dtype
    if not seed.model:
        raise ValueError("the seed command names no model")
    return seed
