"""Engine kernel and compiler settings as one compact string, `Config.kernels`.

The string is `key=value` pairs joined by commas, sorted by key, so one setting has one spelling and one config key
whichever order it was typed in: `attn=TRITON_ATTN,cg=FULL`. A list value joins its items with `+`
(`cgsizes=1+2+4+8`). With `Config.kernels` unset nothing here runs and every launch is what it was.

A setting reaches a launch only when the engine probe (rigtune.backends.facts) confirmed that the engine that will run
has it. There is no source-reading fallback: an unanswered question offers nothing and says it was not asked, never
"not supported". None of this has run on hardware.

Engine facts, read in vLLM v0.29.0 / v0.33.0:
  --attention-backend, --compilation-config (-cc), --optimization-level   engine/arg_utils.py:1021 / 1114,
                                                                          1721 / 1891, 1736 / 1911
  -cc takes JSON; `mode` and `cudagraph_mode` accept their enum names     config/compilation.py:849-875 / 853-878
  fields cudagraph_mode, cudagraph_capture_sizes, max_cudagraph_capture_size, use_inductor_graph_partition,
  pass_config                                                             config/compilation.py:607-692 / 610-695
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# key -> what it sets, per engine. `pass.<flag>` (vLLM) is one PassConfig flag.
VLLM_KEYS = {
    "attn": "attention backend",
    "cg": "CUDA-graph mode",
    "cgmax": "largest CUDA-graph capture size",
    "cgsizes": "CUDA-graph capture sizes",
    "O": "optimisation level",
    "mode": "compilation mode",
    "gp": "Inductor graph partition",
}
SGLANG_KEYS = {
    "attn": "attention backend",
    "attnp": "prefill attention backend",
    "attnd": "decode attention backend",
    "cg": "decode CUDA-graph backend",
    "cgp": "prefill CUDA-graph backend",
    "cgmax": "largest decode CUDA-graph batch",
    "cgpmax": "largest prefill CUDA-graph batch",
    "tc": "torch.compile",
    "tcmax": "largest torch.compile batch",
}
# Named in the plan and not built: refused by name, so a profile can never hold a setting no renderer knows.
RESERVED = ("moe", "moecfg", "lin", "fp8gemm")
PASS_PREFIX = "pass."
_INTS = ("cgmax", "cgpmax", "tcmax")
_BOOLS = ("gp", "tc")
_TRUE, _FALSE = ("1", "true", "on", "yes"), ("0", "false", "off", "no")

NOT_ASKED = "the engine probe did not say which kernel settings this engine takes (not asked, or it gave no answer)"


def known_key(key: str) -> bool:
    return key in VLLM_KEYS or key in SGLANG_KEYS or (key.startswith(PASS_PREFIX) and len(key) > len(PASS_PREFIX))


def parse(text: str) -> Dict[str, str]:
    """The settings of a kernels string. Raises ValueError for anything that is not `key=value[,key=value]` over
    known keys: an unknown key, a key given twice, an empty value, or a value its key cannot hold."""
    out: Dict[str, str] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"kernel settings are key=value pairs joined by commas, not {text!r}")
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise ValueError(f"kernel setting {item!r} is not key=value")
        if key in RESERVED:
            raise ValueError(f"kernel setting {key!r} is not built yet")
        if not known_key(key):
            raise ValueError(f"unknown kernel setting {key!r}; known: "
                             f"{', '.join(sorted(set(VLLM_KEYS) | set(SGLANG_KEYS)))}, pass.<flag>")
        if key in out:
            raise ValueError(f"kernel setting {key!r} is given twice")
        out[key] = _value(key, value)
    return out


def _value(key: str, value: str) -> str:
    """`value` in the one spelling its key keeps: an integer without a sign or leading zeros, a boolean as 1 or 0,
    a size list ascending without repeats."""
    if key in _INTS or key == "O":
        if not value.isdigit():
            raise ValueError(f"kernel setting {key} takes a whole number, not {value!r}")
        return str(int(value))
    if key in _BOOLS or key.startswith(PASS_PREFIX):
        low = value.lower()
        if low not in _TRUE + _FALSE:
            raise ValueError(f"kernel setting {key} takes 1 or 0, not {value!r}")
        return "1" if low in _TRUE else "0"
    if key == "cgsizes":
        parts = [p.strip() for p in value.split("+")]
        if not all(p.isdigit() and int(p) > 0 for p in parts):
            raise ValueError(f"kernel setting cgsizes takes sizes joined by +, such as 1+2+4+8, not {value!r}")
        return "+".join(str(n) for n in sorted({int(p) for p in parts}))
    if not all(c.isalnum() or c in "_-" for c in value):
        raise ValueError(f"kernel setting {key} takes a name, not {value!r}")
    return value


def format_settings(settings: Dict[str, str]) -> str:
    return ",".join(f"{k}={settings[k]}" for k in sorted(settings))


def canonical(text: str) -> str:
    """`text` in its one spelling: validated, each value normalised, sorted by key."""
    return format_settings(parse(text))


def _flag(facts: Any, option: str) -> bool:
    return bool(facts.flags) and option in facts.flags


def vllm_unconfirmed(settings: Dict[str, str], facts: Any) -> Optional[str]:
    """Why vLLM cannot be handed `settings`, from what the engine probe said this install has; None when every one
    is confirmed. A question the probe did not answer is "not asked", never "not supported"."""
    if facts is None or facts.kernel is None or facts.flags is None:
        return NOT_ASKED
    k = facts.kernel
    for key, value in sorted(settings.items()):
        if key.startswith(PASS_PREFIX):
            if key[len(PASS_PREFIX):] not in k["pass_fields"]:
                return f"this vLLM has no compilation pass {key[len(PASS_PREFIX):]}"
        elif key not in VLLM_KEYS:
            return f"vLLM has no kernel setting {key}"
        elif key == "attn":
            if not _flag(facts, "--attention-backend"):
                return "this vLLM has no --attention-backend"
            if value not in k["attention_backends"]:
                return f"this vLLM has no attention backend {value}"
        elif key == "O":
            if not _flag(facts, "--optimization-level"):
                return "this vLLM has no --optimization-level"
            if f"O{value}" not in k["opt_levels"]:
                return f"this vLLM has no optimisation level {value}"
        elif key == "cg" and value not in k["cudagraph_modes"]:
            return f"this vLLM has no CUDA-graph mode {value}"
        elif key == "mode" and value not in k["compilation_modes"]:
            return f"this vLLM has no compilation mode {value}"
        elif key in _CC_FIELDS and _CC_FIELDS[key] not in k["compilation_fields"]:
            return f"this vLLM's compilation settings have no {_CC_FIELDS[key]}"
    if any(key in _CC_FIELDS or key.startswith(PASS_PREFIX) for key in settings):
        if not _flag(facts, "--compilation-config"):
            return "this vLLM has no --compilation-config"
        if any(key.startswith(PASS_PREFIX) for key in settings) and "pass_config" not in k["compilation_fields"]:
            return "this vLLM's compilation settings have no pass_config"
    return None


# kernels key -> the CompilationConfig field it sets.
_CC_FIELDS = {"cg": "cudagraph_mode", "cgmax": "max_cudagraph_capture_size", "cgsizes": "cudagraph_capture_sizes",
              "mode": "mode", "gp": "use_inductor_graph_partition"}


def vllm_compilation_config(settings: Dict[str, str]) -> Optional[str]:
    """The one --compilation-config value for `settings`: JSON with sorted keys and no spaces, so it is one argument
    with one spelling. None when no setting belongs in it."""
    cc: Dict[str, Any] = {}
    for key, field in _CC_FIELDS.items():
        if key not in settings:
            continue
        value = settings[key]
        if key == "cgmax":
            cc[field] = int(value)
        elif key == "cgsizes":
            cc[field] = [int(p) for p in value.split("+")]
        elif key == "gp":
            cc[field] = value == "1"
        else:
            cc[field] = value
    passes = {key[len(PASS_PREFIX):]: value == "1" for key, value in settings.items() if key.startswith(PASS_PREFIX)}
    if passes:
        cc["pass_config"] = passes
    return json.dumps(cc, sort_keys=True, separators=(",", ":")) if cc else None


def vllm_args(settings: Dict[str, str]) -> Tuple[List[str], Optional[str]]:
    """(the arguments `settings` add to `vllm serve` other than the attention backend, that backend or None). The
    attention backend is returned apart: the launch decides it together with the cache type's own requirement."""
    args: List[str] = []
    cc = vllm_compilation_config(settings)
    if cc is not None:
        args += ["--compilation-config", cc]
    if "O" in settings:
        args += ["--optimization-level", settings["O"]]
    return args, settings.get("attn")
