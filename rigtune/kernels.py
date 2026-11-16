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


# SGLang: kernels key -> (its server flag, how the value is passed). Flag names are the fields' names with dashes
# (`_field_to_cli_name`); the fields, in v0.5.19 server_args.py / v0.5.23 arg_groups/fields/exec_.py:
#   attention_backend 1657 / 207, decode_attention_backend 1666 / 215, prefill_attention_backend 1680 / 235,
#   cuda_graph_backend_decode 1859 / 613, cuda_graph_backend_prefill 1867 / 621, cuda_graph_max_bs_decode 1875 / 629,
#   cuda_graph_max_bs_prefill 1880 / 632, enable_torch_compile 2056 / 691, torch_compile_max_bs 2064 / 698.
# --cuda-graph-max-bs is a deprecated alias on 0.5.19 (server_args.py:3929-3934) and gone from 0.5.23: never rendered.
SGLANG_FLAGS = {
    "attn": "--attention-backend",
    "attnp": "--prefill-attention-backend",
    "attnd": "--decode-attention-backend",
    "cg": "--cuda-graph-backend-decode",
    "cgp": "--cuda-graph-backend-prefill",
    "cgmax": "--cuda-graph-max-bs-decode",
    "cgpmax": "--cuda-graph-max-bs-prefill",
    "tc": "--enable-torch-compile",
    "tcmax": "--torch-compile-max-bs",
}


def sglang_unconfirmed(settings: Dict[str, str], facts: Any) -> Optional[str]:
    """Why SGLang cannot be handed `settings`, from the flags and choices its own parser reported; None when every
    one is confirmed. Without that answer nothing is offered: "not asked", never "not supported"."""
    if facts is None or facts.flags is None:
        return NOT_ASKED
    for key, value in sorted(settings.items()):
        flag = SGLANG_FLAGS.get(key)
        if flag is None:
            return f"SGLang has no kernel setting {key}"
        if flag not in facts.flags:
            return f"this SGLang has no {flag}"
        choices = facts.flags[flag].get("choices")
        if key not in _INTS and key not in _BOOLS and choices is not None and value not in choices:
            return f"this SGLang's {flag} does not take {value} (it takes {', '.join(str(c) for c in choices)})"
    if settings.get("tc") == "0" and "tcmax" in settings:
        return "tcmax sets a size for torch.compile, which tc=0 leaves off"
    return None


def sglang_args(settings: Dict[str, str]) -> List[str]:
    """The arguments `settings` add to SGLang's server, in key order. tc=0 adds nothing: the flag only turns
    torch.compile on, and it is off unless asked for."""
    args: List[str] = []
    for key in sorted(settings):
        flag, value = SGLANG_FLAGS[key], settings[key]
        if key in _BOOLS:
            if value == "1":
                args.append(flag)
        else:
            args += [flag, value]
    return args


# ---------------------------------------------------------------------------- what an engine refuses or changes
#
# Rules read in the engines' source, each for the releases named; outside them, or with a version that cannot be
# read, a rule says nothing: the setting is neither refused nor called the same as the default, it may be tried,
# and what the engine logged it changed (the trial's start-up record) decides. An error inside a rule is raised,
# never turned into a refusal.

VLLM_RULES = ((0, 29), (0, 33))  # the vLLM releases the rules below were read in, first and last
SGLANG_RULES = ((0, 5, 19), (0, 5, 23))

# vLLM: the KV-cache types each general attention backend lists (supported_kv_cache_dtypes), the same lists in
# v0.29.0 / v0.33.0: v1/attention/backends/flash_attn.py:83 / 290, flashinfer.py:410 / 573, triton_attn.py:294 / 326,
# flex_attention.py:96 both. FLASH_ATTN's fp8 types also need FA3 or FA4 on the card; that is left to the engine.
VLLM_ATTN_KV = {
    "FLASH_ATTN": ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3"),
    "FLASHINFER": ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "nvfp4", "nvfp4_4over6"),
    "TRITON_ATTN": ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "int4_per_token_head",
                    "int8_per_token_head", "fp8_per_token_head"),
    "FLEX_ATTENTION": ("auto", "float16", "bfloat16"),
}
# vLLM's CUDA-graph mode at its default optimisation level: FULL_AND_PIECEWISE at O2 and O3
# (config/vllm.py:301, 324 / 311, 332); "O3: Currently the same as -O2" (:121-122 / 143-144).
VLLM_DEFAULT_CG = "FULL_AND_PIECEWISE"
VLLM_DEFAULT_O = ("2", "3")


def release(version: Optional[str], parts: int) -> Optional[Tuple[int, ...]]:
    """The first `parts` numbers of an engine version, or None when it does not start with that many."""
    out: List[int] = []
    for piece in (version or "").split("."):
        digits = "".join(ch for ch in piece[:4] if ch.isdigit()) if piece[:1].isdigit() else ""
        if not digits or len(out) == parts:
            break
        out.append(int(digits))
    return tuple(out) if len(out) == parts else None


def _within(version: Optional[str], bounds: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Optional[Tuple[int, ...]]:
    rel = release(version, len(bounds[0]))
    return rel if rel is not None and bounds[0] <= rel <= bounds[1] else None


def vllm_refusal(settings: Dict[str, str], kv_dtype: str, version: Optional[str]) -> Optional[str]:
    """Why vLLM would refuse `settings` with this KV-cache type, where a rule read for its release says so."""
    if _within(version, VLLM_RULES) is None:
        return None
    attn = settings.get("attn")
    if attn in VLLM_ATTN_KV and kv_dtype not in VLLM_ATTN_KV[attn]:
        return (f"vLLM's {attn} attention does not take a {kv_dtype} KV cache "
                f"(it lists {', '.join(VLLM_ATTN_KV[attn])})")
    return None


def vllm_same_as_default(settings: Dict[str, str], version: Optional[str], attention: Optional[str] = None,
                         speculating: bool = False, capability: Optional[Tuple[int, int]] = None) -> Optional[str]:
    """Why every one of `settings` is what vLLM does anyway, naming each rule; None when any setting is not, or when
    no rule was read for this release. `attention` is the attention backend when the launch fixes one, `speculating`
    whether the config proposes more than one token a step, `capability` the card's. Only certain equivalences are claimed: cg=FULL on a backend without full
    support becomes FULL_AND_PIECEWISE or FULL_DECODE_ONLY depending on how the model is compiled
    (config/compilation.py:1393-1419 @0.33), which is not known here, so it is left to be tried."""
    if _within(version, VLLM_RULES) is None or not settings:
        return None
    level_is_default = settings.get("O", VLLM_DEFAULT_O[0]) in VLLM_DEFAULT_O
    why: List[str] = []
    for key, value in sorted(settings.items()):
        if key == "O" and value in VLLM_DEFAULT_O:
            why.append(f"O={value} is vLLM's default optimisation level (O3 is the same as O2)")
        elif key == "cg" and value == VLLM_DEFAULT_CG and level_is_default:
            why.append(f"cg={value} is vLLM's default CUDA-graph mode at that level")
        elif (key == "cg" and value == "PIECEWISE" and level_is_default and speculating
              and attention == "FLASHINFER" and capability is not None and capability < (10, 0)
              and release(version, 2) == (0, 29)):
            # 0.29 only: below SM100 FlashInfer supports full graphs for single-token decode alone
            # (v1/attention/backends/flashinfer.py:974-1015), so speculation is given PIECEWISE
            # (config/compilation.py:1448-1465). 0.33 keeps full graphs there (flashinfer.py:1187-1191).
            why.append("cg=PIECEWISE is what vLLM 0.29 already runs with FlashInfer under speculation on this card")
        else:
            return None
    return "; ".join(why)


def sglang_default_decode_cap(vram_mib: float, batch: int, tp: int, version: Optional[str]) -> Optional[int]:
    """The decode CUDA-graph batch cap SGLang picks for a card when none is given, or None where no rule was read.

    arg_groups/memory_hook.py, by the card's memory in MiB, for tp < 4 / tp >= 4:
      under 20 GiB 8; under 35 GiB 24 / 80 on 0.5.19 (:74-87) and 48 / 160 on 0.5.23 (:101-117); under 60 GiB
      32 / 160 (:88-101 / 118-131); under 160 GiB 256 / 512; above 512.
    0.5.23 then lowers a default cap to --max-running-requests (:182-188), which rigtune always passes as the
    batch; 0.5.19 has no such step."""
    rel = _within(version, SGLANG_RULES)
    if rel is None or vram_mib <= 0:
        return None
    wide = tp >= 4
    if vram_mib < 20 * 1024:
        cap = 8
    elif vram_mib < 35 * 1024:
        cap = (160 if wide else 48) if rel >= (0, 5, 23) else (80 if wide else 24)
    elif vram_mib < 60 * 1024:
        cap = 160 if wide else 32
    elif vram_mib < 160 * 1024:
        cap = 512 if wide else 256
    else:
        cap = 512
    return min(cap, batch) if rel >= (0, 5, 23) else cap


def sglang_same_as_default(settings: Dict[str, str], vram_mib: float, batch: int, tp: int,
                           version: Optional[str]) -> Optional[str]:
    """Why `settings` are what SGLang does anyway: a decode cap equal to the one it would pick for this card."""
    if set(settings) != {"cgmax"}:
        return None
    default = sglang_default_decode_cap(vram_mib, batch, tp, version)
    if default is not None and int(settings["cgmax"]) == default:
        return f"cgmax={default} is the decode cap SGLang picks for this card and batch"
    return None


SGLANG_GRAPH_MIB_PER_BATCH = 2  # SGLang's own reserve per unit of decode cap (arg_groups/memory_hook.py:292-303
#                                 @0.5.19 / 357-368 @0.5.23), which it adds only when it computes the memory fraction
#                                 itself; rigtune always passes the fraction, so rigtune must count it.


def sglang_graph_extra_bytes(settings: Dict[str, str], vram_mib: float, batch: int, tp: int, version: Optional[str],
                             second_model: bool = False) -> int:
    """Bytes of CUDA-graph memory a raised decode cap adds beyond what the card's default cap already takes: 2 MiB
    for each unit above it, twice that with a second model, whose runner captures graphs of its own (a prior: not
    measured). 0 without cgmax or at or below the default. Where the default is not known (a release no rule was
    read for) the whole cap is counted: a plan that is too careful, never one that is too bold."""
    if "cgmax" not in settings:
        return 0
    default = sglang_default_decode_cap(vram_mib, batch, tp, version) or 0
    above = max(0, int(settings["cgmax"]) - default)
    return above * SGLANG_GRAPH_MIB_PER_BATCH * (2 if second_model else 1) * 1024 * 1024
