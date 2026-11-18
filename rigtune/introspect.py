"""What a checkpoint says about itself: its own MTP head, quantization format, thinking switch and modalities.

Each feature is read from what the checkpoint ships (config.json, its chat template, its file listing and its
safetensors headers), never from its name, for the reason quantized.verified_bits gives: names are not evidence.
A feature the checkpoint does not declare, or one that could not be read, is None and describes itself as "none";
a file that is missing or cannot be fetched adds a note and never raises.

  * MTP head. Qwen3.5-family checkpoints carry a multi-token-prediction head beside the model: config.json declares
    its layers (`mtp_num_hidden_layers`, in `text_config` for the multimodal ones) and the weights hold them under
    `mtp.`. vLLM reads that key for Qwen3.5 (v0.29.0 config/speculative.py SpeculativeConfig.hf_config_override,
    line 893; v0.33.0 line 998, falling back to text_config) and loads the `mtp.` tensors into its draft, besides
    its own copy of the embedding and output head, which it then drops for the target's
    (model_executor/models/qwen3_5_mtp.py Qwen3_5MTP.load_weights, both tags); SGLang loads only names containing
    "mtp" and strips the `mtp.` prefix (srt/models/qwen3_5_mtp.py Qwen3_5ForCausalLMMTP.load_weights, both tags).
    `num_nextn_predict_layers`, the key DeepSeek-style heads use, is read too. A declared head is reported only when
    the weights hold `mtp.` tensors: a config key without them is a note. The weights are listed (their headers
    only) just when the config declares a head, so a model without one costs no request beyond its config; when the
    Hub cannot be asked, the headers of a fully cached snapshot in the local Hugging Face cache are read instead.
    The listing also gives the size of the target's embedding and output head, which each engine's draft first
    builds a copy of (MtpHead.shared_bytes).
  * Quantization: what `quantization_config` (config.json, or its text_config) or ModelOpt's hf_quant_config.json
    declares: method, format, weight and activation bits, KV-cache scheme.
  * Thinking switch: an `enable_thinking` variable in the chat template, and what the template does without it.
    Qwen3's template thinks unless it is false ("enable_thinking is defined and enable_thinking is false"), and so
    does the Qwen/Qwen3.5-9B checkpoint's own chat_template.jinja (revision c2022362, line 149): served with its
    default, it thinks. vLLM v0.33.0's test copy rust/src/chat/tests/templates/qwen35.jinja (line 149) is the
    other kind, thinking only when it is true ("... is true"), so a template's own lines decide, never its family
    name. Other switches are not read.
  * Modalities: "text", plus "image" for a vision_config, "audio" for an audio_config, and "video" for a
    video_config or a video_preprocessor_config.json in the listing.

Only the MTP head is used so far (speculative.find_mtp_head); the rest is recorded for later.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rigtune.models import ModelSpec

logger = logging.getLogger(__name__)

FEATURES: Tuple[str, ...] = ("mtp", "quantization", "thinking", "modalities")
MTP_LAYER_KEYS: Tuple[str, ...] = ("mtp_num_hidden_layers", "num_nextn_predict_layers")
MTP_PREFIX = "mtp."
NONE = "none"

# Bytes per element of each safetensors dtype (the format's own list); a tensor of another dtype has no size here.
SAFETENSORS_DTYPE_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2,
                           "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

# ModelOpt's quant_algo -> (weight bits, activation bits), where the name says them.
MODELOPT_BITS = {"FP8": (8, 8), "NVFP4": (4, 4)}


@dataclass(frozen=True)
class MtpHead:
    layers: int  # MTP layers the config declares
    config_key: str  # the key that declared them
    tensors: int  # tensors under "mtp." in the weights
    bytes: Optional[int]  # their summed size; None when the listing gave names without sizes
    # The target's input embedding and output head in the listing (one tensor when they are tied). Both engines'
    # MTP drafts first build a copy of each and later swap in the target's (vLLM loads them too), so these bytes
    # are held twice for a while. None when the listing gave no size for them.
    shared_bytes: Optional[int] = None

    def describe(self) -> str:
        size = f", {self.bytes / 1e9:.2f} GB" if self.bytes is not None else ""
        return f"{self.layers} layer{'s' if self.layers != 1 else ''} ({self.config_key}), {self.tensors} tensors{size}"


@dataclass(frozen=True)
class QuantFacts:
    method: str  # quant_method ("fp8", "awq", "gptq", "compressed-tensors", ...) or "modelopt"
    fmt: Optional[str] = None  # format / fmt, or ModelOpt's quant_algo
    weight_bits: Optional[int] = None
    activation_bits: Optional[int] = None
    kv_scheme: Optional[str] = None
    source: str = "config.json"

    def describe(self) -> str:
        parts = [self.method + (f" ({self.fmt})" if self.fmt else "")]
        if self.weight_bits:
            parts.append(f"{self.weight_bits}-bit weights")
        if self.activation_bits:
            parts.append(f"{self.activation_bits}-bit activations")
        if self.kv_scheme:
            parts.append(f"KV cache {self.kv_scheme}")
        return ", ".join(parts)


@dataclass(frozen=True)
class ThinkingSwitch:
    variable: str  # the chat-template variable that switches thinking
    default: str  # what the template does when it is unset: "on", "off" or "unknown"

    def describe(self) -> str:
        return f"{self.variable}, {self.default} by default" if self.default != "unknown" else self.variable


@dataclass
class ModelFacts:
    mtp: Optional[MtpHead] = None
    quantization: Optional[QuantFacts] = None
    thinking: Optional[ThinkingSwitch] = None
    modalities: Tuple[str, ...] = ("text",)
    notes: List[str] = field(default_factory=list)

    def describe(self) -> Dict[str, str]:
        """One line per feature; "none" for one the checkpoint does not declare or that could not be read."""
        return {"mtp": self.mtp.describe() if self.mtp else NONE,
                "quantization": self.quantization.describe() if self.quantization else NONE,
                "thinking": self.thinking.describe() if self.thinking else NONE,
                "modalities": ", ".join(self.modalities)}


# ---------------------------------------------------------------- the rules, on what was fetched


def _sections(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The config itself, then its text_config (multimodal checkpoints keep the language model's fields there)."""
    text = config.get("text_config")
    return [config] + ([text] if isinstance(text, dict) else [])


def mtp_layers(config: Optional[Dict[str, Any]]) -> Optional[Tuple[int, str]]:
    """(layers, key) of the MTP head a config declares, or None."""
    for section in _sections(config or {}):
        for key in MTP_LAYER_KEYS:
            value = section.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value, key
    return None


def mtp_head(config: Optional[Dict[str, Any]], tensors: Optional[Dict[str, Optional[int]]],
             name: str = "the checkpoint") -> Tuple[Optional[MtpHead], List[str]]:
    """The MTP head a config declares and its weights hold, with notes when only one of the two says so.
    `tensors` maps tensor names to bytes (None where unknown); None means the weights could not be listed."""
    declared = mtp_layers(config)
    found = sorted(n for n in tensors if n.startswith(MTP_PREFIX)) if tensors is not None else None
    if declared is None:
        notes = [f"{name} has {len(found)} {MTP_PREFIX}* tensors but its config declares no MTP layers; no MTP "
                 "head is reported"] if found else []
        return None, notes
    layers, key = declared
    if found is None:
        return None, [f"{name} declares {layers} MTP layer(s) ({key}) but its weights could not be listed; no MTP "
                      "head is reported"]
    if not found:
        return None, [f"{name} declares {layers} MTP layer(s) ({key}) but its weights hold no {MTP_PREFIX}* "
                      "tensors; no MTP head is reported"]
    return MtpHead(layers=layers, config_key=key, tensors=len(found), bytes=_total(tensors[n] for n in found),
                   shared_bytes=_total(tensors[n] for n in tensors if _shared(n))), []


def _shared(name: str) -> bool:
    """The target's input embedding or output head (model.embed_tokens, model.language_model.embed_tokens,
    lm_head), not a copy under mtp."""
    return not name.startswith(MTP_PREFIX) and (name.endswith("embed_tokens.weight") or name == "lm_head.weight")


def _total(sizes) -> Optional[int]:
    sizes = list(sizes)
    return sum(sizes) if sizes and all(s is not None for s in sizes) else None


def _int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def _group_bits(groups: Any, part: str) -> Optional[int]:
    """The bit width every compressed-tensors config group gives `part` ("weights", "input_activations"), when they
    agree; None otherwise."""
    if not isinstance(groups, dict) or not groups:
        return None
    bits = {_int(g.get(part, {}).get("num_bits")) if isinstance(g, dict) and isinstance(g.get(part), dict)
            else None for g in groups.values()}
    return bits.pop() if len(bits) == 1 else None


def quantization(config: Optional[Dict[str, Any]], hf_quant: Optional[Dict[str, Any]] = None) -> Optional[QuantFacts]:
    """What the checkpoint declares about its quantization, or None for an unquantized one."""
    for section in _sections(config or {}):
        q = section.get("quantization_config")
        if isinstance(q, dict) and q:
            method = str(q.get("quant_method") or "unknown").lower()
            fmt = q.get("format") or q.get("fmt") or q.get("quant_algo")  # quant_algo: ModelOpt in config.json
            weight_bits = _int(q.get("bits") or q.get("w_bit")) or _group_bits(q.get("config_groups"), "weights")
            if weight_bits is None and method == "fp8":
                weight_bits = 8
            kv = q.get("kv_cache_scheme")
            kv_scheme = None
            if isinstance(kv, dict) and kv:
                kv_scheme = f"{kv.get('type', '')}{kv.get('num_bits', '')}" or "declared"
            return QuantFacts(method=method, fmt=str(fmt) if fmt else None, weight_bits=weight_bits,
                              activation_bits=_group_bits(q.get("config_groups"), "input_activations"),
                              kv_scheme=kv_scheme)
    hq = (hf_quant or {}).get("quantization")
    if isinstance(hq, dict) and hq.get("quant_algo"):
        algo = str(hq["quant_algo"])
        wb, ab = MODELOPT_BITS.get(algo.upper(), (None, None))
        kv = hq.get("kv_cache_quant_algo")
        return QuantFacts(method="modelopt", fmt=algo, weight_bits=wb, activation_bits=ab,
                          kv_scheme=str(kv) if kv else None, source="hf_quant_config.json")
    return None


_THINKS_UNLESS_FALSE = (r"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+false",
                        r"enable_thinking\s+is\s+defined\s+and\s+not\s+enable_thinking",
                        r"enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+true")
_THINKS_ONLY_IF_TRUE = (r"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+true",
                        r"enable_thinking\s+if\s+enable_thinking\s+is\s+defined\s+else\s+false")


def thinking_switch(template: Optional[str]) -> Optional[ThinkingSwitch]:
    """The template's thinking switch, or None when it has none this reads (only `enable_thinking`)."""
    if not template or "enable_thinking" not in template:
        return None
    on = any(re.search(p, template, re.IGNORECASE) for p in _THINKS_UNLESS_FALSE)
    off = any(re.search(p, template, re.IGNORECASE) for p in _THINKS_ONLY_IF_TRUE)
    return ThinkingSwitch("enable_thinking", "on" if on and not off else "off" if off and not on else "unknown")


def chat_template(tokenizer_config: Optional[Dict[str, Any]], template_file: Optional[str]) -> Optional[str]:
    """The chat template: chat_template.jinja where the checkpoint has one, else tokenizer_config.json's
    chat_template (a string, or a list of named templates of which "default" is the one used)."""
    if template_file:
        return template_file
    t = (tokenizer_config or {}).get("chat_template")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        named = {x.get("name"): x.get("template") for x in t if isinstance(x, dict)}
        found = named.get("default")
        return found if isinstance(found, str) else None
    return None


def modalities(config: Optional[Dict[str, Any]], files: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """"text" plus what the config's sub-configs and the processor files in the listing show."""
    cfg = config or {}
    names = {os.path.basename(f) for f in (files or ())}
    out = ["text"]
    if isinstance(cfg.get("vision_config"), dict):
        out.append("image")
    if isinstance(cfg.get("audio_config"), dict):
        out.append("audio")
    if isinstance(cfg.get("video_config"), dict) or "video_preprocessor_config.json" in names:
        out.append("video")
    return tuple(out)


# ---------------------------------------------------------------- fetching, injectable


def _local(spec: ModelSpec) -> Optional[str]:
    return spec.hf_id if os.path.isdir(spec.hf_id) else None


def _default_config(spec: ModelSpec, token: Optional[str] = None) -> Dict[str, Any]:
    from rigtune.hfconfig import fetch_config

    return fetch_config(spec, token=token)


def _default_text(spec: ModelSpec, filename: str, token: Optional[str] = None) -> Optional[str]:
    """A file's text, or None when the checkpoint has no such file. Raises when the Hub cannot be asked."""
    local = _local(spec)
    if local is not None:
        path = os.path.join(local, filename)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(spec.hf_id, filename, revision=spec.revision, token=token)
    except Exception as exc:
        if "EntryNotFound" in type(exc).__name__:  # the repository exists and has no such file
            return None
        raise
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _default_files(spec: ModelSpec, token: Optional[str] = None) -> List[str]:
    local = _local(spec)
    if local is not None:
        return sorted(os.listdir(local))
    from huggingface_hub import HfApi

    return sorted(HfApi(token=token).list_repo_files(spec.hf_id, revision=spec.revision))


def safetensors_header_tensors(path: str) -> Dict[str, Optional[int]]:
    """Tensor name -> bytes from one local safetensors file's header (an 8-byte little-endian length, then JSON)."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n).decode("utf-8"))
    return {name: _tensor_bytes(info.get("dtype"), info.get("shape"), info.get("data_offsets"))
            for name, info in header.items() if name != "__metadata__" and isinstance(info, dict)}


def _tensor_bytes(dtype: Any, shape: Any, offsets: Any) -> Optional[int]:
    if isinstance(offsets, (list, tuple)) and len(offsets) == 2:
        return int(offsets[1]) - int(offsets[0])
    size = SAFETENSORS_DTYPE_BYTES.get(str(dtype))
    if size is None or not isinstance(shape, (list, tuple)):
        return None
    count = 1
    for d in shape:
        count *= int(d)
    return count * size


def _default_tensors(spec: ModelSpec, token: Optional[str] = None) -> Dict[str, Optional[int]]:
    """Tensor name -> bytes over every safetensors file: read from the headers alone (huggingface_hub fetches each
    header with a range request; no weights are downloaded)."""
    local = _local(spec)
    if local is not None:
        out: Dict[str, Optional[int]] = {}
        for name in sorted(os.listdir(local)):
            if name.endswith(".safetensors"):
                out.update(safetensors_header_tensors(os.path.join(local, name)))
        return out
    from huggingface_hub import HfApi

    try:
        meta = HfApi(token=token).get_safetensors_metadata(spec.hf_id, revision=spec.revision)
    except Exception as exc:
        cached = cached_tensors(spec)
        if cached is None:
            raise
        logger.info("%s: the Hub could not be asked (%s); read the cached snapshot's safetensors headers",
                    spec.hf_id, exc)
        return cached
    out = {}
    for file_meta in meta.files_metadata.values():
        for name, info in file_meta.tensors.items():
            out[name] = _tensor_bytes(getattr(info, "dtype", None), getattr(info, "shape", None),
                                      getattr(info, "data_offsets", None))
    return out


def cached_tensors(spec: ModelSpec) -> Optional[Dict[str, Optional[int]]]:
    """Tensor name -> bytes from the safetensors files of the model's snapshot in the local Hugging Face cache, for
    when the Hub cannot be asked; None unless every file the checkpoint lists is cached. The files come from
    model.safetensors.index.json's weight_map, or are the single model.safetensors."""
    from huggingface_hub import try_to_load_from_cache

    def cached(filename: str) -> Optional[str]:
        path = try_to_load_from_cache(spec.hf_id, filename, revision=spec.revision)
        return path if isinstance(path, str) and os.path.isfile(path) else None

    index = cached("model.safetensors.index.json")
    if index is not None:
        with open(index, "r", encoding="utf-8") as fh:
            files = sorted(set((json.load(fh).get("weight_map") or {}).values()))
    else:
        files = ["model.safetensors"]
    paths = [cached(f) for f in files]
    if not files or any(p is None for p in paths):
        return None
    out: Dict[str, Optional[int]] = {}
    for p in paths:
        out.update(safetensors_header_tensors(p))
    return out


def inspect_model(
    spec: ModelSpec,
    features: Sequence[str] = FEATURES,
    fetch_config: Optional[Callable[[ModelSpec], Dict[str, Any]]] = None,
    fetch_text: Optional[Callable[[ModelSpec, str], Optional[str]]] = None,
    list_files: Optional[Callable[[ModelSpec], List[str]]] = None,
    list_tensors: Optional[Callable[[ModelSpec], Dict[str, Optional[int]]]] = None,
    token: Optional[str] = None,
) -> ModelFacts:
    """What `spec`'s checkpoint declares, for the `features` asked for; only what those need is fetched. Every
    fetcher can be replaced (tests never touch the network). A fetch that fails is a note, never an exception."""
    get_config = fetch_config or (lambda s: _default_config(s, token))
    get_text = fetch_text or (lambda s, f: _default_text(s, f, token))
    get_files = list_files or (lambda s: _default_files(s, token))
    get_tensors = list_tensors or (lambda s: _default_tensors(s, token))
    facts = ModelFacts()
    name = spec.hf_id

    def attempt(what: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:
            facts.notes.append(f"{name}: could not read {what} ({exc})")
            return None

    wanted = set(features)
    config = attempt("config.json", lambda: get_config(spec)) if wanted else None
    if "mtp" in wanted:
        tensors = attempt("the safetensors headers", lambda: get_tensors(spec)) if mtp_layers(config) else {}
        facts.mtp, notes = mtp_head(config, tensors, name)
        facts.notes += notes
    if "quantization" in wanted:
        hf_quant_text = attempt("hf_quant_config.json", lambda: get_text(spec, "hf_quant_config.json"))
        try:
            hf_quant = json.loads(hf_quant_text) if hf_quant_text else None
        except ValueError as exc:
            facts.notes.append(f"{name}: hf_quant_config.json is not JSON ({exc})")
            hf_quant = None
        facts.quantization = quantization(config, hf_quant)
    if "thinking" in wanted:
        template_file = attempt("chat_template.jinja", lambda: get_text(spec, "chat_template.jinja"))
        tokenizer_config = None
        if not template_file:
            text = attempt("tokenizer_config.json", lambda: get_text(spec, "tokenizer_config.json"))
            try:
                tokenizer_config = json.loads(text) if text else None
            except ValueError as exc:
                facts.notes.append(f"{name}: tokenizer_config.json is not JSON ({exc})")
        template = chat_template(tokenizer_config, template_file)
        if template is None:
            facts.notes.append(f"{name}: no chat template found; no thinking switch is reported")
        facts.thinking = thinking_switch(template)
    if "modalities" in wanted:
        files = attempt("the file listing", lambda: get_files(spec))
        facts.modalities = modalities(config, files)
    return facts
