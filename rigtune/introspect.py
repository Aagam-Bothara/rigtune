"""What a checkpoint says about itself: its own MTP head, quantization format, thinking switch and modalities.

Each feature is read from what the checkpoint ships (config.json, its chat template, its file listing and its
safetensors headers), never from its name, for the reason quantized.verified_bits gives: names are not evidence.
A feature the checkpoint does not declare, or one that could not be read, is None and describes itself as "none";
a file that is missing or cannot be fetched adds a note and never raises.

  * MTP head: a multi-token-prediction head a checkpoint ships beside the model. Whether an engine can load it is
    the engine's own question (rigtune.backends.own_head); this says whether the checkpoint declares one and holds
    its weights. Families declare it three ways, each read in config.json, in its text_config, and in a sub-config
    whose name has "mtp" in it:
      `mtp_num_hidden_layers` (Qwen3.5, Qwen3.6; vLLM v0.33.0 config/speculative.py hf_config_override line 998),
      `num_nextn_predict_layers` (DeepSeek, GLM, MiMo, Ernie and most others),
      `mtp_config.num_nextn_predict_layers` (Inkling; vLLM line 1088, SGLang configs/model_config.py line 1101).
    Qwen3-Next, LongCat-Flash and MiMo-V2-Flash declare no count and ship the tensors: the count is then taken from
    the tensors' layer indices, which the engine's own rule is asked about (rigtune.backends.own_head).
    And they store it two ways, read from the weights' names (the safetensors headers of 36 checkpoints, checked on
    the Hub on 20 November 2026):
      under a segment that starts with "mtp" or "nextn": Qwen3.5's `mtp.layers.0...`, Inkling-Small's
        `model.mtp.layers...`, MiMo-7B's `model.mtp_layers.0...`;
      as decoder layers past the last: DeepSeek-V3's `model.layers.61...` (61 layers), GLM-4.5-Air's
        `model.layers.46...` (46). Only under the prefix whose layer indices are the language model's own
        (0 to num_hidden_layers - 1, then at most the declared count) and which has its own embedding or final norm
        beside them, and not when a named head was found, so a vision or audio tower's `layers.N`, which can run
        past the language model's count, is not taken for a head. (An unnamed stack with its own norm and exactly
        the declared count past the language model's would be; none of the 36 checkpoints has one.)
    A declared head is reported only when the weights hold it: a config key without the tensors is a note. The
    weights are listed (their headers only) just when the config declares a head, so a model without one costs no
    request beyond its config; when the Hub cannot be asked, the headers of a fully cached snapshot in the local
    Hugging Face cache are read instead. The listing also gives the size of the target's embedding and output head,
    which Qwen3.5's draft first builds a copy of in both engines (MtpHead.shared_bytes;
    model_executor/models/qwen3_5_mtp.py Qwen3_5MTP.load_weights, srt/models/qwen3_5_mtp.py).
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
# A name segment that starts an MTP head's tensors: "mtp.", "model.mtp.", "model.mtp_layers.", "...nextn...".
MTP_SEGMENT = re.compile(r"(?:^|\.)(?:mtp|nextn)[^.]*\.")
_LAYER = re.compile(r"^(?:(.*)\.)?layers\.(\d+)\.")
NONE = "none"

# Bytes per element of each safetensors dtype (the format's own list); a tensor of another dtype has no size here.
SAFETENSORS_DTYPE_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2,
                           "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

# ModelOpt's quant_algo -> (weight bits, activation bits), where the name says them.
MODELOPT_BITS = {"FP8": (8, 8), "NVFP4": (4, 4)}


@dataclass(frozen=True)
class MtpHead:
    layers: int  # MTP layers the config declares
    config_key: str  # the key that declared them ("mtp_config.num_nextn_predict_layers" for one in a sub-config)
    tensors: int  # the head's tensors in the weights (head_tensors)
    bytes: Optional[int]  # their summed size; None when the listing gave names without sizes
    # The target's input embedding and output head in the listing (one tensor when they are tied). Both engines'
    # MTP drafts first build a copy of each and later swap in the target's (vLLM loads them too), so these bytes
    # are held twice for a while. None when the listing gave no size for them.
    shared_bytes: Optional[int] = None

    def describe(self) -> str:
        size = f", {self.bytes / 1e9:.2f} GB" if self.bytes is not None else ""
        return f"{self.layers} layer{'s' if self.layers != 1 else ''} ({self.config_key}), {self.tensors} tensors{size}"


# The precision label of a checkpoint served as its author quantized it ("ckpt-fp8", "ckpt-nvfp4"), apart from
# rigtune's own precisions: "fp8" is vLLM or SGLang quantizing 16-bit weights, "awq" a Hub copy rigtune found.
STORED_PREFIX = "ckpt-"


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

    @property
    def label(self) -> str:
        """STORED_PREFIX and a short name: ModelOpt's algorithm, a compressed-tensors scheme by its format or its
        bits ("w8a8", "w4a16": activations not quantized stay 16-bit), else the quant_method."""
        fmt = (self.fmt or "").lower()
        if self.method == "modelopt" and fmt:
            short = fmt
        elif self.method in ("compressed-tensors", "compressed_tensors"):
            named = next((n for n in ("nvfp4", "mxfp4", "mxfp8") if n in fmt), None)
            if named:
                short = named
            elif fmt.startswith("float") and self.weight_bits == 8:
                short = "fp8"
            elif self.weight_bits:
                short = f"w{self.weight_bits}a{self.activation_bits or 16}"
            else:
                short = "compressed-tensors"
        else:
            short = self.method
        return STORED_PREFIX + re.sub(r"[^a-z0-9_.-]+", "-", short.lower())


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


def _mtp_sections(config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """(prefix, section) where a head's layer count can be declared: the config, its text_config, and in either a
    sub-config whose name has "mtp" in it (Inkling's mtp_config)."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for section in _sections(config):
        out.append(("", section))
        out += [(f"{k}.", v) for k, v in section.items() if "mtp" in k.lower() and isinstance(v, dict)]
    return out


def mtp_layers(config: Optional[Dict[str, Any]]) -> Optional[Tuple[int, str]]:
    """(layers, key) of the MTP head a config declares, or None."""
    for prefix, section in _mtp_sections(config or {}):
        for key in MTP_LAYER_KEYS:
            value = section.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value, prefix + key
    return None


def _num_layers(config: Dict[str, Any]) -> Optional[int]:
    """The language model's decoder layers: text_config's num_hidden_layers first, as for every other text field."""
    for section in reversed(_sections(config)):
        value = section.get("num_hidden_layers")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def head_tensors(tensors: Dict[str, Optional[int]], config: Dict[str, Any], declared: int) -> List[str]:
    """The names of an MTP head's tensors in a listing. Those under an "mtp"/"nextn" segment (MTP_SEGMENT) when there
    are any; else decoder layers past the language model's last. The language model is the prefix whose layer indices
    run 0..n-1 with none missing and at most `declared` more after them, and which has its own embedding or final norm
    beside its layers (DeepSeek-V3's model.embed_tokens, model.norm): a vision or audio tower's `layers.N` can run
    past the language model's count, even by exactly one, and has neither."""
    named = sorted(n for n in tensors if MTP_SEGMENT.search(n))
    n = _num_layers(config)
    if named or n is None:
        return named
    layer_of: Dict[str, Tuple[str, int]] = {}  # name -> (prefix, layer index)
    for name in tensors:
        m = _LAYER.match(name)
        if m:
            layer_of[name] = (m.group(1) or "", int(m.group(2)))
    indices: Dict[str, set] = {}
    for prefix, i in layer_of.values():
        indices.setdefault(prefix, set()).add(i)
    found: List[str] = []
    for prefix, seen in indices.items():
        extra = {i for i in seen if i >= n}
        if extra and set(range(n)) <= seen and extra <= set(range(n, n + declared)) and _decoder_of(prefix, tensors):
            found += [name for name, (p, i) in layer_of.items() if p == prefix and i in extra]
    return sorted(found)


def _decoder_of(prefix: str, tensors: Dict[str, Optional[int]]) -> bool:
    """Whether `prefix` holds a language model: not a tower by its name, and an embedding or a final norm sits beside
    its layers."""
    lead = prefix + "." if prefix else ""
    return not _TOWER.search(lead) and any(f"{lead}{leaf}.weight" in tensors
                                           for leaf in ("embed_tokens", "embed", "embeddings", "norm"))


def mtp_head(config: Optional[Dict[str, Any]], tensors: Optional[Dict[str, Optional[int]]],
             name: str = "the checkpoint") -> Tuple[Optional[MtpHead], List[str]]:
    """The MTP head a config declares and its weights hold, with notes when only one of the two says so.
    `tensors` maps tensor names to bytes (None where unknown); None means the weights could not be listed."""
    declared = mtp_layers(config)
    notes: List[str] = []
    if declared is None:
        # Qwen3-Next, LongCat-Flash and MiMo-V2-Flash ship mtp.* tensors and declare no count (vLLM's rules for them
        # default or fix it); the depth is then counted from the tensors.
        named = sorted(n for n in tensors if MTP_SEGMENT.search(n)) if tensors is not None else []
        if not named:
            return None, []
        depths = {m.group(1) for n in named if (m := _HEAD_LAYER.search(n))}
        declared = (len(depths) or 1, "the tensors")
        notes.append(f"{name} declares no MTP layers but ships {len(named)} MTP-named tensors; counted "
                     f"{declared[0]} layer(s) from them")
    layers, key = declared
    if tensors is None:
        return None, [f"{name} declares {layers} MTP layer(s) ({key}) but its weights could not be listed; no MTP "
                      "head is reported"]
    found = head_tensors(tensors, config or {}, layers)
    if not found:
        return None, [f"{name} declares {layers} MTP layer(s) ({key}) but its weights hold no MTP tensors (none "
                      "named mtp or nextn, and no decoder layer past the last); no MTP head is reported"]
    head = set(found)
    return MtpHead(layers=layers, config_key=key, tensors=len(found), bytes=_total(tensors[n] for n in found),
                   shared_bytes=_shared_bytes(tensors, head)), notes


# The target's input embedding and output head, as families name them: model.embed_tokens (Qwen, Llama, DeepSeek),
# model.language_model.embed_tokens (multimodal Qwen), model.llm.embed and model.llm.unembed (Inkling),
# backbone.embeddings (Nemotron-H), model.word_embeddings (Ling), lm_head. A tower's are not the language model's.
_EMBEDDING = re.compile(r"(?:^|\.)(?:embed_tokens|embed|embeddings|word_embeddings|wte)\.weight$")
_OUTPUT = re.compile(r"(?:^|\.)(?:lm_head|unembed)\.weight$")
_TOWER = re.compile(r"(?:^|\.)(?:visual|vision\w*|audio\w*|speech\w*)\.")
# A head's own copy of the output head (GLM-4.5's and DeepSeek's model.layers.<n>.shared_head.head, Step-3.5's
# .transformer.shared_head.output); its own embedding copy matches _EMBEDDING (model.layers.<n>.embed_tokens,
# LongCat's model.mtp.embed_tokens). GLM-4.6's layer 92 has only shared_head.norm, which is neither.
_OWN_OUTPUT = re.compile(r"(?:^|\.)(?:shared_head\.head|shared_head\.output|lm_head|unembed)\.weight$")
# A head's layer index: mtp.layers.<i>, model.mtp.layers.<i>, model.mtp_layers.<i>, ...nextn...layers.<i>.
_HEAD_LAYER = re.compile(r"(?:mtp|nextn)(?:[^.]*\.(?:.*\.)?layers?|_layers?)\.(\d+)\.")


def _shared_bytes(tensors: Dict[str, Optional[int]], head: set) -> Optional[int]:
    """The target's embedding and output head that a draft builds a copy of: each one the head does not bring its own
    of. 0 when it brings both; None when a size is unknown or the target's embedding is not found. A tied checkpoint
    has no separate output head to add."""
    own_embedding = any(_EMBEDDING.search(n) for n in head)
    own_output = any(_OWN_OUTPUT.search(n) for n in head)
    target = [n for n in tensors if n not in head and not _TOWER.search(n)]
    embedding = [] if own_embedding else [n for n in target if _EMBEDDING.search(n)]
    output = [] if own_output else [n for n in target if _OUTPUT.search(n)]
    if not own_embedding and not embedding:
        return None
    return _total(tensors[n] for n in embedding + output) if embedding or output else 0


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


def _default_index_names(spec: ModelSpec, token: Optional[str] = None) -> List[str]:
    """The tensor names model.safetensors.index.json maps to files: one small file, read before the headers it
    points to. Empty for a checkpoint without one (a single safetensors file)."""
    text = _default_text(spec, "model.safetensors.index.json", token)
    return list((json.loads(text).get("weight_map") or {}).keys()) if text else []


def inspect_model(
    spec: ModelSpec,
    features: Sequence[str] = FEATURES,
    fetch_config: Optional[Callable[[ModelSpec], Dict[str, Any]]] = None,
    fetch_text: Optional[Callable[[ModelSpec, str], Optional[str]]] = None,
    list_files: Optional[Callable[[ModelSpec], List[str]]] = None,
    list_tensors: Optional[Callable[[ModelSpec], Dict[str, Optional[int]]]] = None,
    list_index: Optional[Callable[[ModelSpec], List[str]]] = None,
    token: Optional[str] = None,
) -> ModelFacts:
    """What `spec`'s checkpoint declares, for the `features` asked for; only what those need is fetched. Every
    fetcher can be replaced (tests never touch the network). A fetch that fails is a note, never an exception.

    For the MTP head the safetensors headers are listed when the config declares one, and when it declares none but
    the weight index names MTP tensors (Qwen3-Next, LongCat, MiMo-V2): one small file for a model without a head."""
    get_config = fetch_config or (lambda s: _default_config(s, token))
    get_text = fetch_text or (lambda s, f: _default_text(s, f, token))
    get_files = list_files or (lambda s: _default_files(s, token))
    get_tensors = list_tensors or (lambda s: _default_tensors(s, token))
    get_index = list_index or (lambda s: _default_index_names(s, token))
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
        if mtp_layers(config):
            tensors = attempt("the safetensors headers", lambda: get_tensors(spec))
        else:
            index = attempt("model.safetensors.index.json", lambda: get_index(spec)) if config is not None else None
            named = bool(index) and any(MTP_SEGMENT.search(n) for n in index)
            tensors = attempt("the safetensors headers", lambda: get_tensors(spec)) if named else {}
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
