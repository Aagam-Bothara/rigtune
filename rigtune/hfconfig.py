"""Read what the memory planner needs from a Hugging Face model config."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from rigtune.models import ArchInfo, ModelSpec

logger = logging.getLogger(__name__)

_COMMIT = re.compile(r"[0-9a-f]{40}")

DTYPE_BYTES: Dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "auto": 2,
}


def dtype_bytes(name: str) -> int:
    return DTYPE_BYTES.get(name.lower(), 2)


# Bytes per KV element for llama.cpp's block-quantized caches: 32 values plus their scale(s). vLLM's
# int8_per_token_head keeps a scale per token and head beside one byte per value; sized here as a 4-byte
# scale over a 64-wide head, which overstates it for wider heads (Qwen2.5's are 128).
KV_ELEMENT_BYTES: Dict[str, float] = {"q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32,
                                      "q5_1": 24 / 32, "int8_per_token_head": 1 + 4 / 64}


def kv_element_bytes(name: str) -> float:
    return KV_ELEMENT_BYTES.get(name.lower(), float(dtype_bytes(name)))


def _first(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


# The keys SGLang reads a model's context length from, first present wins (utils/hf_transformers/common.py
# CONTEXT_LENGTH_KEYS and get_context_length: lines 558-584 at v0.5.19, 668-694 at v0.5.23, identical).
SGLANG_CONTEXT_KEYS = ("max_sequence_length", "seq_length", "max_seq_len", "model_max_length",
                       "max_position_embeddings")


def sglang_context_length(cfg: Dict[str, Any]) -> int:
    """The context length SGLang derives from a (text) config: the first of SGLANG_CONTEXT_KEYS it has, times the
    rope scaling factor unless the scaling names original_max_position_embeddings or is llama3's, else 2048. It can be
    smaller than max_position_embeddings: Qwen/Qwen-7B-Chat declares seq_length 8192 beside 32768. SGLang reads the
    transformers config object, whose class may default a key the file leaves out; read from the file, a missing key
    gives 2048 here, never more than SGLang would see."""
    rope = cfg.get("rope_scaling")
    factor = 1
    if isinstance(rope, dict) and rope:
        factor = rope.get("factor", 1)
        if (not isinstance(factor, (int, float)) or "original_max_position_embeddings" in rope
                or rope.get("rope_type") == "llama3"):
            factor = 1
    for key in SGLANG_CONTEXT_KEYS:
        value = cfg.get(key)
        if value is not None:
            return int(factor * value)
    return 2048


# The keys vLLM derives a model's maximum length from, the smallest present wins, and model_max_length then replaces
# it wherever it is set (transformers_utils/model_arch_config_convertor.py derive_max_model_len_and_key, lines
# 374-407 at v0.33.0). Inkling 2's config sets only model_max_length (1,048,576), which vLLM's own config class also
# copies to max_position_embeddings (models/inkling2/configs.py line 88 on main).
VLLM_CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "model_max_length",
                     "max_target_positions", "max_sequence_length", "max_seq_length", "seq_len")


def vllm_context_length(cfg: Dict[str, Any]) -> int:
    """The maximum length vLLM derives from a (text) config before rope scaling: the smallest of VLLM_CONTEXT_KEYS
    present, replaced by model_max_length where it is set. A config with none of them falls back to n_ctx, else
    4096, as rigtune did before (vLLM itself would stop and ask for --max-model-len)."""
    found = [int(cfg[k]) for k in VLLM_CONTEXT_KEYS if isinstance(cfg.get(k), (int, float))]
    if isinstance(cfg.get("model_max_length"), (int, float)) and cfg["model_max_length"]:
        return int(cfg["model_max_length"])
    if found:
        return min(found)
    return int(_first(cfg, "n_ctx", default=4096))


def arch_from_config(cfg: Dict[str, Any]) -> ArchInfo:
    """Build ArchInfo from a raw config.json dict. Handles nested text_config (multimodal)."""
    from rigtune.introspect import mtp_layers, quantization

    mm = any(isinstance(cfg.get(k), dict) for k in ("vision_config", "audio_config"))
    mtp = mtp_layers(cfg)
    quant = quantization(cfg)
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        merged = dict(cfg)
        merged.update(cfg["text_config"])
        cfg = merged
    hidden = int(_first(cfg, "hidden_size", "n_embd", "d_model", default=4096))
    layers = int(_first(cfg, "num_hidden_layers", "n_layer", "num_layers", default=32))
    heads = int(_first(cfg, "num_attention_heads", "n_head", default=32))
    kv_heads = int(_first(cfg, "num_key_value_heads", "num_kv_heads", default=heads))
    head_dim = int(_first(cfg, "head_dim", default=hidden // max(heads, 1)))
    vocab = int(_first(cfg, "vocab_size", default=32000))
    max_pos = vllm_context_length(cfg)
    dtype = str(_first(cfg, "torch_dtype", "dtype", default="bfloat16"))
    archs = cfg.get("architectures") or ["unknown"]
    intermediate = int(_first(cfg, "intermediate_size", "n_inner", default=4 * hidden))
    tie = bool(cfg.get("tie_word_embeddings", False))
    params = estimate_params(hidden, layers, heads, kv_heads, head_dim, vocab, intermediate, tie)
    return ArchInfo(
        architecture=str(archs[0]),
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        sglang_context=sglang_context_length(cfg),
        num_params=params,
        torch_dtype=dtype,
        has_mm_config=mm,
        mtp_layers=mtp[0] if mtp else 0,
        checkpoint_quant=quant.describe() if quant else None,
        stored_as=quant.label if quant else None,
        **hybrid_layout(cfg, layers, dtype),
    )


# Layer-type names as transformers writes them (configuration_utils.ALLOWED_ATTN_LAYER_TYPES), with the two
# legacy names it remaps (_LEGACY_LAYER_TYPE_REMAP: "attention" -> "full_attention"). "mamba" is remapped to
# linear_attention there too, but its state is shaped differently from Gated DeltaNet's, so it stays unknown.
_FULL_TYPES = {"full_attention", "attention"}
_SLIDING_TYPES = {"sliding_attention"}
_LINEAR_TYPES = {"linear_attention"}
# Gated DeltaNet fields (Qwen3.5, Qwen3-Next); both engines size the recurrent state from exactly these.
_GDN_KEYS = ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
             "linear_conv_kernel_dim")
_warned: set = set()


def _layer_types(cfg: Dict[str, Any], layers: int) -> Optional[list]:
    """The per-layer attention types: `layer_types` itself, else derived the way transformers derives it."""
    lt = cfg.get("layer_types")
    if isinstance(lt, list):
        return [str(t) for t in lt]
    # transformers Qwen3_5TextConfig.__post_init__: layer i is full when (i + 1) % full_attention_interval == 0.
    interval = cfg.get("full_attention_interval")
    if isinstance(interval, int) and interval > 0 and all(cfg.get(k) for k in _GDN_KEYS):
        return ["linear_attention" if (i + 1) % interval else "full_attention" for i in range(layers)]
    # transformers Gemma3TextConfig.__init__: layer i is sliding unless (i + 1) % sliding_window_pattern == 0.
    # Gemma 3 configs carry `sliding_window_pattern` (google/gemma-3-12b-it: 6, window 1024). Gemma 2 configs
    # carry no pattern (transformers hard-codes alternation), so they keep the every-layer-full estimate.
    pattern = cfg.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 0 and cfg.get("sliding_window"):
        return ["sliding_attention" if (i + 1) % pattern else "full_attention" for i in range(layers)]
    return None


def gdn_state_bytes(cfg: Dict[str, Any], model_dtype: str) -> int:
    """Recurrent state of one Gated DeltaNet layer for one sequence (tensor parallel 1).

    vLLM 0.29 mamba_utils.MambaStateShapeCalculator.gated_delta_net_state_shape: a conv state of
    (2 * key_heads * key_dim + value_heads * value_dim) x (conv_kernel - 1) and a temporal state of
    value_heads x value_dim x key_dim. SGLang 0.5.19 configs/mamba_utils.BaseLinearStateParams.mamba_cache_per_req
    sums the same two tensors. Conv state: model dtype in vLLM (mamba_cache_dtype "auto",
    MambaStateDtypeCalculator._mamba_state_dtype), bfloat16 in SGLang (SGLANG_MAMBA_CONV_DTYPE). Temporal state:
    the config's `mamba_ssm_dtype` in both (vLLM Qwen3_5ForConditionalGenerationConfig.verify_and_update_config,
    SGLang mamba2_state_dtype). Without that field the engines disagree (vLLM: model dtype, SGLang: float32);
    float32, the larger, is used.
    """
    nv = int(cfg["linear_num_value_heads"])
    dk, dv = int(cfg["linear_key_head_dim"]), int(cfg["linear_value_head_dim"])
    kernel = int(cfg["linear_conv_kernel_dim"])
    conv = (kernel - 1) * gdn_conv_step_bytes(cfg, model_dtype)
    temporal = nv * dv * dk * dtype_bytes(str(cfg.get("mamba_ssm_dtype") or "float32"))
    return conv + temporal


def gdn_conv_step_bytes(cfg: Dict[str, Any], model_dtype: str) -> int:
    """One column of a Gated DeltaNet layer's conv state: what each speculative token adds. vLLM's
    gated_delta_net_state_shape sizes the conv state conv_dim x (conv_kernel - 1 + num_spec), and
    Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config passes num_speculative_tokens as num_spec."""
    conv_dim = 2 * int(cfg["linear_num_key_heads"]) * int(cfg["linear_key_head_dim"])         + int(cfg["linear_num_value_heads"]) * int(cfg["linear_value_head_dim"])
    return conv_dim * dtype_bytes(model_dtype)


def _unrecognised(cfg: Dict[str, Any], note: str) -> Dict[str, Any]:
    arch = (cfg.get("architectures") or ["unknown"])[0]
    if (arch, note) not in _warned:
        _warned.add((arch, note))
        logger.warning("%s: %s; planning KV as if every layer kept a full cache", arch, note)
    return {"layout_note": note}


def hybrid_layout(cfg: Dict[str, Any], layers: int, model_dtype: str) -> Dict[str, Any]:
    """ArchInfo fields for a hybrid-attention layout; {} for a dense model (every layer full attention).

    A layout is recognised only when every layer is full, sliding (with a window) or Gated DeltaNet linear
    attention. Anything else returns just a `layout_note`, so the planner keeps today's estimate.
    """
    types = _layer_types(cfg, layers)
    if types is None or all(t in _FULL_TYPES for t in types):
        return {}
    if len(types) != layers:
        return _unrecognised(cfg, f"layer_types lists {len(types)} layers for {layers}")
    unknown = sorted(set(types) - _FULL_TYPES - _SLIDING_TYPES - _LINEAR_TYPES)
    if unknown:
        return _unrecognised(cfg, f"unrecognised layer types {', '.join(unknown)}")
    if int(cfg.get("num_kv_shared_layers") or 0) > 0:
        # Gemma 4's KV-shared layers reuse an earlier layer's cache (vLLM gemma4.Gemma4Attention
        # kv_sharing_target_layer_name); not modelled.
        return _unrecognised(cfg, "KV-shared layers (num_kv_shared_layers) are not modelled")
    full = sum(t in _FULL_TYPES for t in types)
    sliding = sum(t in _SLIDING_TYPES for t in types)
    linear = sum(t in _LINEAR_TYPES for t in types)
    out: Dict[str, Any] = {"full_attention_layers": full}
    if sliding:
        window = cfg.get("sliding_window")
        if not isinstance(window, int) or window <= 0:
            return _unrecognised(cfg, "sliding layers without a sliding_window")
        out.update(sliding_layers=sliding, sliding_window=window)
    if linear:
        if not all(cfg.get(k) for k in _GDN_KEYS):
            return _unrecognised(cfg, "linear-attention layers without Gated DeltaNet state fields")
        out.update(linear_layers=linear, linear_state_bytes=linear * gdn_state_bytes(cfg, model_dtype),
                   linear_conv_step_bytes=linear * gdn_conv_step_bytes(cfg, model_dtype))
    # Gemma 4 full layers use `global_head_dim`, and with `attention_k_eq_v` `num_global_key_value_heads` KV heads
    # (vLLM transformers_utils/configs/gemma4.gemma4_layer_config). SGLang 0.5.19 applies both global fields
    # without the k_eq_v condition (utils/hf_transformers/config.py); the two agree whenever k_eq_v is set.
    if full and cfg.get("global_head_dim"):
        out["full_head_dim"] = int(cfg["global_head_dim"])
    if full and cfg.get("attention_k_eq_v") and cfg.get("num_global_key_value_heads"):
        out["full_kv_heads"] = int(cfg["num_global_key_value_heads"])
    return out


def estimate_params(
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    vocab: int,
    intermediate: int,
    tie_embeddings: bool = False,
) -> int:
    """Dense-transformer parameter estimate (within a few percent for Llama-style models)."""
    q_o = 2 * hidden * heads * head_dim
    k_v = 2 * hidden * kv_heads * head_dim
    mlp = 3 * hidden * intermediate  # gated MLP (gate, up, down)
    norms = 2 * hidden
    per_layer = q_o + k_v + mlp + norms
    embed = vocab * hidden * (1 if tie_embeddings else 2)
    return layers * per_layer + embed + hidden


def _load_local_config(path: str) -> Optional[Dict[str, Any]]:
    cand = os.path.join(path, "config.json") if os.path.isdir(path) else path
    if os.path.isfile(cand):
        with open(cand, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def fetch_config(spec: ModelSpec, token: Optional[str] = None) -> Dict[str, Any]:
    """config.json for a model: local dir, or HF hub (cached)."""
    local = _load_local_config(spec.hf_id)
    if local is not None:
        return local
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    try:
        path = hf_hub_download(spec.hf_id, "config.json", revision=spec.revision, token=token)
    except GatedRepoError as exc:
        raise RuntimeError(f"{spec.hf_id} is gated on the Hugging Face Hub: accept its licence there and set "
                           "HF_TOKEN, or use an ungated copy (for Meta's Llama models, unsloth's)") from exc
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


INDEX = "model.safetensors.index.json"


def loaded_bytes(sizes: Dict[str, Optional[int]], index_files: Optional[Iterable[str]]) -> Optional[int]:
    """Bytes of the safetensors files an engine loads, given each file's size by its path in the checkpoint. With an
    index, the files it maps tensors to and no others: both engines filter by it, which leaves out a second copy of
    the weights such as gpt-oss's original/model.safetensors (SGLang model_loader/loader.py and vLLM's default
    loader, filter_duplicate_safetensors_files). Without one, every safetensors file at the top level, the folder the
    engines glob. None when a file is missing or has no size: a partial sum would plan too little memory."""
    files = list(index_files) if index_files is not None else [n for n in sizes if "/" not in n]
    if not files or any(not sizes.get(f) for f in files):
        return None
    return int(sum(sizes[f] for f in files))  # type: ignore[misc]


def _index_files(path: Union[str, Path]) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return sorted(set((json.load(fh).get("weight_map") or {}).values()))


# What the engines leave out of a checkpoint's .bin files, by the end of the name: transformers' training files
# (filter_files_not_needed_for_inference in model_loader/weight_utils.py, vLLM v0.33.0 lines 619-634 and SGLang
# v0.5.23 lines 868-884).
NOT_WEIGHTS = ("training_args.bin", "optimizer.bin")
BIN_INDEX = "pytorch_model.bin.index.json"


def bin_complete(bins: Iterable[str], index_files: Optional[Iterable[str]] = None) -> bool:
    """Whether a set of .bin files is whole: every shard its names count (prefix-00001-of-00003.bin, padded or not,
    numbered from 1) and every file its index maps a tensor to is there. A cache snapshot keeps what was downloaded."""
    names = set(bins)
    if index_files is not None and not set(index_files) <= names:
        return False
    shards: Dict[Tuple[str, int], set] = {}
    for name in names:
        shard = re.search(r"-(\d+)-of-(\d+)\.bin$", name)
        if shard:
            shards.setdefault((name[:shard.start()], int(shard[2])), set()).add(int(shard[1]))
    return all(found == set(range(1, total + 1)) for (_, total), found in shards.items())


def _bin_sizes(sizes: Dict[str, Optional[int]], index_files: Optional[Iterable[str]] = None
               ) -> Dict[str, Optional[int]]:
    """The .bin files the engines load from a checkpoint with no safetensors at its top level. Both take a format
    pattern by pattern, "*.safetensors" first, then "*.bin", from the top folder (vLLM model_loader/default_loader.py
    _prepare_weights lines 185-234; SGLang model_loader/loader.py lines 500-552), with no index for .bin:
    lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B and AngelSlim/Qwen3-8B_eagle3 ship a pytorch_model.bin only. A set
    that is not whole (bin_complete; its index, when given, read for that only) gives a size of None, not a partial
    sum."""
    if any(n.endswith(".safetensors") and "/" not in n for n in sizes):
        return {}
    bins = {n: s for n, s in sizes.items() if n.endswith(".bin") and "/" not in n and not n.endswith(NOT_WEIGHTS)}
    if bins and not bin_complete(bins, index_files):
        return {next(iter(bins)): None}
    return bins


def listed_bytes(listing: Dict[str, Optional[int]], fetch_index: Callable[[], Union[str, Path]]) -> Optional[int]:
    """loaded_bytes for a Hub repository's file listing (every file's path -> size), fetching its index through
    `fetch_index` (a local path back) when the listing has one; a checkpoint of .bin files only is sized from them
    (_bin_sizes). Raises whatever fetch_index raises."""
    if INDEX not in listing:
        bins = _bin_sizes(listing)
        if bins:
            return loaded_bytes(bins, None)
    sizes = {name: size for name, size in listing.items() if name.endswith(".safetensors")}
    return loaded_bytes(sizes, _index_files(fetch_index()) if INDEX in listing else None)


def local_checkpoint_bytes(root: Union[str, Path]) -> Optional[int]:
    """loaded_bytes for a local checkpoint directory; None when it cannot be read."""
    root = Path(root)
    try:
        sizes: Dict[str, Optional[int]] = {f.name: f.stat().st_size for f in root.glob("*.safetensors")}
        index = root / INDEX
        if not sizes and not index.is_file():
            bin_index = root / BIN_INDEX  # a local copy may be partial: its index says which shards it needs
            bins = _bin_sizes({f.name: f.stat().st_size for f in root.glob("*.bin")},
                              _index_files(bin_index) if bin_index.is_file() else None)
            if bins:
                return loaded_bytes(bins, None)
        files = _index_files(index) if index.is_file() else None
        for name in files or []:
            if name not in sizes and (root / name).is_file():
                sizes[name] = (root / name).stat().st_size
        return loaded_bytes(sizes, files)
    except (OSError, ValueError, AttributeError) as exc:
        logger.debug("could not size the checkpoint in %s: %s", root, exc)
        return None


def hub_weights(spec: ModelSpec, token: Optional[str] = None
                ) -> Tuple[Optional[int], Optional[int], Dict[str, Optional[int]]]:
    """(parameter count, loaded_bytes, the file listing) for a Hub model from one model_info call, plus its index
    when it has one. The count is the Hub's safetensors total, which counts elements, so it says nothing of bytes: an
    FP8 checkpoint holds one byte a parameter, and the Hub counts AWQ's packed int4 values unpacked but NVFP4's
    packed. Each is None (the listing empty) when the Hub does not say."""
    from huggingface_hub import HfApi, hf_hub_download

    try:
        info = HfApi(token=token).model_info(spec.hf_id, revision=spec.revision, files_metadata=True)
    except Exception as exc:
        logger.debug("safetensors metadata unavailable for %s: %s", spec.hf_id, exc)
        return None, None, {}
    st = getattr(info, "safetensors", None)
    params = int(st.total) if st and getattr(st, "total", None) else None
    sha = getattr(info, "sha", None)
    listing = hub_listing(info)
    try:
        return params, listed_bytes(listing, lambda: hf_hub_download(
            spec.hf_id, INDEX, revision=sha if isinstance(sha, str) else spec.revision, token=token)), listing
    except Exception as exc:  # a gated repository answers model_info but not a download without a token
        logger.debug("could not read %s of %s: %s", INDEX, spec.hf_id, exc)
        return params, None, listing


HF_QUANT = "hf_quant_config.json"  # ModelOpt's own file, which some of its exports declare their quantization in only


def _modelopt_quant(arch: ArchInfo, read: Callable[[], Union[str, Path]]) -> None:
    """Set the checkpoint's quantization from hf_quant_config.json (introspect.quantization); unset when unreadable."""
    from rigtune.introspect import quantization

    try:
        with open(read(), "r", encoding="utf-8") as fh:
            quant = quantization(None, json.load(fh))
    except Exception as exc:
        logger.debug("could not read %s: %s", HF_QUANT, exc)
        return
    if quant:
        arch.checkpoint_quant, arch.stored_as = quant.describe(), quant.label


def hub_listing(info: Any) -> Dict[str, Optional[int]]:
    """Every file of a model_info(files_metadata=True) answer: its path -> size."""
    return {getattr(s, "rfilename", ""): getattr(s, "size", None) for s in (getattr(info, "siblings", None) or [])}


def snapshot_revision(spec: ModelSpec, cache_dir: Union[str, Path, None] = None) -> Optional[str]:
    """The commit of the model's snapshot in the local Hugging Face cache, read from the snapshot's own path
    (.../snapshots/<commit>/config.json). Never asks the Hub. None for a local directory or an uncached model."""
    if os.path.isdir(spec.hf_id):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(spec.hf_id, "config.json", cache_dir=cache_dir, revision=spec.revision)
    except Exception as exc:
        logger.debug("no cached snapshot for %s: %s", spec.hf_id, exc)
        return None
    if not isinstance(path, str):
        return None
    snapshot = Path(path).parent
    if snapshot.parent.name == "snapshots" and _COMMIT.fullmatch(snapshot.name):
        return snapshot.name
    return None


def hub_revision(spec: ModelSpec, token: Optional[str] = None) -> Optional[str]:
    """The commit the Hub resolves the requested revision (default: main) to. Asks the Hub."""
    from huggingface_hub import HfApi

    sha = getattr(HfApi(token=token).model_info(spec.hf_id, revision=spec.revision), "sha", None)
    return sha if isinstance(sha, str) and _COMMIT.fullmatch(sha) else None


def resolve_revision(spec: ModelSpec, fetch: Optional[Callable[[ModelSpec], Optional[str]]] = None,
                     cache_dir: Union[str, Path, None] = None) -> Optional[str]:
    """The Hub commit a calibration measured, for its profile: the cached snapshot's, since that is what the
    engines load (fetch_config has just downloaded config.json into it), else what the Hub says (`fetch`,
    default hub_revision). None for a local directory, or when neither can be read: never an error."""
    found = snapshot_revision(spec, cache_dir)
    if found is not None or os.path.isdir(spec.hf_id):
        return found
    try:
        return (fetch or hub_revision)(spec)
    except Exception as exc:
        logger.debug("could not resolve the Hub revision of %s: %s", spec.hf_id, exc)
        return None


def load_arch(spec: ModelSpec, token: Optional[str] = None) -> ArchInfo:
    arch = arch_from_config(fetch_config(spec, token=token))
    if os.path.isdir(spec.hf_id):
        root = Path(spec.hf_id)
        arch.checkpoint_bytes = local_checkpoint_bytes(root)
        if arch.checkpoint_quant is None and (root / HF_QUANT).is_file():
            _modelopt_quant(arch, lambda: root / HF_QUANT)
        if arch.checkpoint_quant is None:  # a quantized one is sized by its files alone (quantized.as_stored)
            # From the files, like the Hub's count, rather than the dense estimate above, which leaves out an MoE's
            # experts and an MTP head: exact from the headers, else the files' size over the config's dtype.
            from rigtune.pairing import _local_params

            safetensors = sorted(root.glob("*.safetensors"))
            counted = _local_params(root, safetensors)
            if counted:
                arch.num_params = counted
            elif arch.checkpoint_bytes:
                by_size = arch.checkpoint_bytes // dtype_bytes(arch.torch_dtype)
                # .bin files have no header to count from, and their dtype need not be the config's: fp16 files under
                # a float32 config would halve the count, and the planned weights with it. Never below the estimate.
                arch.num_params = by_size if safetensors else max(arch.num_params or 0, by_size)
        return arch
    exact, arch.checkpoint_bytes, listing = hub_weights(spec, token=token)
    if exact:
        arch.num_params = exact
    if arch.checkpoint_quant is None and HF_QUANT in listing:
        from huggingface_hub import hf_hub_download

        _modelopt_quant(arch, lambda: hf_hub_download(spec.hf_id, HF_QUANT, revision=spec.revision, token=token))
    return arch
