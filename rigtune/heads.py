"""Whether a drafter that reads the target's hidden states can draft for a target, from its files alone.

Three kinds of drafter read the target's hidden states rather than its tokens alone, so each is trained for one target
and fits no other: an EAGLE3 head, a DFlash drafter and a DSpark drafter. A repository's name proposes one; its
config.json decides, per engine, since the engines load them by different rules:

  vLLM v0.33.0 wraps the draft's config for "eagle3" and "dflash" (transformers_utils/configs/eagle.py EAGLEConfig,
  lines 55-76): an architecture X is renamed Eagle3X (DFlashX) unless it already starts or ends so, and the renamed
  name must be in its model registry (model_executor/models/registry.py _SPECULATIVE_DECODING_MODELS, lines 655-714).
  So a head declaring plain LlamaForCausalLM loads as Eagle3LlamaForCausalLM. A config of model_type "speculators" (or
  with speculators_config and no model_type: transformers_utils/config.py lines 361-367) is converted while it is read
  (configs/speculators/base.py, algos.py): its kind is speculators_model_type, its layers transformer_layer_config.

  SGLang v0.5.23 has no such rename and no speculators converter: the draft's own architecture must be one of its
  model classes (models/registry.py, the EntryClass lists in models/llama_eagle3.py line 387, kimi_k25_eagle3.py
  line 523, dflash.py lines 1317-1324, dspark.py line 995, lfm2_dspark.py line 16, dspark_minimax.py line 326).
  A plain LlamaForCausalLM head would load as a full Llama.

What each engine checks of the pair is little (vLLM checks no vocabulary or hidden size of an EAGLE3 head; a wrong
target_layer_ids fails at the first forward), so the checks here are rigtune's: what the drafter reads from the
target must exist there. Missing facts are unknown, never proof of fit.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Optional

from rigtune.models import ArchInfo
from rigtune.pairing import PairVerdict, text_config

HEAD_KINDS = ("eagle3", "dflash", "dspark")

# The registry names each engine loads a drafter of each kind as, read at vLLM v0.33.0 and SGLang v0.5.23. Left out on
# purpose: P-EAGLE (PEagle*, Peagle*: needs parallel drafting, which rigtune does not plan), beam DFlash
# (DFlash2BeamDraftModel: beam drafting, likewise), vLLM's DeepSeek-V4 DSpark (DSparkV41DraftModel, and a
# DSparkDraftModel that is not of model_type qwen3: it reuses the target's own config and checkpoint, config/
# speculative.py lines 1547-1566), Qwen3-Omni's DSpark (a long contract of its own, lines 167-392) and the XPress
# refiners. vLLM's rename applies to EAGLE3 only here: a full model renamed DFlash{arch} (DFlashLagunaForCausalLM)
# would be accepted as a drafter.
ENGINE_ARCHS: Dict[str, Dict[str, FrozenSet[str]]] = {
    "vllm": {
        "eagle3": frozenset({"Eagle3LlamaForCausalLM", "LlamaForCausalLMEagle3", "Eagle3Qwen3ForCausalLM",
                             "Eagle3MiniMaxM2ForCausalLM", "Eagle3Qwen2_5vlForCausalLM", "Eagle3Qwen3vlForCausalLM",
                             "Eagle3DeepseekV2ForCausalLM", "Eagle3DeepseekV3ForCausalLM"}),
        # MuseGlimmerAssistantModel: a registry alias of its own (registry.py line 674), and vLLM picks dflash for it
        # (config/speculative.py lines 1449-1454).
        "dflash": frozenset({"DFlashDraftModel", "DFlash2DraftModel", "DFlashMuseGlimmerAssistantModel",
                             "MuseGlimmerAssistantModel", "DFlashLagunaForCausalLM"}),
        "dspark": frozenset({"Qwen3DSparkModel", "Gemma4DSparkModel", "K3DSparkModel", "MuseSparkDSparkModel"}),
    },
    "sglang": {
        "eagle3": frozenset({"LlamaForCausalLMEagle3", "Eagle3DeepseekV2ForCausalLM"}),
        "dflash": frozenset({"DFlashDraftModel", "DFlash2DraftModel", "DFlashLagunaForCausalLM",
                             "MuseGlimmerAssistantModel", "MiniMaxDFlash2DraftModel"}),
        # DSparkDraftModel: SGLang's generic dense DSpark (models/dspark.py lines 967 and 995).
        "dspark": frozenset({"Qwen3DSparkModel", "LingDSparkModel", "DSparkDraftModel", "Lfm2DSparkDraftModel",
                             "DSparkMiniMaxDraftModel"}),
    },
}
# vLLM renames a DSparkDraftModel of model_type qwen3 to Qwen3DSparkModel (config/speculative.py lines 1456-1468 and
# 1535-1542).
VLLM_QWEN3_DSPARK = "DSparkDraftModel"
# DSpark drafters whose num_target_layers counts the layers they read, not the target's depth: vLLM's MLA DSpark
# refuses any other count ("MLA DSpark requires non-empty target_layer_ids and a matching num_target_layers",
# transformers_utils/configs/k3_dspark.py lines 70-77); Inferact's Kimi-K3 and Muse Spark drafters declare 5.
COUNTS_READ_LAYERS = frozenset({"K3DSparkModel", "MuseSparkDSparkModel"})
# SGLang captures EAGLE3 layer i + 1 (models/llama.py set_eagle3_layers_to_capture, lines 938-941) in a loop over the
# target's layers (lines 486-493), so an id above num_layers - 2 is never captured; gpt-oss keeps ids whose largest
# is num_layers as they are (models/gpt_oss.py lines 1346-1351).
SGLANG_FINAL_LAYER_ARCHS = frozenset({"GptOssForCausalLM"})

# vLLM's speculators converter (configs/speculators/algos.py): the kind each speculators_model_type becomes, and the
# layer types its eagle3 converter knows (lines 39-45, "Unsupported model_type ... for Eagle3 speculator" otherwise).
SPECULATORS_KINDS = {"eagle3": "eagle3", "dflash": "dflash", "dflash2": "dflash", "dspark": "dspark"}
SPECULATORS_EAGLE3_TYPES = frozenset({"llama", "qwen3"})


def is_speculators(cfg: Dict[str, Any]) -> bool:
    """The speculators format, as vLLM recognises it (transformers_utils/config.py lines 361-367)."""
    return cfg.get("model_type") == "speculators" or (cfg.get("model_type") is None
                                                       and cfg.get("speculators_config") is not None)


def _vllm_eagle3_name(arch: str) -> str:
    """The registry name vLLM looks an EAGLE3 head's architecture up by (EAGLEConfig's rename)."""
    return arch if arch.startswith("Eagle3") or arch.endswith("Eagle3") else f"Eagle3{arch}"


def head_kind(cfg: Dict[str, Any], engine: str) -> Optional[str]:
    """The kind of drafter `engine` loads this config as (one of HEAD_KINDS), or None when it loads none.

    On vLLM a plain architecture renamed into the registry counts as an EAGLE3 head only with an EAGLE3 head's marks:
    one decoder layer and a draft_vocab_size, which no full model declares (yuhuili's and nm-testing's heads declare
    LlamaForCausalLM; a full Llama would be renamed and loaded the same way)."""
    tables = ENGINE_ARCHS.get(engine)
    if not tables or not isinstance(cfg, dict):
        return None
    if is_speculators(cfg):
        if engine != "vllm":
            return None
        kind = SPECULATORS_KINDS.get(str(cfg.get("speculators_model_type")))
        layer = cfg.get("transformer_layer_config") or {}
        if kind == "eagle3" and layer.get("model_type", "llama") not in SPECULATORS_EAGLE3_TYPES:
            return None
        return kind
    archs = [str(a) for a in (cfg.get("architectures") or [])]
    for kind in HEAD_KINDS:
        if any(a in tables[kind] for a in archs):
            return kind
    if engine != "vllm":
        return None
    if VLLM_QWEN3_DSPARK in archs and cfg.get("model_type") == "qwen3":
        return "dspark"
    if (any(_vllm_eagle3_name(a) in tables["eagle3"] for a in archs)
            and cfg.get("num_hidden_layers") == 1 and "draft_vocab_size" in cfg):
        return "eagle3"
    return None


def _layer_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The decoder layer's fields: transformer_layer_config under the speculators format, else the text config."""
    if is_speculators(cfg):
        return {**(cfg.get("transformer_layer_config") or {}),
                **{k: v for k, v in cfg.items() if k != "transformer_layer_config" and v is not None}}
    return text_config(cfg)


def _listed(value: Any) -> Optional[List[Any]]:
    return list(value) if isinstance(value, (list, tuple)) else None


def target_layer_ids(cfg: Dict[str, Any], kind: str) -> Optional[List[int]]:
    """The target layers the drafter was trained to read, in the engines' numbering; None when it names none.

    DFlash and DSpark: dflash_config.target_layer_ids, else target_layer_ids; the speculators format stores them one
    higher (vLLM configs/speculators/algos.py: aux_id - 1, lines 114-120 and 192-195). EAGLE3: what the head sizes its
    input from, eagle_aux_hidden_state_layer_ids at the top, else in eagle_config (vLLM models/qwen3_eagle3.py lines
    206-209; llama_eagle3.py lines 213-216 and SGLang models/llama_eagle3.py lines 156-161 read eagle_config); the
    speculators format keeps them as stored (algos.py lines 47-50)."""
    if is_speculators(cfg):
        if kind == "eagle3":
            return _listed(cfg.get("eagle_aux_hidden_state_layer_ids"))
        ids = _listed(cfg.get("aux_hidden_state_layer_ids"))
        return [i - 1 for i in ids] if ids is not None else None
    if kind == "eagle3":
        top = _listed(cfg.get("eagle_aux_hidden_state_layer_ids"))
        return top if top else _listed((cfg.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids"))
    found = _listed((cfg.get("dflash_config") or {}).get("target_layer_ids"))
    return found if found is not None else _listed(cfg.get("target_layer_ids"))


def engine_eagle3_ids(cfg: Dict[str, Any], engine: str) -> Optional[List[Any]]:
    """The layers `engine` tells the target to capture for an EAGLE3 head; None when it falls back to the target's
    defaults (eagle3_defaults). vLLM 0.33's default runner (Model Runner V2, v1/worker/gpu/spec_decode/eagle/
    eagle3_utils.py get_eagle3_aux_layers_from_config, lines 113-151) reads the top-level ids, then eagle_config's
    "layer_ids", never eagle_config's eagle_aux_hidden_state_layer_ids, which V1 read (gpu_model_runner.py lines
    5207-5221). SGLang 0.5.23 reads eagle_config's only (model_executor/model_runner_components/
    spec_aux_hidden_state.py lines 184-196)."""
    eagle = cfg.get("eagle_config") or {}
    if not isinstance(eagle, dict):
        eagle = {}
    if engine == "vllm":
        return _listed(cfg.get("eagle_aux_hidden_state_layer_ids")) or _listed(eagle.get("layer_ids")) or None
    return _listed(eagle.get("eagle_aux_hidden_state_layer_ids")) or None


def eagle3_defaults(target: ArchInfo) -> List[int]:
    """The layers a target captures for an EAGLE3 head that names none: vLLM model_executor/models/interfaces.py line
    1772, SGLang models/llama.py set_eagle3_layers_to_capture line 936."""
    n = target.num_layers
    return [2, n // 2, n - 3]


def _ids_fit(ids: List[Any], kind: str, engine: str, target: ArchInfo) -> Optional[bool]:
    """Whether every id names a layer the engine captures on `target`; None when that depends on the target's model
    code, unread. DFlash and DSpark ids are layers, read after the layer (vLLM eagle3_utils.py lines 119-141 adds one;
    SGLang speculative/dflash_utils.py lines 578-583 refuses an id outside them). An EAGLE3 id L is captured before
    layer L, so on vLLM 0..num_layers are all captured (models/llama.py lines 428-441: num_layers is the last layer's
    output). SGLang captures id + 1 inside the layer loop, so num_layers - 2 at most; gpt-oss also captures the final
    output (SGLANG_FINAL_LAYER_ARCHS), and some other targets (MiniMax-M3, DeepSeek) capture num_layers - 1 too."""
    n = target.num_layers
    if not ids or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids):
        return False
    top = max(ids)
    if kind != "eagle3":
        return top < n
    if engine == "vllm" or target.architecture in SGLANG_FINAL_LAYER_ARCHS:
        return top <= n
    if top <= n - 2:
        return True
    return None if top == n - 1 else False


def _positive(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_head(target: ArchInfo, cfg: Dict[str, Any], engine: str, *, files: Optional[Iterable[str]] = None,
               aliases: Iterable[str] = ()) -> PairVerdict:
    """Whether `engine` can load the drafter whose config.json is `cfg` for `target`, and what it reads exists there.

    `files` is the repository's file listing, when read (a ModelOpt hf_quant_config.json beside the config marks a
    quantized head); `aliases` the names the target goes by (its repository, and the one it was quantized from), to
    check a speculators verifier against. Status "ok", "refused" or "unknown", as pairing.check_pair's."""
    refused: List[str] = []
    unknown: List[str] = []
    kind = head_kind(cfg, engine)
    evidence: Dict[str, Any] = {"engine": engine, "kind": kind,
                                "architectures": list((cfg or {}).get("architectures") or [])}
    if kind is None:
        return PairVerdict(status="refused", reasons=["not_a_head_here"], evidence=evidence)
    layer = _layer_config(cfg)
    vocab = layer.get("vocab_size")
    if not _positive(vocab):
        unknown.append("vocab_unread")
    elif vocab != target.vocab_size:
        refused.append("vocab_differs")
    hidden = cfg.get("target_hidden_size") or layer.get("hidden_size")
    evidence["sizes"] = {"vocab": [target.vocab_size, vocab], "hidden": [target.hidden_size, hidden]}
    if not _positive(hidden):
        unknown.append("hidden_unread")
    elif hidden != target.hidden_size:
        refused.append("hidden_differs")
    ids = target_layer_ids(cfg, kind)
    layers = cfg.get("num_target_layers")
    if layers is not None:
        counts_read = any(a in COUNTS_READ_LAYERS for a in evidence["architectures"])
        expected = (len(ids) if ids else None) if counts_read else target.num_layers
        evidence["target_layers"] = [expected, layers]
        if layers != expected:
            refused.append("target_layers_differ")
    if ids is not None:
        evidence["target_layer_ids"] = ids
        fits = _ids_fit(ids, kind, engine, target)
        if fits is False:
            refused.append("layer_ids_outside_target")
        elif fits is None:
            unknown.append("layer_ids_edge_unchecked")
        elif kind == "eagle3":
            # The engine must capture the layers the head was trained on; one that falls back to its defaults, or
            # reads other ids, runs the head on other layers with no error.
            captured = engine_eagle3_ids(cfg, engine)
            evidence["captured_layer_ids"] = captured if captured is not None else "defaults"
            # Both engines capture in layer order whatever order the config lists (vLLM interfaces.py line 1617:
            # tuple(sorted(set(layers)))), so the lists are compared as sets.
            if set(captured if captured is not None else eagle3_defaults(target)) != set(ids):
                unknown.append("layer_ids_unread_by_engine")
    elif kind != "eagle3":  # an EAGLE3 head without ids is trained on, and gets, the target's defaults
        unknown.append("layer_ids_unread")
    mask = (cfg.get("dflash_config") or {}).get("mask_token_id", cfg.get("mask_token_id"))
    if kind in ("dflash", "dspark"):
        if mask is None:
            unknown.append("mask_token_unread")
        elif not isinstance(mask, int) or not 0 <= mask < target.vocab_size:
            refused.append("mask_token_outside_vocab")
    if layer.get("quantization_config") or cfg.get("quantization_config") or (
            files is not None and "hf_quant_config.json" in set(files)):
        refused.append("quantized_head")
    verifier = ((cfg.get("speculators_config") or {}).get("verifier") or {}).get("name_or_path")
    names = {a.lower() for a in aliases if a}
    if verifier and names:
        evidence["verifier"] = verifier
        if str(verifier).lower() not in names:
            refused.append("verifier_differs")
    return PairVerdict(status="refused" if refused else "unknown" if unknown else "ok",
                       reasons=refused + unknown, evidence=evidence)
