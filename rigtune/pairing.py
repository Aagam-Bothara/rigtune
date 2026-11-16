"""Evidence for a second model. A name proposes a pair; files decide it.

Checking is pure. Readers are lazy and injectable; importing this module never asks
the Hub or loads model code. Missing facts are unknown, never proof of compatibility.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PairFiles(BaseModel):
    repo: Optional[str] = None
    sha: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tokenizer_sha256: Optional[str] = None
    token_ids: Optional[Dict[str, int]] = None
    tokenizer_bytes: Optional[int] = None
    template_sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    num_params: Optional[int] = None
    gated: Optional[bool] = None
    source: Optional[str] = None


class PairVerdict(BaseModel):
    status: str
    reasons: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)


# Engines' draft-config rewrites, not model-name heuristics. Sources:
# vLLM v0.29.0/v0.33.0 config/speculative.py hf_config_override;
# SGLang v0.5.19/v0.5.23 configs/model_config.py _config_draft_model.
REWRITTEN_TYPES = frozenset({
    "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen3_next", "deepseek_v3", "deepseek_v32", "deepseek_v4", "mimo", "mimo_v2",
    "glm4_moe", "glm4_moe_lite", "ernie4_5_moe", "nemotron_h", "exaone_moe", "exaone4_5",
    "gemma4_assistant", "gemma4_unified_assistant", "longcat_flash", "dots1",
})
REWRITTEN_AS_HEAD = {
    "vllm": frozenset({"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
                       "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3NextForCausalLM",
                       "Gemma4AssistantForCausalLM", "Gemma4UnifiedAssistantForCausalLM"}),
    "sglang": frozenset({"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
                         "Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM", "Qwen3NextForCausalLM",
                         "Gemma4AssistantForCausalLM", "Gemma4UnifiedAssistantForCausalLM"}),
}


def text_config(config: Optional[dict]) -> dict:
    raw = config or {}
    return {**raw, **(raw.get("text_config") or {})}


def _positive(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_pair(target: PairFiles, second: PairFiles, kind: str, ctx: int,
               *, engine: str = "vllm") -> PairVerdict:
    """Check raw vocabulary, token ids, context, loader semantics and cost."""
    a, b = text_config(target.config), text_config(second.config)
    refused, unknown = [], []
    evidence: Dict[str, Any] = {
        "engine": engine, "context": ctx, "kind": kind,
        "target_tokenizer_sha256": target.tokenizer_sha256,
        "second_tokenizer_sha256": second.tokenizer_sha256,
        "template": ("unread" if not (target.template_sha256 and second.template_sha256)
                     else "same" if target.template_sha256 == second.template_sha256 else "differs"),
    }
    av, bv = a.get("vocab_size"), b.get("vocab_size")
    evidence["vocab_sizes"] = [av, bv]
    if not (_positive(av) and _positive(bv)):
        unknown.append("vocab_unread")
    elif av != bv:
        refused.append("vocab_differs")
    same_hash = bool(target.tokenizer_sha256 and target.tokenizer_sha256 == second.tokenizer_sha256)
    same_maps = bool(target.token_ids and second.token_ids and target.token_ids == second.token_ids)
    if kind == "draft":
        if same_hash or same_maps:
            evidence["ids"] = "same_hash" if same_hash else "same_map"
        elif target.token_ids and second.token_ids:
            refused.append("ids_differ")
        else:
            unknown.append("ids_unread")
        length = b.get("max_position_embeddings")
        if not _positive(length):
            unknown.append("context_unread")
        elif length < ctx:
            refused.append("context_too_short")
        archs = b.get("architectures") or []
        if not archs or not b.get("model_type"):
            unknown.append("architecture_unread")
        if (set(archs) & REWRITTEN_AS_HEAD.get(engine, frozenset())
                or b.get("model_type") in REWRITTEN_TYPES
                or any(word in str(b.get("model_type", "")).lower() for word in ("mtp", "eagle", "dflash", "dspark"))):
            refused.append("rewritten_as_head")
        if engine == "vllm" and any(s in (second.repo or "").lower()
                                    for s in ("eagle-", "eagle3", "dflash", "dspark")):
            refused.append("method_changing_name")
        if not (_positive(target.num_params) and _positive(second.num_params)):
            unknown.append("cost_unread")
        else:
            evidence["parameter_ratio"] = second.num_params / target.num_params
            if second.num_params > target.num_params * .25:
                refused.append("over_cost_bound")
    else:
        refused.append("unsupported_kind")
    if not _positive(second.size_bytes):
        unknown.append("size_unread")
    if not target.sha or not second.sha:
        unknown.append("revision_unread")
    return PairVerdict(status="refused" if refused else "unknown" if unknown else "ok",
                       reasons=refused + unknown, evidence=evidence)
