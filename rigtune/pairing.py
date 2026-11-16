"""Evidence for a second model. A name proposes a pair; files decide it.

Checking is pure. Readers are lazy and injectable; importing this module never asks
the Hub or loads model code. Missing facts are unknown, never proof of compatibility.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PureWindowsPath
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


TOKENIZER_LIMIT = 64_000_000


class HubLookupError(RuntimeError):
    def __init__(self, why: str):
        self.why = why
        super().__init__(f"pair files not checked: {why}")


def _lookup_error(exc: Exception, source: str) -> HubLookupError:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    why = {401: "gated", 403: "gated", 404: "not_found", 429: "rate_limited"}.get(code)
    if why is None:
        why = "offline" if source == "cache" or type(exc).__name__ in (
            "OfflineModeIsEnabled", "LocalEntryNotFoundError") else "error"
    return HubLookupError(why)


def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def local_alias(repo: str, digest: str) -> str:
    """Portable, path-free identity; works with Windows paths on Linux too."""
    name = PureWindowsPath(repo).name if "\\" in repo or re.match(r"^[A-Za-z]:", repo) else Path(repo).name
    return f"local/{name}-{digest[:8]}"


def is_local(repo: str) -> bool:
    return Path(repo).is_dir() or Path(repo).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", repo))


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _tokens(path: Path) -> Dict[str, int]:
    data = _json(path)
    vocab = data.get("model", {}).get("vocab")
    if isinstance(vocab, list):  # SentencePiece/Unigram: [[piece, score], ...]
        vocab = {entry[0]: i for i, entry in enumerate(vocab)}
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer has no vocabulary")
    out = dict(vocab)
    for entry in data.get("added_tokens", []):
        token, ident = entry["content"], entry["id"]
        if token in out and out[token] != ident:
            raise ValueError("token has conflicting ids")
        out[token] = ident
    if any(not isinstance(k, str) or not isinstance(v, int) or v < 0 for k, v in out.items()):
        raise ValueError("invalid vocabulary")
    if len(set(out.values())) != len(out):
        raise ValueError("duplicate token ids")
    return out


def _template(root: Path) -> Optional[str]:
    if (root / "chat_template.jinja").is_file():
        return _hash_file(root / "chat_template.jinja")
    if (root / "tokenizer_config.json").is_file():
        template = _json(root / "tokenizer_config.json").get("chat_template")
        if template is not None:
            return hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()
    return None


def _shards_missing(root: Path, weights: List[Path]) -> bool:
    """Whether the directory holds only some of its weight files: a shard its index names, or one a present shard's
    own name counts (`-00001-of-00003`), is absent. A cache snapshot keeps whatever was downloaded so far."""
    present = {f.name for f in weights}
    declared = set()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = _json(index).get("weight_map")
        declared |= {v for v in mapping.values() if isinstance(v, str)} if isinstance(mapping, dict) else set()
    for name in present:
        shard = re.search(r"-(\d+)-of-(\d+)\.safetensors$", name)
        if shard:
            declared |= {f"{name[:shard.start()]}-{i:0{len(shard[1])}d}-of-{shard[2]}.safetensors"
                         for i in range(1, int(shard[2]) + 1)}
    return bool(present) and not declared <= present


def _from_directory(root: Path, repo: Optional[str] = None, sha: Optional[str] = None,
                    source: str = "local") -> PairFiles:
    config = _json(root / "config.json")
    weights = sorted(root.glob("*.safetensors"))
    # Some shards of several: the size is unknown, not the sum of those present, which would plan too little memory.
    size = None if _shards_missing(root, weights) else sum(f.stat().st_size for f in weights) or None
    if size is None and not weights and (root / "model.safetensors.index.json").is_file():
        size = _json(root / "model.safetensors.index.json").get("metadata", {}).get("total_size")
    tokenizer = root / "tokenizer.json"
    ts = tokenizer.stat().st_size if tokenizer.is_file() else None
    digest = _hash_file(tokenizer) if ts and ts <= TOKENIZER_LIMIT else None
    if source == "local":
        # Content, including the weights, identifies a mutable local directory.
        entries = sorted({*weights, *(f for f in root.iterdir() if f.suffix in (".json", ".jinja"))})
        sha = hashlib.sha256(json.dumps([(f.name, _hash_file(f)) for f in entries]).encode()).hexdigest()
        repo = local_alias(str(root), sha)
    cfg = text_config(config)
    dtype = cfg.get("torch_dtype", cfg.get("dtype"))
    elem = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype)
    params = size // elem if size and elem and not config.get("quantization_config") else None
    return PairFiles(repo=repo, sha=sha, config=config, size_bytes=size, num_params=params,
                     tokenizer_bytes=ts, tokenizer_sha256=digest,
                     token_ids=_tokens(tokenizer) if digest else None, template_sha256=_template(root), source=source)


def read_files(repo: str, source: str = "hub", *, revision: Optional[str] = None,
               info_fetcher=None, file_fetcher=None) -> PairFiles:
    """Read metadata and config at one immutable revision, never model weights from the Hub.

    A cache read uses only local_files_only downloads (no API call). A local read
    hashes content so a directory changed in place cannot keep an old verdict.
    """
    try:
        if is_local(repo):
            return _from_directory(Path(repo))
        if file_fetcher is None:
            from huggingface_hub import hf_hub_download
            file_fetcher = hf_hub_download
        if source == "cache":
            config_file = Path(file_fetcher(repo, "config.json", revision=revision, local_files_only=True))
            # Cache snapshots have immutable commit directories; a missing pin is unknown.
            sha = next((part for part in config_file.parts if re.fullmatch(r"[a-f0-9]{40}", part)), None)
            return _from_directory(config_file.parent, repo, sha, "cache")
        if info_fetcher is None:
            from huggingface_hub import HfApi
            info_fetcher = HfApi().model_info
        info = info_fetcher(repo, revision=revision, files_metadata=True)
        sha = _get(info, "sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{40}", sha):
            raise HubLookupError("error")
        config = _json(Path(file_fetcher(repo, "config.json", revision=sha)))
        files = {_get(s, "rfilename"): s for s in (_get(info, "siblings") or [])}
        sizes = [_get(s, "size") for name, s in files.items() if name.endswith(".safetensors")]
        size = sum(sizes) if sizes and all(_positive(s) for s in sizes) else None
        tok = files.get("tokenizer.json")
        digest = _get(_get(tok, "lfs"), "sha256")  # a git blob id is NOT sha256
        ts = _get(tok, "size")
        params = _get(_get(info, "safetensors"), "total")
        template = None
        if "chat_template.jinja" in files:
            template = _hash_file(Path(file_fetcher(repo, "chat_template.jinja", revision=sha)))
        elif "tokenizer_config.json" in files:
            cfg = _json(Path(file_fetcher(repo, "tokenizer_config.json", revision=sha)))
            if cfg.get("chat_template") is not None:
                template = hashlib.sha256(json.dumps(cfg["chat_template"], sort_keys=True).encode()).hexdigest()
        return PairFiles(repo=repo, sha=sha, config=config, tokenizer_sha256=digest, tokenizer_bytes=ts,
                         size_bytes=size, num_params=params, gated=bool(_get(info, "gated")),
                         template_sha256=template, source=source)
    except HubLookupError:
        raise
    except Exception as exc:
        raise _lookup_error(exc, source) from None


def read_tokenizers(target: PairFiles, second: PairFiles, *, source: str = "hub", file_fetcher=None) -> None:
    """Download only when hashes cannot prove equality and BOTH files fit the bound."""
    if target.tokenizer_sha256 and target.tokenizer_sha256 == second.tokenizer_sha256:
        return
    if not all(f.tokenizer_bytes and f.tokenizer_bytes <= TOKENIZER_LIMIT for f in (target, second)):
        return
    if file_fetcher is None:
        from huggingface_hub import hf_hub_download
        file_fetcher = hf_hub_download
    for f in (target, second):
        if f.token_ids is not None or not f.repo or f.repo.startswith("local/"):
            continue
        try:
            path = Path(file_fetcher(f.repo, "tokenizer.json", revision=f.sha, local_files_only=source == "cache"))
            if path.stat().st_size > TOKENIZER_LIMIT:
                continue
            digest = _hash_file(path)
            if f.tokenizer_sha256 and digest != f.tokenizer_sha256:
                raise HubLookupError("error")
            f.token_ids = _tokens(path)
            f.tokenizer_sha256 = digest
        except HubLookupError:
            raise
        except Exception as exc:
            raise _lookup_error(exc, source) from None
