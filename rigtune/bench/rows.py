"""Rows `rigtune compare` adds on request: the engines' own tuning guides' settings, and other tools' picks.

A row is a named engine configuration, measured like every other row: same runner, workload, limits and
repeats, interleaved with the others. Two sources:

- `--expert`: rigtune/data/expert_rows.json. Each row is a stock row (`vllm-default`, ...) with one setting
  an engine's own documentation recommends, quoted from the page at the release tags it was read at. A setting
  a page makes conditional on what a running server logs cannot be fixed before launch, so it is listed under
  `considered` with the reason instead. A row is skipped with a note, never an error, when its engine is not
  part of this comparison, when the engine's own default on this card already does what the page says, on a
  card where the engine's own source says the setting hurts, when the card or model lacks what the row needs
  (a KV dtype rigtune runs there, a CUDA GPU, an EAGLE-3 head), or when rigtune knows the configuration fails.
- `--rows FILE`: rows from other tools' output, such as vLLM's `benchmarks/auto_tune/auto_tune.sh`
  (`benchmarks/vllm_auto_tune.py row` writes one). The file is JSON:

      {"rows": [{"label": "autotune-vllm", "kind": "tuner", "source": "how it was produced",
                 "config": {"backend": "vllm", "quant": "bf16", "ctx": 4096, "batch": 256, ...}}]}

  `config` takes rigtune's Config fields (rigtune/models.py); an unknown field is an error rather than
  ignored, so a typo cannot silently drop a setting. `kind` is "tuner" (another tool's pick, the default) or
  "expert". A row with `"skipped": "<why>"` and no config stands for a tool that produced no pick: it is
  recorded with its reason, not measured, rather than left out, and the verdict calls it not applicable.

Every added row carries its kind and source into the results file, and the verdict (`rigtune verdict`,
rigtune/bench/verdict.py) holds each kind to its own threshold.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rigtune.models import Config, HardwareDescriptor, PreparedModel

KINDS = ("tuner", "expert")
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Labels the shipped expert rows use (vllm-expert-ngram, ...): reserved by pattern, so checking a --rows file
# needs no read of the expert data.
EXPERT_LABEL = re.compile(r"-expert(-|$)")


@dataclass
class ExtraRow:
    """One added row. `config` is None only for a skipped row whose engine is not part of the comparison."""

    config: Optional[Config]
    kind: str = "tuner"
    source: str = ""
    skipped: Optional[str] = None  # why it is not measured; None = measure it
    notes: List[str] = field(default_factory=list)  # added to the results' notes when the row is measured
    changes_numerics: bool = False  # an expert setting that changes what the model computes (quantized KV)
    # Measured instead when `config` fails to start, with this note (the page's other format, say).
    fallback: Optional[Config] = None
    fallback_note: str = ""
    # The busiest level at which its setting acts as the page means it, and what to note above it.
    acts_up_to_users: Optional[int] = None
    acts_note: str = ""


class RowsFileError(ValueError):
    pass


def _config(fields: Any, where: str) -> Config:
    if not isinstance(fields, dict):
        raise RowsFileError(f"{where}: config must be an object of Config fields")
    unknown = sorted(set(fields) - set(Config.model_fields))
    if unknown:
        raise RowsFileError(f"{where}: unknown Config field(s) {', '.join(unknown)}")
    try:
        return Config.model_validate(fields)
    except ValueError as exc:
        raise RowsFileError(f"{where}: {exc}") from None


def check_label(label: Any, where: str, taken: Dict[str, Any]) -> str:
    """A label must be safe in a stage name, must not stand for rigtune's own row, a stock row (`*-default`)
    or a shipped expert row (`*-expert-*`), whose thresholds and meaning differ, and must not repeat."""
    if not isinstance(label, str) or not LABEL.match(label):
        raise RowsFileError(f"{where}: label {label!r} must be letters, digits, '.', '_' or '-'")
    if label == "rigtune" or label.endswith("-default") or EXPERT_LABEL.search(label):
        raise RowsFileError(f"{where}: label {label!r} is reserved for rigtune's own row, the stock rows and "
                            "the shipped expert rows")
    if label in taken:
        raise RowsFileError(f"{where}: label {label!r} is already used by another row")
    return label


def read_rows_file(path: Path, taken: Optional[Dict[str, Any]] = None) -> Dict[str, ExtraRow]:
    """The rows in one `--rows` file, in file order. `taken` holds labels already in use (other files, the
    expert rows)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RowsFileError(f"{path}: {exc}") from None
    entries = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise RowsFileError(f'{path}: expected {{"rows": [...]}}')
    taken = dict(taken or {})
    out: Dict[str, ExtraRow] = {}
    for i, entry in enumerate(entries, 1):
        where = f"{path}: row {i}"
        if not isinstance(entry, dict):
            raise RowsFileError(f"{where}: expected an object")
        unknown = sorted(set(entry) - {"label", "kind", "source", "config", "skipped"})
        if unknown:
            raise RowsFileError(f"{where}: unknown key(s) {', '.join(unknown)}")
        label = check_label(entry.get("label"), where, {**taken, **out})
        kind = entry.get("kind", "tuner")
        if kind not in KINDS:
            raise RowsFileError(f"{where}: kind must be one of {', '.join(KINDS)}, not {kind!r}")
        skipped = entry.get("skipped")
        if skipped is not None and (not isinstance(skipped, str) or not skipped.strip()):
            raise RowsFileError(f"{where}: skipped must be the reason, as text")
        config = _config(entry.get("config"), where) if "config" in entry or not skipped else None
        out[label] = ExtraRow(config=config, kind=kind, source=str(entry.get("source") or ""),
                              skipped=skipped)
    return out


# --------------------------------------------------------------------------- expert rows


def expert_data() -> Dict[str, Any]:
    """rigtune/data/expert_rows.json, read through the installed package (only under --expert)."""
    text = resources.files("rigtune").joinpath("data", "expert_rows.json").read_text(encoding="utf-8")
    return json.loads(text)


def _vllm_batched_tokens(hw: HardwareDescriptor) -> int:
    from rigtune.backends.vllm import vllm_default_batched_tokens

    return vllm_default_batched_tokens(hw)


# The engine's own default for a setting on this card, for `applies_when_default_at_most`.
ENGINE_DEFAULTS: Dict[Tuple[str, str], Callable[[HardwareDescriptor], int]] = {
    ("vllm", "prefill_budget"): _vllm_batched_tokens,
}


def _eagle3_head(prepared: PreparedModel) -> Optional[str]:
    from rigtune.speculative import eagle3_heads_for

    heads = eagle3_heads_for(prepared.spec.hf_id)
    return heads[0] if heads else None


def _cite(s: Dict[str, Any]) -> str:
    quotes = "; ".join(f"\"{c['quote']}\" ({c['page']}, read at {' and '.join(c['read_at'])})"
                       for c in s["cites"])
    return f"`{s['flag']}`: {quotes}"


def _unmet(need: str, cfg: Config, backend: str, hw: HardwareDescriptor, reg: Dict[str, Any],
           model: PreparedModel) -> Optional[str]:
    """Why this card or model cannot have what a row `requires`, or None."""
    if need == "cuda" and not any(g.vendor == "nvidia" for g in hw.gpus):
        return "the page makes it CUDA-only, and this machine has no NVIDIA GPU"
    if need == "kv_dtype_offered" and cfg.kv_dtype not in reg[backend].kv_dtypes(hw):
        offered = ", ".join(reg[backend].kv_dtypes(hw)) or "none"
        return (f"rigtune does not run {backend}'s {cfg.kv_dtype} KV cache on this card (compute capability "
                f"{'.'.join(map(str, hw.gpu.cc)) if hw.gpu else '-'}; it runs: {offered})")
    if need == "eagle3_head" and _eagle3_head(model) is None:
        return f"rigtune lists no EAGLE-3 head for {model.spec.hf_id} (rigtune/speculative.py EAGLE3_HEADS)"
    return None


def expert_rows(hw: HardwareDescriptor, prepared: Dict[str, PreparedModel], reg: Dict[str, Any],
                data: Optional[Dict[str, Any]] = None) -> Dict[str, ExtraRow]:
    """The expert rows for this comparison's engines, card and model, each to measure or skipped with why."""
    from rigtune.bench.references import reference_configs

    data = data or expert_data()
    stock = reference_configs(hw, prepared, reg)
    rows: Dict[str, ExtraRow] = {}
    gpu = (hw.gpu.name if hw.gpu else "").lower()
    for row in data["rows"]:
        label, backend = row["label"], row["backend"]
        numerics = bool(row.get("changes_numerics"))
        source = "; ".join(_cite(s) for s in row["settings"])

        def skip(why: str, cfg: Optional[Config] = None) -> ExtraRow:
            return ExtraRow(config=cfg, kind="expert", source=source, skipped=why, changes_numerics=numerics)

        base = stock.get(row["from"])
        if backend not in prepared or base is None:
            rows[label] = skip(f"{backend} is not one of this comparison's engines")
            continue
        update: Dict[str, Any] = {}
        extra = dict(base.extra)
        reasons: List[str] = []
        notes: List[str] = []
        fallback: Optional[Dict[str, Any]] = None
        acts: Optional[Dict[str, Any]] = None
        for s in row["settings"]:
            limit, on = s.get("applies_when_default_at_most"), s.get("skip_on_gpu")
            if on is not None and on["match"] in gpu:
                reasons.append(f"not on an {on['match'].upper()}: {on['why']}")
                continue
            if limit is not None:
                default = ENGINE_DEFAULTS[(backend, s["field"])](hw)
                if default > limit:
                    quote = s["cites"][0]["quote"]
                    reasons.append(f"{backend}'s own default on this card, {s['field']} {default}, "
                                   f"already meets \"{quote}\"")
                    continue
            value = s["value"]
            if value == "{eagle3_head}":
                value = _eagle3_head(prepared[backend])
            alt = s.get("on_cards_without_it")
            if alt is not None and s["field"] == "kv_dtype":
                offered = reg[backend].kv_dtypes(hw)
                if value not in offered and alt["value"] in offered:
                    notes.append(f"{label}: this card does not have {backend}'s {value} KV cache as rigtune "
                                 f"runs it, so the row uses {alt['value']}: {_cite(alt)}; {alt['why']}")
                    value = alt["value"]
            if s.get("fallback") is not None and s["field"] == "kv_dtype":
                fallback = s["fallback"]
            acts = s.get("acts_up_to_users") or acts
            if s["field"].startswith("extra:"):
                extra[s["field"].split(":", 1)[1]] = value
            else:
                update[s["field"]] = value
        if reasons:  # a row stands for its guide's settings together: one that cannot apply drops the row
            rows[label] = skip("; ".join(reasons) + ", so the row would not be what the page recommends")
            continue
        cfg = base.model_copy(update={**update, "extra": extra})
        unmet = [why for need in row.get("requires", ())
                 if (why := _unmet(need, cfg, backend, hw, reg, prepared[backend])) is not None]
        if unmet:
            rows[label] = skip("; ".join(unmet), cfg)
            continue
        refusal = reg[backend].known_failure(cfg)
        if refusal:
            rows[label] = skip(f"rigtune expects it to fail: {refusal}", cfg)
            continue
        version = reg[backend].version(hw)
        tags = sorted({t.lstrip("v") for s in row["settings"] for c in s["cites"] for t in c["read_at"]})
        if version and version not in tags:
            notes.append(f"{label}: the page was read at {backend} {' and '.join(tags)}; this is {version}")
        out = ExtraRow(config=cfg, kind="expert", source=f"{row['from']} plus {source}", notes=notes,
                       changes_numerics=numerics)
        if fallback is not None:
            out.fallback = cfg.model_copy(update={"kv_dtype": fallback["value"]})
            out.fallback_note = (f"{label}: {cfg.kv_dtype} failed to start, so the row was measured with "
                                 f"{fallback['value']}: {_cite(fallback)}; {fallback['why']}")
        if acts is not None:
            out.acts_up_to_users = int(acts["users"])
            out.acts_note = acts["why"]
        rows[label] = out
    return rows
