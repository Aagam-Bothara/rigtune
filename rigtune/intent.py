"""Why to change a seed: rules that read what a seed's settings are for and propose the changes worth measuring.

A seed (rigtune.seed) is the expert's command. Each rule in data/intent_rules.json names when it holds (the model,
the GPU, the workload, what the seed already does), one change to measure against the seed, why, and the session or
source it rests on. The calibration measures each proposal right after the seed (search.py, stage "rule"), so the
variant stages and the confirm round see it like any other trial. A rule proposes; the measurements decide.

Conditions a rule's `when` may hold (all must):
  engine                       the seed's engine ("vllm", "sglang")
  architecture                 the model's architecture starts with this
  seed_spec                    "none", "any", or a drafter kind the seed uses ("dspark", "mtp", ...)
  seed_batch_at_least          the seed's batch
  seed_prefill_budget_unset    the seed leaves the token budget to the engine
  gpu_memory_gib_below         the GPU's memory
  prefill_tokens_at_most / prefill_tokens_at_least   the workload's prompt length
  max_users_at_least / max_users_at_most            the workload's busiest level
  model_has_mtp_head           the checkpoint ships an MTP head
A rule's `try` is Config fields; "half" for the batch halves the seed's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rigtune import speculative as sp
from rigtune.models import ArchInfo, Config

GiB = 1 << 30
CONDITIONS = ("engine", "architecture", "seed_spec", "seed_batch_at_least", "seed_prefill_budget_unset", "gpu_memory_gib_below",
              "prefill_tokens_at_most", "prefill_tokens_at_least", "max_users_at_least", "max_users_at_most",
              "model_has_mtp_head")
FIELDS = ("batch", "prefill_budget", "spec_decode", "kv_dtype", "ctx", "gpu_memory_utilization", "prefix_cache")


@dataclass
class Proposal:
    rule: str
    config: Config
    because: str
    evidence: str


def load_rules() -> List[Dict[str, Any]]:
    """The shipped rules, each checked: known conditions and fields, a reason and evidence. Raises ValueError."""
    data = json.loads(resources.files("rigtune").joinpath("data", "intent_rules.json").read_text(encoding="utf-8"))
    rules = data["rules"]
    for r in rules:
        for key in ("id", "when", "try", "because", "evidence"):
            if not r.get(key):
                raise ValueError(f"intent rule {r.get('id', '?')} has no {key}")
        unknown = set(r["when"]) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"intent rule {r['id']}: unknown conditions {sorted(unknown)}")
        bad = set(r["try"]) - set(FIELDS)
        if bad:
            raise ValueError(f"intent rule {r['id']}: cannot change {sorted(bad)}")
    if len({r["id"] for r in rules}) != len(rules):
        raise ValueError("intent rules share an id")
    return rules


def _spec_kind(spec: Optional[str]) -> str:
    return "none" if not spec else sp.parse(spec)[0]


def holds(when: Dict[str, Any], seed: Config, arch: ArchInfo, gpu_gib: float, prefill_tokens: int,
          max_users: int) -> bool:
    """Whether every condition of a rule holds for this seed, model, GPU and workload."""
    checks = {
        "engine": lambda v: seed.backend == v,
        "architecture": lambda v: (arch.architecture or "").startswith(v),
        "seed_spec": lambda v: (v == "any" and seed.spec_decode is not None) or _spec_kind(seed.spec_decode) == v,
        "seed_batch_at_least": lambda v: seed.batch >= v,
        "seed_prefill_budget_unset": lambda v: (seed.prefill_budget is None) == v,
        "gpu_memory_gib_below": lambda v: gpu_gib < v,
        "prefill_tokens_at_most": lambda v: prefill_tokens <= v,
        "prefill_tokens_at_least": lambda v: prefill_tokens >= v,
        "max_users_at_least": lambda v: max_users >= v,
        "max_users_at_most": lambda v: max_users <= v,
        "model_has_mtp_head": lambda v: (arch.mtp_layers > 0) == v,
    }
    return all(checks[k](v) for k, v in when.items())


def _changed(seed: Config, change: Dict[str, Any]) -> Config:
    update = dict(change)
    if update.get("batch") == "half":
        update["batch"] = max(1, seed.batch // 2)
    return seed.model_copy(update=update)


def propose(seed: Config, arch: ArchInfo, gpu_gib: float, prefill_tokens: int, max_users: int,
            offered_specs: Callable[[Config], Optional[Sequence[str]]],
            refusal: Callable[[Config], Optional[str]] = lambda cfg: None,
            rules: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Proposal], List[str]]:
    """The changes worth measuring against this seed, and notes on rules that held but were not tried.
    `offered_specs(cfg)` lists the speculation settings the engine offers for `cfg` with no speculation (None:
    speculation is off); a proposed drafter is tried only as offered. `refusal(cfg)` is the engine's reason it would
    not launch a configuration (backend.config_refusal), None when it would."""
    proposals: List[Proposal] = []
    notes: List[str] = []
    for r in rules if rules is not None else load_rules():
        if not holds(r["when"], seed, arch, gpu_gib, prefill_tokens, max_users):
            continue
        cfg = _changed(seed, r["try"])
        if cfg.key() == seed.key():
            continue
        spec = r["try"].get("spec_decode")
        if spec:
            offered = offered_specs(seed.model_copy(update={"spec_decode": None}))
            if offered is None:
                notes.append(f"rule {r['id']} held, but speculation is off (--speculative off): not tried")
                continue
            if spec not in offered:
                same = sorted(s for s in offered if _spec_kind(s) == _spec_kind(spec))
                notes.append(f"rule {r['id']} held, but this engine is not offered {spec} here"
                             + (f" (only {', '.join(same)})" if same else "") + ": not tried")
                continue
        why_not = refusal(cfg)
        if why_not:
            notes.append(f"rule {r['id']} held, but {cfg.key()} would not launch ({why_not}): not tried")
            continue
        proposals.append(Proposal(rule=r["id"], config=cfg, because=r["because"], evidence=r["evidence"]))
    return proposals, notes
