"""Orchestration: probe -> select -> prepare -> plan -> calibrate -> cache -> serve."""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rigtune import __version__
from rigtune import cache as profile_cache
from rigtune.backends import BaseBackend, registry as backend_registry
from rigtune.calibrate.objectives import Constraints, close_call_level
from rigtune.calibrate.search import ProgressFn, StagedSearch, SubprocessTrialRunner, TrialRunner
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.hardware import hardware_hash, llmtrace_version, probe
from rigtune.memory import plan as memory_plan
from rigtune.models import (
    Config,
    GiB,
    HardwareDescriptor,
    MemoryEstimate,
    MiB,
    ModelSpec,
    PreparedModel,
    Profile,
    TrialResult,
)
from rigtune.quantized import PREQUANTIZED
from rigtune.selector import select_backends

logger = logging.getLogger(__name__)

AUTO_QUANT = "auto"


def allowed_quants(supported: List[str], quants: Optional[List[str]]) -> Optional[List[str]]:
    """The precisions a backend may prepare under --quant. None: the backend lists none, so it decides.

    `auto` (the default) is every supported precision except the Hub's pre-quantized checkpoints:
    4-bit AWQ and GPTQ, which on Qwen2.5 3B and 7B raised perplexity by 26-36% where fp8 cost 1-2%,
    and 8-bit W8A8, whose quality has not been measured. A server should not trade quality away
    unless asked: `--quant auto,awq` or `--quant auto,w8a8` adds them back.
    """
    if not supported:
        return None
    wanted = quants if quants is not None else [AUTO_QUANT]
    return [q for q in supported if q in wanted or (AUTO_QUANT in wanted and q not in PREQUANTIZED)]


@dataclass
class PlanResult:
    hw: HardwareDescriptor
    spec: ModelSpec
    candidates: List[str]
    prepared: Dict[str, PreparedModel] = field(default_factory=dict)
    feasible: Dict[str, List[Tuple[Config, MemoryEstimate]]] = field(default_factory=dict)
    considered: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)  # for the profile, such as a Hub lookup that failed

    @property
    def all_feasible(self) -> List[Config]:
        return [c for cfgs in self.feasible.values() for c, _ in cfgs]

    @property
    def total_considered(self) -> int:
        return sum(self.considered.values())


@dataclass
class SearchOptions:
    """Switches for the optional search dimensions. The defaults explore everything that is safe."""

    # None = auto: every supported precision but 4-bit checkpoints. A list may include "auto".
    quants: Optional[List[str]] = None
    kv_quant: bool = True  # quantized KV caches
    speculative: bool = True  # n-gram and draft-model speculative decoding
    # --speculative all: also the methods that have never run on hardware (SGLang's n-gram, EAGLE3 head and draft
    # model, vLLM's EAGLE3 head), the engines' choice after speculation and a third speculation setting per engine.
    speculative_all: bool = False
    prefix_cache: bool = True  # keep prefix caching on, and tune it for shared-prefix workloads
    phase_tuning: bool = True  # the prefill-knob stage
    combine: bool = True  # combinations of the strategies that were promising on their own
    budget_s: Optional[float] = None  # --budget: seconds of calibration, after which later trials are skipped
    ttft_percentile: int = 95  # --ttft-percentile: which time to first token the ceiling applies to
    confirm: bool = True  # --confirm: re-measure the best three configurations and choose on those runs
    all_levels: bool = False  # --all-levels: measure every concurrency level of every trial (no early stop)
    explore: bool = False  # --explore: random draws around the leader after the stages (off: see search.py)
    # --quality-probe: measure how far each precision's answers drift from the most faithful precision that
    # runs, and record it in the profile (quality.py). It does not gate the search: the comparison saturates,
    # so there is no threshold worth enforcing. Off by default because it costs two short passes per precision.
    quality_probe: bool = False
    # --eval-set: grade each weight precision on labelled items and serve one below the most precise format only
    # when its accuracy loss is certified to be under max_accuracy_loss points (accuracy_gate.py). None = off.
    eval_set: Optional[str] = None
    max_accuracy_loss: float = 3.0  # --max-accuracy-loss, points
    eval_items: int = 300  # --eval-items: the first N items (0 = all)
    # --report-reuse: launch engines with their per-request cached-token reports for a flat workload too (a session
    # file always has them), so a flattened control records the reuse it got. It adds a launch flag, so a profile
    # calibrated with it is cached under its own name.
    report_reuse: bool = False
    # --kv-tier auto: after the variant stages, try a host-RAM (and perhaps disk) KV-cache tier on each tuned
    # engine's leader when its session trial shows the working set outgrowing the GPU pool (tier_variants_fn).
    # Session files only. "off" (the default) changes nothing: no trial, no launch flag, no cache-key entry.
    kv_tier: str = "off"

    # --arrival-rate (efficiency only): requests per second the server will see. Trials are then scored by joules
    # per request at that rate, the card's idle draw and wind-down included (calibrate/rate.py). None = J/token.
    arrival_rate: Optional[float] = None
    # --rate-check: with an arrival rate, re-measure the pick once with Poisson arrivals at it (search.py).
    rate_check: bool = True

    def key(self, objective: Optional[str] = None) -> Dict[str, str]:
        """Options that change the pick, recorded in the profile and in its cache path: the non-default
        ones, and for `balanced` (the objective with a TTFT ceiling) the percentile unless it is the
        median. Balanced profiles from before the percentile existed were picked by the median and
        carry no key for it, so they are served only where the median is asked for."""
        out: Dict[str, str] = {}
        if self.quants is not None:
            out["quant"] = "+".join(self.quants)
        if not self.kv_quant:
            out["kv"] = "off"
        if not self.speculative:
            out["spec"] = "off"
        elif self.speculative_all:
            out["spec"] = "all"
        if not self.prefix_cache:
            out["prefix"] = "off"
        if not self.phase_tuning:
            out["prefill"] = "off"
        if not self.combine:
            out["combine"] = "off"
        if not self.confirm:
            out["confirm"] = "off"
        if self.all_levels:
            out["levels"] = "all"
        if self.explore:
            out["explore"] = "on"
        if self.quality_probe:  # the probe costs trials' time, so a probed profile is cached under its own name
            out["quality"] = "on"
        if self.budget_s is not None:  # a budgeted profile is never served where a full one was asked for
            out["budget"] = f"{int(self.budget_s)}s"
        if self.eval_set:  # a gated profile answers a different question; ungated cache paths are untouched
            out["eval"] = f"{Path(self.eval_set).stem}-{self.eval_items or 'all'}"
            out["maxloss"] = f"{self.max_accuracy_loss:g}"
        if self.report_reuse:
            out["reuse"] = "on"
        if self.kv_tier != "off":
            out["tier"] = self.kv_tier
        if self.arrival_rate is not None:  # picked for one rate; J/token profiles keep their names
            out["rate"] = f"{self.arrival_rate:g}rps"
            if not self.rate_check:  # a profile without the check's evidence is not served where it was asked for
                out["ratecheck"] = "off"
        if objective == "balanced" and self.ttft_percentile != 50:
            out["ttft"] = f"p{self.ttft_percentile}"
        return out

    def search_quants(self) -> Optional[List[str]]:
        """The precisions calibration prepares. `auto` leaves the Hub's pre-quantized checkpoints out on quality
        measured elsewhere (4-5 GSM8K points on Qwen2.5-3B); with an eval set the gate measures that cost here and
        refuses what it cannot certify, so `auto` takes them too: on a 14B at 24 GB, GPTQ measured 3.9x the best
        fp8 configuration. An explicit list without `auto` is left as the user wrote it."""
        if not self.eval_set or (self.quants is not None and AUTO_QUANT not in self.quants):
            return self.quants
        return list(dict.fromkeys([*(self.quants or [AUTO_QUANT]), *PREQUANTIZED]))

    def allows(self, quant: str) -> bool:
        """Whether --quant permits this precision (see allowed_quants and search_quants)."""
        return quant in (allowed_quants([quant], self.search_quants()) or [])


def _usable(cached: Optional[Profile], opts: SearchOptions, force_backend: Optional[str]) -> bool:
    """Serve a cached profile only if it matches the forced backend and the current --quant: a
    profile calibrated when `auto` still picked 4-bit checkpoints must not outlive that default. A gated
    profile must also have been graded on the same eval set: its name is in the cache path, its contents are not."""
    if cached is None or (force_backend is not None and cached.backend != force_backend):
        return False
    if opts.eval_set:
        from rigtune.evalset import eval_set_sha256

        if cached.eval_set is None or cached.eval_set.sha256 != eval_set_sha256(opts.eval_set):
            return False
    return opts.allows(cached.config.quant)


def level_rule(objective: str, constraints: Constraints, opts: SearchOptions,
               power_points: Optional[list] = None) -> Optional[Callable]:
    """How a calibration trial measures its concurrency levels: busiest first, stopping once one settles the
    score (objectives.enough_level), or every level (None). Every level with --all-levels, which exists to
    check that shortcut, and with a power stage, which compares energy per token over every level."""
    if power_points or opts.all_levels:
        return None
    from rigtune.calibrate.objectives import enough_level

    return enough_level(objective, constraints)


EFFICIENCY_IDLE_TAIL_S = 3.0
RATE_IDLE_TAIL_S = 6.0


def idle_tail_for(objective: str, opts: SearchOptions) -> float:
    """Seconds each trial stays measured after its last request, for the card's idle draw and wind-down.

    Only `efficiency` records them: the other objectives never look at energy beyond a tie-break, and their
    trials keep their timing exactly. 3 s holds about 30 samples at NVML's 100 ms, of which the last second gives
    idle power once a wind-down of a second or two is over. With an arrival rate the pick depends on both
    figures, so the tail is doubled to leave a slower wind-down room to finish before idle is read.
    """
    if objective != "efficiency":
        return 0.0
    return RATE_IDLE_TAIL_S if opts.arrival_rate is not None else EFFICIENCY_IDLE_TAIL_S


def spec_state_notes_for(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult", cfg: Config,
                         opts: SearchOptions) -> List[str]:
    """Notes for the speculation settings a hybrid model's recurrent state keeps from the pick's batch
    (rigtune.memory.spec_state_notes). Empty for models without linear-attention layers."""
    pm = plan.prepared.get(cfg.backend)
    if not opts.speculative or pm is None or not pm.arch.linear_layers:
        return []
    from rigtune.memory import spec_state_notes

    be = reg[cfg.backend]
    gmu = cfg.gmu_before_spec if cfg.gmu_before_spec is not None else cfg.gpu_memory_utilization
    base = cfg.model_copy(update={"spec_decode": None, "gpu_memory_utilization": gmu, "gmu_before_spec": None})
    try:
        variants = be.spec_variants(base, pm, hw, opts.speculative_all)
        return spec_state_notes(hw, pm, be.memory_model(hw), base, variants)
    except Exception as exc:
        logger.debug("no speculation state notes for %s: %s", cfg.key(), exc)
        return []


def combination_fits(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult"):
    """Whether a combined config can run: it must fit in memory, and on vLLM and SGLang the prefill
    budget must cover the batch (each stage keeps that true on its own; a combination may not)."""
    from rigtune.memory import estimate

    small_gpu = hw.gpu is not None and hw.gpu.vram_total_bytes < 32 * 2**30

    def fn(c: Config) -> bool:
        # SGLang 0.5.19 captures prefill CUDA graphs up to its prefill budget, outside its memory fraction: on an
        # RTX 4090 (24 GB) every trial with a budget of 8192 or 16384 ran out of memory at start-up, at 93% and
        # at 88% reserved, in four calibrations.
        if c.backend == "sglang" and small_gpu and (c.prefill_budget or 0) >= 8192:
            return False
        pm = plan.prepared.get(c.backend)
        if pm is None or reg[c.backend].known_failure(c):
            return False
        if c.kv_tier is not None and tier_refusal(hw, reg[c.backend], pm, c) is not None:
            return False  # a combination may carry a tier to a config its engine or the host cannot give one
        # SGLang: a draft or EAGLE3 head needs its weights' room outside the reservation (SglangBackend.spec_fraction);
        # a second model of unknown size gets a limit of 0, so it is never combined in.
        cap = getattr(reg[c.backend], "spec_fraction", None)
        limit = cap(c.spec_decode, pm, hw) if cap is not None and c.spec_decode else None
        if limit is not None and (c.gpu_memory_utilization or 0) > limit + 1e-9:
            return False
        if c.prefill_budget is not None and c.prefill_budget < c.batch and not c.backend.startswith("llamacpp"):
            return False
        try:
            return estimate(hw, pm, c, reg[c.backend].memory_model(hw)).feasible
        except Exception:
            return False

    return fn


# --------------------------------------------------------------------------- KV-cache tiers (--kv-tier auto)

# The gate and the sizing, from the design reviewed for this stage (docs/decisions.md). Starting points, not
# measurements: the GPU validation in docs/benchmarks.md is what would confirm or move them.
TIER_WORKING_SET_GATE = 0.8  # try a tier once the working set is over 0.8x the GPU pool
TIER_HEADROOM = 1.25  # a tier sized at 1.25x the working set
TIER_OVER_POOL = 1.2  # and offered only above 1.2x the GPU pool: offloading is immediate, so a smaller tier
#                       only mirrors the pool (vLLM 0.29 docs/features/kv_offloading_usage.md)
# The gate is the working set against the pool alone. A reuse shortfall cannot open it on its own: below
# 0.8x the pool, 1.25x the working set is under 1.0x the pool, never the 1.2x a tier needs.


def _host_room(hw: HardwareDescriptor, be: BaseBackend) -> Tuple[int, str]:
    """Host bytes a tier may take on `be`, and what binds: the planner's budget with the engine's reserve
    ("host RAM"), or the engine's own cap where that is lower (vLLM: "/dev/shm")."""
    from rigtune.memory import host_tier_budget

    room = host_tier_budget(hw, be.tier_host_reserve())
    cap = be.tier_host_cap()
    if cap is not None and cap < room:
        return cap, "/dev/shm, where vLLM keeps its CPU tier,"
    return room, "host RAM"


def tier_refusal(hw: HardwareDescriptor, be: BaseBackend, pm: PreparedModel, c: Config) -> Optional[str]:
    """Why `c`, which carries a KV-cache tier, cannot run here, or None when it can: the engine's own rules,
    the host RAM it would take, and for a disk tier the engine's file tier, the directory rule and disk."""
    from rigtune import kvtier
    from rigtune.memory import disk_tier_budget

    reason = be.tier_refusal(c, pm)
    if reason:
        return reason
    host_gib, disk_gib = c.tier_gib
    need = be.tier_host_need(c, host_gib * GiB)
    room, binding = _host_room(hw, be)
    if need > room:
        return (f"the tier takes {need / GiB:.0f} GiB of host RAM and {room / GiB:.0f} GiB of {binding} is "
                "free")
    if disk_gib:
        reason = be.tier_disk_refusal()
        if reason:
            return reason
        root = kvtier.tier_root()
        reason = kvtier.refusal(root)
        if reason:
            return reason
        if disk_gib * GiB > disk_tier_budget(root):
            return f"{disk_gib} GiB of disk is not free under {root}"
    return None


def gpu_pool_bytes(leader: TrialResult, hw: HardwareDescriptor, pm: PreparedModel, mm) -> Tuple[int, str]:
    """The GPU KV-cache pool `leader` ran with, in bytes, and where the figure came from. "measured": what the
    engine logged it allocated (rigtune.memlog: vLLM's "Available KV cache memory", which is num_blocks x the
    page size of every KV cache group, recurrent state included; SGLang's pools summed). Not vLLM's "GPU KV
    cache size" line: that is max_concurrency x max_model_len (v1/core/kv_cache_utils
    .update_kv_cache_capacity), not tokens per pool, for a hybrid model. "planned": the memory fraction less
    weights and workspace."""
    measured = leader.memory.measured if leader.memory is not None else None
    if measured is not None and measured.kv_mb:
        return int(measured.kv_mb * MiB), "measured"
    c = leader.config
    if hw.gpu is None or c.gpu_memory_utilization is None:
        return 0, "unknown"
    reserved = int(c.gpu_memory_utilization * hw.gpu.vram_total_bytes)
    return max(0, reserved - pm.weights_for(c.quant) - mm.runtime_workspace), "planned"


def tier_variants_fn(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult", objective: str,
                     constraints: Constraints, workload: Optional[Workload] = None, counter=None):
    """--kv-tier auto: (leader trial) -> (configs to try, a note saying what was tried or why nothing was).

    Gate, on the leader's own session trial at the level the objective scored it: its working set W (the live
    sessions' last prompts and answers, in the GPU pool's units: memory.working_set_bytes) is over 0.8x the
    GPU pool G (gpu_pool_bytes). Otherwise no trial at all. A system prompt the sessions share (`workload`'s,
    as session replay finds it) is one prefix in the cache, not one per session, so the working set counts it
    once. Sizing: one host point S = min(1.25 W, the host budget), offered only when S >= 1.2 G, and one
    host + disk point (disk 1.25 W) only when the host budget cannot hold W and the engine bounds its disk
    tier."""
    import math

    from rigtune import kvtier
    from rigtune.calibrate.measure import _count, _shared_system
    from rigtune.calibrate.workload import TURN_OVERHEAD
    from rigtune.memory import disk_tier_budget, working_set_bytes
    from rigtune.models import format_kv_tier

    shared = _shared_system(workload) if workload is not None and workload.sessions else ""

    def shared_tokens(pm: PreparedModel) -> int:
        if not shared:
            return 0
        from rigtune.calibrate.tokens import TokenCounter

        return _count(shared, counter or TokenCounter.for_model(pm.spec.hf_id)) + TURN_OVERHEAD

    def fn(leader: TrialResult) -> Tuple[List[Config], Optional[str]]:
        c = leader.config
        be, pm = reg[c.backend], plan.prepared.get(c.backend)
        if c.kv_tier is not None or pm is None:
            return [], None
        lead = f"no KV-cache tier tried on {c.backend}"
        reason = be.tier_refusal(c, pm)
        if reason:
            return [], f"{lead}: {reason}"
        m = scored_level(leader, [leader], objective, constraints)
        if not m.working_set_tokens:
            return [], f"{lead}: its trial recorded no working set"
        mm = be.memory_model(hw)
        sessions = math.ceil(m.effective_pool or m.concurrency or 1)
        once = shared_tokens(pm)
        tokens = max(once, m.working_set_tokens - (sessions - 1) * once)
        w = working_set_bytes(pm, c, tokens, sessions, mm)
        g, source = gpu_pool_bytes(leader, hw, pm, mm)
        if g <= 0:
            return [], f"{lead}: the GPU pool's size is not known"
        seen = (f"working set {tokens:,} tokens ({w / GiB:.1f} GiB, {sessions} sessions"
                + (f", a {once:,}-token shared prefix counted once" if once else "")
                + f") is {w / g:.2f}x the GPU pool ({g / GiB:.1f} GiB, {source})"
                + (f", reuse {m.prefix_reuse:.2f} of {m.reuse_expected:.2f} expected"
                   if m.reuse_expected is not None and m.prefix_reuse is not None else ""))
        if w <= TIER_WORKING_SET_GATE * g:
            return [], f"{lead}: {seen}"
        room, binding = _host_room(hw, be)
        per_byte = be.tier_host_need(c, GiB) / GiB  # SGLang adds a second model's host pool
        host = min(TIER_HEADROOM * w, room / per_byte)
        if host < TIER_OVER_POOL * g:
            short = (f"{TIER_HEADROOM}x the working set ({TIER_HEADROOM * w / GiB:.1f} GiB)"
                     if TIER_HEADROOM * w < TIER_OVER_POOL * g
                     else f"the {room / GiB:.0f} GiB of {binding} free for a tier")
            return [], (f"{lead}: {seen}; {short} is under {TIER_OVER_POOL}x the GPU pool, and a tier that "
                        "small only mirrors the pool")
        host_gib = int(host // GiB)
        if host_gib < 1:  # tiers are sized in whole GiB
            return [], f"{lead}: {seen}; a tier of {host / GiB:.2f} GiB is under the 1 GiB step tiers come in"
        out = [c.model_copy(update={"kv_tier": format_kv_tier(host_gib)})]
        disk_note = ""
        if room / per_byte < w:  # the host cannot hold the working set: a disk tier behind it
            root = kvtier.tier_root()
            disk_gib = math.ceil(TIER_HEADROOM * w / GiB)
            why = be.tier_disk_refusal() or kvtier.refusal(root)
            if why is None and disk_tier_budget(root) < disk_gib * GiB:
                why = f"{disk_gib} GiB of disk is not free under {root}"
            if why is None:
                out.append(c.model_copy(update={"kv_tier": format_kv_tier(host_gib, disk_gib)}))
            else:
                disk_note = f"; no disk tier: {why}"
        tried = ", ".join(v.kv_tier for v in out)
        return out, f"KV-cache tier tried on {c.backend} ({tried}): {seen}{disk_note}"

    return fn


def retry_floor_fn(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult"):
    """The out-of-memory retry floor for configs that reserve less on purpose (SglangBackend.spec_retry_floor)."""
    def fn(c: Config) -> Optional[float]:
        floor = getattr(reg.get(c.backend), "spec_retry_floor", None)
        pm = plan.prepared.get(c.backend)
        return floor(c, pm, hw) if floor is not None and pm is not None and c.spec_decode else None

    return fn


ENGINE_NAMES = {"vllm": "vLLM", "sglang": "SGLang"}
# Engine bugs a load failure's log line identifies: (backend, text in the line) -> what the note adds.
KNOWN_LOAD_BUGS = {("sglang", "lm_head_is_tied"): ", a known SGLang bug for Gemma 4 unified models, fixed in SGLang 0.5.21 "
                   "(sgl-project/sglang #38820, #42091)"}
TEXT_ONLY_NOTE = ("served text-only (--language-model-only): image, video and audio inputs are not accepted, and "
                  "rigtune serve answers them with HTTP 400")


def startup_failure_fn(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult"):
    """What the search does after a start-up failure whose cause the engine log shows (search.startup_cause): a
    multimodal model whose multimodal profiling failed is retried text-only where the engine can serve it so
    (BaseBackend.text_only_refusal); anything else is noted, never retried. Returns (config to retry, note)."""
    from rigtune.calibrate.search import LOAD_FAILED, startup_cause

    def fn(c: Config, res) -> Tuple[Optional[Config], Optional[str]]:
        be, pm = reg.get(c.backend), plan.prepared.get(c.backend)
        kind, line = startup_cause(res.error)
        if be is None or pm is None or kind is None:
            return None, None
        engine = f"{ENGINE_NAMES.get(c.backend, c.backend)} {be.version(hw) or '(version unknown)'}"
        if kind == LOAD_FAILED:
            known = next((why for (b, mark), why in KNOWN_LOAD_BUGS.items() if b == c.backend and mark in line), "")
            return None, (f"{pm.spec.hf_id} could not start on {engine}: the engine's model code failed while "
                          f"loading it ({line}){known}; a load failure, not memory, so it is not retried")
        # Seen with vLLM 0.29.0 and transformers 5.23.0, a newer transformers than that vLLM was released against.
        what = (f"{pm.spec.hf_id} could not start on {engine} with its multimodal profiling ({line}), as when the "
                "engine and its transformers mismatch")
        if c.text_only:
            return None, f"{what}, even text-only"
        if not pm.arch.multimodal_arch:
            return None, f"{what}; not retried text-only: {pm.arch.architecture} is not a multimodal architecture"
        why = be.text_only_refusal()
        if why:
            return None, f"{what}; not retried text-only: {why}"
        return c.model_copy(update={"text_only": True}), f"{what}; retried text-only (--language-model-only)"

    return fn


def kv_variants_fn(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult"):
    """Leader variants with a quantized KV cache, plus the batch step that the smaller cache admits."""
    from rigtune.memory import estimate

    def fn(c: Config) -> List[Config]:
        be = reg[c.backend]
        pm = plan.prepared.get(c.backend)
        if pm is None:
            return []
        mm = be.memory_model(hw)

        def fits(x: Config) -> bool:
            if be.known_failure(x):
                return False
            try:
                return estimate(hw, pm, x, mm).feasible
            except Exception:
                return False

        out: List[Config] = []
        for kd in be.kv_dtypes(hw):
            if kd == c.kv_dtype:
                continue
            v = c.model_copy(update={"kv_dtype": kd})
            if fits(v):
                out.append(v)
            bigger = [b for b in be.batch_ladder() if b > c.batch]
            if bigger:
                update: Dict[str, object] = {"batch": bigger[0]}
                if v.prefill_budget is not None and v.prefill_budget < bigger[0]:
                    update["prefill_budget"] = None  # vLLM: the budget must cover the batch
                vb = v.model_copy(update=update)
                if fits(vb) and not fits(vb.model_copy(update={"kv_dtype": c.kv_dtype})):
                    out.append(vb)
        return out

    return fn


def search_space(hw: HardwareDescriptor, plan: "PlanResult", reg: Dict[str, BaseBackend], workload: Workload,
                 options: Optional[SearchOptions] = None) -> List[Config]:
    """Every runnable configuration calibration could reach, each once: the planner's configurations, then each
    stage's variants of everything so far (prefill budgets, quantized KV caches with the batch step they allow,
    prefix-cache settings, speculative methods), so every combination of them is included. The explore stage
    draws from it, and so does random search in benchmarks/search_baselines.py."""
    opts = options or SearchOptions()
    stages: List[Callable[[Config], List[Config]]] = []
    if opts.phase_tuning:
        stages.append(lambda c: reg[c.backend].prefill_variants(c))
    if opts.kv_quant:
        stages.append(kv_variants_fn(hw, reg, plan))
    if opts.prefix_cache and workload.reuses_prefix:
        stages.append(lambda c: reg[c.backend].prefix_variants(c))
    if opts.speculative:
        stages.append(lambda c: reg[c.backend].spec_variants(c, plan.prepared[c.backend], hw, opts.speculative_all))
    _warn_if_only_quantized_fits(plan, opts)
    configs = list(plan.all_feasible)
    if not opts.prefix_cache:
        configs = [c.model_copy(update={"prefix_cache": False}) for c in configs]
    for stage in stages:
        configs += [v for c in configs for v in stage(c)]
    fits = combination_fits(hw, reg, plan)
    out: Dict[str, Config] = {}
    for c in configs:
        if c.key() not in out and fits(c):
            out[c.key()] = c
    return list(out.values())


def scored_level(winner, results, objective: str, constraints: Constraints):
    """The concurrency level the objective scored the winning trial at (objectives.Ranked.metrics): what a session
    profile records as served, rather than the trial's highest-throughput level, which the objective may have
    rejected for breaking a ceiling. Falls back to the trial's summary when the winner is not among the ranked."""
    from rigtune.calibrate.objectives import rank

    return next((r.metrics for r in rank(results, objective, constraints) if r.result is winner), winner.metrics)


def select(
    hw: HardwareDescriptor, spec: ModelSpec, force: Optional[str] = None, workload: Optional[Workload] = None,
    report_reuse: bool = False,
) -> Tuple[List[str], Dict[str, BaseBackend]]:
    """The candidate backends. A session `workload`, or `report_reuse` (--report-reuse), builds them to report
    per-request cached tokens, so what prefix reuse each trial achieved is measured; no other launch arguments
    change."""
    reg = backend_registry(report_reuse=report_reuse or bool(workload is not None and workload.sessions))
    return select_backends(hw, spec, reg, force=force), reg


FULL_PRECISION = ("bf16", "fp16", "fp32")


def _warn_if_only_quantized_fits(plan, opts) -> None:
    """Say so when the card cannot hold the model at full precision and 4-bit checkpoints are not being tried.

    `--quant auto` leaves the Hub's pre-quantized checkpoints out, because they can cost answer quality, and that
    is the right default — but it is the wrong silence. On a 24 GB card a 14B fits in none of bf16, and the 4-bit
    checkpoints `auto` skips measured 3.9x the best configuration it does consider, for about 2 GSM8K points. A
    user who never hears that cannot make the trade; one who hears it can, which is why this is a message and not
    a change of default.
    """
    if opts.quants is not None or opts.eval_set:  # the user chose the list, or the gate is trying them
        return
    feasible = plan.all_feasible
    if not feasible or any(c.quant in FULL_PRECISION for c in feasible):
        return
    logging.getLogger(__name__).warning(
        "%s does not fit this GPU at full precision, so calibration is limited to %s. The Hub's pre-quantized "
        "4-bit checkpoints are excluded by --quant auto because they can cost answer quality: on a 14B at 24 GB "
        "they measured 3.9x faster for about 2 GSM8K points. Add --quant auto,awq,gptq to consider them, and "
        "benchmarks/task_quality.py to grade the result, or --eval-set gsm8k to have calibration grade them.",
        plan.spec.hf_id, ", ".join(sorted({c.quant for c in feasible})))


def quality_probe_for(opts, workload):
    """The quality probe for a calibration, or None when `--quality-probe` was not asked for.

    Prompts come from a copy of the workload. File and real-text workloads carry none until the measurement
    path materialises them per concurrency level (calibrate/workload.py), and asking the original for them
    would change what the first level measures. The first version of this guard tested `workload.prompts`
    directly, which is empty here for exactly the workloads a user gates, so the gate silently did nothing on
    its first hardware run: a 14B calibration kept a 4-bit checkpoint no one had compared against anything.
    A probe that cannot be built now says so rather than disabling itself quietly.
    """
    if not opts.quality_probe:
        return None
    import logging
    from dataclasses import replace

    from rigtune.quality import QualityProbe

    # Up to 48 prompts, answered twice per precision at 32 tokens each: enough that one changed answer is about
    # 2% rather than the 6% sixteen gave, and still seconds of generation. A workload materialises one
    # concurrency level's prompts at a time (32 on the chat preset), so later levels are drawn at their own
    # offsets until there are enough; a synthetic workload regenerates the same prompts, hence the de-duplication.
    wanted, seen, prompts = 48, set(), []
    base = replace(workload, prompts=[], fitted=False)
    for k in range(4):
        level = replace(base, prompts=[], fitted=False,
                        sample_offset=base.sample_offset + k * max(1, base.n_prompts)).ensure_prompts()
        for text in level.prompts:
            if text not in seen:
                seen.add(text)
                prompts.append(text)
        if len(prompts) >= wanted:
            break
    prompts = prompts[:wanted]
    if not prompts:
        logging.getLogger(__name__).warning(
            "--quality-probe was asked for, but workload %s produced no prompts to compare answers on; "
            "no drift will be measured", workload.name)
        return None
    return QualityProbe(prompts=prompts)


def accuracy_gate_for(opts: SearchOptions, feasible: List[Config]):
    """The accuracy gate for a calibration, or None without --eval-set.

    The set is read and every item checked here, before any engine starts, so a bad line fails in seconds rather
    than after stage 1. Items are also checked against the context the feasible configurations offer: an engine
    rejects a prompt that, with its answer, does not fit, and an unanswered item narrows the comparison."""
    if not opts.eval_set:
        return None
    from rigtune.accuracy_gate import AccuracyGate, best_certifiable
    from rigtune.evalset import EvalSetError, load_eval_set
    from rigtune.models import EvalSetInfo

    if not 0 < opts.max_accuracy_loss < 100:
        raise EvalSetError(f"--max-accuracy-loss is in points and must be between 0 and 100, not "
                           f"{opts.max_accuracy_loss:g}")
    es = load_eval_set(opts.eval_set, opts.eval_items)
    gate = AccuracyGate(items=es.items, max_loss=opts.max_accuracy_loss,
                        info=EvalSetInfo(name=es.name, path=es.path, items=len(es.items), total=es.total,
                                         sha256=es.sha256, max_accuracy_loss=opts.max_accuracy_loss))
    if feasible:
        # Three characters a token undercounts English (about four), so this errs towards warning.
        longest = max(es.items, key=lambda i: i.chars)
        need = longest.chars // 3 + (longest.max_tokens or gate.max_tokens)
        smallest, largest = min(c.ctx for c in feasible), max(c.ctx for c in feasible)
        where = f"item {longest.id}" if longest.id else "the longest item"
        if need > largest:
            raise EvalSetError(f"{es.path}: {where} needs about {need} tokens with its answer, more than the largest "
                               f"feasible context ({largest}); shorten it or lower its max_tokens")
        if need > smallest:
            logger.warning("%s: %s needs about %d tokens with its answer, more than the smallest feasible context "
                           "(%d); a precision graded at that context may leave it unanswered", es.path, where, need,
                           smallest)
    logger.info("accuracy gate: %d %s items, tolerance %.1f points; with no answer changed they certify down to "
                "%.1f points", len(es.items), es.name, opts.max_accuracy_loss, best_certifiable(len(es.items)))
    if best_certifiable(len(es.items)) >= opts.max_accuracy_loss:
        logger.warning("accuracy gate: %d items cannot certify %.1f points even if no answer changes, so every "
                       "precision below the most precise one will be refused; use more items", len(es.items),
                       opts.max_accuracy_loss)
    return gate


def prepare_and_plan(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    candidates: List[str],
    reg: Dict[str, BaseBackend],
    materialize: bool = False,
    workload: Optional[Workload] = None,
    quants: Optional[List[str]] = None,
    speculative_all: bool = False,
) -> PlanResult:
    """Prepare each candidate backend's model, enumerate its grid, keep what fits the workload.

    `quants` restricts weight precisions (the --quant option); None allows everything supported. With
    `speculative_all` (--speculative all), each backend also looks up its second models on the Hub; nothing is
    looked up otherwise.
    """
    result = PlanResult(hw=hw, spec=spec, candidates=list(candidates))
    min_ctx = workload.min_ctx if workload else 0
    for name in candidates:
        backend = reg[name]
        try:
            allowed = allowed_quants(backend.supported_quants(hw), quants)
            if quants is not None and allowed == []:
                raise RuntimeError(f"none of --quant {','.join(quants)} is available on {name}")
            prepared = backend.prepare(spec, hw, quants=allowed)
            if speculative_all:
                result.notes += backend.find_second_models(prepared)
            grid = backend.candidate_configs(hw, prepared, min_ctx=min_ctx)
            if not grid:
                raise RuntimeError(
                    f"model max context {prepared.arch.max_position_embeddings} < workload minimum {min_ctx}"
                )
            kept = memory_plan(hw, prepared, grid, backend.memory_model(hw))
            result.prepared[name] = prepared
            result.feasible[name] = kept
            result.considered[name] = len(grid)
            logger.info("%s: %d/%d configs feasible", name, len(kept), len(grid))
            if materialize and kept:
                survivors = sorted({c.quant for c, _ in kept})  # not `quants`: that filter applies to every backend
                backend.materialize(prepared, survivors)
                # Re-plan with true file sizes (GGUF sizes are estimates until downloaded).
                result.feasible[name] = memory_plan(hw, prepared, grid, backend.memory_model(hw))
        except Exception as exc:
            logger.exception("%s: prepare/plan failed", name)
            result.errors[name] = str(exc)
    return result


def _profile_for(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    objective: str,
    backend: BaseBackend,
    cfg: Config,
    prepared: PreparedModel,
    table=None,
    notes: Optional[List[str]] = None,
    workload: Optional[Workload] = None,
    prepared_all: Optional[Dict[str, PreparedModel]] = None,
    served: Optional[object] = None,
) -> Profile:
    launch = backend.launch_spec(cfg, prepared, 0)
    workload = workload or get_workload("default")
    return Profile(
        rigtune_version=__version__,
        hardware_hash=hardware_hash(hw),
        hardware=hw,
        model_id=spec.hf_id,
        objective=objective,
        workload=workload.name,
        workload_spec=workload.spec(served),
        backend=backend.name,
        backend_version=backend.version(hw),
        config=cfg,
        prepared=prepared,
        prepared_all=dict(prepared_all or {}),
        launch_args=launch.args,
        launch_env=launch.env,
        calibration_table=list(table or []),
        llmtrace_version=llmtrace_version(),
        notes=list(notes or []),
    )


def default_profile(hw: HardwareDescriptor, spec: ModelSpec, plan: PlanResult, reg: Dict[str, BaseBackend],
                    objective: str = "balanced", workload: Optional[Workload] = None) -> Profile:
    """Uncalibrated: first candidate backend with its default config (or the largest feasible one)."""
    workload = workload or get_workload("default")
    for name in plan.candidates:
        prepared = plan.prepared.get(name)
        if prepared is None:
            continue
        backend = reg[name]
        cfg = backend.default_config(hw, prepared, min_ctx=workload.min_ctx)
        feasible = plan.feasible.get(name) or []
        if feasible and not any(c.key() == cfg.key() for c, _ in feasible):
            # Default does not fit; take the feasible config with the most headroom for ctx/batch.
            cfg = max(feasible, key=lambda ce: (ce[0].ctx, ce[0].batch))[0]
        if cfg.quant not in prepared.weights_bytes:
            cfg.quant = next(iter(prepared.weights_bytes))
        backend.materialize(prepared, [cfg.quant])
        return _profile_for(hw, spec, objective, backend, cfg, prepared, notes=["uncalibrated defaults"],
                            workload=workload, prepared_all=plan.prepared)
    raise RuntimeError(f"no usable backend for {spec.hf_id}; errors: {plan.errors}")


def calibrate(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    objective: str,
    plan: PlanResult,
    reg: Dict[str, BaseBackend],
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    progress: Optional[ProgressFn] = None,
    runner: Optional[TrialRunner] = None,
    log_dir: Optional[Path] = None,
    power_mode: str = "off",
    power_controller: Optional[object] = None,
    power_points: Optional[list] = None,
    phase_tuning: bool = True,
    options: Optional[SearchOptions] = None,
) -> Profile:
    opts = options or SearchOptions()
    workload = workload or get_workload("default")
    constraints = constraints or Constraints(ttft_ceiling_ms=workload.ttft_ceiling_ms,
                                             tpot_ceiling_ms=workload.tpot_ceiling_ms,
                                             ttft_percentile=opts.ttft_percentile)
    if opts.arrival_rate is not None:
        if objective != "efficiency":
            raise RuntimeError(f"--arrival-rate scores energy per request, so it applies to --objective efficiency "
                               f"only, not {objective}")
        # The options are the one source of the rate: the constraints score with it, the profile's workload records
        # it so the server can compare live traffic against it.
        constraints = dataclasses.replace(constraints, arrival_rate=opts.arrival_rate)
        workload = dataclasses.replace(workload, arrival_rate_rps=opts.arrival_rate)
    feasible = plan.all_feasible
    if not feasible:
        raise RuntimeError("memory planner left no feasible configuration; try a smaller model or quant")
    if not opts.prefix_cache:
        feasible = [c.model_copy(update={"prefix_cache": False}) for c in feasible]
    stages = []
    if opts.kv_quant:
        stages.append(("kv", kv_variants_fn(hw, reg, plan)))
    if opts.prefix_cache and workload.reuses_prefix:
        stages.append(("prefix", lambda c: reg[c.backend].prefix_variants(c)))
    if opts.speculative:
        stages.append(("spec", lambda c: reg[c.backend].spec_variants(c, plan.prepared[c.backend], hw,
                                                                         opts.speculative_all)))
    controller, points, power_notes = None, [], []
    if power_points is not None:  # injected (tests, or a caller that already probed the GPU)
        controller, points = power_controller, list(power_points)
    elif power_mode != "off":
        controller, points, power_notes = setup_power(hw, power_mode, power_controller)
    from rigtune import kvtier

    kvtier.sweep()  # disk tiers a killed rigtune left behind, whatever this calibration asks for
    tier_fn, tier_notes = None, []
    if opts.kv_tier == "auto":
        if workload.sessions:
            tier_fn = tier_variants_fn(hw, reg, plan, objective, constraints, workload)
        else:  # the CLI refuses this; a caller of the API gets the note
            tier_notes.append("--kv-tier auto needs a session file (the working set it is sized from); no "
                              "KV-cache tier was tried")
    probe = quality_probe_for(opts, workload)
    gate = accuracy_gate_for(opts, feasible)
    if runner is not None and gate is not None and getattr(runner, "gate", False) is None:
        runner.gate = gate  # an injected runner that grades (tests) gets the same gate the search decides with
    if runner is None:
        runner = SubprocessTrialRunner(
            backends={n: reg[n] for n in plan.candidates},
            models=plan.prepared,
            hw=hw,
            workload=workload,
            log_dir=log_dir or (profile_cache.logs_dir() / spec.safe_id),
            power=controller,
            enough=level_rule(objective, constraints, opts, points),
            close_call=close_call_level(objective, constraints),
            quality=probe,
            gate=gate,
            idle_tail_s=idle_tail_for(objective, opts),
        )
    from rigtune.predict import Predictor

    search = StagedSearch(objective=objective, runner=runner, constraints=constraints, progress=progress,
                          predictor=Predictor(hw), models=plan.prepared, workload=workload,
                          power_points=points, variant_stages=stages, combine=opts.combine,
                          budget_s=opts.budget_s, confirm_top=3 if opts.confirm else 0,
                          feasible_fn=combination_fits(hw, reg, plan), quality=probe, gate=gate,
                          explore_space=search_space(hw, plan, reg, workload, opts) if opts.explore else [],
                          spec_first=True, spec_beam=3 if opts.speculative_all else 2, rate_check=opts.rate_check,
                          decide_after_spec=opts.speculative_all,
                          retry_floor=retry_floor_fn(hw, reg, plan) if opts.speculative_all else None,
                          prefill_variants=((lambda c: reg[c.backend].prefill_variants(c))
                                            if phase_tuning and opts.phase_tuning else None),
                          tier_variants=tier_fn, startup_failure=startup_failure_fn(hw, reg, plan))
    t0 = time.monotonic()
    try:
        winner, notes = search.run(feasible)
    finally:
        if controller is not None and getattr(controller, "applied", None) is not None:
            controller.restore()  # type: ignore[attr-defined]
    if winner is None:
        raise RuntimeError("calibration failed: " + "; ".join(notes))
    elapsed = time.monotonic() - t0
    notes = power_notes + list(plan.notes) + tier_notes + notes
    notes += spec_state_notes_for(hw, reg, plan, winner.config, opts)
    if winner.config.text_only:
        notes.append(TEXT_ONLY_NOTE)
    trials =sum(1 for r in search.results if not r.metrics.open_loop)  # the rate check is a check, not a trial
    notes.append(f"calibration took {elapsed:.0f}s over {trials} trials")
    notes.append(f"workload: {workload.describe()}")
    backend = reg[winner.config.backend]
    prepared = plan.prepared[winner.config.backend]
    profile = _profile_for(hw, spec, objective, backend, winner.config, prepared, table=search.results,
                           notes=notes, workload=workload, prepared_all=plan.prepared,
                           served=scored_level(winner, search.results, objective, constraints))
    profile.calibration_seconds = elapsed
    profile.calibration_trials = trials
    profile.power_mode = power_mode
    profile.options = opts.key(objective)
    if gate is not None:
        profile.accuracy = gate.records()
        profile.eval_set = gate.info
    return profile


def setup_power(hw: HardwareDescriptor, mode: str, controller: Optional[object] = None):
    """Probe what this GPU permits and build the stage-4 points. Returns (controller, points, notes)."""
    from rigtune.power import PowerControlUnavailable, candidate_points, controller_for

    if hw.gpu is None:
        return None, [], [f"--power {mode} requested but no GPU is visible; power stage skipped"]
    ctl = controller or controller_for(hw.gpu.index)
    try:
        caps = ctl.capabilities()  # type: ignore[attr-defined]
    except PowerControlUnavailable as exc:
        return None, [], [f"--power {mode} requested but unavailable: {exc}; power stage skipped"]
    notes = []
    if mode in ("cap", "both") and not caps.can_cap:
        notes.append(f"power capping not permitted: {caps.reasons.get('cap', 'unknown reason')}")
    if mode in ("clock", "both") and not caps.can_lock:
        notes.append(f"clock locking not permitted: {caps.reasons.get('clock', 'unknown reason')}")
    points = candidate_points(caps, mode)
    trying = [p for p in points if not p.is_default]
    if not trying:
        notes.append("no power settings available to try; power stage skipped")
        return None, [], notes
    notes.append(f"power stage tried {len(trying)} settings: " + ", ".join(p.label() for p in trying))
    return ctl, points, notes


def resolve_profile(
    spec: ModelSpec,
    objective: str = "balanced",
    force_backend: Optional[str] = None,
    recalibrate: bool = False,
    skip_calibration: bool = False,
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    progress: Optional[ProgressFn] = None,
    hw: Optional[HardwareDescriptor] = None,
    on_stage: Optional[Callable[[str], None]] = None,
    power_mode: str = "off",
    phases: str = "unified",
    kv_connector: str = "nixl",
    options: Optional[SearchOptions] = None,
    layout: str = "single",
) -> Profile:
    """Cached profile if valid, else run the full pipeline and cache the result.

    `phases` other than "unified" resolves the unified profile first, then measures disaggregated
    prefill/decode pairs built around it (see rigtune.disagg).
    """
    say = on_stage or (lambda s: None)
    workload = workload or get_workload("default")
    say("probe")
    hw = hw or probe()
    opts = options or SearchOptions()
    if phases != "unified" and layout != "single":
        raise RuntimeError("choose either --phases (disaggregated prefill/decode) or --layout (replicas, tp), not both")
    if phases != "unified" and not skip_calibration:
        return _resolve_phased(spec, objective, force_backend, recalibrate, workload, constraints, progress, hw,
                               on_stage, power_mode, phases, kv_connector, opts)
    if layout != "single" and not skip_calibration:
        return _resolve_layout(spec, objective, force_backend, recalibrate, workload, constraints, progress, hw,
                               on_stage, power_mode, opts, layout)
    if not recalibrate and not skip_calibration:
        cached = profile_cache.load(hw, spec, objective, workload.name, power_mode, options=opts.key(objective))
        if _usable(cached, opts, force_backend):
            say("cache hit")
            return cached
    say("select")
    candidates, reg = select(hw, spec, force=force_backend, workload=workload, report_reuse=opts.report_reuse)
    if not candidates:
        raise RuntimeError("no backend supports this machine/model; run `rigtune probe` for details")
    say("prepare + plan")
    plan = prepare_and_plan(hw, spec, candidates, reg, materialize=not skip_calibration, workload=workload,
                            quants=opts.search_quants(), speculative_all=opts.speculative_all)
    if skip_calibration:
        say("defaults")
        return default_profile(hw, spec, plan, reg, objective, workload=workload)
    say("calibrate")
    profile = calibrate(hw, spec, objective, plan, reg, workload=workload, constraints=constraints, progress=progress,
                        power_mode=power_mode, options=opts)
    profile_cache.save(profile)
    return profile


def _resolve_phased(spec: ModelSpec, objective: str, force_backend: Optional[str], recalibrate: bool,
                    workload: Workload, constraints: Optional[Constraints], progress: Optional[ProgressFn],
                    hw: HardwareDescriptor, on_stage: Optional[Callable[[str], None]], power_mode: str,
                    phases: str, kv_connector: str, options: Optional[SearchOptions] = None) -> Profile:
    from rigtune.disagg import calibrate_disaggregated, check_support

    say = on_stage or (lambda s: None)
    if not recalibrate:
        cached = profile_cache.load(hw, spec, objective, workload.name, power_mode, phases,
                                    options=(options or SearchOptions()).key())
        if _usable(cached, options or SearchOptions(), force_backend):
            say("cache hit")
            return cached
    if phases == "disaggregated":
        # Refuse before spending a unified calibration on a machine that cannot disaggregate.
        reason = check_support(hw, "vllm", kv_connector)
        if reason is None and not hw.backend_available("vllm"):
            reason = f"vLLM is not available ({hw.backends.get('vllm').reason if hw.backends.get('vllm') else 'unknown'})"
        if reason:
            raise RuntimeError(f"--phases disaggregated is not possible here: {reason}")
    # Disaggregation runs on vLLM; `auto` lets the unified search choose freely and falls back if needed.
    base = resolve_profile(spec, objective, force_backend=force_backend or ("vllm" if phases == "disaggregated" else None),
                           recalibrate=recalibrate, workload=workload, constraints=constraints, progress=progress,
                           hw=hw, on_stage=on_stage, power_mode=power_mode if phases == "auto" else "off",
                           options=options)
    _, reg = select(hw, spec, force=base.backend, workload=workload,
                    report_reuse=bool(options is not None and options.report_reuse))
    say("disaggregate")
    profile = calibrate_disaggregated(hw, spec, objective, base, reg, workload=workload, constraints=constraints,
                                      phases=phases, connector=kv_connector, progress=progress, power_mode=power_mode,
                                      log_dir=profile_cache.logs_dir() / spec.safe_id)
    profile_cache.save(profile)
    return profile


def _resolve_layout(spec: ModelSpec, objective: str, force_backend: Optional[str], recalibrate: bool,
                    workload: Workload, constraints: Optional[Constraints], progress: Optional[ProgressFn],
                    hw: HardwareDescriptor, on_stage: Optional[Callable[[str], None]], power_mode: str,
                    options: SearchOptions, layout: str) -> Profile:
    from rigtune.layout import calibrate_layout

    say = on_stage or (lambda s: None)
    key = {**options.key(objective), "layout": layout}
    if not recalibrate:
        cached = profile_cache.load(hw, spec, objective, workload.name, power_mode, options=key)
        if _usable(cached, options, force_backend):
            say("cache hit")
            return cached
    gpus = [g for g in hw.gpus if g.vendor == "nvidia"]
    if layout != "auto" and len(gpus) < 2:
        raise RuntimeError(f"--layout {layout} is not possible here: multi-GPU layouts need two NVIDIA GPUs; "
                           f"{len(gpus)} visible")
    single = options
    if options.kv_tier != "off":  # a tier is budgeted for one engine on one GPU: no tier stage under a layout
        import dataclasses

        single = dataclasses.replace(options, kv_tier="off")
    base = resolve_profile(spec, objective, force_backend=force_backend, recalibrate=recalibrate, workload=workload,
                           constraints=constraints, progress=progress, hw=hw, on_stage=on_stage,
                           power_mode=power_mode, options=single)
    _, reg = select(hw, spec, force=base.backend, workload=workload, report_reuse=options.report_reuse)
    say("layout")
    profile = calibrate_layout(hw, objective, base, reg, layout, workload=workload, constraints=constraints,
                               progress=progress, log_dir=profile_cache.logs_dir() / spec.safe_id)
    if single is not options:
        profile.options = {**profile.options, "tier": options.kv_tier}  # cached where it is looked up
        profile.notes.append(f"--kv-tier {options.kv_tier} was not tried under --layout {layout}: a KV-cache "
                             "tier's host RAM is budgeted for one engine on one GPU")
    profile_cache.save(profile)
    return profile
