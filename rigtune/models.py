"""Core data types shared across the pipeline.

Everything here is plain data (pydantic) so it can be serialised into the
profile cache and printed by the CLI without importing any backend.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_serializer

Objective = Literal["throughput", "latency", "balanced", "efficiency"]
OBJECTIVES: Tuple[str, ...] = ("throughput", "latency", "balanced", "efficiency")

MiB = 1024 * 1024
GiB = 1024 * MiB


# --------------------------------------------------------------------------- hardware


class GPUInfo(BaseModel):
    vendor: str  # "nvidia" | "amd" | "apple" | "unknown"
    name: str
    index: int = 0
    compute_capability: Optional[Tuple[int, int]] = None
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    uuid: Optional[str] = None

    @property
    def cc(self) -> Tuple[int, int]:
        return self.compute_capability or (0, 0)


class CPUInfo(BaseModel):
    model_name: str = "unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    avx2: bool = False
    avx512: bool = False
    ram_total_bytes: int = 0
    ram_free_bytes: int = 0
    arch: str = "unknown"  # x86_64 | aarch64 | ...


class BackendAvailability(BaseModel):
    name: str
    available: bool
    version: Optional[str] = None
    reason: Optional[str] = None
    # Versions of the libraries an engine release depends on most (torch, transformers, flashinfer, tokenizers),
    # read in the engine's own environment. No part of the hardware hash, but compared when a cached profile is
    # reused (cache.library_changes): torch, transformers and tokenizers by major.minor, FlashInfer exactly, since
    # it ships the attention kernels. vLLM 0.29.0 declared only transformers>=5.10.4, so two installs of the same
    # vLLM could differ here, and one that resolved transformers 5.23.0 failed Gemma 4's multimodal start-up.
    libs: Dict[str, str] = Field(default_factory=dict)


class HardwareDescriptor(BaseModel):
    os: str
    python: str
    gpus: List[GPUInfo] = Field(default_factory=list)
    cpu: CPUInfo
    backends: Dict[str, BackendAvailability] = Field(default_factory=dict)
    probed_at: float = Field(default_factory=time.time)

    @property
    def gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    def backend_available(self, name: str) -> bool:
        b = self.backends.get(name)
        return bool(b and b.available)

    def backend_version(self, name: str) -> Optional[str]:
        b = self.backends.get(name)
        return b.version if b else None

    @property
    def available_memory_bytes(self) -> int:
        """Memory the planner budgets against: free VRAM if a GPU exists, else free RAM."""
        if self.gpu is not None:
            return self.gpu.vram_free_bytes
        return self.cpu.ram_free_bytes


# --------------------------------------------------------------------------- model


class ModelSpec(BaseModel):
    hf_id: str
    revision: Optional[str] = None

    @property
    def safe_id(self) -> str:
        return self.hf_id.replace("/", "__")


class ArchInfo(BaseModel):
    """What the memory planner needs from an HF config."""

    architecture: str = "unknown"
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    num_params: Optional[int] = None
    torch_dtype: str = "bfloat16"
    # Hybrid attention layouts, read from the config's `layer_types` by rigtune.hfconfig. All defaulted, so
    # profiles cached before these existed still load: None / 0 means every layer keeps a full KV cache.
    full_attention_layers: Optional[int] = None  # set only for a recognised hybrid layout
    full_kv_heads: Optional[int] = None  # full layers' KV heads where they differ (Gemma 4); else num_kv_heads
    full_head_dim: Optional[int] = None  # full layers' head size where it differs (Gemma 4); else head_dim
    sliding_layers: int = 0  # layers that keep at most `sliding_window` tokens per sequence
    sliding_window: Optional[int] = None
    linear_layers: int = 0  # linear-attention (Gated DeltaNet) layers: fixed recurrent state, no KV
    linear_state_bytes: int = 0  # recurrent state one sequence holds, summed over the linear layers
    # What that state grows by per speculative token: the conv state keeps one more column for each.
    linear_conv_step_bytes: int = 0
    layout_note: Optional[str] = None  # why a hybrid-looking layout fell back to the every-layer-full estimate
    # The config has a vision or audio sub-config (vision_config, audio_config). Defaulted: profiles saved before it
    # load as False, and `multimodal_arch` still recognises them by name.
    has_mm_config: bool = False
    # Layers of the checkpoint's own multi-token-prediction head, as its config declares them
    # (rigtune.introspect.mtp_layers); 0 for none. The planner counts them only for an "mtp" speculation setting,
    # which is offered only once the head's weights were found. Defaulted, so older profiles load.
    mtp_layers: int = 0
    # Bytes of the safetensors files the engines load (hfconfig.loaded_bytes); None when unknown. Not saved: a profile
    # keeps the weights_bytes worked out from it.
    checkpoint_bytes: Optional[int] = Field(default=None, exclude=True)

    @property
    def multimodal_arch(self) -> bool:
        """A model with an image, video or audio encoder beside its language model."""
        return self.has_mm_config or self.architecture.endswith("ForConditionalGeneration")

    def kv_bytes_per_token(self, kv_dtype_bytes: int) -> int:
        # 2 (K and V) x layers x kv_heads x head_dim x bytes.  For MHA models
        # kv_heads*head_dim == hidden_size, which matches the spec formula; for GQA
        # models this is the correct (smaller) number.  It counts every layer as full
        # attention; see is_hybrid for the layouts where that is not so.
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * kv_dtype_bytes

    @property
    def is_hybrid(self) -> bool:
        """A recognised layout where some layers keep no full KV cache (sliding or linear attention)."""
        return self.full_attention_layers is not None

    def full_kv_bytes_per_token(self, kv_dtype_bytes: float) -> float:
        """KV bytes per token over the full-attention layers only (hybrid layouts)."""
        heads = self.full_kv_heads or self.num_kv_heads
        dim = self.full_head_dim or self.head_dim
        return 2 * (self.full_attention_layers or 0) * heads * dim * kv_dtype_bytes

    def sliding_kv_bytes_per_token(self, kv_dtype_bytes: float) -> float:
        """KV bytes per token over the sliding-window layers only (hybrid layouts)."""
        return 2 * self.sliding_layers * self.num_kv_heads * self.head_dim * kv_dtype_bytes


class PreparedModel(BaseModel):
    spec: ModelSpec
    backend: str
    arch: ArchInfo
    # vLLM/SGLang: the HF id (or local snapshot dir). llama.cpp: path to a GGUF file per quant.
    hf_path: Optional[str] = None
    gguf_paths: Dict[str, str] = Field(default_factory=dict)  # quant -> path
    weights_bytes: Dict[str, int] = Field(default_factory=dict)  # quant/precision -> bytes
    hf_paths: Dict[str, str] = Field(default_factory=dict)  # quant -> pre-quantized repo (AWQ, GPTQ)
    draft_paths: Dict[str, str] = Field(default_factory=dict)  # draft model id -> GGUF path (llama.cpp)
    # Second models for speculative decoding that the Hub has and that fit this target: spec kind -> repository
    # ("eagle3" -> a verified head), and repository -> size in bytes, for the memory left outside the engine's
    # reservation. The checkpoint's own MTP head, once its weights were found, is "mtp" -> the model itself; its size
    # is under "mtp", and that of the target's embedding and output head (which the draft first copies) under
    # "mtp_shared". Empty in profiles saved before they existed, which still load.
    spec_repos: Dict[str, str] = Field(default_factory=dict)
    spec_bytes: Dict[str, int] = Field(default_factory=dict)
    spec_revisions: Dict[str, str] = Field(default_factory=dict)
    spec_paths: Dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)
    # Only verified mode populates pairs. Paths stay private; records use portable aliases.
    spec_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    drafts_mode: str = "table"

    @model_serializer(mode="wrap")
    def _omit_unverified_drafts(self, handler) -> Dict[str, Any]:
        """Written only by a verified-drafts calibration, so a table-mode profile is saved as it always was."""
        data = handler(self)
        for name, unset in (("spec_revisions", {}), ("spec_pairs", []), ("drafts_mode", "table")):
            if data.get(name) == unset:
                data.pop(name, None)
        return data

    def weights_for(self, quant: str) -> int:
        if quant in self.weights_bytes:
            return self.weights_bytes[quant]
        raise KeyError(f"no weight size recorded for quant {quant!r}")


# --------------------------------------------------------------------------- config


class Config(BaseModel):
    """One concrete launch configuration for a backend.

    Fields are the union of what all v1 backends understand; each backend
    ignores the ones that do not apply to it.
    """

    backend: str
    quant: str  # "bf16" | "fp16" | "fp8" | "Q4_K_M" | ...
    ctx: int = 4096  # max model length / context window
    batch: int = 64  # max_num_seqs (vLLM/SGLang) or n_parallel (llama.cpp)
    gpu_memory_utilization: Optional[float] = None  # vLLM/SGLang
    n_gpu_layers: Optional[int] = None  # llama.cpp
    n_batch: Optional[int] = None  # llama.cpp logical batch
    # Prefill knob: chunked-prefill token budget (vLLM max_num_batched_tokens, SGLang
    # chunked_prefill_size) or llama.cpp micro-batch (-ub). None = the engine's own default.
    prefill_budget: Optional[int] = None
    kv_dtype: str = "auto"
    prefix_cache: Optional[bool] = None  # None = the engine default (on for vLLM and SGLang)
    # speculative decoding: "ngram|ngram_gpu|suffix|mtp:<k>" | "draft|eagle3:<hf_id>:<k>"
    spec_decode: Optional[str] = None
    spec_head_quant: Optional[str] = None
    tp: int = 1  # tensor-parallel degree across GPUs
    # A KV-cache tier below the GPU pool (vLLM's offloading connector, SGLang's hierarchical cache):
    # "host:<GiB>" of host RAM, or "host:<GiB>+disk:<GiB>" with a directory under rigtune's own cache dir
    # behind it. None = GPU only, as in every profile before tiers. Only --kv-tier auto tries one.
    kv_tier: Optional[str] = None
    # Engine kernel and compiler settings (rigtune.kernels), e.g. "attn=TRITON_ATTN,cg=FULL": kept in one spelling,
    # sorted by key, so one setting has one config key. None launches the engine with its own choices, as before.
    kernels: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    # Serve a multimodal model's language model alone (vLLM --language-model-only): set only by the retry after the
    # engine's multimodal profiling failed at start-up. Such an engine accepts no image, video or audio input.
    text_only: bool = False
    # The memory fraction before a speculation setting lowered it to leave a second model room (SGLang under
    # --speculative all), so undoing or swapping the speculation restores it. Not part of the key: the launch is
    # the same whichever fraction it came from.
    gmu_before_spec: Optional[float] = None
    # Energy tuning, applied through NVML while the backend runs rather than as launch flags.
    power_limit_w: Optional[int] = None  # board power cap
    sm_clock_mhz: Optional[int] = None  # upper bound of the locked SM clock range

    def key(self) -> str:
        parts = [self.backend, self.quant, f"ctx{self.ctx}", f"b{self.batch}"]
        if self.gpu_memory_utilization is not None:
            parts.append(f"gmu{self.gpu_memory_utilization:.2f}")
        if self.n_gpu_layers is not None:
            parts.append(f"ngl{self.n_gpu_layers}")
        if self.n_batch is not None:
            parts.append(f"nb{self.n_batch}")
        if self.prefill_budget is not None:
            parts.append(f"pb{self.prefill_budget}")
        if self.kv_dtype != "auto":
            parts.append(f"kv{self.kv_dtype}")
        if self.prefix_cache is not None:
            parts.append("pc" if self.prefix_cache else "nopc")
        if self.spec_decode is not None:
            parts.append(f"sd={self.spec_decode}")  # no brackets: rich reads [..] as console markup
        if self.tp > 1:
            parts.append(f"tp{self.tp}")
        if self.kv_tier is not None:
            parts.append("tier=" + self.kv_tier.replace(":", ""))  # host16, host16+disk40
        for k in sorted(self.extra):
            if k != "threads":
                parts.append(f"{k}={self.extra[k]}")
        if self.text_only:  # only when set, so every key from before it is unchanged
            parts.append("textonly")
        if self.power_limit_w is not None:
            parts.append(f"pl{self.power_limit_w}")
        if self.sm_clock_mhz is not None:
            parts.append(f"clk{self.sm_clock_mhz}")
        if self.spec_head_quant is not None:
            parts.append(f"shq={self.spec_head_quant}")
        if self.kernels is not None:  # only when set, so every key from before it is unchanged
            parts.append(f"kern={self.kernels}")
        return "/".join(parts)

    @field_validator("kernels")
    @classmethod
    def _one_spelling(cls, value: Optional[str]) -> Optional[str]:
        from rigtune.kernels import canonical

        return None if value is None else canonical(value)

    @model_serializer(mode="wrap")
    def _omit_unset_head_quant(self, handler) -> Dict[str, Any]:
        data = handler(self)
        if self.spec_head_quant is None:
            data.pop("spec_head_quant", None)
        if self.kernels is None:  # written only when set, so every profile from before it is saved as it was
            data.pop("kernels", None)
        return data

    def base_key(self) -> str:
        """Key without energy settings, so power variants of one configuration share it."""
        return self.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None}).key()

    def short(self) -> str:
        return self.key()

    @property
    def tier_gib(self) -> Tuple[int, int]:
        """(host GiB, disk GiB) of the KV-cache tier; (0, 0) without one."""
        return parse_kv_tier(self.kv_tier)


def format_kv_tier(host_gib: int, disk_gib: int = 0) -> str:
    """Config.kv_tier for `host_gib` GiB of host RAM, with `disk_gib` GiB of disk behind it if given."""
    return f"host:{int(host_gib)}" + (f"+disk:{int(disk_gib)}" if disk_gib else "")


def parse_kv_tier(tier: Optional[str]) -> Tuple[int, int]:
    """(host GiB, disk GiB) from Config.kv_tier; (0, 0) for None. Raises ValueError for anything else."""
    if tier is None:
        return 0, 0
    sizes = {"host": 0, "disk": 0}
    for part in tier.split("+"):
        kind, _, gib = part.partition(":")
        if kind not in sizes or not gib.isdigit() or sizes[kind]:
            raise ValueError(f"not a KV-cache tier: {tier!r} (expected host:<GiB> or host:<GiB>+disk:<GiB>)")
        sizes[kind] = int(gib)
    if sizes["host"] <= 0:
        raise ValueError(f"not a KV-cache tier: {tier!r} (a tier starts with host RAM)")
    return sizes["host"], sizes["disk"]


class MemoryEstimate(BaseModel):
    config_key: str
    weights: int
    kv_cache: int
    runtime_workspace: int
    safety_margin: int
    total: int
    budget: int
    feasible: bool

    @property
    def total_gib(self) -> float:
        return self.total / GiB


# --------------------------------------------------------------------------- calibration


class AnswerStats(BaseModel):
    """What a level's answers looked like, beside how fast they came: a configuration can win on speed by
    answering longer, or by looping on text that n-gram speculation accepts easily. Computed from what the
    load generator already receives (calibrate/measure.py answer_stats)."""

    answers: int = 0  # successful requests
    mean_tokens: Optional[float] = None
    median_tokens: Optional[float] = None
    # The engine's finish_reason per answer: "stop" (ended on its own), "length" (cut at max_tokens), ...;
    # "none" where the stream sent none.
    finish_reasons: Dict[str, int] = Field(default_factory=dict)
    # Over answers of at least 8 words: the mean share of word 4-grams that already occurred earlier in the
    # same answer (0 for text that never repeats itself, near 1 for a loop), and how many answers are at least
    # half repeats. None when no answer text was kept (session replay keeps none).
    repeated_4gram_share: Optional[float] = None
    looping_answers: int = 0
    judged_for_repetition: int = 0


class SpecAcceptance(BaseModel):
    """How much of what speculative decoding drafted the model kept, over a trial's measured requests, as the
    engine logged it (memlog.parse_acceptance). Read only from the log lines written while the trial's workload
    ran, so the answers a gate or quality probe graded afterwards on the same engine are not in it."""

    rate: float  # accepted / drafted tokens; from SGLang's log, its "accept rate" averaged over its lines
    mean_length: float  # tokens per verification step, 1 + accepted / drafts; from a log, the engine's own figure
    accepted_tokens: Optional[int] = None  # vLLM only: SGLang logs rates, not counts
    drafted_tokens: Optional[int] = None
    lines: Optional[int] = None  # from a log: how many of its lines; None from the counters
    # "vllm_metrics" (the change in vLLM's /metrics counters over the workload), "vllm_log" (its SpecDecoding lines,
    # where the counters could not be read) or "sglang_log" (its decode-batch lines)
    source: str


class TrialMetrics(BaseModel):
    tok_s: float = 0.0  # aggregate output tokens per second
    ttft_ms: Optional[float] = None  # p50 time to first token (None = not measured / failed)
    ttft_p95_ms: Optional[float] = None
    tpot_ms: Optional[float] = None  # mean time per output token after the first
    peak_mem_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    power_w: Optional[float] = None  # mean device power during the trial
    sm_clock_mhz: Optional[float] = None  # mean observed SM clock; confirms a clock lock took effect
    joules_per_token: Optional[float] = None
    # The same window's energy from NVML's total-energy counter, where the driver reports it (Volta and newer).
    # joules_per_token stays the integral of sampled power, so no recorded number moves; the two
    # side by side check every J/tok measured that way.
    counter_energy_j: Optional[float] = None
    duration_s: float = 0.0
    requests: int = 0
    failed: int = 0
    output_tokens: int = 0
    concurrency: int = 1
    telemetry_source: str = "none"  # "llmtrace" | "pynvml" | "psutil" | "none"
    token_count_source: str = "none"  # "usage" | "tokenizer" | "chunks" (approximate) | "none"
    prompt_tokens: int = 0  # mean measured prompt length (0 if no tokenizer)
    by_concurrency: Dict[str, "TrialMetrics"] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)  # up to three distinct request errors, when requests failed
    # Every successful request's time to first token, sorted, in ms: what a percentile's confidence interval
    # needs (calibrate/tail.py). Kept per concurrency level.
    ttft_samples_ms: List[float] = Field(default_factory=list)
    resampled: bool = False  # the level was too close to its ceiling to call and was measured again
    # Session files, and flat files with --report-reuse (calibrate/measure.py _session_stats). Prompt tokens the
    # engine counted, and how many it said it served from its prefix cache; prefix_reuse is their ratio, from the
    # replies' usage or, when no reply in the level carried a count, from the engine's /metrics counters
    # (reuse_source says which, or "none"), split into first steps (a shared system prompt) and later steps (the
    # history). reuse_expected is an estimate of what the conversations allowed, counted in characters and blind to
    # the engines' cache blocks: compare engines on prefix_reuse_later, not on the gap. Achieved far below expected
    # means eviction, or a chat template that does not re-render an earlier step identically.
    prompt_tokens_total: int = 0
    cached_tokens_total: int = 0
    prefix_reuse: Optional[float] = None
    prefix_reuse_first: Optional[float] = None
    prefix_reuse_later: Optional[float] = None
    reuse_expected: Optional[float] = None
    reuse_expected_first: Optional[float] = None
    reuse_expected_later: Optional[float] = None
    reuse_source: str = "none"  # "usage" | "metrics" | "none"
    sessions: int = 0
    sessions_cut: int = 0  # stopped before a step whose prompt would not fit
    sessions_truncated: int = 0  # stopped with steps left because the level's turn budget was spent
    mean_in_flight: Optional[float] = None  # time-weighted requests in flight: below the concurrency, a thinned level
    effective_pool: Optional[float] = None  # mean sessions live at dispatch: what a --session-pool actually reached
    reuse_distance: Optional[float] = None  # median requests dispatched between two steps of one session (P - 1)
    idle_gap: Optional[float] = None  # median of those dispatched after its previous step finished (P - c)
    working_set_tokens: Optional[int] = None  # peak of the live sessions' last prompts and answers, in tokens
    ttft_by_turn_ms: Dict[str, float] = Field(default_factory=dict)  # median time to first token, by step number
    # The engine's prompt counts for the steps served (p50/p90/max/mean), and steps reached per session (mean/max):
    # what a session profile records for the drift watch, from the level its winning trial was scored at.
    prompt_tokens_seen: Dict[str, int] = Field(default_factory=dict)
    turns_seen: Dict[str, float] = Field(default_factory=dict)
    # Trials with a KV-cache tier only (calibrate/measure.py _tier_stats); empty or None on every other trial.
    # Cached prompt tokens by where they came from: SGLang's own split per reply ("device", "host",
    # "storage"), else vLLM's /metrics counters ("device", and "tier" for host and disk: vLLM does not split).
    cached_by_source: Dict[str, int] = Field(default_factory=dict)
    ttft_returning_ms: Optional[float] = None  # median time to first token over steps after a session's first
    ttft_returning_p95_ms: Optional[float] = None
    # Median time to first token of those returning steps, by where their cache hit came from: "storage",
    # "host" or "device" (the slowest tier any of its tokens came from), "hit" where the engine does not say,
    # or "miss".
    ttft_by_hit_ms: Dict[str, float] = Field(default_factory=dict)

    # Explicit requests (agent traces; calibrate/measure.py _behind_edit): the share of prompt tokens a session had
    # sent before but placed behind a change, so the engine prefilled them again. Only what a rewriting layout refills
    # beyond an appending one of the same events is the layout's doing; eviction refills either way.
    refill_behind_edit: Optional[float] = None
    # By tag (a request's kind unless the file tagged it): requests, prompt_tokens, reuse_expected, prefix_reuse,
    # refill_behind_edit and median ttft_ms; and the median time from a chain's first send to its last request's first
    # token (_by_kind). Report only: no objective reads them.
    reuse_by_kind: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    chain_latency_ms: Optional[float] = None

    # The card after the trial's last request (measure.IdleSummary), recorded on the trial when an idle tail was
    # measured: under `efficiency`, or with an arrival rate. Closed-loop levels never let the GPU idle, so none
    # of this is in joules_per_token; a server at a modest arrival rate pays it between requests.
    idle_power_w: Optional[float] = None  # from the energy counter where the driver reports it, else sampled
    idle_power_sampled_w: Optional[float] = None  # the sampled figure, recorded either way
    hold_joules: Optional[float] = None  # energy above idle while winding down, from the same source as idle
    hold_below_idle: bool = False  # the tail measured below idle (noise): hold_joules recorded as 0
    hold_s: Optional[float] = None  # until power was within 10% of the drop; an upper bound as NVML samples it
    hold_truncated: bool = False  # the tail ended while power was still falling
    hold_after_concurrency: Optional[int] = None  # the level the tail followed
    energy_source: Optional[str] = None  # "nvml_counter" | "trapezoid": what idle and hold were computed from
    idle_tail_s: Optional[float] = None
    # An open-loop run (the rate check): requests sent at Poisson arrival times rather than c always in flight.
    # Never ranked with the closed-loop trials; it checks the J/request the rate model predicted for the pick.
    open_loop: bool = False
    arrival_rate_rps: Optional[float] = None
    joules_per_request: Optional[float] = None  # the whole window's energy, gaps included, over its requests
    # What the answers looked like (AnswerStats), for a comparison under --verdict only. Left out of the file
    # when unset, so every trial measured without it is written exactly as before.
    answers: Optional[AnswerStats] = None
    # Speculative decoding's acceptance over the trial, for a trial that drafted and whose engine logged it. Left out
    # of the file when unset, like answers, so every other trial is written exactly as before.
    spec_acceptance: Optional[SpecAcceptance] = None

    @model_serializer(mode="wrap")
    def _omit_unset_answers(self, handler) -> Dict[str, Any]:
        data = handler(self)
        for name in ("answers", "spec_acceptance"):
            if data.get(name) is None:
                data.pop(name, None)
        return data

    # A trial is not invalidated by a couple of degenerate requests. Small models sometimes emit
    # end-of-sequence immediately, producing no tokens; a backend that is actually broken fails
    # everything, not 2 requests in 48.
    FAILURE_TOLERANCE: ClassVar[float] = 0.10

    @property
    def ok(self) -> bool:
        if self.requests <= 0 or self.output_tokens <= 0:
            return False
        return self.failed <= self.FAILURE_TOLERANCE * self.requests

    def failure_summary(self) -> str:
        """'12/48 requests failed: <first error>' for a trial's error message."""
        msg = f"{self.failed}/{self.requests} requests failed"
        return f"{msg}: {self.errors[0]}" if self.errors else msg


class DisaggSpec(BaseModel):
    """Disaggregated serving: one engine per phase, each on its own GPU, joined by KV-cache transfer."""

    prefill: Config
    decode: Config
    prefill_gpu: int = 0
    decode_gpu: int = 1
    connector: str = "nixl"
    kv_transfer_config: Dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:
        return f"pd[gpu{self.prefill_gpu}:{self.prefill.key()} -> gpu{self.decode_gpu}:{self.decode.key()}]"


class MemoryObservation(BaseModel):
    """Planner prediction next to what the backend actually allocated, for one trial."""

    predicted: Optional[MemoryEstimate] = None
    measured: Optional["MeasuredMemory"] = None


class StartupCost(BaseModel):
    """What one engine launch cost before it served its first request. A value that was not observed is None,
    never 0: an engine that never became ready has no start-up time, and a log line that was not found says
    nothing about what it would have reported."""

    seconds: Optional[float] = None  # launch to the first healthy answer; None when it never became ready
    # Read from the engine's own log (memlog.parse_startup); each None unless its line was found.
    capture_s: Optional[float] = None  # CUDA-graph capture time, all passes; the slowest worker's when several report
    cudagraph_gib: Optional[float] = None  # memory the kept graphs took (the last pass), of that same worker
    capture_lines: int = 0  # capture lines read: one per worker per pass
    # seconds torch.compile took, summed over the compiled parts (vLLM's own total; a cache load reads its load time)
    compile_s: Optional[float] = None
    compile_cached: Optional[bool] = None  # every range loaded from the compile cache; None when no line says
    # The first forward's profiling/warm-up run, summed over the compiled parts: where Triton compiles and autotunes
    # its kernels when its cache is cold (74 s on Qwen3.5-9B on an A100, 0.3 s warm)
    warmup_s: Optional[float] = None
    rewrites: List[str] = Field(default_factory=list)  # settings the engine said it changed, "cudagraph_mode=NAME"
    parsed: List[str] = Field(default_factory=list)  # which lines were found: an empty list read nothing


class TrialResult(BaseModel):
    config: Config
    stage: str
    metrics: TrialMetrics
    launched: bool = True
    error: Optional[str] = None
    started_at: float = Field(default_factory=time.time)
    memory: Optional[MemoryObservation] = None
    disagg: Optional[DisaggSpec] = None  # set for trials of a disaggregated prefill/decode pair
    replicas: int = 1  # engines measured behind a load balancer (the replicas layout)
    # Why the trial failed, one of calibrate/search.py FAILURE_KINDS, set where its error text is written; None for a
    # trial that succeeded. The error keeps its prefix, so everything that reads the cause from it still does, and
    # search.failure_kind reads it back for results recorded without this field. Left out of every file a model dump
    # writes (profiles, comparison results), which stay as they were; the trial ledger records it.
    failure_kind: Optional[str] = Field(default=None, exclude=True)
    # What the launch behind this trial cost to start; None for a trial that launched nothing. Left out of every
    # model dump like failure_kind, so profiles and comparison results stay as they were; the trial ledger records it
    # beside the trial, not inside it.
    startup: Optional[StartupCost] = Field(default=None, exclude=True)

    @property
    def ok(self) -> bool:
        return self.launched and self.error is None and self.metrics.ok


from rigtune.memlog import MeasuredMemory  # noqa: E402  (after MemoryEstimate is defined)

MemoryObservation.model_rebuild()


class EvalSetInfo(BaseModel):
    """The labelled items an accuracy-gated profile was graded on (--eval-set)."""

    name: str  # "gsm8k", or the file's stem
    path: str
    items: int  # items graded: the first `items` of the file
    total: int = 0  # items in the file
    sha256: str  # of the whole file: a cached profile is reused only on the same set
    max_accuracy_loss: float  # the tolerance, in points


class PrecisionAccuracy(BaseModel):
    """One weight precision's grades and the accuracy gate's decision on them (rigtune/accuracy_gate.py)."""

    backend: str
    quant: str
    items: int
    answered: int
    correct: int
    accuracy: float  # correct / answered
    ci95: Tuple[float, float]  # Wilson interval
    reference: bool = False
    # Against `against` (its engine's reference, or another engine's), over the items both answered.
    against: Optional[str] = None  # "backend/quant"
    cross_engine: bool = False  # compared with another engine's reference: engine and template are confounded
    paired: Optional[int] = None
    lost: Optional[int] = None  # the reference right, this one wrong
    gained: Optional[int] = None
    delta_points: Optional[float] = None
    delta_ci95: Optional[Tuple[float, float]] = None  # score interval, points
    mcnemar_p: Optional[float] = None
    admitted: Optional[bool] = None  # None: not gated (the reference, or its precision on another engine)
    reason: str = ""
    # The reference answers twice; how its grades moved between the passes (its own noise).
    self_lost: Optional[int] = None
    self_gained: Optional[int] = None
    seconds: float = 0.0  # grading time
    # Per item, in eval-set order: '1' right, '0' wrong, '-' unanswered. With the set's sha256 this is enough to
    # decide again at another tolerance without a GPU.
    outcomes: str = ""
    outcomes_repeat: Optional[str] = None  # the reference's second pass
    # Set on the record of the pick's own configuration (KV cache and speculation included), graded during confirm.
    config: Optional[str] = None


class Profile(BaseModel):
    """What gets cached: the winner plus everything needed to reproduce the decision."""

    # Written as polyserve_version before the rename; profiles saved then still load.
    rigtune_version: str = Field(validation_alias=AliasChoices("rigtune_version", "polyserve_version"))
    hardware_hash: str
    hardware: HardwareDescriptor
    model_id: str
    # The Hub commit the calibration measured (hfconfig.resolve_revision), None for a local directory or when it
    # could not be read. A reused profile is compared with the local snapshot's commit when both are known.
    model_revision: Optional[str] = None
    # Every field of the Constraints the calibration was scored with. Empty in profiles written before they were
    # recorded: those cannot show they answer today's request, so they are recalibrated once (pipeline._usable).
    calibration_constraints: Dict[str, Any] = Field(default_factory=dict)
    # The busiest measured concurrency at which the pick met its constraints (objectives.capacity_level), else the
    # level it was scored at. `rigtune serve --max-in-flight measured` derives its limit from it. None in older
    # profiles.
    served_concurrency: Optional[int] = None
    objective: str
    workload: str = "default"
    workload_spec: Dict[str, Any] = Field(default_factory=dict)
    power_mode: str = "off"  # "off" | "cap" | "clock" | "both"
    phases: str = "unified"  # requested mode: "unified" | "disaggregated" | "auto"
    layout: str = "single"  # requested multi-GPU layout: "single" | "replicas" | "tp" | "auto"
    replicas: int = 1  # engines served behind the load balancer
    # Non-default search options that shaped this profile; part of its cache path.
    options: Dict[str, str] = Field(default_factory=dict)
    backend: str
    backend_version: Optional[str]
    config: Config
    prepared: Optional[PreparedModel] = None
    # Every candidate backend's prepared model, so trials from losing backends stay analysable.
    prepared_all: Dict[str, PreparedModel] = Field(default_factory=dict)
    disagg: Optional[DisaggSpec] = None  # set when serving disaggregated prefill/decode
    launch_args: List[str]
    launch_env: Dict[str, str] = Field(default_factory=dict)
    calibration_table: List[TrialResult] = Field(default_factory=list)
    calibration_seconds: float = 0.0
    calibration_trials: int = 0
    llmtrace_version: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    notes: List[str] = Field(default_factory=list)
    # --eval-set: what each weight precision scored and what the accuracy gate decided. Empty when ungated.
    accuracy: List[PrecisionAccuracy] = Field(default_factory=list)
    eval_set: Optional[EvalSetInfo] = None
    # The calibration's id in the trial ledger (rigtune/ledger.py), where every trial behind this profile is
    # recorded. Left out of the file when unset, so a profile calibrated without the ledger, and every profile
    # written before it, is written exactly as before.
    calibration_id: Optional[str] = None

    @model_serializer(mode="wrap")
    def _omit_unset_calibration_id(self, handler) -> Dict[str, Any]:
        data = handler(self)
        if data.get("calibration_id") is None:
            data.pop("calibration_id", None)
        return data
