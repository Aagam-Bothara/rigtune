"""Extract measured memory components from backend logs.

The planner predicts weights + kv + workspace. To calibrate it we need the same components as the
backend actually allocated. vLLM, SGLang and llama.cpp all print them at startup; NVML peak
(measured by the trial telemetry) is the ground truth for the total.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

MiB = 1024 * 1024
GiB = 1024 * MiB


class MeasuredMemory(BaseModel):
    weights_mb: Optional[float] = None
    kv_mb: Optional[float] = None  # KV cache (and hybrid models' recurrent state) the backend actually allocated
    workspace_mb: Optional[float] = None  # everything else on the device (context, graphs, activations)
    non_kv_mb: Optional[float] = None  # weights + workspace as the backend reports it
    # Tokens the allocated KV cache can hold, as the engine reports it. Recorded, not read back by the planner.
    # Hybrid models: vLLM reports max-length-request capacity, SGLang the full-attention pool (see the parsers).
    kv_tokens: Optional[int] = None
    device_peak_mb: Optional[float] = None  # NVML peak during the trial, minus what was used before launch
    baseline_mb: Optional[float] = None  # device memory used before launch
    source: str = "none"  # "vllm-log" | "llamacpp-log" | "sglang-log" | "nvml" | "psutil"
    details: Dict[str, float] = Field(default_factory=dict)

    @property
    def total_mb(self) -> Optional[float]:
        """Best available total: device peak, else the log's components."""
        if self.device_peak_mb is not None:
            return self.device_peak_mb
        parts = [p for p in (self.weights_mb, self.kv_mb, self.workspace_mb) if p is not None]
        return sum(parts) if parts else None


_SIZE = r"([\d,.]+)\s*(GiB|MiB|GB|MB|KiB|KB)"


def _to_mb(num: str, unit: str) -> float:
    v = float(num.replace(",", ""))
    unit = unit.lower()
    if unit in ("gib", "gb"):
        return v * 1024
    if unit in ("kib", "kb"):
        return v / 1024
    return v


def parse_vllm(text: str) -> MeasuredMemory:
    """vLLM 0.10/0.11 startup lines.

    Examples:
      Model loading took 6.4265 GiB memory and 8.16 seconds
      Total non KV cache memory: 7.85GiB; torch peak memory increase: 1.0GiB; non-torch forward increase
        memory: 0.1GiB; weights memory: 6.4GiB.
      Available KV cache memory: 12.34 GiB
      GPU KV cache size: 353,296 tokens
    """
    m = MeasuredMemory(source="none")
    d: Dict[str, float] = {}
    if r := re.search(r"weights memory:\s*" + _SIZE, text):
        m.weights_mb = _to_mb(*r.groups())
    elif r := re.search(r"Model loading took\s*" + _SIZE, text):
        # vLLM 0.11: "Model loading took 5.7916 GiB and 1.16 seconds" (no "memory" word).
        m.weights_mb = _to_mb(*r.groups())
    if r := re.search(r"Total non KV cache memory:\s*" + _SIZE, text):
        m.non_kv_mb = _to_mb(*r.groups())
    elif r := re.search(r"Desired GPU memory utilization is \([\d.]+,\s*" + _SIZE + r"\)"
                        r".*?Current kv cache memory in use is\s*" + _SIZE, text):
        # vLLM 0.33 logs the line above at debug level only (utils/mem_utils.py:219). Its start-up summary
        # (v1/worker/gpu_worker.py:1066) gives what it requested and the KV cache it kept; the difference is what it
        # reserved outside the KV cache: consumed memory, activation headroom and its CUDA-graph estimate. Not the
        # summary's three terms added up: its "peak activation" already includes that estimate.
        m.non_kv_mb = _to_mb(*r.groups()[:2]) - _to_mb(*r.groups()[2:])
        if c := re.search(r"Actual usage is\s*" + _SIZE + r" for consumed memory.*?,\s*" + _SIZE
                          + r" for peak activation, and\s*" + _SIZE + r" for CUDAGraph memory", text):
            g = c.groups()
            d["consumed_mb"], d["peak_activation_mb"], d["cudagraph_mb"] = (
                _to_mb(*g[0:2]), _to_mb(*g[2:4]), _to_mb(*g[4:6]))
    if r := re.search(r"torch peak memory increase:\s*" + _SIZE, text):
        d["torch_peak_increase_mb"] = _to_mb(*r.groups())
    if r := re.search(r"non-torch forward increase memory:\s*" + _SIZE, text):
        d["non_torch_increase_mb"] = _to_mb(*r.groups())
    if r := re.search(r"Available KV cache memory:\s*" + _SIZE, text):
        m.kv_mb = _to_mb(*r.groups())
    if r := re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text):
        # vLLM 0.29 (v1/core/kv_cache_utils.update_kv_cache_capacity) prints max_concurrency x max_model_len,
        # where max_concurrency counts the blocks one max-length request takes from every KV cache group. For a
        # hybrid model that is max-length-request capacity, not tokens per group; "Available KV cache memory"
        # already covers the whole shared pool, recurrent state included.
        m.kv_tokens = int(r.group(1).replace(",", ""))
    if m.non_kv_mb is not None and m.weights_mb is not None:
        m.workspace_mb = max(0.0, m.non_kv_mb - m.weights_mb)
    if any(v is not None for v in (m.weights_mb, m.kv_mb, m.non_kv_mb)):
        m.source = "vllm-log"
    m.details = d
    return m


def parse_llamacpp(text: str) -> MeasuredMemory:
    """llama-server startup lines (device buffers only; CPU_Mapped / host buffers are recorded separately).

    Examples:
      load_tensors: CUDA0 model buffer size =  2003.50 MiB
      load_tensors:   CPU_Mapped model buffer size =   308.23 MiB
      llama_kv_cache_unified: CUDA0 KV buffer size =   576.00 MiB
      llama_context:      CUDA0 compute buffer size =   300.25 MiB
      llama_context:  CUDA_Host compute buffer size =    16.01 MiB
    """
    m = MeasuredMemory(source="none")
    d: Dict[str, float] = {}
    dev_w = host_w = dev_kv = host_kv = dev_c = host_c = 0.0
    found = False
    for r in re.finditer(r"(\S+)\s+model buffer size\s*=\s*" + _SIZE, text):
        found = True
        dev, num, unit = r.groups()
        if dev.upper().startswith(("CUDA", "ROCM", "METAL", "VULKAN", "SYCL")) and "HOST" not in dev.upper():
            dev_w += _to_mb(num, unit)
        else:
            host_w += _to_mb(num, unit)
    for r in re.finditer(r"(\S+)\s+KV buffer size\s*=\s*" + _SIZE, text):
        found = True
        dev, num, unit = r.groups()
        if dev.upper().startswith(("CUDA", "ROCM", "METAL", "VULKAN", "SYCL")) and "HOST" not in dev.upper():
            dev_kv += _to_mb(num, unit)
        else:
            host_kv += _to_mb(num, unit)
    for r in re.finditer(r"(\S+)\s+compute buffer size\s*=\s*" + _SIZE, text):
        found = True
        dev, num, unit = r.groups()
        if dev.upper().startswith(("CUDA", "ROCM", "METAL", "VULKAN", "SYCL")) and "HOST" not in dev.upper():
            dev_c += _to_mb(num, unit)
        else:
            host_c += _to_mb(num, unit)
    if r := re.search(r"n_ctx\s*=\s*(\d+)", text):
        m.kv_tokens = int(r.group(1))
    if not found:
        return m
    gpu = "CUDA" in text.upper() and dev_w + dev_kv + dev_c > 0
    if gpu:
        m.weights_mb, m.kv_mb, m.workspace_mb = dev_w, dev_kv, dev_c
        d.update(host_weights_mb=host_w, host_kv_mb=host_kv, host_compute_mb=host_c)
    else:
        m.weights_mb, m.kv_mb, m.workspace_mb = host_w, host_kv, host_c
    m.non_kv_mb = (m.weights_mb or 0) + (m.workspace_mb or 0)
    m.source = "llamacpp-log"
    m.details = d
    return m


def parse_sglang(text: str) -> MeasuredMemory:
    """SGLang startup lines.

    Examples:
      Load weight end. type=Qwen2ForCausalLM, dtype=torch.bfloat16, avail mem=15.20 GB, mem usage=6.21 GB.
      KV Cache is allocated. #tokens: 123456, K size: 2.12 GB, V size: 2.12 GB
      Memory pool end. avail mem=8.10 GB
    """
    m = MeasuredMemory(source="none")
    if r := re.search(r"Load weight end\..*?mem usage=" + _SIZE, text):
        m.weights_mb = _to_mb(*r.groups())
    if _parse_sglang_pools(text, m):
        if m.weights_mb is not None or m.kv_mb is not None:
            m.source = "sglang-log"
        return m
    ks = vs = None
    if r := re.search(r"K size:\s*" + _SIZE, text):
        ks = _to_mb(*r.groups())
    if r := re.search(r"V size:\s*" + _SIZE, text):
        vs = _to_mb(*r.groups())
    if ks is not None or vs is not None:
        m.kv_mb = (ks or 0) + (vs or 0)
    if r := re.search(r"#tokens:\s*([\d,]+)", text):
        m.kv_tokens = int(r.group(1).replace(",", ""))
    if m.weights_mb is not None or m.kv_mb is not None:
        m.source = "sglang-log"
    return m


def _sglang_rank(line: str) -> Optional[int]:
    """The tensor-parallel rank in an SGLang log line's prefix ("[2026-11-12 10:00:00 TP1] ..."), if any."""
    r = re.search(r"\bTP(\d+)\b", line)
    return int(r.group(1)) if r else None


_STATE_KEYS = ("conv_state size", "ssm_state size", "intermediate_ssm_state_cache size",
               "intermediate_conv_window_cache size")


def _parse_sglang_pools(text: str, m: MeasuredMemory) -> bool:
    """Every cache pool SGLang 0.5.19 logs for one rank, summed into kv_mb. False when no pool line is present.

    Hybrid models log more than one pool, and the first "K size" alone undercounts them; the remainder would
    then be derived as workspace (merge) and inflate `memory-report`'s fitted workspace for the machine.
      mem_cache/memory_pool._finalize_allocation_log, one line per pool, labelled "Full" / "SWA" for hybrid SWA
        (mem_cache/swa_memory_pool.py), and "VA upper bound" instead of "is allocated" while post-capture KV
        sizing is on:
        SWA KV Cache is allocated. dtype: torch.bfloat16, #tokens: 81920, K size: 12.50 GB, V size: 12.50 GB
      model_runner_components/kv_pool_runtime.py, the backed size once post-capture sizing has resized the pool:
        Post-capture KV sizing: KV cache allocated. dtype: torch.bfloat16, #tokens: 90000, KV size: 9.10 GB, ...
      MambaPool.__init__, the per-request recurrent state of linear-attention layers (and its speculative
        intermediate caches):
        Mamba Cache is allocated. max_mamba_cache_size: 93, conv_state size: 0.11GB, ssm_state size: 4.38GB
    With tensor parallelism every rank logs its own pools, and weights are read from one rank, so only the
    lowest rank's lines are summed. kv_mb is everything that rank allocated for cache and state; kv_tokens is the
    full-attention pool's token count, with the SWA pool's in details.
    """
    tagged = [ln for ln in text.splitlines()
              if re.search(r"KV Cache (?:is allocated|VA upper bound)\.|Mamba Cache is allocated\.|"
                           r"Post-capture KV sizing: KV cache allocated\.", ln)]
    if not tagged:
        return False
    ranks = [r for r in (_sglang_rank(ln) for ln in tagged) if r is not None]
    rank = min(ranks) if ranks else None
    lines = [ln for ln in tagged if _sglang_rank(ln) == rank]
    attention = state = 0.0
    backed = None
    for ln in lines:
        if "Post-capture KV sizing" in ln:
            if r := re.search(r"KV size:\s*" + _SIZE, ln):
                backed = _to_mb(*r.groups())
            if r := re.search(r"#tokens:\s*([\d,]+)", ln):
                m.kv_tokens = int(r.group(1).replace(",", ""))
            continue
        for key in ("K size", "V size", "KV size"):
            if r := re.search(r"(?<![\w_])" + key + r":\s*" + _SIZE, ln):
                attention += _to_mb(*r.groups())
        for key in _STATE_KEYS:
            if r := re.search(r"(?<![\w_])" + key + r":\s*" + _SIZE, ln):
                state += _to_mb(*r.groups())
        if r := re.search(r"max_mamba_cache_size:\s*(\d+)", ln):
            m.details["mamba_slots"] = float(r.group(1))
        if r := re.search(r"#tokens:\s*([\d,]+)", ln):
            tokens = int(r.group(1).replace(",", ""))
            if "SWA KV Cache" in ln:
                m.details["swa_kv_tokens"] = float(tokens)
            elif m.kv_tokens is None:
                m.kv_tokens = tokens
    if backed is not None:
        attention = backed  # what the resized pool holds, not the virtual upper bound
    if "Mamba Cache is allocated" in "\n".join(lines):
        m.details["mamba_mb"] = state
    m.kv_mb = attention + state
    if rank is not None:  # weights from the same rank as the pools
        for ln in text.splitlines():
            if _sglang_rank(ln) == rank and (r := re.search(r"Load weight end\..*?mem usage=" + _SIZE, ln)):
                m.weights_mb = _to_mb(*r.groups())
                break
    return True


PARSERS = {
    "vllm": parse_vllm,
    "vllm-cpu": parse_vllm,
    "sglang": parse_sglang,
    "llamacpp-cuda": parse_llamacpp,
    "llamacpp-cpu": parse_llamacpp,
}


# vLLM's start-up lines, each matched on its whole phrase with its unit, case-sensitively. Sources, v0.29.0 / v0.33.0:
#   "Graph capturing finished in %.0f secs, took %.2f GiB"   v1/worker/gpu_model_runner.py:7002 / 6517 and
#                                                            v1/worker/gpu/model_runner.py:961 / 1164
#   "Encoder graph capturing finished in ... (%d graphs)"    v1/worker/mm_encoder_model_runner.py:95 (0.33 only):
#                                                            a different capture, never read as the decoder's
#   "[prefix]Compiling a graph for compile range %s takes %.2f s"       compilation/backends.py:394 / 418
#   "[prefix]Directly load the compiled graph(s) for compile range %s from the cache, took %.3f s"   :293 / 311
#   "...; setting cudagraph_mode=<NAME>"                     config/compilation.py, the resolver's warnings
#   "[prefix]Directly load AOT compilation from path %s"     compilation/decorators.py:326 (0.33): the whole model
#                                                            from the AOT cache, where no compile-range line follows.
#                                                            Read from a real log: Gemma 4 12B's relaunch on an A40,
#                                                            16 Nov 2026, "torch.compile took 0.54 s in total"
_CAPTURE = re.compile(r"(?<!Encoder )Graph capturing finished in (\d+) secs, took (-?\d+\.\d+) GiB")
_COMPILED = re.compile(r"Compiling a graph for compile range (.+?) takes (\d+\.\d+) s")
_CACHED = re.compile(r"Directly load the compiled graph\(s\) for compile range .+? from the cache, took \d+\.\d+ s"
                     r"|Directly load AOT compilation from path \S")
#   "[prefix] Initial profiling/warmup run took %.2f s"      compilation/monitor.py:91 (0.33): the first forward of
#                                                            each compiled part, where Triton compiles and autotunes
#                                                            on a cold cache (Qwen3.5's GDN kernels: 74-82 s cold,
#                                                            0.3 s warm, benchmarks/sessions/2026-11/modal-triton-cache)
_WARMUP = re.compile(r"(\[[\w.-]+\] )?Initial profiling/warmup run took (\d+\.\d+) s")
_REWRITE = re.compile(r"; setting (cudagraph_mode=[A-Z_]+)")
# The other wording, for piecewise compilation with nothing to split (config/compilation.py:1204 / 1226); the
# source breaks the phrase across two string literals, the log does not.
_REWRITE_TO = re.compile(r"Setting cudagraph_mode to ([A-Z_]+)")


def parse_startup(backend: str, text: str) -> Dict[str, object]:
    """What an engine's log says its start-up cost: the fields of models.StartupCost other than `seconds`.

    Only vLLM's lines are known; another engine's log, an empty log and a log cut before these lines all give
    nothing, with `parsed` empty to say so. A value is never guessed: no capture line leaves the capture unknown,
    and without a compile line or a cache line `compile_cached` stays None rather than False."""
    out: Dict[str, object] = {"parsed": []}
    if backend != "vllm" or not text:
        return out
    parsed: List[str] = []
    captures = [(float(s), float(g)) for s, g in _CAPTURE.findall(text)]
    if captures:  # one line per worker under tensor parallelism: the slowest one bounds the start-up
        out["capture_s"], out["cudagraph_gib"] = max(captures)
        out["capture_lines"] = len(captures)
        parsed.append("capture")
    # One figure per compile range, the slowest: 0.33 prints the line once per tensor-parallel worker, all
    # compiling at once, and 0.29 once in all, so summing lines would double the time on one release only.
    ranges: Dict[str, float] = {}
    for compile_range, seconds in _COMPILED.findall(text):
        ranges[compile_range] = max(ranges.get(compile_range, 0.0), float(seconds))
    compiled, cached = list(ranges.values()), _CACHED.findall(text)
    if compiled:
        out["compile_s"] = round(sum(compiled), 2)
        parsed.append("compile")
    if cached:
        parsed.append("compile_cache")
    if compiled or cached:
        out["compile_cached"] = not compiled
    # One figure per compiled part (the backbone, an MTP head), the slowest worker's, summed over the parts.
    warmups: Dict[str, float] = {}
    for prefix, seconds in _WARMUP.findall(text):
        warmups[prefix] = max(warmups.get(prefix, 0.0), float(seconds))
    if warmups:
        out["warmup_s"] = round(sum(warmups.values()), 2)
        parsed.append("warmup")
    found = [(m.start(), m.group(1)) for m in _REWRITE.finditer(text)]
    found += [(m.start(), f"cudagraph_mode={m.group(1)}") for m in _REWRITE_TO.finditer(text)]
    rewrites = list(dict.fromkeys(name for _, name in sorted(found)))  # each once, in the order the engine said them
    if rewrites:
        out["rewrites"] = rewrites
        parsed.append("rewrites")
    out["parsed"] = parsed
    return out


# Speculative decoding's acceptance as each engine logs it:
#   vLLM, v1/spec_decode/metrics.py SpecDecodingLogging.log (v0.33.0), every logging interval:
#     "SpecDecoding metrics: Mean acceptance length: %.2f, ... Accepted: %d tokens, Drafted: %d tokens, ..."
#   SGLang, each decode-batch line under speculation (0.5.23, read from the A100 session's MTP trial):
#     "Decode batch, #running-req: 8, ..., accept len: 2.71, accept rate: 0.57, cuda graph: True, ..."
_VLLM_SPEC = re.compile(r"SpecDecoding metrics: Mean acceptance length: (\d+(?:\.\d+)?), .*?"
                        r"Accepted: (\d+) tokens, Drafted: (\d+) tokens")
_SGLANG_SPEC = re.compile(r"accept len: (\d+(?:\.\d+)?), accept rate: (\d+(?:\.\d+)?)")
# vLLM's Prometheus counters (v1/spec_decode/metrics.py, v0.33.0), without their _total suffix
SPEC_DRAFTS, SPEC_DRAFT_TOKENS, SPEC_ACCEPTED = (
    "vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")


def acceptance_from_counters(drafts: float, drafted: float, accepted: float) -> Optional[Dict[str, object]]:
    """Acceptance from the change in vLLM's counters over the workload; None when nothing was drafted."""
    if drafts <= 0 or drafted <= 0 or accepted < 0:
        return None
    return {"rate": round(accepted / drafted, 4), "mean_length": round(1 + accepted / drafts, 3),
            "accepted_tokens": int(accepted), "drafted_tokens": int(drafted), "source": "vllm_metrics"}


def parse_acceptance(backend: str, text: str) -> Optional[Dict[str, object]]:
    """Acceptance from the engine's log lines in `text`, or None when none were logged (or nothing was drafted)."""
    if backend == "vllm":
        rows = [(float(m), int(a), int(d)) for m, a, d in _VLLM_SPEC.findall(text or "")]
        accepted, drafted = sum(r[1] for r in rows), sum(r[2] for r in rows)
        if not rows or drafted <= 0:
            return None
        return {"rate": round(accepted / drafted, 4), "mean_length": round(sum(r[0] for r in rows) / len(rows), 3),
                "accepted_tokens": accepted, "drafted_tokens": drafted, "lines": len(rows), "source": "vllm_log"}
    if backend == "sglang":
        rows = [(float(length), float(rate)) for length, rate in _SGLANG_SPEC.findall(text or "")]
        if not rows:
            return None
        return {"rate": round(sum(r[1] for r in rows) / len(rows), 4),
                "mean_length": round(sum(r[0] for r in rows) / len(rows), 3), "lines": len(rows),
                "source": "sglang_log"}
    return None


def parse_log(backend: str, text: str) -> MeasuredMemory:
    fn = PARSERS.get(backend)
    return fn(text) if fn else MeasuredMemory()


def merge(log: MeasuredMemory, device_peak_mb: Optional[float], baseline_mb: Optional[float],
          telemetry_source: str) -> MeasuredMemory:
    """Combine log components with the NVML/psutil peak observed during the trial.

    vLLM 0.11 reports weights and the KV pool but not the rest, so workspace is derived as
    peak - weights - kv when all three are known. That residual is exactly the non-weight,
    non-KV device memory (CUDA context, graphs, activations) the planner budgets for.
    """
    out = log.model_copy()
    if device_peak_mb is not None:
        peak = device_peak_mb - (baseline_mb or 0.0)
        out.device_peak_mb = max(0.0, peak)
        out.baseline_mb = baseline_mb
        if out.source == "none":
            out.source = "nvml" if telemetry_source in ("llmtrace", "pynvml") else telemetry_source
    if out.workspace_mb is None and out.device_peak_mb and out.weights_mb is not None and out.kv_mb is not None:
        out.workspace_mb = max(0.0, out.device_peak_mb - out.weights_mb - out.kv_mb)
        out.details["workspace_derived"] = 1.0
    if out.non_kv_mb is None and out.weights_mb is not None and out.workspace_mb is not None:
        out.non_kv_mb = out.weights_mb + out.workspace_mb
    return out


def device_used_mb(gpu_ids: List[int]) -> Optional[float]:
    """Device memory in use right now (before launching a backend), via NVML."""
    if not gpu_ids:
        return None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            return sum(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used for i in gpu_ids) / MiB
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None
