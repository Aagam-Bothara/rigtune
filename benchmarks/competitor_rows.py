#!/usr/bin/env python
"""Turn two other tuners' picks into rows for `rigtune compare --rows`: BentoML llm-optimizer and NVIDIA
AIConfigurator. Each pick is re-measured by `compare` like every other row (same runner, prompts, levels,
limits and repeats); the numbers the tool reported for itself are kept in the row's source, never scored.

    python benchmarks/competitor_rows.py llm-optimizer-cmd --framework vllm --model MODEL --ttft-ms 500 \
        --tpot-ms 50 --input-len 63 --output-len 395 --max-model-len 4096 --levels 1,4,8 --out lo-vllm.json
    python benchmarks/competitor_rows.py llm-optimizer-row --result lo-vllm.json --framework vllm \
        --max-model-len 4096 --tool-version <commit> --out lo-vllm/rows.json
    python benchmarks/competitor_rows.py aiconfigurator-cmd --model MODEL --gpu "NVIDIA L40S" --backend vllm \
        --ttft-ms 500 --tpot-ms 50 --input-len 63 --output-len 395 --save-dir aic-vllm
    python benchmarks/competitor_rows.py aiconfigurator-row --save-dir aic-vllm --backend vllm \
        --max-model-len 4096 --tool-version 0.12.0 --out aic-vllm/rows.json
    python benchmarks/competitor_rows.py not-applicable --label aiconfigurator-vllm --reason "..." --out rows.json

What each tool is, as read from its own repository (README and source) on 15 November 2026:

- **llm-optimizer** (github.com/bentoml/llm-optimizer, package version 0.0.1, installed from a clone with
  `pip install -e .`): a grid runner. `--server-args "max_num_seqs=[64,128];..."` and `--client-args` are
  expanded to their Cartesian product; every server point is launched (`vllm serve {model} --host {host}
  --port {port}`, or `python3 -m sglang.launch_server --model-path {model} ...`, with each server argument
  appended as `--name-with-hyphens=value`) and driven by its own client, adapted from SGLang's
  bench_serving. `--constraints "ttft:p95<500ms;itl:mean<50ms"` filters results (metrics ttft, itl,
  e2e_latency; statistics mean, median, p95, p99; a value equal to the limit fails). `--output-json` writes
  {"metadata", "best_configurations", "test_results"}; best_configurations has best_input_throughput,
  best_output_throughput and their `_constrained` versions, each {"throughput", "config": {"client_args",
  "server_args", "server_cmd_args"}, "cmd"} or null. The row is best_output_throughput_constrained: rigtune
  also counts output tokens per second.
- **AIConfigurator** (github.com/ai-dynamo/aiconfigurator, `pip install aiconfigurator`, 0.12.0, Linux
  x86-64 only, "maintenance-only" per its README): predicts from timings collected per system and engine
  release, launches nothing. `aiconfigurator cli default --model M --total-gpus 1 --system S --backend vllm
  --isl I --osl O --ttft T --tpot P --save-dir D` writes D/<run>/agg/top1/agg/node_0_run.sh, a Dynamo launch
  script whose worker line is `python3 -m dynamo.vllm --model "$MODEL_PATH" <engine flags>` (or
  dynamo.sglang). The engine flags are plain vLLM/SGLang flags rendered from its cli_args.j2 templates; this
  script reads them back into a rigtune Config. Copying them to `vllm serve` is rigtune's inference, not
  something AIConfigurator documents. Its README lists measured data for h100_sxm, h200_sxm, b200_sxm, gb200
  and a100_sxm (vLLM 0.12.0 on h100/h200/a100, SGLang 0.5.6.post2 on h100/h200/b200); the plan for this
  session read l40s data at its commit 6cd6f96. Which pairs it supports is decided on the pod by its own
  `aiconfigurator cli support`, not here; a card with no system name below is not applicable before that.

Fairness: both get the comparison's model, card, latency limits (p95 TTFT, and TPOT; llm-optimizer judges
mean inter-token latency, which equals mean TPOT without speculation) and request lengths (the calibration's
mean prompt and answer tokens, as vLLM's auto_tune gets them), and the row records the tool's version. Their
workloads are still their own: random fixed-length prompts (llm-optimizer) or a model (AIConfigurator), not
the comparison's real prompts, and the row's source says so.

Where a pick carries a setting a rigtune row cannot launch faithfully (pipeline or data parallelism beyond
one card, a list-valued flag, speculative settings, no tokenizer), the row is written as skipped, with the
reason, rather than measured as something else. A setting the tool left at the engine's default stays at
it, except the two every rigtune launch passes, the context length and batch cap. The context is always the
comparison's (--max-model-len): llm-optimizer's grid fixes it there, and AIConfigurator's own context (sized
from --isl and --osl) is replaced by it, with a note, so no row gains memory for its batch by serving a shorter
context than the others. A batch cap the tool did not set is 256, the value rigtune's stock rows take as the
engine's default, which the row's source states. vLLM's own default is 1024 on cards with 70 GiB or more
(except an A100: vllm/engine/arg_utils.py, EngineArgs.get_batch_defaults); 256 keeps the rows comparable with
the stock rows, and neither tool's row depends on it in practice, because both tools set the batch cap.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LO_LABEL = "llm-optimizer-{framework}"
AIC_LABEL = "aiconfigurator-{backend}"
STOCK_BATCH = 256
BACKENDS = ("vllm", "sglang")

# Engine argument (underscored, as llm-optimizer records it) -> rigtune Config field.
FIELDS: Dict[str, Dict[str, str]] = {
    "vllm": {"max_model_len": "ctx", "max_num_seqs": "batch", "max_num_batched_tokens": "prefill_budget",
             "gpu_memory_utilization": "gpu_memory_utilization", "tensor_parallel_size": "tp",
             "kv_cache_dtype": "kv_dtype"},
    "sglang": {"context_length": "ctx", "max_running_requests": "batch", "chunked_prefill_size": "prefill_budget",
               "mem_fraction_static": "gpu_memory_utilization", "tp_size": "tp", "tensor_parallel_size": "tp",
               "kv_cache_dtype": "kv_dtype"},
}
# Set by rigtune itself on every launch, or naming the model: never carried from a tool.
IGNORED = {"model", "model_path", "host", "port", "served_model_name"}
PARALLEL = {"pipeline_parallel_size", "pp_size", "data_parallel_size", "dp_size"}
# Settings a rigtune row cannot launch as the tool meant them.
UNCARRIED = {"skip_tokenizer_init": "it starts the engine without a tokenizer, so it cannot serve chat requests"}
# List-valued CUDA-graph batch sizes: a rigtune row passes an extra flag one value only, so these are left at
# the engine's own list (which also covers batch sizes up to its cap) and the row's source says so.
DROPPED = {"cudagraph_capture_sizes", "cuda_graph_bs"}


class AdapterError(ValueError):
    pass


def to_config(backend: str, args: Dict[str, Any], quant: str, max_model_len: int) -> Tuple[
        Optional[Dict[str, Any]], List[str], Optional[str]]:
    """(Config fields, notes, why not applicable) for an engine argument dict as a tool set it."""
    if backend not in FIELDS:
        raise AdapterError(f"backend must be one of {', '.join(BACKENDS)}")
    cfg: Dict[str, Any] = {"backend": backend, "quant": quant}
    extra: Dict[str, Any] = {}
    notes: List[str] = []
    for raw_key, value in sorted(args.items()):
        key = raw_key.lstrip("-").replace("-", "_")
        if key in IGNORED:
            continue
        if key in DROPPED:
            notes.append(f"its {key} ({value}) is left at the engine's own list: a rigtune row cannot pass a list")
            continue
        if key in UNCARRIED:
            return None, notes, f"its pick sets {key}: {UNCARRIED[key]}"
        if key.startswith("speculative"):
            return None, notes, (f"its pick sets {key}; a rigtune row names speculation its own way and this "
                                 "adapter does not translate it")
        if key in PARALLEL:
            if int(value) != 1:
                return None, notes, f"its pick sets {key}={value}, more than the one card this compare runs on"
            continue
        if isinstance(value, list):
            return None, notes, f"its pick sets {key} to a list, which a rigtune row passes as one value"
        field_name = FIELDS[backend].get(key)
        if field_name in ("ctx", "batch", "prefill_budget", "tp"):
            cfg[field_name] = int(value)
        elif field_name == "gpu_memory_utilization":
            cfg[field_name] = round(float(value), 2)
        elif field_name == "kv_dtype":
            cfg[field_name] = str(value)
        elif key in ("enable_prefix_caching", "no_enable_prefix_caching", "disable_radix_cache"):
            on = bool(value) if key == "enable_prefix_caching" else not bool(value)
            cfg["prefix_cache"] = on
        elif key == "quantization":
            if str(value) != "fp8":
                return None, notes, f"its pick sets quantization={value}, which rigtune launches only as fp8"
            cfg["quant"] = "fp8"
        elif key == "dtype":
            dt = {"bfloat16": "bf16", "float16": "fp16", "half": "fp16"}.get(str(value))
            if dt:
                cfg["quant"] = dt
        else:
            extra[key] = value
    if "ctx" not in cfg:
        cfg["ctx"] = max_model_len
        notes.append(f"the tool set no context length; the row uses {max_model_len}, the one it was given")
    if "batch" not in cfg:
        cfg["batch"] = STOCK_BATCH
        notes.append(f"the tool set no batch cap; the row uses {STOCK_BATCH}, the value rigtune's stock rows "
                     "take as the engine's default (vLLM's own is 1024 on cards with 70 GiB or more, except "
                     "an A100)")
    if extra:
        cfg["extra"] = extra
    return cfg, notes, None


def _row(label: str, source: str, cfg: Optional[Dict[str, Any]], notes: List[str], skipped: Optional[str]) -> dict:
    row: Dict[str, Any] = {"label": label, "kind": "tuner", "source": source + "".join(f"; {n}" for n in notes)}
    if skipped:
        row["skipped"] = skipped
    else:
        row["config"] = cfg
    return row


def write_rows(rows: List[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- llm-optimizer


def lo_grid(framework: str, max_model_len: int) -> str:
    """The 3 x 3 server grid the session's plan gives llm-optimizer (foundation plan section 11.3): the batch
    cap by the prefill budget, the context fixed at the comparison's, so its points are what rigtune's
    batch and prefill stages vary."""
    if framework == "vllm":
        return (f"max_num_seqs=[64,128,256];max_num_batched_tokens=[2048,8192,16384];"
                f"max_model_len={max_model_len}")
    return (f"max_running_requests=[64,128,256];chunked_prefill_size=[2048,4096,8192];"
            f"context_length={max_model_len}")


def lo_constraints(ttft_ms: float, tpot_ms: Optional[float]) -> str:
    out = f"ttft:p95<{ttft_ms:g}ms"
    return out + (f";itl:mean<{tpot_ms:g}ms" if tpot_ms else "")


def lo_command(framework: str, model: str, ttft_ms: float, tpot_ms: Optional[float], input_len: int,
               output_len: int, max_model_len: int, levels: List[int], out: str, num_prompts: int = 256,
               server_grid: Optional[str] = None, server_cmd: Optional[str] = None,
               binary: str = "llm-optimizer") -> List[str]:
    """The llm-optimizer command line for this comparison's model and limits. `server_cmd` replaces its
    built-in launch template (SGLang in its own environment, through $SGLANG_PYTHON)."""
    client = (f"max_concurrency=[{','.join(str(n) for n in levels)}];num_prompts={num_prompts};"
              f"dataset_name=random;random_input_len={input_len};random_output_len={output_len};"
              f"random_range_ratio=1.0")
    argv = [binary, "--framework", framework, "--model", model, "--gpus", "1",
            "--server-args", server_grid or lo_grid(framework, max_model_len), "--client-args", client,
            "--constraints", lo_constraints(ttft_ms, tpot_ms), "--output-json", out]
    if server_cmd:
        argv += ["--server-cmd", server_cmd]
    return argv


# Run in llm-optimizer's own environment: its end-of-run summary (cli.py), built from the JSONL it appends after
# every run, for a run that was cut off before it could write --output-json itself.
LO_FINISH = """
import json, sys
from llm_optimizer.cli import extract_token_lengths, find_best_throughput_configs
from llm_optimizer.performance import parse_slo_constraints
from llm_optimizer.visualization.visualize import convert_constraints_for_visualization
jsonl, constraints, budget_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
results = []
for line in open(jsonl):
    try:
        results.append(json.loads(line))
    except ValueError:
        pass
parsed = parse_slo_constraints(constraints)
first = (results[0].get("config") or {}).get("client_args") or {} if results else {}
input_len, output_len = extract_token_lengths(first) if results else (None, None)
gpu = (results[0].get("metadata") or {}).get("gpu_type") if results else None
json.dump({"metadata": {"gpu_type": gpu, "total_tests": len(results),
                        "constraints": convert_constraints_for_visualization(parsed),
                        "input_len": input_len, "output_len": output_len,
                        "cut_off": {"budget_s": budget_s, "runs": len(results)}},
           "best_configurations": find_best_throughput_configs(results, parsed),
           "test_results": results}, sys.stdout)
"""


def lo_finish(python: str, jsonl: str, constraints: str, budget_s: float) -> dict:
    """llm-optimizer's --output-json for a run cut off at `budget_s`, made by its own code from its JSONL."""
    got = subprocess.run([python, "-c", LO_FINISH, jsonl, constraints, str(budget_s)],
                         capture_output=True, text=True)
    if got.returncode != 0:
        raise AdapterError(f"llm-optimizer's summary of {jsonl} failed: {got.stderr.strip()[-500:]}")
    return json.loads(got.stdout)


def lo_row(result: dict, framework: str, max_model_len: int, tool_version: str, quant: str = "bf16",
           label: Optional[str] = None) -> dict:
    """llm-optimizer's --output-json as one --rows entry."""
    if not isinstance(result, dict) or "best_configurations" not in result:
        raise AdapterError("not llm-optimizer --output-json output: no best_configurations")
    label = label or LO_LABEL.format(framework=framework)
    meta = result.get("metadata") or {}
    best = result["best_configurations"] or {}
    constraints = meta.get("constraints")
    total = meta.get("total_tests", len(result.get("test_results") or []))
    source = (f"BentoML llm-optimizer at {tool_version}, --framework {framework}: its highest output "
              f"throughput among {total} measured runs that met {json.dumps(constraints)}, on its own "
              f"client with random prompts of {meta.get('input_len')} tokens and answers of "
              f"{meta.get('output_len')}; its card {meta.get('gpu_type')!r}; this row is re-measured on the "
              "comparison's prompts")
    if meta.get("cut_off"):
        source += (f"; its search was cut off at the calibration's budget of {meta['cut_off']['budget_s']:g} s, "
                   f"so this is its pick among the {total} runs it finished, made by its own end-of-run code")
    if not constraints:
        return _row(label, source, None, [], "llm-optimizer ran without --constraints, so its pick is not "
                                             "judged by the comparison's limits")
    pick = best.get("best_output_throughput_constrained")
    if not total:
        return _row(label, source, None, [], "llm-optimizer measured nothing: no run produced a result")
    if pick is None:
        return _row(label, source, None, [], f"no run llm-optimizer measured met {constraints}")
    conf = pick.get("config") or {}
    cfg, notes, skipped = to_config(framework, conf.get("server_args") or {}, quant, max_model_len)
    client = conf.get("client_args") or {}
    notes = [f"its own figure: {float(pick.get('throughput') or 0):.1f} output tok/s at max_concurrency "
             f"{client.get('max_concurrency')}", f"its server command: {pick.get('cmd')}"] + notes
    return _row(label, source, cfg, notes, skipped)


# --------------------------------------------------------------------------- AIConfigurator


# The card name nvidia-smi reports (lower case) -> AIConfigurator's system name, checked in order: the first
# needle found decides, so a name that contains another's ("gb200" holds "b200", "gh200" and "h200 nvl" hold
# "h200") is listed before it. None, and any card not here, has no system in AIConfigurator, so its row is not
# applicable before any run: Grace Hopper and the NVL / PCIe cards are different machines from the SXM boards
# its data was collected on.
AIC_SYSTEMS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("gb200", "gb200"), ("gh200", None), ("h200 nvl", None), ("h200", "h200_sxm"), ("b200", "b200_sxm"),
    ("h100 nvl", None), ("h100 pcie", None), ("h100 80gb hbm3", "h100_sxm"), ("h100 sxm", "h100_sxm"),
    ("a100-sxm", "a100_sxm"), ("a100 sxm", "a100_sxm"), ("l40s", "l40s"))


def aic_system(gpu_name: str) -> Optional[str]:
    name = gpu_name.lower()
    for needle, system in AIC_SYSTEMS:
        if needle in name:
            return system
    return None


def aic_not_applicable(gpu_name: str) -> str:
    return (f"AIConfigurator has no system for this card ({gpu_name}): it predicts from timings collected "
            "per system, and lists none for it (its README; systems such as h100_sxm, a100_sxm, l40s)")


def aic_command(model: str, system: str, backend: str, ttft_ms: float, tpot_ms: Optional[float],
                input_len: int, output_len: int, save_dir: str, backend_version: Optional[str] = None,
                binary: str = "aiconfigurator") -> List[str]:
    argv = [binary, "cli", "default", "--model", model, "--total-gpus", "1", "--system", system,
            "--backend", backend, "--isl", str(input_len), "--osl", str(output_len), "--ttft", f"{ttft_ms:g}",
            "--save-dir", save_dir]
    if tpot_ms:
        argv += ["--tpot", f"{tpot_ms:g}"]
    if backend_version:
        argv += ["--backend-version", backend_version]
    return argv


ENDS = ("&", "&&", "|", "||", ";", ")", "2>&1")
WORKER = {"vllm": ("dynamo.vllm", "vllm serve"), "sglang": ("dynamo.sglang", "sglang.launch_server")}


def _logical_lines(text: str) -> List[str]:
    out, cur = [], ""
    for line in text.splitlines():
        if line.rstrip().endswith("\\"):
            cur += line.rstrip()[:-1] + " "
            continue
        out.append(cur + line)
        cur = ""
    if cur:
        out.append(cur)
    return out


def engine_flags(script: str, backend: str) -> Dict[str, Any]:
    """The engine flags on the first worker line of a generated launch script (AIConfigurator's
    node_0_run.sh, or a plain `vllm serve` / `sglang.launch_server` line), as {underscored name: value}; a
    bare flag is True. Shell tokens around the command (`(`, `&`, redirections, `|`) end it."""
    markers = WORKER[backend]
    for line in _logical_lines(script):
        if line.lstrip().startswith("#") or not any(m in line for m in markers):
            continue
        try:
            words = shlex.split(line, comments=True)
        except ValueError as exc:
            raise AdapterError(f"cannot split the worker line: {exc}") from None
        start = next(i for i, w in enumerate(words) if any(w == m.split()[-1] or w.endswith(m.split()[-1])
                                                           for m in markers))
        flags: Dict[str, Any] = {}
        i = start + 1
        while i < len(words):
            w = words[i]
            if w in ENDS or w.startswith((">", "2>")):
                break
            if not w.startswith("--"):
                i += 1  # `serve MODEL` or a positional: not a setting
                continue
            name, eq, val = w[2:].partition("=")
            key = name.replace("-", "_")
            i += 1
            if eq:
                flags[key] = _value(val)
                continue
            values = []
            while (i < len(words) and not words[i].startswith("--") and words[i] not in ENDS
                   and not words[i].startswith((">", "2>"))):
                values.append(_value(words[i]))
                i += 1
            flags[key] = True if not values else values[0] if len(values) == 1 else values
        return flags
    raise AdapterError(f"no {backend} worker line ({' or '.join(markers)}) in the script")


def _value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def find_script(save_dir: Path) -> Path:
    """The aggregated top-1 launch script under a --save-dir (one run per directory)."""
    found = sorted(Path(save_dir).rglob("agg/top1/agg/node_0_run.sh"), key=lambda p: p.as_posix())
    if not found:
        raise AdapterError(f"{save_dir}: no agg/top1/agg/node_0_run.sh (did aiconfigurator cli default run "
                           "with --save-dir here?)")
    if len(found) > 1:
        raise AdapterError(f"{save_dir}: {len(found)} runs; give one run's directory")
    return found[0]


def aic_row(script_text: str, backend: str, max_model_len: int, tool_version: str, quant: str = "bf16",
            system: Optional[str] = None, label: Optional[str] = None, script_name: str = "node_0_run.sh") -> dict:
    label = label or AIC_LABEL.format(backend=backend)
    source = (f"NVIDIA AIConfigurator {tool_version}, cli default --backend {backend}"
              + (f" --system {system}" if system else "")
              + f": the engine flags of its top aggregated configuration, read from {script_name} (a Dynamo "
                "launch script; running them under the engine's own server is rigtune's reading, not "
                "AIConfigurator's documentation); a prediction from its timing data, never measured by it; "
                "this row is measured on the comparison's prompts")
    flags = engine_flags(script_text, backend)
    cfg, notes, skipped = to_config(backend, flags, quant, max_model_len)
    if cfg is not None and cfg["ctx"] != max_model_len:
        notes.append(f"AIConfigurator sized the context at {cfg['ctx']} (from its --isl and --osl); the row uses "
                     f"the comparison's {max_model_len}, as every other row does")
        cfg["ctx"] = max_model_len
    return _row(label, source, cfg, notes, skipped)


# --------------------------------------------------------------------------- command line


def _levels(text: str) -> List[int]:
    try:
        out = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        out = []
    if not out or min(out) < 1:
        raise argparse.ArgumentTypeError(f"{text!r}: numbers of users, comma-separated")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def limits(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True)
        p.add_argument("--ttft-ms", type=float, required=True, help="the comparison's p95 TTFT ceiling")
        p.add_argument("--tpot-ms", type=float, default=None, help="the comparison's TPOT ceiling")
        p.add_argument("--input-len", type=int, required=True, help="mean prompt tokens of the calibration")
        p.add_argument("--output-len", type=int, required=True, help="mean answer tokens of the calibration")

    p = sub.add_parser("llm-optimizer-cmd", help="print the llm-optimizer command for this comparison")
    p.add_argument("--framework", choices=BACKENDS, required=True)
    limits(p)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--levels", type=_levels, default=[1, 4, 8], help="its max_concurrency values")
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--server-cmd", default=None)
    p.add_argument("--binary", default="llm-optimizer")
    p.add_argument("--out", required=True, help="its --output-json")
    p.add_argument("--shell", action="store_true", help="print one shell-quoted line instead of JSON")

    p = sub.add_parser("llm-optimizer-row", help="its --output-json as a --rows file")
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--framework", choices=BACKENDS, required=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tool-version", required=True, help="the clone's commit (it has no release)")
    p.add_argument("--quant", default="bf16")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("llm-optimizer-cut-off-row",
                       help="a --rows file from the JSONL of a run cut off before it wrote --output-json")
    p.add_argument("--jsonl", required=True, help="the llm-optimizer.jsonl beside its --output-json")
    p.add_argument("--python", required=True, help="the Python of llm-optimizer's environment")
    p.add_argument("--ttft-ms", type=float, required=True)
    p.add_argument("--tpot-ms", type=float, default=None)
    p.add_argument("--budget-s", type=float, required=True, help="where it was cut off")
    p.add_argument("--framework", choices=BACKENDS, required=True)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tool-version", required=True)
    p.add_argument("--quant", default="bf16")
    p.add_argument("--summary", type=Path, default=None, help="also keep the summary it made here")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("aiconfigurator-cmd", help="print the AIConfigurator command, or say it does not apply")
    limits(p)
    p.add_argument("--gpu", required=True, help="the card's name as nvidia-smi reports it")
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--backend-version", default=None)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--binary", default="aiconfigurator")
    p.add_argument("--shell", action="store_true")

    p = sub.add_parser("aiconfigurator-row", help="its generated launch script as a --rows file")
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--script", type=Path, default=None, help="a launch script, instead of --save-dir")
    p.add_argument("--backend", choices=BACKENDS, required=True)
    p.add_argument("--system", default=None)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--tool-version", required=True)
    p.add_argument("--quant", default="bf16")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("not-applicable", help="a skipped row with its reason")
    p.add_argument("--label", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--source", default="")
    p.add_argument("--out", type=Path, required=True)

    a = ap.parse_args(argv)
    try:
        if a.cmd == "llm-optimizer-cmd":
            cmd = lo_command(a.framework, a.model, a.ttft_ms, a.tpot_ms, a.input_len, a.output_len,
                             a.max_model_len, a.levels, a.out, a.num_prompts, server_cmd=a.server_cmd,
                             binary=a.binary)
            print(shlex.join(cmd) if a.shell else json.dumps(cmd))
        elif a.cmd == "aiconfigurator-cmd":
            system = aic_system(a.gpu)
            if system is None:
                print(aic_not_applicable(a.gpu), file=sys.stderr)
                return 3  # not applicable: the plan writes a skipped row with this reason
            cmd = aic_command(a.model, system, a.backend, a.ttft_ms, a.tpot_ms, a.input_len, a.output_len,
                              a.save_dir, a.backend_version, a.binary)
            print(shlex.join(cmd) if a.shell else json.dumps(cmd))
        elif a.cmd == "llm-optimizer-row":
            data = json.loads(a.result.read_text(encoding="utf-8"))
            row = lo_row(data, a.framework, a.max_model_len, a.tool_version, a.quant)
            write_rows([row], a.out)
            print(json.dumps(row, indent=1))
        elif a.cmd == "llm-optimizer-cut-off-row":
            data = lo_finish(a.python, a.jsonl, lo_constraints(a.ttft_ms, a.tpot_ms), a.budget_s)
            if a.summary:
                a.summary.write_text(json.dumps(data, indent=1), encoding="utf-8")
            row = lo_row(data, a.framework, a.max_model_len, a.tool_version, a.quant)
            write_rows([row], a.out)
            print(json.dumps(row, indent=1))
        elif a.cmd == "aiconfigurator-row":
            if (a.save_dir is None) == (a.script is None):
                raise AdapterError("give --save-dir or --script")
            script = a.script or find_script(a.save_dir)
            row = aic_row(script.read_text(encoding="utf-8"), a.backend, a.max_model_len, a.tool_version,
                          a.quant, a.system, script_name=script.name)
            write_rows([row], a.out)
            print(json.dumps(row, indent=1))
        else:
            write_rows([{"label": a.label, "kind": "tuner", "source": a.source, "skipped": a.reason}], a.out)
    except (AdapterError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
