"""Today's launch commands, candidates, plans and replayed picks, per engine release, for the golden files.

The golden files under tests/golden/ are what this module produced: launch commands for about forty configurations
per engine, the candidates and plans for three cards and four models, and the pick every recorded profile
replays to. Each engine release is seen through tests/engine_trees.py, so the feature checks that read the
engine's source answer as they would on that release. They pin today's behaviour so that a later change that is
meant to move nothing (the engine probe, a refactor) can show it moved nothing, and a change that is meant to
move something shows exactly what.

    python -m tests.golden_cases            # rewrite every golden file from the code as it is
    python -m tests.golden_cases --check    # list the entries that differ, write nothing

The files were first captured on ee9ead1, before the three fixes that came just before them, and only the entries
those fixes moved were regenerated:
- launch/vllm-0.33.0.json, the five reuse/* entries: vLLM 0.33 keeps its server arguments in
  entrypoints/common/cli_args.py, where rigtune now finds --enable-prompt-tokens-details. Every launch is now the
  one vLLM 0.29.0 gets.
- plans/sglang-0.5.19.json and plans/sglang-0.5.23.json, the gemma-4 entries of each card: the nine speculation
  rows with the radix cache off now hold an eviction interval per draft token (on the A40, n-gram at batch 16 no
  longer fits at ctx 2048, 4096 and 8192); on 0.5.23 the 4090 and the L4 also plan 4096-token prefill chunks
  wherever the radix cache is off and no prefill budget is set (those speculation rows, and nine more each).
Added later, as new entries only (every existing entry unchanged): qwen3.5-9b-mtp, Qwen3.5-9B with its own MTP head
found (the head's 486,581,248 B and the 4,068,474,880 B of embedding and output head its listing gives), in the
candidates and plans of every card, and the qwen3.5-mtp-* launch cases.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest import mock

from tests import engine_trees

GOLDEN = Path(__file__).parent / "golden"
FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
PORT = 8000

# (compute capability, total MiB as nvidia-smi reports it). The 4090 and the L4 sit in SGLang's 20-35 GiB tier,
# the A40 in its 35-60 GiB one.
CARDS = {"a40": ("NVIDIA A40", (8, 6), 46068), "rtx4090": ("NVIDIA GeForce RTX 4090", (8, 9), 24564),
         "l4": ("NVIDIA L4", (8, 9), 23034)}

# meta-llama/Llama-3.1-8B-Instruct config.json, the fields that matter for planning.
LLAMA_31_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "intermediate_size": 14336,
    "max_position_embeddings": 131072, "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8,
    "head_dim": 128, "rope_theta": 500000.0, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
    "vocab_size": 128256,
}
DRAFT = "meta-llama/Llama-3.2-1B-Instruct"
HEAD = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
# name: (hf id, config fixture or dict)
MODELS = {
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "qwen2_5_7b_instruct_config.json"),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", LLAMA_31_8B_CONFIG),
    "qwen3.5-9b": ("Qwen/Qwen3.5-9B", "qwen3_5_9b_config.json"),
    "qwen3.5-9b-mtp": ("Qwen/Qwen3.5-9B", "qwen3_5_9b_config.json"),  # with its own MTP head found
    "gemma-4": ("google/gemma-4-12B-it", "gemma4_12b_it_config.json"),
}


def hw(card: str, engine: str, version: str):
    from rigtune.models import BackendAvailability, CPUInfo, GiB, GPUInfo, HardwareDescriptor

    name, cc, mib = CARDS[card]
    cpu = CPUInfo(model_name="Test CPU", physical_cores=16, logical_cores=32, avx2=True, avx512=False,
                  ram_total_bytes=128 * GiB, ram_free_bytes=120 * GiB, arch="x86_64")
    gpus = [GPUInfo(vendor="nvidia", name=name, compute_capability=cc, vram_total_bytes=mib * 2**20,
                    vram_free_bytes=(mib - 500) * 2**20)]
    backends = {engine: BackendAvailability(name=engine, available=True, version=version)}
    return HardwareDescriptor(os="Linux test", python="3.12", gpus=gpus, cpu=cpu, backends=backends)


def prepared(model: str, engine: str, revision: Optional[str] = None):
    from rigtune.hfconfig import arch_from_config
    from rigtune.models import ModelSpec, PreparedModel

    hf_id, config = MODELS[model]
    if isinstance(config, str):
        config = json.loads((FIXTURES / config).read_text(encoding="utf-8"))
    arch = arch_from_config(config)
    p = arch.num_params or 0
    pm = PreparedModel(spec=ModelSpec(hf_id=hf_id, revision=revision), backend=engine, arch=arch, hf_path=hf_id,
                       hf_paths={"awq": hf_id + "-AWQ"},
                       weights_bytes={"bf16": 2 * p, "fp16": 2 * p, "fp8": p, "awq": p // 2})
    if model == "llama-3.1-8b":  # the second models --speculative all finds for it
        pm.spec_repos = {"eagle3": HEAD, "draft": DRAFT}
        pm.spec_bytes = {HEAD: 850_000_000, DRAFT: 2_471_645_608}
    if model == "qwen3.5-9b-mtp":  # what --speculative all records once the checkpoint's own head is found
        pm.spec_repos = {"mtp": hf_id}
        pm.spec_bytes = {"mtp": 486_581_248, "mtp_shared": 4_068_474_880}
    return pm


@contextlib.contextmanager
def release(engine: str, version: str, tmp: Path) -> Iterator[None]:
    """rigtune as it sees `engine` `version` installed, with nothing of this machine's state."""
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl

    with contextlib.ExitStack() as stack:
        # No engine is asked what it offers: the trees stand in for the engines' source (the facts suites give
        # the probe's answers themselves).
        stack.enter_context(mock.patch.dict(os.environ, {"RIGTUNE_HOME": str(tmp / "home"),
                                                         "RIGTUNE_ENGINE_PROBE": "off"}))
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        stack.enter_context(engine_trees.installed(engine, version, tmp / "trees" / f"{engine}-{version}"))
        # The engines' own registries are never imported, so an installed engine changes nothing.
        stack.enter_context(mock.patch.object(vl, "vllm_registry_archs", lambda: None))
        stack.enter_context(mock.patch.object(sg, "sglang_registry_archs", lambda: None))
        # Two PCIe cards: the tensor-parallel case shows the peer-to-peer workaround.
        stack.enter_context(mock.patch.object(vl, "nvlink_between", lambda gpus: False))
        stack.enter_context(mock.patch.object(sg, "nvlink_between", lambda gpus: False))
        yield


# ---------------------------------------------------------------- launch commands

# (case name, model, Config fields). Each runs without and, where marked, with per-request reuse reports.
def launch_cases(engine: str) -> List[Tuple[str, str, Dict[str, Any], bool]]:
    fp8_kv = "fp8_e5m2"
    lookup = "ngram_gpu:4" if engine == "vllm" else "ngram:12"
    k = 4 if engine == "vllm" else 3
    cases = [
        ("default", "llama-3.1-8b", {}, True),
        ("bf16-small", "llama-3.1-8b", {"ctx": 8192, "batch": 16, "gpu_memory_utilization": 0.80}, False),
        ("fp16", "llama-3.1-8b", {"quant": "fp16"}, False),
        ("fp8", "llama-3.1-8b", {"quant": "fp8"}, False),
        ("awq", "llama-3.1-8b", {"quant": "awq"}, False),
        ("no-fraction", "llama-3.1-8b", {"gpu_memory_utilization": None}, False),
        ("kv-fp8", "llama-3.1-8b", {"kv_dtype": "fp8"}, False),
        ("kv-fp8-e5m2", "llama-3.1-8b", {"kv_dtype": fp8_kv}, False),
        ("prefix-on", "llama-3.1-8b", {"prefix_cache": True}, False),
        ("prefix-off", "llama-3.1-8b", {"prefix_cache": False}, True),
        ("prefill-2048", "llama-3.1-8b", {"prefill_budget": 2048}, False),
        ("prefill-16384", "llama-3.1-8b", {"prefill_budget": 16384}, False),
        ("spec-lookup", "llama-3.1-8b", {"spec_decode": lookup}, False),
        ("spec-draft", "llama-3.1-8b", {"spec_decode": f"draft:{DRAFT}:{k}"}, False),
        ("spec-eagle3", "llama-3.1-8b", {"spec_decode": f"eagle3:{HEAD}:3"}, False),
        ("tp2", "llama-3.1-8b", {"tp": 2}, False),
        ("tier-host", "llama-3.1-8b", {"kv_tier": "host:16"}, True),
        ("revision-bf16", "llama-3.1-8b@abc123", {}, False),
        ("revision-awq", "llama-3.1-8b@abc123", {"quant": "awq"}, False),
        ("extra-flag", "llama-3.1-8b", {"extra": {"enforce_eager": True, "max_loras": 2, "skip": False}}, False),
        ("everything", "llama-3.1-8b", {"quant": "fp8", "kv_dtype": fp8_kv, "prefix_cache": False,
                                        "prefill_budget": 8192, "spec_decode": f"draft:{DRAFT}:{k}",
                                        "batch": 16, "ctx": 16384}, True),
        ("qwen2.5-default", "qwen2.5-7b", {}, False),
        ("qwen3.5-default", "qwen3.5-9b", {}, False),
        ("qwen3.5-text-only", "qwen3.5-9b", {"text_only": True}, True),
        ("qwen3.5-prefix-off-spec", "qwen3.5-9b", {"prefix_cache": False, "spec_decode": lookup}, False),
        ("gemma-4-default", "gemma-4", {}, False),
        ("gemma-4-prefix-off", "gemma-4", {"prefix_cache": False, "batch": 16}, False),
        ("qwen3.5-mtp-1", "qwen3.5-9b-mtp", {"spec_decode": "mtp:1", "batch": 16}, False),
        ("qwen3.5-mtp-3", "qwen3.5-9b-mtp", {"spec_decode": "mtp:3", "batch": 16}, False),
        ("qwen3.5-mtp-revision", "qwen3.5-9b-mtp@abc123", {"spec_decode": "mtp:1", "batch": 16}, False),
        ("qwen3.5-mtp-prefix-off", "qwen3.5-9b-mtp", {"spec_decode": "mtp:3", "prefix_cache": False, "batch": 16},
         False),
    ]
    if engine == "vllm":
        cases += [
            ("spec-ngram-cpu", "llama-3.1-8b", {"spec_decode": "ngram:4"}, False),
            ("spec-suffix", "llama-3.1-8b", {"spec_decode": "suffix:24"}, False),
            ("kv-int8", "llama-3.1-8b", {"kv_dtype": "int8_per_token_head"}, False),
            ("attention-forced", "llama-3.1-8b", {"extra": {"attention_backend": "FLASHINFER"}}, False),
            ("tier-disk", "llama-3.1-8b", {"kv_tier": "host:16+disk:40"}, False),
            ("tp2-tier", "llama-3.1-8b", {"tp": 2, "kv_tier": "host:16"}, False),
        ]
    else:
        cases += [
            ("no-overlap", "llama-3.1-8b", {"extra": {"disable_overlap_schedule": True}}, False),
            ("mamba-strategy", "qwen3.5-9b", {"extra": {"mamba_radix_cache_strategy": "no_buffer"}}, False),
            ("fp8-prefill-8192", "llama-3.1-8b", {"quant": "fp8", "prefill_budget": 8192}, False),
            ("tier-host-prefix-off", "llama-3.1-8b", {"kv_tier": "host:16", "prefix_cache": False}, False),
        ]
    return cases


def _model(spec: str, engine: str):
    name, _, revision = spec.partition("@")
    return prepared(name, engine, revision or None)


def launches(engine: str, version: str, tmp: Path) -> Dict[str, Any]:
    from rigtune.backends import registry
    from rigtune.models import Config

    out: Dict[str, Any] = {}
    with release(engine, version, tmp):
        plain, reporting = registry()[engine], registry(report_reuse=True)[engine]
        card = hw("rtx4090", engine, version)
        for name, model, fields, reuse in launch_cases(engine):
            pm = _model(model, engine)
            base = plain.default_config(card, pm).model_dump()
            base.update(fields)
            cfg = Config(**base)
            for be, label in ((plain, name), (reporting, f"reuse/{name}")):
                if label.startswith("reuse/") and not reuse:
                    continue
                try:
                    spec = be.launch_spec(cfg, pm, PORT)
                    out[label] = {"key": cfg.key(), "args": list(spec.args), "env": dict(sorted(spec.env.items()))}
                except ValueError as exc:
                    out[label] = {"key": cfg.key(), "error": str(exc)}
    return out


# ---------------------------------------------------------------- candidates and plans


def _grouped(keys: List[str]) -> Dict[str, Any]:
    import hashlib

    parts = [k.split("/") for k in keys]
    return {"n": len(keys), "quants": sorted({p[1] for p in parts}), "ctxs": sorted({int(p[2][3:]) for p in parts}),
            "batches": sorted({int(p[3][1:]) for p in parts}), "fractions": sorted({p[4] for p in parts}),
            "sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest()}


def candidates(engine: str, version: str, tmp: Path) -> Dict[str, Any]:
    from rigtune.backends import registry
    from rigtune.models import ModelSpec

    out: Dict[str, Any] = {}
    with release(engine, version, tmp):
        be = registry()[engine]
        be_mod = sys.modules[type(be).__module__]
        for card in sorted(CARDS):
            h = hw(card, engine, version)
            for model in sorted(MODELS):
                pm = prepared(model, engine)
                with mock.patch.object(be_mod, "load_arch", lambda spec, token=None, arch=pm.arch: arch):
                    supported = be.supports(h, ModelSpec(hf_id=pm.spec.hf_id))
                default = be.default_config(h, pm)
                entry = {
                    "supports": supported,
                    "precisions": be.precisions(h),
                    "kv_dtypes": be.kv_dtypes(h),
                    "default": default.key(),
                    "candidates": _grouped([c.key() for c in be.candidate_configs(h, pm)]),
                    "candidates_16k": _grouped([c.key() for c in be.candidate_configs(h, pm, min_ctx=16384)]),
                    "prefill_variants": [c.key() for c in be.prefill_variants(default)],
                    "spec_variants": [c.key() for c in be.spec_variants(default, pm, h)],
                    "spec_variants_all": [c.key() for c in be.spec_variants(default, pm, h, all_methods=True)],
                }
                out[f"{card}/{model}"] = entry
    return out


def plan_grid(be, h, pm) -> List[Any]:
    """Every candidate; and, for the first precision at the middle fraction, each with every KV dtype, the prefix
    cache off, every prefill budget, and every speculation setting with the prefix cache on and off."""
    base = be.candidate_configs(h, pm)
    grid = {c.key(): c for c in base}
    fractions = sorted({c.gpu_memory_utilization for c in base})
    middle = fractions[len(fractions) // 2] if fractions else None
    for c in base:
        if c.gpu_memory_utilization != middle or c.quant != base[0].quant:
            continue
        off = c.model_copy(update={"prefix_cache": False})
        variants = [c.model_copy(update={"kv_dtype": d}) for d in be.kv_dtypes(h)] + [off] \
            + be.prefill_variants(c) + be.prefill_variants(off)
        for s in be.spec_variants(c, pm, h, all_methods=True):
            variants += [s, s.model_copy(update={"prefix_cache": False})]
        for v in variants:
            grid[v.key()] = v
    return [grid[k] for k in sorted(grid)]


def plans(engine: str, version: str, tmp: Path) -> Dict[str, Any]:
    """[key, weights, KV cache, total, budget, feasible] for every configuration of plan_grid."""
    from rigtune.backends import registry
    from rigtune.memory import estimate

    out: Dict[str, Any] = {}
    with release(engine, version, tmp):
        be = registry()[engine]
        for card in sorted(CARDS):
            h = hw(card, engine, version)
            mm = be.memory_model(h)
            for model in sorted(MODELS):
                pm = prepared(model, engine)
                rows = []
                for c in plan_grid(be, h, pm):
                    e = estimate(h, pm, c, mm)
                    rows.append([c.key(), e.weights, e.kv_cache, e.total, e.budget, e.feasible])
                out[f"{card}/{model}"] = rows
    return out


# ---------------------------------------------------------------- replayed picks


def replays() -> Dict[str, Any]:
    import importlib.util

    script = ROOT / "benchmarks" / "replay_search.py"
    spec = importlib.util.spec_from_file_location("replay_search", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out: Dict[str, Any] = {}
    for path in sorted(mod.STRATEGIES.glob("**/profiles/**/*.json"), key=lambda p: p.as_posix()):
        profile = mod.load(path)
        if profile is None or not profile.calibration_table:
            continue
        r = mod.replay(profile, str(path))
        name = path.relative_to(mod.STRATEGIES).as_posix()
        out[name] = {"skipped": r.skipped} if r.skipped else {
            "verdict": r.verdict, "recorded": r.recorded, "replayed": r.replayed, "trials": r.trials,
            "missing": [list(m) for m in r.missing], "unused": r.unused, "cannot": r.cannot}
    return out


def _no_network():
    """What tests/conftest.py's no_network fixture pins, for the replay: no Hub lookups."""
    import rigtune.backends.llamacpp as lc
    import rigtune.backends.sglang as sg
    import rigtune.backends.vllm as vl
    import rigtune.quantized as qz
    import rigtune.speculative as sp
    from rigtune.hfconfig import arch_from_config
    from tests.conftest import LLAMA_3B_CONFIG

    arch = arch_from_config(LLAMA_3B_CONFIG)
    stack = contextlib.ExitStack()
    for mod in (lc, sg, vl):
        stack.enter_context(mock.patch.object(mod, "load_arch", lambda spec, token=None: arch))
    stack.enter_context(mock.patch.object(vl, "vllm_registry_archs", lambda: None))
    stack.enter_context(mock.patch.object(sg, "sglang_registry_archs", lambda: None))
    stack.enter_context(mock.patch.object(lc, "search_hub_gguf", lambda spec, quants, **kw: {}))
    stack.enter_context(mock.patch.object(lc, "llama_server_binary", lambda: "/usr/local/bin/llama-server"))
    stack.enter_context(mock.patch.object(qz, "find_prequantized_repos", lambda spec, **kw: {}))
    stack.enter_context(mock.patch.object(sp, "find_eagle3_head", lambda spec, arch, **kw: None))
    stack.enter_context(mock.patch.object(sp, "repo_bytes", lambda repo, token=None: None))
    return stack


# ---------------------------------------------------------------- files


def generate(tmp: Path) -> Dict[str, Any]:
    """Every golden file's content, by path relative to tests/golden."""
    out: Dict[str, Any] = {}
    for name in engine_trees.VERSIONS:
        engine, version = name.split("-")
        out[f"launch/{name}.json"] = launches(engine, version, tmp / "launch")
        out[f"candidates/{name}.json"] = candidates(engine, version, tmp / "candidates")
        out[f"plans/{name}.json"] = plans(engine, version, tmp / "plans")
    with _no_network():
        out["replay.json"] = replays()
    return out


def dumps(obj: Any) -> str:
    """JSON with one entry, or one plan row, per line, so a change shows as the lines it changed."""
    if isinstance(obj, dict) and obj and all(isinstance(v, list) and v and isinstance(v[0], list)
                                             for v in obj.values()):
        lines = []
        for k in sorted(obj):
            rows = ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in obj[k])
            lines.append(f" {json.dumps(k)}: [\n{rows}\n ]")
        return "{\n" + ",\n".join(lines) + "\n}\n"
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def load(rel: str) -> Any:
    return json.loads((GOLDEN / rel).read_text(encoding="utf-8"))


def differences(got: Dict[str, Any]) -> List[str]:
    """`file: entry` for every entry that differs from the golden files."""
    out = []
    for rel in sorted(got):
        path = GOLDEN / rel
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        for k in sorted(set(old) | set(got[rel])):
            if old.get(k) != got[rel].get(k):
                out.append(f"{rel}: {k}")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="list the entries that differ; write nothing")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        got = generate(Path(tmp))
    diff = differences(got)
    for line in diff:
        print(line)
    if args.check:
        return 1 if diff else 0
    for rel, content in got.items():
        path = GOLDEN / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(content), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
