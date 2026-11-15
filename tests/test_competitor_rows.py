"""benchmarks/competitor_rows.py: llm-optimizer's and AIConfigurator's picks as `rigtune compare --rows` rows.

The files in tests/fixtures/competitors/ are CONSTRUCTED SAMPLES, built by hand from each tool's source and
templates (each file says so in its first lines), not output of a real run. They stand in until the GPU session
produces the real files, which must replace them; until then these tests show the adapters read the documented
format, not that the tools write it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rigtune.backends import get_backend
from rigtune.bench.rows import read_rows_file
from rigtune.models import Config

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "competitors"
_spec = importlib.util.spec_from_file_location("competitor_rows", ROOT / "benchmarks" / "competitor_rows.py")
cr = importlib.util.module_from_spec(_spec)
sys.modules["competitor_rows"] = cr
_spec.loader.exec_module(cr)


def sample(name: str):
    text = (FIX / name).read_text(encoding="utf-8")
    return json.loads(text) if name.endswith(".json") else text


def via_rigtune(row: dict, tmp_path: Path):
    """The row as `rigtune compare --rows` reads it."""
    path = tmp_path / "rows.json"
    cr.write_rows([row], path)
    return read_rows_file(path)[row["label"]]


def test_every_sample_says_it_is_constructed():
    names = sorted(p.name for p in FIX.iterdir())
    assert names and all(".SAMPLE." in n for n in names)
    for p in FIX.iterdir():
        assert "CONSTRUCTED SAMPLE, not output of a real run" in p.read_text(encoding="utf-8")[:600]


# --------------------------------------------------------------------------- llm-optimizer


def test_llm_optimizer_row_is_its_constrained_best_and_launches_as_it_did(tmp_path, prepared_vllm):
    data = sample("llm-optimizer-vllm.SAMPLE.json")
    row = cr.lo_row(data, "vllm", 4096, "7be87d7")
    pick = data["best_configurations"]["best_output_throughput_constrained"]["config"]["server_args"]
    unconstrained = data["best_configurations"]["best_output_throughput"]["config"]["server_args"]
    assert pick != unconstrained  # the sample's fastest run missed the TTFT limit, so the choice matters
    extra = via_rigtune(row, tmp_path)
    assert extra.kind == "tuner" and extra.skipped is None
    cfg = extra.config
    assert (cfg.backend, cfg.quant, cfg.ctx, cfg.batch, cfg.prefill_budget) == (
        "vllm", "bf16", pick["max_model_len"], pick["max_num_seqs"], pick["max_num_batched_tokens"])
    assert cfg.gpu_memory_utilization is None and cfg.prefix_cache is None  # left at the engine's defaults
    args = get_backend("vllm").launch_spec(cfg, prepared_vllm, 1).args
    for flag, value in (("--max-num-seqs", "128"), ("--max-num-batched-tokens", "8192"),
                        ("--max-model-len", "4096")):
        assert args[args.index(flag) + 1] == value
    assert "--gpu-memory-utilization" not in args
    src = row["source"]
    assert "llm-optimizer at 7be87d7" in src and "ttft:p95<500ms;itl:mean<50ms" in src
    assert "at max_concurrency 8" in src and "random prompts of 63 tokens" in src


def test_llm_optimizer_with_no_run_within_the_limits_is_not_applicable(tmp_path):
    row = cr.lo_row(sample("llm-optimizer-vllm-none-met.SAMPLE.json"), "vllm", 4096, "7be87d7")
    extra = via_rigtune(row, tmp_path)
    assert extra.config is None and "no run llm-optimizer measured met ttft:p95<5ms" in extra.skipped


def test_llm_optimizer_without_constraints_is_not_used(tmp_path):
    data = sample("llm-optimizer-vllm.SAMPLE.json")
    data["metadata"]["constraints"] = None
    assert "without --constraints" in cr.lo_row(data, "vllm", 4096, "x")["skipped"]


def test_llm_optimizer_sglang_pick_carries_its_other_settings(tmp_path, prepared_vllm):
    row = cr.lo_row(sample("llm-optimizer-sglang.SAMPLE.json"), "sglang", 4096, "7be87d7")
    cfg = via_rigtune(row, tmp_path).config
    assert (cfg.batch, cfg.prefill_budget, cfg.ctx) == (64, 4096, 4096)
    assert cfg.extra == {"schedule_conservativeness": 0.3}
    args = get_backend("sglang").launch_spec(cfg, prepared_vllm.model_copy(update={"backend": "sglang"}), 1).args
    assert args[args.index("--schedule-conservativeness") + 1] == "0.3"
    assert args[args.index("--max-running-requests") + 1] == "64"
    assert args[args.index("--chunked-prefill-size") + 1] == "4096"


def test_not_llm_optimizer_output_is_an_error():
    with pytest.raises(cr.AdapterError, match="best_configurations"):
        cr.lo_row({"rows": []}, "vllm", 4096, "x")


def test_llm_optimizer_gets_the_comparisons_model_limits_lengths_and_levels():
    cmd = cr.lo_command("vllm", "Qwen/Qwen3.5-9B", 500, 50, 63, 395, 4096, [1, 4, 8], "out.json")
    opt = {cmd[i]: cmd[i + 1] for i in range(1, len(cmd) - 1, 2)}
    assert opt["--framework"] == "vllm" and opt["--model"] == "Qwen/Qwen3.5-9B" and opt["--gpus"] == "1"
    assert opt["--constraints"] == "ttft:p95<500ms;itl:mean<50ms"
    assert opt["--server-args"] == "max_num_seqs=[64,128,256];max_num_batched_tokens=[2048,8192,16384];max_model_len=4096"
    client = dict(kv.split("=") for kv in opt["--client-args"].split(";"))
    assert client["max_concurrency"] == "[1,4,8]" and client["random_input_len"] == "63"
    assert client["random_output_len"] == "395" and client["dataset_name"] == "random"
    sg = cr.lo_command("sglang", "M", 500, None, 63, 395, 4096, [8], "o.json", server_cmd="py -m sglang.launch_server")
    assert sg[sg.index("--constraints") + 1] == "ttft:p95<500ms" and sg[-2:] == ["--server-cmd", "py -m sglang.launch_server"]
    assert "context_length=4096" in sg[sg.index("--server-args") + 1]


# --------------------------------------------------------------------------- AIConfigurator


def test_aiconfigurator_vllm_row_reads_the_worker_flags(tmp_path):
    row = cr.aic_row(sample("aiconfigurator-vllm-node_0_run.SAMPLE.sh"), "vllm", 4096, "0.12.0", system="l40s")
    cfg = via_rigtune(row, tmp_path).config
    assert (cfg.ctx, cfg.batch, cfg.prefill_budget, cfg.kv_dtype, cfg.tp) == (4096, 96, 8192, "fp8", 1)
    # Its own context (4458, sized from --isl and --osl) is replaced by the comparison's, and the row says so.
    assert "sized the context at 4458" in row["source"] and "the comparison's 4096" in row["source"]
    assert cfg.extra == {"block_size": 64}
    assert "cudagraph_capture_sizes ([1, 2, 4, 8, 16, 32, 64, 96]) is left at the engine's own list" in row["source"]
    assert "AIConfigurator 0.12.0" in row["source"] and "--system l40s" in row["source"]


def test_aiconfigurator_sglang_row_reads_quoted_values_and_bare_flags(tmp_path, prepared_vllm):
    row = cr.aic_row(sample("aiconfigurator-sglang-node_0_run.SAMPLE.sh"), "sglang", 4096, "0.12.0")
    cfg = via_rigtune(row, tmp_path).config
    assert (cfg.ctx, cfg.batch, cfg.prefill_budget, cfg.gpu_memory_utilization, cfg.kv_dtype) == (
        4096, 128, 8192, 0.85, "fp8_e4m3")
    assert cfg.extra == {"page_size": 64, "cuda_graph_max_bs": 128, "enable_mixed_chunk": True}
    args = get_backend("sglang").launch_spec(cfg, prepared_vllm.model_copy(update={"backend": "sglang"}), 1).args
    assert "--enable-mixed-chunk" in args and args[args.index("--mem-fraction-static") + 1] == "0.85"


@pytest.mark.parametrize("gpu, system", [
    ("NVIDIA L40S", "l40s"), ("NVIDIA H100 80GB HBM3", "h100_sxm"), ("NVIDIA A100-SXM4-80GB", "a100_sxm"),
    ("NVIDIA H200", "h200_sxm"), ("NVIDIA A40", None), ("NVIDIA GeForce RTX 4090", None), ("NVIDIA L4", None),
    ("NVIDIA H100 NVL", None), ("NVIDIA A100 80GB PCIe", None), ("NVIDIA GB200", "gb200"),
    ("NVIDIA GH200 480GB", None), ("NVIDIA H200 NVL", None), ("NVIDIA H100 PCIe", None),
    ("NVIDIA B200", "b200_sxm"), ("NVIDIA H100 SXM5 80GB", "h100_sxm")])
def test_aiconfigurator_system_for_a_card(gpu, system):
    assert cr.aic_system(gpu) == system


def test_aiconfigurator_on_a_card_it_has_no_data_for_is_not_applicable(tmp_path, capsys):
    rc = cr.main(["aiconfigurator-cmd", "--model", "Qwen/Qwen3.5-9B", "--gpu", "NVIDIA A40", "--backend", "vllm",
                  "--ttft-ms", "500", "--tpot-ms", "50", "--input-len", "63", "--output-len", "395",
                  "--save-dir", "aic"])
    assert rc == 3
    reason = capsys.readouterr().err.strip()
    assert "no system for this card (NVIDIA A40)" in reason
    out = tmp_path / "rows.json"
    assert cr.main(["not-applicable", "--label", "aiconfigurator-vllm", "--reason", reason, "--out", str(out)]) == 0
    row = read_rows_file(out)["aiconfigurator-vllm"]
    assert row.config is None and row.skipped == reason


def test_aiconfigurator_gets_the_comparisons_limits_and_lengths(capsys):
    assert cr.main(["aiconfigurator-cmd", "--model", "Qwen/Qwen3-8B", "--gpu", "NVIDIA L40S", "--backend", "vllm",
                    "--ttft-ms", "500", "--tpot-ms", "50", "--input-len", "63", "--output-len", "395",
                    "--save-dir", "aic"]) == 0
    cmd = json.loads(capsys.readouterr().out)
    opt = {cmd[i]: cmd[i + 1] for i in range(3, len(cmd) - 1, 2)}
    assert cmd[:3] == ["aiconfigurator", "cli", "default"]
    assert opt == {"--model": "Qwen/Qwen3-8B", "--total-gpus": "1", "--system": "l40s", "--backend": "vllm",
                   "--isl": "63", "--osl": "395", "--ttft": "500", "--save-dir": "aic", "--tpot": "50"}


def test_aiconfigurator_row_from_a_save_dir(tmp_path):
    run = tmp_path / "aic" / "Qwen3-8B_l40s_vllm" / "agg" / "top1" / "agg"
    run.mkdir(parents=True)
    (run / "node_0_run.sh").write_text(sample("aiconfigurator-vllm-node_0_run.SAMPLE.sh"), encoding="utf-8")
    out = tmp_path / "rows.json"
    assert cr.main(["aiconfigurator-row", "--save-dir", str(tmp_path / "aic"), "--backend", "vllm",
                    "--tool-version", "0.12.0", "--out", str(out)]) == 0
    assert read_rows_file(out)["aiconfigurator-vllm"].config.batch == 96
    assert cr.main(["aiconfigurator-row", "--save-dir", str(tmp_path / "empty"), "--backend", "vllm",
                    "--tool-version", "0.12.0", "--out", str(out)]) == 2


# --------------------------------------------------------------------------- what cannot be carried


@pytest.mark.parametrize("args, why", [
    ({"pipeline_parallel_size": 2}, "more than the one card"),
    ({"speculative_config": "{}"}, "speculation"),
    ({"skip_tokenizer_init": True}, "without a tokenizer"),
    ({"quantization": "awq"}, "only as fp8"),
    ({"max_num_seqs": [64, 128]}, "to a list"),
])
def test_a_setting_a_row_cannot_launch_faithfully_makes_it_not_applicable(args, why):
    cfg, _, skipped = cr.to_config("vllm", args, "bf16", 4096)
    assert cfg is None and why in skipped


def test_settings_the_tool_left_alone_stay_at_the_engines_defaults_and_say_so():
    cfg, notes, skipped = cr.to_config("vllm", {"max_num_batched_tokens": 2048, "enable_prefix_caching": False},
                                       "bf16", 8192)
    assert skipped is None and Config.model_validate(cfg).key() == "vllm/bf16/ctx8192/b256/pb2048/nopc"
    assert any("no batch cap" in n and "1024" in n for n in notes) and any("no context length" in n for n in notes)


def test_engine_flags_from_a_plain_serve_line():
    line = "vllm serve Qwen/Qwen3-8B --max-num-seqs 64 --enable-prefix-caching --kv-cache-dtype=fp8 > log 2>&1 &"
    assert cr.engine_flags(line, "vllm") == {"max_num_seqs": 64, "enable_prefix_caching": True, "kv_cache_dtype": "fp8"}
    with pytest.raises(cr.AdapterError, match="no sglang worker line"):
        cr.engine_flags(line, "sglang")
