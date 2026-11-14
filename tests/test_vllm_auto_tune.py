"""benchmarks/vllm_auto_tune.py: vLLM's auto_tune.sh patched to a rule, run, and its pick made a --rows file.

No vLLM file is copied here. SCRIPT has the lines the patch rewrites, as many times as vLLM v0.29.0's and
v0.33.0's benchmarks/auto_tune/auto_tune.sh (byte-identical at the two tags) have them; the patch was also
applied to both real files and their copies checked with `bash -n` when this was written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from rigtune.bench.rows import read_rows_file

_spec = importlib.util.spec_from_file_location(
    "vllm_auto_tune", Path(__file__).resolve().parents[1] / "benchmarks" / "vllm_auto_tune.py")
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)

BENCH = """    vllm bench serve \\
        --request-rate inf \\
        --percentile-metrics ttft,tpot,itl,e2el \\
        --goodput e2el:"$MAX_LATENCY_ALLOWED_MS" \\
        --port 8004 &> "$bm_log"
"""
SCRIPT = ("#!/bin/bash\n\nTAG=$(date +\"%Y_%m_%d_%H_%M\")\nHOSTNAME=$(hostname)\n"
          "start_server() {\n    pkill -if \"vllm serve\" || true\n}\n"
          "run_benchmark() {\n    pkill -if \"vllm serve\" || true\n" + BENCH
          + "    e2el=$(grep \"P99 E2EL (ms):\" \"$bm_log\" | awk '{print $NF}')\n" + BENCH
          + "    e2el=$(grep \"P99 E2EL (ms):\" \"$bm_log\" | awk '{print $NF}')\n"
          "    pkill -if \"vllm serve\" || true\n}\n"
          "if (( $(echo \"$best_throughput > 0\" | bc -l) )); then\n" + BENCH
          + "fi\npkill -if \"vllm serve\" || true\n")

STDOUT = """====================== AUTO TUNE PARAMETERS ====================
MODEL=Qwen/Qwen2.5-7B-Instruct
TP=1
INPUT_LEN=63
OUTPUT_LEN=395
MAX_MODEL_LEN=4096
MIN_CACHE_HIT_PCT=0
MAX_LATENCY_ALLOWED_MS=500
NUM_SEQS_LIST=128 256
NUM_BATCHED_TOKENS_LIST=512 1024 2048 4096
RESULT_FILE={result}
current_hash: 4ba787ec9959e566d3b587a796dff07673324da5
Using gpu_memory_utilization=.97 to serve model.
max_num_seqs: 128, max_num_batched_tokens: 512, request_rate: 4, e2el: 431.2, throughput: 3.92, goodput: 3.90
best_max_num_seqs: 128, best_num_batched_tokens: 512, best_throughput: 3.92
max_num_seqs: 256, max_num_batched_tokens: 4096, request_rate: 5, e2el: 470.0, throughput: 4.60, goodput: 4.51
best_max_num_seqs: 256, best_num_batched_tokens: 4096, best_throughput: 4.60
max_num_seqs: 256, max_num_batched_tokens: 512 does not meet latency requirement 500
best_max_num_seqs: 256, best_num_batched_tokens: 4096, best_throughput: 4.60
finish permutations
No configuration met the latency requirements. Skipping final profiling run.
best_max_num_seqs: 256, best_num_batched_tokens: 4096, best_throughput: 4.60, profile saved in: /p
"""


def test_the_ttft_rule_rewrites_every_latency_read_and_nothing_else():
    new = at.patch(SCRIPT, "ttft95")
    assert new.count("--metric-percentiles 95,99") == 3 and new.count("--goodput ttft:") == 3
    assert new.count('grep "P95 TTFT (ms):"') == 2 and "E2EL" not in new and "goodput e2el" not in new
    assert new.count("; at_gpu_wait") == 4 and new.count("at_gpu_wait() {") == 1
    assert 'AT_FINAL_PROFILE:-0' in new and "HOSTNAME=$(hostname)" in new
    own = at.patch(SCRIPT, "e2el99")
    assert 'grep "P99 E2EL (ms):"' in own and "--metric-percentiles" not in own and "; at_gpu_wait" in own
    assert at.patch(SCRIPT, "e2el99", final_profile=True, gpu_hygiene=False) == SCRIPT
    assert "HOSTNAME=127.0.0.1" in at.patch(SCRIPT, "e2el99", host_ip=True)


def test_a_different_script_is_refused_rather_than_half_patched():
    with pytest.raises(at.AutoTuneError, match="occurs 2 times, expected 3"):
        at.patch(SCRIPT.replace(BENCH, "", 1), "ttft95")
    with pytest.raises(at.AutoTuneError, match="rule must be"):
        at.patch(SCRIPT, "p50")


def test_the_copy_and_its_diff_are_written_beside_each_other(tmp_path):
    src = tmp_path / "auto_tune.sh"
    src.write_text(SCRIPT, encoding="utf-8")
    copy, diff = at.write_patched(src, tmp_path / "out", "ttft95")
    assert copy.read_bytes() == at.patch(SCRIPT, "ttft95").encode("utf-8") and src.read_text() == SCRIPT
    text = diff.read_text(encoding="utf-8")
    assert '-    e2el=$(grep "P99 E2EL (ms):"' in text and '+    e2el=$(grep "P95 TTFT (ms):"' in text
    _, empty = at.write_patched(src, tmp_path / "plain", "e2el99", final_profile=True, gpu_hygiene=False)
    assert empty.read_text(encoding="utf-8") == ""


def test_its_pick_becomes_a_config_with_real_weights(tmp_path):
    out = at.read_output(STDOUT.format(result="/r/result.txt"))
    assert out["gpu_memory_utilization"] == 0.97 and out["finished"]  # bc's ".97"
    assert out["best"] == {"max_num_seqs": 256, "max_num_batched_tokens": 4096, "req_per_s": 4.6}
    assert [m["max_num_seqs"] for m in out["measured"]] == ["128", "256"]
    assert out["failed_combinations"] == [{"max_num_seqs": 256, "max_num_batched_tokens": 512}]
    row = at.to_row(out, "ttft95")
    assert row["config"] == {"backend": "vllm", "quant": "bf16", "ctx": 4096, "batch": 256,
                             "gpu_memory_utilization": 0.97, "prefill_budget": 4096, "prefix_cache": True}
    assert row["label"] == "autotune-vllm" and row["kind"] == "tuner"
    for words in ("at 4ba787ec9959,", "p95 time to first token <= 500 ms", "auto_tune.diff", "INPUT_LEN=63",
                  "--load-format dummy (random weights)", "open-loop", "re-measured with the real weights"):
        assert words in row["source"], words
    assert "P99 end-to-end latency <= 500 ms" in at.to_row(out, "e2el99")["source"]


def test_no_pick_is_a_skipped_row_and_a_cut_short_run_says_so():
    no_start = at.read_output("MAX_MODEL_LEN=4096\nCannot find a proper gpu_memory_utilization over 0.9\n")
    assert at.to_row(no_start, "ttft95")["skipped"].startswith("auto_tune found no --gpu-memory-utilization")
    none_met = at.read_output("MAX_MODEL_LEN=4096\nUsing gpu_memory_utilization=0.98 to serve model.\n"
                              "best_max_num_seqs: 0, best_num_batched_tokens: 0, best_throughput: 0\n")
    row = at.to_row(none_met, "ttft95")
    assert row["skipped"] == "no combination auto_tune tried met its latency limit" and "config" not in row
    cut = at.read_output(STDOUT.format(result="").replace("finish permutations\n", ""))
    assert "did not finish its grid" in at.to_row(cut, "ttft95")["source"]


def test_the_lengths_are_the_calibrations_mean_prompt_and_answer():
    levels = {"1": {"requests": 10, "failed": 0, "output_tokens": 4000, "prompt_tokens": 60},
              "8": {"requests": 30, "failed": 2, "output_tokens": 11000, "prompt_tokens": 64}}
    profile = {"calibration_table": [{"metrics": {"by_concurrency": levels}},
                                     {"metrics": {"requests": 5, "failed": 5, "output_tokens": 0}}]}
    assert at.lengths_from_profile(profile) == (63, 395, 38)  # (60*10 + 64*28) / 38, 15000 / 38
    assert at.lengths_from_profile({"calibration_table": []}) is None


def _args(tmp_path, src, **kw):
    a = dict(vllm_src=str(src), model="Qwen/Qwen2.5-7B-Instruct", rule="ttft95", limit_ms=500.0,
             input_len=63, output_len=395, max_model_len=4096, tp=1, num_seqs=None, num_batched_tokens=None,
             download_dir="/hf", quant="bf16", out=tmp_path / "at", host_ip=False, no_gpu_hygiene=False,
             final_profile=False)
    a.update(kw)
    return argparse.Namespace(**a)


def _checkout(tmp_path, name="vllm"):
    src = tmp_path / "src" / name
    (src / "benchmarks" / "auto_tune").mkdir(parents=True)
    (src / "benchmarks" / "auto_tune" / "auto_tune.sh").write_text(SCRIPT, encoding="utf-8")
    return src


def test_run_patches_runs_and_writes_a_rows_file_compare_reads(tmp_path):
    src = _checkout(tmp_path)
    seen = {}

    def fake_launch(cmd, env, cwd, log):
        seen.update(cmd=cmd, env=env, cwd=cwd)
        folder = Path(env["BASE"]) / "auto-benchmark" / "2026_11_14_08_00"
        folder.mkdir(parents=True)
        (folder / "result.txt").write_text("hash:4ba787ec\nbest_max_num_seqs: 256, best_num_batched_tokens: "
                                           "4096, best_throughput: 4.60, profile saved in: /p\n",
                                           encoding="utf-8")
        (folder / "bm_log_256_4096_requestrate_inf.txt").write_text("P95 TTFT (ms): 470.0\n",
                                                                    encoding="utf-8")
        (folder / "profile").mkdir()
        log.write_text(STDOUT.format(result=folder / "result.txt"), encoding="utf-8")
        return 0

    assert at.run(_args(tmp_path, src), launch=fake_launch) == 0
    out = tmp_path / "at"
    assert seen["cmd"] == ["bash", str(out / "auto_tune.sh")] and seen["cwd"] == out
    env = seen["env"]
    assert env["BASE"] == str(src.parent) and env["SYSTEM"] == "GPU"
    assert [env[k] for k in ("MAX_LATENCY_ALLOWED_MS", "INPUT_LEN", "OUTPUT_LEN", "MAX_MODEL_LEN",
                             "DOWNLOAD_DIR")] == ["500", "63", "395", "4096", "/hf"]
    params = json.loads((out / "params.json").read_text())
    assert "NUM_SEQS_LIST" not in params["env"]  # left to the script's own grid
    assert (out / "auto_tune.diff").read_text().count("P95 TTFT") == 2
    assert (out / "result" / "result.txt").exists() and not (out / "result" / "profile").exists()
    rows = read_rows_file(out / "rows.json")
    assert rows["autotune-vllm"].config.key() == "vllm/bf16/ctx4096/b256/gmu0.97/pb4096/pc"
    summary = json.loads((out / "autotune_summary.json").read_text())
    assert summary["rule"] == "ttft95" and summary["vllm_commit"].startswith("4ba787ec")
    assert summary["exit_code"] == 0 and "exited with status" not in rows["autotune-vllm"].source


def test_run_returns_auto_tunes_own_exit_status_and_still_writes_the_row(tmp_path):
    src = _checkout(tmp_path)

    def cut_short(cmd, env, cwd, log):
        log.write_text(STDOUT.format(result="").replace("finish permutations\n", ""), encoding="utf-8")
        return 143  # killed by a deadline guard

    assert at.run(_args(tmp_path, src), launch=cut_short) == 143
    out = tmp_path / "at"
    row = read_rows_file(out / "rows.json")["autotune-vllm"]
    assert row.config is not None and row.source.endswith("so this is its best so far; auto_tune.sh exited "
                                                          "with status 143")
    assert json.loads((out / "autotune_summary.json").read_text())["exit_code"] == 143


def test_run_refuses_what_auto_tune_would_fail_on(tmp_path):
    with pytest.raises(at.AutoTuneError, match="directory named `vllm`"):
        at.run(_args(tmp_path, _checkout(tmp_path, "vllm-src")), launch=None)
    with pytest.raises(at.AutoTuneError, match="more than --max-model-len 400"):
        at.run(_args(tmp_path, _checkout(tmp_path), max_model_len=400), launch=None)
