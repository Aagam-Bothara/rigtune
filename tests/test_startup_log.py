"""vLLM's start-up lines are read from its log into the trial's start-up record, and nothing is guessed.

Each line is matched on its whole phrase with its unit. The fixtures are the format strings of vLLM v0.29.0 and
v0.33.0 (cited in rigtune/memlog.py) filled in; none of these lines has been read from a real log yet.
"""

from __future__ import annotations

import pytest

import rigtune.calibrate.search as search_mod
from rigtune import ledger as lg
from rigtune.calibrate.search import SubprocessTrialRunner
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.ledger import Ledger
from rigtune.memlog import parse_startup
from rigtune.models import Config, TrialMetrics
from tests.test_failure_kind import _Live, _LiveBackend

CAPTURE = "INFO 11-16 02:10:44 [gpu_model_runner.py:6517] Graph capturing finished in 23 secs, took 0.71 GiB"
ENCODER = "INFO 11-16 02:10:40 [mm_encoder_model_runner.py:95] Encoder graph capturing finished in 4 secs, took 0.20 GiB (6 graphs)"
MISS_029 = "INFO 11-16 02:09:58 [backends.py:394] Compiling a graph for compile range (1, 8192) takes 31.42 s"
MISS_033 = "INFO 11-16 02:09:58 [backends.py:418] [decoder] Compiling a graph for compile range (1, 8192) takes 31.42 s"
HIT_029 = ("INFO 11-16 02:09:31 [backends.py:293] Directly load the compiled graph(s) for compile range (1, 8192) "
           "from the cache, took 0.412 s")
HIT_033 = ("INFO 11-16 02:09:31 [backends.py:311] [decoder] Directly load the compiled graph(s) for compile range "
           "(1, 8192) from the cache, took 0.412 s")
REWRITE = ("WARNING 11-16 02:09:20 [compilation.py:1461] CUDAGraphMode.FULL_AND_PIECEWISE is not supported with "
           "spec-decode for attention backend FlashInferBackend (support: UNIFORM_SINGLE_TOKEN_DECODE); "
           "setting cudagraph_mode=PIECEWISE")


def _log(*lines: str, newline: str = "\n") -> str:
    return newline.join(("INFO 11-16 02:09:00 [api_server.py:1] vLLM API server version 0.33.0",) + lines) + newline


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_capture_line_parsed_both_releases(newline):
    """The capture line is the same text in 0.29 and 0.33, from either model runner."""
    got = parse_startup("vllm", _log(CAPTURE, newline=newline))
    assert (got["capture_s"], got["cudagraph_gib"], got["capture_lines"]) == (23.0, 0.71, 1)
    assert got["parsed"] == ["capture"]


def test_encoder_capture_line_is_not_the_decoder_one():
    assert parse_startup("vllm", _log(ENCODER))["parsed"] == []
    both = parse_startup("vllm", _log(ENCODER, CAPTURE))
    assert (both["capture_s"], both["cudagraph_gib"], both["capture_lines"]) == (23.0, 0.71, 1)


@pytest.mark.parametrize("line", [
    "(pid=4096) took 512 MiB",
    "INFO graph capturing finished in 23 secs, took 0.71 GiB",  # not the engine's capitalisation
    "INFO Graph capturing finished in 23 secs",  # no size: not the whole phrase
    "INFO Graph capturing finished in 23 secs, took 404 MiB",  # another unit
    "INFO Compiling a graph for compile range (1, 8192)",  # no time
    "INFO 14040 graphs were compiled in 31.42 s",
    "WARNING cudagraph_mode=PIECEWISE was requested",  # a request, not the engine's change
])
def test_pid_and_byte_counts_do_not_match(line):
    assert parse_startup("vllm", _log(line)) == {"parsed": []}


def test_compile_prefix_of_0_33_parsed():
    for miss in (MISS_029, MISS_033):
        got = parse_startup("vllm", _log(miss, miss.replace("31.42", "8.08").replace("(1, 8192)", "(8193, 16384)")))
        assert got["compile_s"] == 39.5 and got["compile_cached"] is False and got["parsed"] == ["compile"]


def test_cache_hit_and_miss_lines():
    for hit in (HIT_029, HIT_033):
        got = parse_startup("vllm", _log(hit))
        assert got["compile_cached"] is True and "compile_s" not in got and got["parsed"] == ["compile_cache"]
    mixed = parse_startup("vllm", _log(HIT_033, MISS_033))  # one range from the cache, one compiled: not a hit
    assert mixed["compile_cached"] is False and mixed["compile_s"] == 31.42


# From the A40 session's Gemma 4 12B relaunch (16 Nov 2026), the only cache hit read from a real log so far: vLLM 0.33
# loaded the whole model from its AOT cache and printed no compile-range line at all.
AOT_HIT_033 = ("(EngineCore pid=289984) INFO 11-17 16:15:55 [decorators.py:326] [backbone] Directly load AOT "
               "compilation from path /root/.cache/vllm/torch_compile_cache/torch_aot_compile/4fc34eff6932ab7a/"
               "rank_0_0/model")
AOT_SAVED_033 = ("(EngineCore pid=276944) INFO 11-17 15:35:58 [decorators.py:752] [backbone] saved AOT compiled "
                 "function to /root/.cache/vllm/torch_compile_cache/torch_aot_compile/4fc34eff6932ab7a/rank_0_0/model")


def test_an_aot_cache_load_is_a_hit():
    got = parse_startup("vllm", _log(AOT_HIT_033, CAPTURE))
    assert got["compile_cached"] is True and "compile_s" not in got
    assert got["parsed"] == ["capture", "compile_cache"]


def test_an_aot_save_after_compiling_is_a_miss():
    got = parse_startup("vllm", _log(MISS_033, AOT_SAVED_033))
    assert got["compile_cached"] is False and got["compile_s"] == 31.42


def test_an_aot_load_beside_a_compiled_range_is_not_a_hit():
    """MTP: the backbone loads from the cache while the draft head compiles."""
    head = MISS_033.replace("[decoder]", "[eagle_head]").replace("31.42", "3.29")
    got = parse_startup("vllm", _log(AOT_HIT_033, head))
    assert got["compile_cached"] is False and got["compile_s"] == 3.29


# From the H200 session's Qwen3.5-35B-A3B MTP launch and its AOT-cached relaunch (20 Nov 2026).
RANGE_H200 = ("(EngineCore pid=1619) INFO 11-20 23:28:36 [backends.py:417] [backbone] Compiling a graph for compile "
              "range (1, 8192) takes 16.41 s")
TOTAL_H200 = "(EngineCore pid=1619) INFO 11-20 23:28:43 [monitor.py:62] [backbone] torch.compile took 34.83 s in total"
TOTAL_HEAD = "(EngineCore pid=1619) INFO 11-20 23:29:10 [monitor.py:62] [eagle_head] torch.compile took 4.83 s in total"


def test_compile_reads_vllms_own_total_not_the_ranges():
    """The compile-range line leaves out Dynamo's transform and the cache save; the total has them."""
    got = parse_startup("vllm", _log(RANGE_H200, TOTAL_H200))
    assert got["compile_s"] == 34.83 and got["compile_cached"] is False and got["parsed"] == ["compile"]


def test_compile_total_summed_over_parts_once_per_worker():
    other_worker = TOTAL_H200.replace("pid=1619", "pid=1620").replace("34.83", "33.10")
    assert parse_startup("vllm", _log(TOTAL_H200, other_worker, TOTAL_HEAD))["compile_s"] == 39.66


def test_a_cache_load_reads_its_load_time():
    got = parse_startup("vllm", _log(AOT_HIT_033, TOTAL_H200.replace("34.83", "0.48")))
    assert got["compile_cached"] is True and got["compile_s"] == 0.48


# From the A100 session's Qwen3.5-9B MTP launch (18 Nov 2026), cold, and the Modal run's warm start (19 Nov 2026).
WARMUP_COLD = ("(EngineCore pid=19238) INFO 11-18 05:57:41 [monitor.py:91] [backbone] Initial profiling/warmup run "
               "took 81.98 s")
WARMUP_HEAD = ("(EngineCore pid=19238) INFO 11-18 05:57:45 [monitor.py:91] [eagle_head] Initial profiling/warmup run "
               "took 0.01 s")


def test_warmup_is_summed_over_the_compiled_parts():
    got = parse_startup("vllm", _log(WARMUP_COLD, WARMUP_HEAD))
    assert got["warmup_s"] == 81.99 and got["parsed"] == ["warmup"]


def test_warmup_printed_by_every_worker_counts_once_per_part():
    other_worker = WARMUP_COLD.replace("pid=19238", "pid=19239").replace("81.98", "80.10")
    assert parse_startup("vllm", _log(WARMUP_COLD, other_worker))["warmup_s"] == 81.98


def test_a_warm_start_reads_its_small_warmup():
    assert parse_startup("vllm", _log(WARMUP_COLD.replace("81.98", "0.34")))["warmup_s"] == 0.34


def test_no_compile_line_is_unknown_not_cached():
    got = parse_startup("vllm", _log(CAPTURE))
    assert "compile_cached" not in got and "compile_s" not in got


def test_rewrite_warning_gives_mode_name_only():
    got = parse_startup("vllm", _log(REWRITE, REWRITE, CAPTURE))
    assert got["rewrites"] == ["cudagraph_mode=PIECEWISE"]  # named once, and nothing else of the line is kept
    assert got["parsed"] == ["capture", "rewrites"]


def test_tp_workers_reduced_by_max_with_count():
    got = parse_startup("vllm", _log(CAPTURE, CAPTURE.replace("23 secs, took 0.71", "31 secs, took 0.64")))
    assert (got["capture_s"], got["cudagraph_gib"], got["capture_lines"]) == (31.0, 0.64, 2)


# From the H200 session (20 Nov 2026): one worker captures to size the graphs' memory, then for real.
SIZING_H200 = ("(EngineCore pid=972) INFO 11-20 23:19:24 [model_runner.py:1163] Graph capturing finished in 9 secs, "
               "took 0.42 GiB")
KEPT_H200 = ("(EngineCore pid=972) INFO 11-20 23:19:35 [model_runner.py:1163] Graph capturing finished in 8 secs, "
             "took 0.22 GiB")


def test_both_passes_of_one_worker_are_summed_and_the_last_is_kept():
    got = parse_startup("vllm", _log(SIZING_H200, KEPT_H200))
    assert (got["capture_s"], got["cudagraph_gib"], got["capture_lines"]) == (17.0, 0.22, 2)


def test_two_workers_each_with_two_passes_give_the_slowest_worker():
    other = [line.replace("pid=972", "pid=973").replace("9 secs", "12 secs") for line in (SIZING_H200, KEPT_H200)]
    got = parse_startup("vllm", _log(SIZING_H200, other[0], KEPT_H200, other[1]))
    assert (got["capture_s"], got["cudagraph_gib"], got["capture_lines"]) == (20.0, 0.22, 4)


@pytest.mark.parametrize("text", ["", "INFO 11-16 02:09:00 [api_server.py:1] vLLM API ser", CAPTURE[:60]])
def test_empty_or_truncated_log_parses_nothing_and_says_so(text):
    assert parse_startup("vllm", text) == {"parsed": []}


def test_sglang_log_gets_seconds_only():
    """No SGLang start-up line has been read from a real log, so none is matched, even one shaped like vLLM's."""
    assert parse_startup("sglang", _log(CAPTURE, MISS_033)) == {"parsed": []}
    assert parse_startup("llamacpp", _log(CAPTURE)) == {"parsed": []}


class _Logged(_Live):
    started_at, ready_at = 10.0, 70.0

    def tail_log(self, n: int = 50) -> str:
        return _log(MISS_033, REWRITE, CAPTURE)


class _LoggedBackend(_LiveBackend):
    def launch(self, cfg, model, port, log_path=None):
        return _Logged()


def test_capture_seconds_and_gib_reach_the_ledger(tmp_path, hw_a100, spec, monkeypatch):
    good = TrialMetrics(requests=8, failed=0, output_tokens=800, tok_s=100.0, ttft_ms=50.0, tpot_ms=10.0)
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: good)
    runner = SubprocessTrialRunner(backends={"vllm": _LoggedBackend("")}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    res = runner.run(Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9), "batch")
    ledger = Ledger.for_calibration(hw_a100, spec, get_workload("default"), "throughput", {}, base=tmp_path)
    ledger.trial(res)
    (record,) = [r for r in lg.read(ledger.path) if r["type"] == "trial"]
    assert record["startup"] == {
        "seconds": 60.0, "capture_s": 23.0, "cudagraph_gib": 0.71, "capture_lines": 1, "compile_s": 31.42,
        "compile_cached": False, "warmup_s": None, "rewrites": ["cudagraph_mode=PIECEWISE"],
        "parsed": ["capture", "compile", "rewrites"]}
    assert "startup" not in res.model_dump(mode="json")
