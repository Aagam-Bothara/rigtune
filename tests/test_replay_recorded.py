"""benchmarks/replay_search.py: recorded calibrations replayed through today's search, classed against a baseline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from rigtune.models import Config, TrialMetrics, TrialResult

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "benchmarks" / "replay_search.py"
PROFILES = ROOT / "benchmarks" / "strategies"
QUALITY_14B = "results-14b/profiles/4090/balanced-file-dolly-d03d5896-quality-0.020-quant-auto+awq+gptq+w8a8-ttft-p95.json"
# Recorded configurations the search no longer asks for. The 14B's calibration tried Qwen2.5-0.5B as a draft for
# the 14B, whose vocabulary differs; rigtune stopped offering that pair (speculative.DRAFTS).
UNUSED = {"results-14b/profiles/4090/balanced-file-dolly-d03d5896-ttft-p95.json": [
    "vllm/fp8/ctx4096/b16/gmu0.90/sd=draft:Qwen/Qwen2.5-0.5B-Instruct:4"]}


def _load():
    spec = importlib.util.spec_from_file_location("replay_search", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


MOD = _load()
EXACT = sorted(name for name, verdict in MOD.BASELINE.items() if verdict == "exact")


@pytest.mark.parametrize("name", EXACT)
def test_an_exact_profile_replays_trial_for_trial_to_its_pick(name, no_network):
    profile = MOD.load(PROFILES / name)
    r = MOD.replay(profile, name)
    assert r.verdict == "exact"
    assert r.missing == [] and r.cannot == []  # every trial asked for was measured, under the recorded options
    assert r.unused == UNUSED.get(name, [])  # and it asked for every configuration that was, save those
    assert r.replayed == r.recorded == profile.config.key()


def test_a_profile_made_under_the_withdrawn_drift_gate_is_inconclusive(no_network):
    """Its trials replay one for one to its pick, but it ran with the drift gate at 0.020, which could refuse a
    precision and which the replay does not apply."""
    profile = MOD.load(PROFILES / QUALITY_14B)
    opts, cannot = MOD.options_of(profile)
    assert opts.quants == ["auto", "awq", "gptq", "w8a8"] and opts.ttft_percentile == 95
    assert cannot == ["quality=0.020"]
    r = MOD.replay(profile, QUALITY_14B)
    assert (r.missing, r.same, r.verdict) == ([], True, "inconclusive")


def test_every_recorded_profile_replays_without_a_regression(no_network, capsys):
    paths = sorted(PROFILES.glob("**/profiles/**/*.json"))
    assert MOD.main([str(p) for p in paths]) == 0
    out = capsys.readouterr().out
    assert "5 exact; 13 inconclusive; 3 different" in out and "REGRESSION" not in out


def test_only_a_worse_class_than_the_baseline_is_a_regression():
    r = MOD.Replay("p", "a", "a", 3)
    assert MOD.regression(EXACT[0], r) is None
    assert MOD.regression(EXACT[0], MOD.Replay("p", "a", "a", 3, missing=[("kv", "k")])) == "was exact, now inconclusive"
    assert MOD.regression("results-real/profiles/extract-run1.profile.json", MOD.Replay("p", "a", "b", 3)) is None
    assert MOD.regression("not/in/the/baseline.json", MOD.Replay("p", "a", "b", 3)) is None


def test_a_budgeted_calibration_is_not_replayed():
    profile = MOD.load(PROFILES / "results-dolly/profiles/balanced-file-dolly-d03d5896-budget-600s.json")
    assert MOD.replay(profile).skipped.startswith("budgeted")


def test_the_runner_answers_by_key_and_lists_what_was_never_measured():
    a = Config(backend="vllm", quant="fp8", batch=16)
    b = a.model_copy(update={"batch": 64})
    m = TrialMetrics(tok_s=100.0, requests=8, output_tokens=800)
    table = [TrialResult(config=a, stage="quant", metrics=m),
             TrialResult(config=a, stage="confirm", metrics=m.model_copy(update={"tok_s": 90.0}))]
    runner = MOD.RecordedRunner(table)
    assert runner.run(a, "batch").metrics.tok_s == 100.0 and runner.run(a, "batch").stage == "batch"
    assert runner.run(a, "confirm").metrics.tok_s == 90.0
    assert runner.run(a, "confirm").metrics.tok_s == 100.0  # recorded confirmations used up: the search trial
    res = runner.run(b, "batch")
    assert not res.ok and not res.launched
    assert runner.missing == [("batch", b.key())]


def test_the_opt_in_search_replays_to_the_same_picks_but_asks_for_more(no_network, capsys):
    """--speculative all with neither SGLang speculation nor EAGLE3 (no profile records them): every pick is
    unchanged, but the third speculation setting asks for trials no profile measured."""
    paths = sorted(PROFILES.glob("**/profiles/**/*.json"))
    assert MOD.main(["--speculative-all"] + [str(p) for p in paths]) == 0
    assert "1 exact; 17 inconclusive; 3 different" in capsys.readouterr().out
