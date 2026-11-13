"""A session profile records what its winning trial served, so the drift watch compares live turns with those."""

from __future__ import annotations

import json

import pytest

import rigtune.calibrate.measure as M
from rigtune.calibrate.workload import get_workload, workload_from_file
from rigtune.drift import TrafficWatch
from rigtune.models import TrialMetrics
from tests.test_session_replay import _file


def _turn(sid, turn, prompt, cached=None):
    return M.RequestOutcome(ok=True, ttft_s=0.1, duration_s=0.5, tokens=10, token_source="usage", prompt_tokens=prompt,
                            cached_tokens=cached, session=sid, turn=turn, live=4, request_chars=10, reusable_chars=5)


def test_a_level_records_served_prompt_lengths_and_turns():
    outcomes = [_turn(s, t, 100 * t, 0 if t == 1 else 90 * (t - 1)) for s in ("a", "b") for t in (1, 2, 3)]
    m = M._metrics_from(outcomes, 1.0, 2)
    assert m.prompt_tokens_seen == {"p50": 200, "p90": 300, "max": 300, "mean": 200}
    assert m.turns_seen == {"mean": 3.0, "max": 3}


def test_the_profile_takes_served_stats_from_the_winning_trial(tmp_path):
    wl = workload_from_file(_file(tmp_path), get_workload("chat"))
    winner = TrialMetrics(prompt_tokens_seen={"p50": 210, "p90": 380, "max": 400, "mean": 230},
                          turns_seen={"mean": 2.9, "max": 3}, prefix_reuse=0.62, effective_pool=8.0, reuse_distance=7)
    spec = wl.ensure_prompts().spec(winner)
    assert spec["prompt_tokens_seen"] == {"p50": 210, "p90": 380, "max": 400, "mean": 230}
    assert spec["turns_seen"] == {"mean": 2.9, "max": 3} and spec["reuse_achieved"] == 0.62
    assert spec["effective_pool"] == 8.0 and spec["reuse_distance"] == 7
    assert "reuse_achieved" not in wl.spec()  # nothing measured, nothing claimed
    assert "turns_seen" not in get_workload("chat").spec(winner)  # a flat workload records none of it


@pytest.mark.usefixtures("no_network")
def test_calibration_records_the_winners_served_stats(hw_gtx1080, spec, monkeypatch, tmp_path):
    import rigtune.backends.llamacpp as lc
    from rigtune.pipeline import calibrate, prepare_and_plan, select
    from tests.test_strategies import FakeRunner

    def fake_materialize(self, m, quants):
        for q in quants:
            m.gguf_paths[q] = f"/models/{q}.gguf"
        return m

    class Served(FakeRunner):
        def run(self, cfg, stage):
            res = super().run(cfg, stage)
            res.metrics.prompt_tokens_seen = {"p50": len(self.calls), "p90": 1, "max": 1, "mean": 1}
            return res

    monkeypatch.setattr(lc.LlamaCppBackend, "materialize", fake_materialize)
    wl = workload_from_file(_file(tmp_path), get_workload("chat"))
    candidates, reg = select(hw_gtx1080, spec, force="llamacpp-cuda", workload=wl)
    plan = prepare_and_plan(hw_gtx1080, spec, candidates, reg, materialize=True, workload=wl)
    profile = calibrate(hw_gtx1080, spec, "throughput", plan, reg, workload=wl, runner=Served())
    winner = next(r for r in reversed(profile.calibration_table) if r.config.key() == profile.config.key() and r.ok)
    assert profile.workload_spec["prompt_tokens_seen"] == winner.metrics.prompt_tokens_seen


def _usage_chunk(prompt, cached=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": 5}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()


def test_live_reuse_is_reported_and_never_warned_about():
    watch = TrafficWatch({"name": "s", "sessions": True, "prefill_tokens": 1000, "decode_tokens": 5,
                          "concurrencies": [1], "prompt_tokens_seen": {"p50": 200, "p90": 300},
                          "reuse_achieved": 0.6}, min_requests=1)
    for i in range(60):
        watch.began()
        watch.record_stream_chunk(_usage_chunk(200, 100 if i % 2 else None))  # SGLang omits a zero count
        watch.ended()
    report = watch.report()
    assert report["prefix_reuse"] == {"live": 0.25, "calibrated": 0.6, "replies_counted": 60}
    assert watch.findings() == [] and not watch.should_warn()  # prompt lengths match the served turns
    assert report["prompt_tokens"]["calibrated"] == 200


def test_a_flat_profiles_report_keeps_its_shape():
    watch = TrafficWatch({"name": "flat"})
    watch.record_stream_chunk(_usage_chunk(9))
    watch.record_response(json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                                "prompt_tokens_details": {"cached_tokens": 5}}}).encode(),
                          "application/json")
    assert "prefix_reuse" not in watch.report() and len(watch.reuse) == 0


def test_served_stats_come_from_the_level_the_objective_scored():
    """balanced rejects the busier level for breaking the TTFT ceiling, so the profile records the level it chose."""
    from rigtune.calibrate.objectives import Constraints
    from rigtune.models import Config, TrialResult
    from rigtune.pipeline import scored_level

    def level(c, tok_s, ttft, p50):
        return TrialMetrics(tok_s=tok_s, ttft_ms=ttft, ttft_p95_ms=ttft, tpot_ms=10, requests=8, output_tokens=800,
                            concurrency=c, prompt_tokens_seen={"p50": p50, "p90": p50, "max": p50, "mean": p50})

    busy, calm = level(8, 900.0, 2000.0, 999), level(4, 600.0, 200.0, 111)
    summary = busy.model_copy(update={"by_concurrency": {"8": busy, "4": calm}})
    winner = TrialResult(config=Config(backend="vllm", quant="bf16"), stage="batch", metrics=summary)
    got = scored_level(winner, [winner], "balanced", Constraints(ttft_ceiling_ms=500))
    assert got.concurrency == 4 and got.prompt_tokens_seen["p50"] == 111
    assert summary.prompt_tokens_seen["p50"] == 999  # what the highest-throughput summary would have recorded
