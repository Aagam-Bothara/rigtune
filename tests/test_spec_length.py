"""rigtune.spec_length: the draft length a chain drafter's acceptance curve and two trials predict best."""

from __future__ import annotations

import pytest

from rigtune.calibrate.search import StagedSearch
from rigtune.models import ArchInfo, Config, ModelSpec, PreparedModel, SpecAcceptance, TrialMetrics, TrialResult
from rigtune.spec_length import (BEFORE, best_length, draft_cost, expected_tokens, probe_tokens, refine,
                                 with_tokens)

# The A100 session's Qwen3.5-9B (18 Nov 2026): its MTP head at 3 tokens accepted 0.844, 0.659, 0.509 a position and ran
# 1,069 tok/s at 8 users, against 565 for stock vLLM.
A100_CURVE = [0.844, 0.659, 0.509]
WEAK = [0.50, 0.25, 0.12, 0.06, 0.03, 0.01]
LEADER = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)


def test_tokens_a_step_yields_at_each_length():
    assert expected_tokens(A100_CURVE, 1) == pytest.approx(1.844)
    assert expected_tokens(A100_CURVE, 3) == pytest.approx(3.012)  # the trial's own mean acceptance length, 3.01


def test_the_a100_head_is_kept_at_three_tokens():
    alpha = draft_cost(A100_CURVE, 3, plain=565, drafting=1069)
    assert alpha == pytest.approx((3.012 * 565 / 1069 - 1) / 3)  # 0.197 of a step a drafted token
    k, gain = best_length(A100_CURVE, alpha, 3)
    assert k == 3 and gain == pytest.approx(1069 / 565, rel=1e-6)  # the model reproduces the trial it was fitted on


def test_a_weak_drafter_drafts_fewer():
    """A Gemma-4-like drafter: half its first tokens kept, few after; at 0.15 of a step a token, two are best."""
    k, gain = best_length(WEAK, 0.15, 6)
    assert k == 2 and gain == pytest.approx(1.75 / 1.30)
    assert best_length(WEAK, 0.0, 6)[0] == 6  # drafting free: every position pays, however little
    assert best_length(WEAK, 0.0, 6, allowed=lambda n: n <= 3)[0] == 3


def test_a_near_tie_goes_to_the_shorter_draft():
    assert best_length([0.9, 0.004], 0.0, 2)[0] == 1  # 1.904 against 1.900: within half a percent


def test_drafting_that_measured_free_is_read_as_free():
    assert draft_cost(A100_CURVE, 3, plain=565, drafting=3000) == 0.0  # noise above E(k) x plain
    assert draft_cost(A100_CURVE, 3, plain=0, drafting=100) is None


def test_the_length_is_rewritten_in_each_spec_kind():
    assert with_tokens("eagle3:org/head:6", 2) == "eagle3:org/head:2"
    assert with_tokens("mtp:3", 1) == "mtp:1"
    assert with_tokens("assist:google/gemma-4-12B-it-assistant:6", 3) == "assist:google/gemma-4-12B-it-assistant:3"


def test_the_probe_stays_inside_vllm_s_graphs_and_token_budget():
    assert probe_tokens("eagle3", 8, 2048) == 6
    assert probe_tokens("eagle3", None, 2048) == 6
    assert probe_tokens("eagle3", 100, 2048) == 4  # 512 graph tokens // 100 users - 1
    assert probe_tokens("eagle3", 8, 40) == 4  # a 40-token budget // 8 users - 1
    assert probe_tokens("eagle3", 128, 2048) == BEFORE["eagle3"] == 3  # never below the length drafted before
    assert probe_tokens("draft", 128, 2048) == BEFORE["draft"] == 4
    assert probe_tokens("assist", 300, 2048) == BEFORE["assist"] == 1


def _trial(cfg, tok_s, curve=None, full=True, levels=None, launched=True):
    if not launched:
        return TrialResult(config=cfg, stage="spec", metrics=TrialMetrics(), launched=False, error="out of memory")
    acc = None
    if curve is not None:
        acc = SpecAcceptance(rate=0.5, mean_length=expected_tokens(curve, len(curve)), source="vllm_metrics",
                             per_position=curve, full_drafts=full)
    by, users = {}, 1
    if levels:
        users = max(levels, key=levels.get)
        tok_s = levels[users]
        by = {str(c): TrialMetrics(tok_s=t, concurrency=c, requests=8, output_tokens=800) for c, t in levels.items()}
    m = TrialMetrics(tok_s=tok_s, spec_acceptance=acc, requests=8, output_tokens=800, ttft_ms=100.0,
                     ttft_p95_ms=150.0, tpot_ms=1000 / tok_s, concurrency=users, by_concurrency=by)
    return TrialResult(config=cfg, stage="spec", metrics=m)


def _models(mtp_layers=1):
    arch = ArchInfo(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8, head_dim=128,
                    mtp_layers=mtp_layers)
    return {"vllm": PreparedModel(spec=ModelSpec(hf_id="m/m"), backend="vllm", arch=arch, weights_bytes={"bf16": 1})}


def _done(*results):
    return {r.config.key(): r for r in results}


def _configs(todo):
    return [c.spec_decode for c, _ in todo if c is not None]


def _notes(todo):
    return [n for c, n in todo if c is None]


def _spec(spec, **more):
    return LEADER.model_copy(update={"spec_decode": spec, **more})


def test_refine_proposes_the_predicted_length_and_says_why():
    leader = _trial(LEADER, 175.0)
    trials = [_trial(_spec("eagle3:org/head:6"), 175 * 1.97 / 1.6, WEAK)]  # 1.97 a step at 6, 1.23x: 0.10 a token
    todo = refine(trials, _done(leader, *trials), _models())
    assert _configs(todo) == ["eagle3:org/head:2"]
    assert "at 2 tokens, predicted" in todo[0][1] and "0.50, 0.25" in todo[0][1]
    measured = _trial(todo[0][0], 200.0, WEAK[:2])
    assert refine(trials, _done(leader, *trials, measured), _models()) == []  # already measured


def test_refine_keeps_a_length_nothing_shorter_beats():
    leader = _trial(LEADER, 565.0)
    probe = _trial(_spec("mtp:3"), 1069.0, A100_CURVE)
    todo = refine([probe], _done(leader, probe), _models())
    assert _configs(todo) == [] and "kept at 3 tokens" in _notes(todo)[0]


def test_a_length_predicted_to_gain_little_is_not_launched():
    leader = _trial(LEADER, 100.0)
    probe = _trial(_spec("eagle3:org/head:6"), 40.0, WEAK)  # drafting lost: even 1 token predicted 0.91x
    todo = refine([probe], _done(leader, probe), _models())
    assert _configs(todo) == [] and "no length tried" in _notes(todo)[0]


@pytest.mark.parametrize("why, trial", [
    ("short drafts", _trial(_spec("eagle3:org/head:6"), 200.0, WEAK, full=False)),
    ("no curve", _trial(_spec("eagle3:org/head:6"), 200.0)),
    ("a block drafter", _trial(_spec("dflash:z-lab/x:15"), 200.0, WEAK)),
    ("n-gram lookup", _trial(_spec("ngram_gpu:4"), 200.0, WEAK[:4])),
    ("SGLang", _trial(_spec("eagle3:org/head:3", backend="sglang"), 200.0, WEAK[:3])),
    ("fewer positions than tokens", _trial(_spec("eagle3:org/head:6"), 200.0, WEAK[:4])),
])
def test_refine_leaves_what_the_model_does_not_describe(why, trial):
    assert _configs(refine([trial], _done(_trial(LEADER, 175.0), trial), _models())) == []


def test_a_curve_it_cannot_read_is_said_so():
    probe = _trial(_spec("eagle3:org/head:6"), 200.0, WEAK, full=False)
    assert "cut short" in _notes(refine([probe], _done(_trial(LEADER, 175.0), probe), _models()))[0]


def test_without_a_plain_trial_at_the_same_settings_nothing_is_weighed():
    probe = _trial(_spec("mtp:3", batch=16), 1069.0, A100_CURVE)  # own_head_fit lowered its batch
    todo = refine([probe], _done(_trial(LEADER, 565.0), probe), _models())
    assert _configs(todo) == [] and "no trial without speculation at the same settings" in _notes(todo)[0]


def test_a_memory_fraction_a_retry_lowered_does_not_matter():
    leader = _trial(LEADER, 175.0)
    probe = _trial(_spec("eagle3:org/head:6", gpu_memory_utilization=0.85), 175 * 1.97 / 1.6, WEAK)
    assert _configs(refine([probe], _done(leader, probe), _models())) == ["eagle3:org/head:2"]


def test_both_throughputs_come_from_one_number_of_users():
    """The plain trial peaked at 64 users and the drafting one at 32: the cost is read at 8, which both measured."""
    leader = _trial(LEADER, 0, levels={8: 100.0, 64: 400.0})
    probe = _trial(_spec("eagle3:org/head:6"), 0, WEAK, levels={8: 100 * 1.97 / 1.6, 32: 300.0})
    todo = refine([probe], _done(leader, probe), _models())
    assert _configs(todo) == ["eagle3:org/head:2"] and "at 8 users" in todo[0][1]


def test_trials_at_no_shared_number_of_users_are_not_weighed():
    leader = _trial(LEADER, 0, levels={64: 400.0})
    probe = _trial(_spec("eagle3:org/head:6"), 0, WEAK, levels={32: 300.0})
    todo = refine([probe], _done(leader, probe), _models())
    assert _configs(todo) == [] and "number of users" in _notes(todo)[0]


def test_a_probe_that_does_not_start_is_tried_at_its_former_length():
    leader = _trial(LEADER, 175.0)
    failed = _trial(_spec("draft:org/draft:6"), 0, launched=False)
    todo = refine([failed], _done(leader, failed), _models())
    assert _configs(todo) == ["draft:org/draft:4"] and "did not start" in todo[0][1]
    again = _trial(todo[0][0], 0, launched=False)
    assert refine([failed], _done(leader, failed, again), _models()) == []  # tried once


def test_a_probe_the_search_skipped_is_not_read_as_failing_to_start():
    from rigtune.calibrate.search import BUDGET_SKIPPED

    leader = _trial(LEADER, 175.0)
    skipped = TrialResult(config=_spec("draft:org/draft:6"), stage="spec", metrics=TrialMetrics(), launched=False,
                          error=BUDGET_SKIPPED, failure_kind="skipped_budget")
    assert refine([skipped], _done(leader, skipped), _models()) == []


def test_a_shared_level_that_failed_in_one_trial_is_said_so():
    leader = _trial(LEADER, 0, levels={8: 100.0, 64: 400.0})
    probe = _trial(_spec("eagle3:org/head:6"), 0, WEAK, levels={8: 0.0001, 32: 300.0})
    probe.metrics.by_concurrency["8"].tok_s = 0.0
    todo = refine([probe], _done(leader, probe), _models())
    assert _configs(todo) == [] and "no throughput at 8 users" in _notes(todo)[0]


def test_an_mtp_head_of_two_layers_is_offered_only_lengths_vllm_takes():
    """Three tokens would be best, but vLLM refuses 3 on a two-layer head (above 2, not a multiple): 4 is kept."""
    curve = [0.9, 0.8, 0.7, 0.05]
    leader = _trial(LEADER, 100.0)
    probe = _trial(_spec("mtp:4"), 100 * expected_tokens(curve, 4) / 1.4, curve)
    todo = refine([probe], _done(leader, probe), _models(mtp_layers=2))
    assert _configs(todo) == [] and "kept at 4 tokens" in _notes(todo)[0]
    assert _configs(refine([probe], _done(leader, probe), _models(mtp_layers=1))) == ["mtp:3"]


class _Runner:
    """Speeds a real model would give: a step's time grows by 0.10 of a plain one per drafted token."""

    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def run(self, cfg, stage):
        self.calls.append(cfg.spec_decode)
        if cfg.spec_decode in self.fail:
            return _trial(cfg, 0, launched=False)
        if not cfg.spec_decode:
            return _trial(cfg, 175.0)
        k = int(cfg.spec_decode.rpartition(":")[2])
        return _trial(cfg, 175.0 * expected_tokens(WEAK, k) / (1 + 0.10 * k), WEAK[:k])


def _search(runner):
    return StagedSearch(objective="throughput", runner=runner,
                        spec_refine=lambda trials, done: refine(trials, done, _models()))


PROBE = [_spec("eagle3:org/head:6")]


def test_the_search_measures_the_predicted_length_once_and_keeps_the_faster():
    runner = _Runner()
    search = _search(runner)
    search._run(LEADER, "batch")
    search.stage_variants("spec", lambda c: PROBE, backend="vllm")
    assert runner.calls == [None, "eagle3:org/head:6", "eagle3:org/head:2"]
    assert search._leader("vllm").config.spec_decode == "eagle3:org/head:2"
    said = [n for n in search.notes if "at 2 tokens, predicted" in n]
    assert len(said) == 1 and "measured" in said[0]
    search.stage_variants("spec", lambda c: PROBE, backend="vllm")  # a budgeted calibration's second pass
    assert runner.calls == [None, "eagle3:org/head:6", "eagle3:org/head:2"]
    assert len([n for n in search.notes if "at 2 tokens" in n]) == 1  # nothing said twice


def test_a_length_that_fails_is_noted_as_not_run():
    search = _search(_Runner(fail={"eagle3:org/head:2"}))
    search._run(LEADER, "batch")
    search.stage_variants("spec", lambda c: PROBE, backend="vllm")
    assert search._leader("vllm").config.spec_decode == "eagle3:org/head:6"
    assert any("at 2 tokens" in n and "did not run" in n for n in search.notes)


def test_without_the_hook_the_stage_is_as_before():
    runner = _Runner()
    search = StagedSearch(objective="throughput", runner=runner)
    search._run(LEADER, "batch")
    search.stage_variants("spec", lambda c: PROBE, backend="vllm")
    assert runner.calls == [None, "eagle3:org/head:6"]
