"""rigtune.reasoning.report: accuracy against GPU time, paired against direct, transitions, bounds and criterion."""

from __future__ import annotations

import pytest

from rigtune.reasoning import problems as rp
from rigtune.reasoning import report as rr

IDS = [f"{i:016x}" for i in range(100)]
REFS = {i: "1" for i in IDS}
SETTINGS = {"model": "m", "max_tokens": 512}


@pytest.fixture(autouse=True)
def _no_math_verify(monkeypatch):
    rp._parsed.cache_clear()  # readings cached under another stub
    monkeypatch.setattr(rp, "_math_verify", lambda: None)


def _run(method, right, wall, k=None, first=None, answers=None, energy=None, errors=0, cut=0, util=0.95,
         settings=SETTINGS, problem_set="set", sequences=256, server="vllm 0.33 stock"):
    attempts = []
    for j, i in enumerate(IDS):
        attempts.append({"problem_id": i, "correct": j < right, "first_correct": (j < first) if first is not None
                         else None, "answers": answers(j) if answers else ["1" if j < right else "2"],
                         "error": "RuntimeError: 400" if j >= 100 - errors else None,
                         "calls": [{"completion_tokens": 100, "prompt_tokens": 50, "cached_tokens": 25,
                                    "finish_reason": "length" if j < cut else "stop"}]})
    name = method if k is None else f"{method}-{k}"
    return {"method": method, "k": k, "name": name, "wall_seconds": wall, "energy_joules": energy,
            "utilisation": util, "settings": settings, "problem_set": problem_set, "sequences": sequences,
            "server": server, "attempts": attempts}


DIRECT = _run("direct", right=50, wall=100.0, energy=20000.0)


def test_a_row_is_accuracy_with_its_interval_cost_and_what_went_wrong():
    r = rr.row(_run("direct", right=50, wall=100.0, energy=20000.0, errors=3, cut=7))
    assert (r.n, r.accuracy, r.gpu_seconds, r.completion_tokens, r.joules) == (100, 0.5, 1.0, 100.0, 200.0)
    assert (r.truncated, r.errors, r.cached_share) == (7, 3, 0.5)
    assert r.interval[0] < 0.5 < r.interval[1]


def test_pairs_count_gains_and_losses_on_the_same_problems():
    better = {i: 1.0 for i in IDS}
    p = rr.paired(rr._correct(DIRECT), better)
    assert (p["n"], p["gained"], p["lost"], p["difference"]) == (100, 50, 0, 0.5)
    assert p["interval"][0] > 0.35 and p["mcnemar_p"] < 1e-6
    assert rr.paired({}, better) == {"n": 0}


def test_the_bootstrap_interval_covers_a_known_difference():
    """Gains on 30 of 100 problems and losses on 10: +20%, a standard error of 6 points, so about 8% to 32%."""
    a = {i: 0.0 if j < 30 else (1.0 if j < 40 else 0.5) for j, i in enumerate(IDS)}
    b = {i: 1.0 if j < 30 else (0.0 if j < 40 else 0.5) for j, i in enumerate(IDS)}
    p = rr.paired(a, b)
    assert p["difference"] == pytest.approx(0.2) and 0.06 < p["interval"][0] < 0.11 and 0.29 < p["interval"][1] < 0.34


def test_self_correction_transitions():
    sc = _run("self-correction", right=55, wall=180.0, first=50)
    assert rr.transitions(sc) == {"wrong->right": 5, "right->wrong": 0, "wrong->wrong": 45, "right->right": 50}


def test_best_of_judges_every_answer_by_the_reference():
    cons = _run("self-consistency", right=50, wall=300.0, k=5,
                answers=lambda j: ["2", "2", "1" if j < 80 else "3", None, "2"])
    assert rr.best_of(cons, REFS) == 0.8


def test_the_criterion_compares_at_equal_gpu_time_along_the_self_consistency_curve():
    """Direct (1 GPU-s, 50%) and self-consistency-3 (3 GPU-s, 70%) bracket self-correction at 2 GPU-s: equal time
    is half of each, 60%, problem by problem."""
    cons3 = _run("self-consistency", right=70, wall=300.0, k=3)
    good = _run("self-correction", right=75, wall=200.0, first=50)
    c = rr.criterion([DIRECT, cons3, good])
    assert c["decided"] and c["met"] and c["rival"] == "direct and self-consistency-3"
    assert c["rival_weight"] == pytest.approx(0.5) and c["vs_rival"]["difference"] == pytest.approx(0.15)
    assert c["dearer"] == "self-consistency-3" and c["vs_dearer"]["difference"] == pytest.approx(0.05)
    weak = _run("self-correction", right=62, wall=200.0, first=50)
    assert rr.criterion([DIRECT, cons3, weak])["met"] is False  # beats its first answer, not equal-time sampling
    cheaper_rival_only = _run("self-consistency", right=52, wall=150.0, k=2)
    assert rr.criterion([DIRECT, cheaper_rival_only, good])["why"] == (
        "no self-consistency runs bracket self-correction's GPU time")  # never compared with a cheaper run alone


@pytest.mark.parametrize("change, why", [
    ({"errors": 2}, "failed requests in self-correction"),
    ({"problem_set": "other"}, "the runs came from different problem sets"),
    ({"settings": {"model": "m", "max_tokens": 1024}}, "the runs used different model settings"),
    ({"sequences": 128}, "the runs ran at different sequence limits"),
    ({"server": "vllm 0.33 rigtune-tuned"}, "the runs were served differently"),
    ({"util": 0.5}, "utilisation below 80% in self-correction"),
])
def test_the_criterion_is_undecided_on_unfair_comparisons(change, why):
    cons3 = _run("self-consistency", right=70, wall=300.0, k=3)
    sc = _run("self-correction", right=75, wall=200.0, first=50, **change)
    assert rr.criterion([DIRECT, cons3, sc]) == {"decided": False, "why": why}
    assert rr.criterion([DIRECT])["why"] == "no self-correction run"


def test_the_markdown_report_states_each_part():
    cons = _run("self-consistency", right=60, wall=300.0, k=3)
    sc = _run("self-correction", right=58, wall=180.0, first=50, cut=4, util=0.4)
    ada = _run("adaptive", right=59, wall=150.0, k=4)
    text = rr.markdown([DIRECT, cons, sc, ada], REFS, title="Qwen3-4B, dev set")
    assert text.startswith("# Qwen3-4B, dev set")
    assert "| direct | 100 | 50.0% |" in text and "| 1.00 | 100 | 0 | 0 | 95% | 50% | 200 |" in text
    assert "| 4 | 0 | 40% (low: wall time, not GPU time) |" in text
    assert "self-consistency-3: +10.0%" in text and "8 repaired, 0 harmed" in text
    assert "adaptive-4: best of its answers by the reference" in text
    assert "Criterion not decided: utilisation below 80% in self-correction." in text


def test_a_cache_the_server_does_not_report_is_unknown_not_zero():
    run = _run("direct", right=50, wall=100.0)
    for a in run["attempts"]:
        a["calls"][0]["cached_tokens"] = None  # vLLM without --enable-prompt-tokens-details
    assert rr.row(run).cached_share is None
    assert "| 95% | — |" in rr.markdown([run], REFS)
