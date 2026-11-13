"""The accuracy gate (rigtune/accuracy_gate.py) inside the staged search, with scripted grades and fake trials."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from rigtune.accuracy_gate import AccuracyGate, chat_answers, pass_wall_seconds
from rigtune.calibrate.search import StagedSearch
from rigtune.evalset import EvalItem
from rigtune.models import Config, EvalSetInfo, Profile, TrialMetrics, TrialResult

Key = Tuple[str, str]
SPEED = {"bf16": 100.0, "fp8": 150.0, "w8a8": 160.0, "awq": 250.0, "gptq": 240.0, "Q8_0": 70.0, "Q4_K_M": 90.0}


def _items(n: int) -> List[EvalItem]:
    return [EvalItem(messages=[{"role": "user", "content": f"question {i}"}], answer=str(i), grader="number",
                     id=f"q{i}") for i in range(n)]


def _flip(outcomes: str, positions: Sequence[int], to: str) -> str:
    chars = list(outcomes)
    for p in positions:
        chars[p] = to
    return "".join(chars)


class Script:
    """Scripted grades per (backend, quant) and pass: '1' answers right, '0' wrong, '-' not at all."""

    def __init__(self, grades: Dict[Key, str], second: Optional[Dict[Key, str]] = None, clock=None,
                 seconds_per_pass: float = 0.0):
        self.grades, self.second = grades, second or {}
        self.passes: List[Key] = []
        self.walls: List[float] = []
        self.clock, self.seconds_per_pass = clock, seconds_per_pass

    def __call__(self, base_url, model, items, concurrency, max_tokens, wall_s) -> List[Optional[str]]:
        key = tuple(base_url.split("//", 1)[1].split("/"))
        again = key in self.passes
        self.passes.append(key)
        self.walls.append(wall_s)
        if self.clock is not None:
            self.clock.now += self.seconds_per_pass
        grades = self.second.get(key, self.grades[key]) if again else self.grades[key]
        return [None if g == "-" else f"The answer is {item.answer if g == '1' else -1}"
                for item, g in zip(items, grades)]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class GradingRunner:
    """Every trial succeeds unless listed in `fail`; cheaper weights are faster. The gate grades on each
    successful trial, as SubprocessTrialRunner does."""

    def __init__(self, gate: Optional[AccuracyGate] = None, fail: Sequence[Key] = (), clock: Optional[Clock] = None,
                 trial_s: float = 60.0):
        self.gate, self.fail, self.clock, self.trial_s = gate, set(fail), clock, trial_s
        self.ran: List[TrialResult] = []

    def run(self, cfg: Config, stage: str) -> TrialResult:
        if self.clock is not None:
            self.clock.now += self.trial_s
        if (cfg.backend, cfg.quant) in self.fail:
            res = TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error="failed to start")
        else:
            tok = SPEED[cfg.quant] * (1 + cfg.batch / 1000) * (1.1 if cfg.kv_dtype != "auto" else 1.0)
            m = TrialMetrics(tok_s=tok, requests=32, output_tokens=2048, concurrency=8, ttft_ms=50.0)
            res = TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))
            if self.gate is not None:
                self.gate.record(f"fake://{cfg.backend}/{cfg.quant}", "model", cfg)
        self.ran.append(res)
        return res


def _feasible(*keys: Key) -> List[Config]:
    return [Config(backend=b, quant=q, ctx=4096, batch=batch, gpu_memory_utilization=0.9)
            for b, q in keys for batch in (16, 64)]


def _gate(n: int, script: Script, max_loss: float = 3.0, clock=None) -> AccuracyGate:
    info = EvalSetInfo(name="test", path="test.jsonl", items=n, total=n, sha256="0" * 64, max_accuracy_loss=max_loss)
    return AccuracyGate(items=_items(n), max_loss=max_loss, info=info, answer=script,
                        **({"clock": clock} if clock is not None else {}))


def _search(gate: AccuracyGate, runner: GradingRunner, **kw) -> StagedSearch:
    return StagedSearch(objective="throughput", runner=runner, gate=gate, **kw)


def test_a_faster_refused_4bit_cannot_win_and_is_not_tuned():
    """6 of 300 lost and none gained: the bound is -4.3 points, so at 3 points awq is refused even though it is
    2.5x faster. The variant stage tunes bf16, the precision that may win, not the refused leader."""
    ref = "1" * 270 + "0" * 30
    script = Script({("vllm", "bf16"): ref, ("vllm", "awq"): _flip(ref, range(6), "0")})
    gate = _gate(300, script)
    kv = ("kv", lambda c: [c.model_copy(update={"kv_dtype": "fp8"})])
    search = _search(gate, GradingRunner(gate), variant_stages=[kv])

    winner, notes = search.run(_feasible(("vllm", "awq"), ("vllm", "bf16")))

    assert winner is not None and winner.config.quant == "bf16"
    assert search.refused == {("vllm", "awq")} and gate.refused == {("vllm", "awq")}
    assert {r.config.quant for r in search.results if r.stage in ("memory", "batch", "kv")} == {"bf16"}
    refusal = next(n for n in notes if n.startswith("accuracy gate: refused vllm/awq"))
    assert "6 lost, 0 gained" in refusal and "95% CI -4.3 to -0.7" in refusal and "cross-engine" not in refusal
    assert "a loss of more than 3.0 points cannot be ruled out" in refusal
    assert any(n.startswith("accuracy gate (300 test items, tolerance 3.0 points): references vllm/bf16; "
                            "admitted none; refused vllm/awq; not graded (never ran) none") for n in notes)


def test_the_14b_counts_are_only_just_refused_at_5_points():
    """docs/usage.md: 4 lost of 200 bound the loss at -5.03, a hair past a 5-point tolerance."""
    from rigtune.calibrate.paired import certifies, score_bounds

    assert round(score_bounds(4, 0, 200)[0] * 100, 2) == -5.03
    assert not certifies(4, 0, 200, 0.05) and certifies(4, 0, 200, 0.051)


def test_no_changed_answer_on_300_items_admits_at_3_points():
    ref = "1" * 250 + "0" * 50
    script = Script({("vllm", "bf16"): ref, ("vllm", "awq"): ref})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    winner, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    assert winner.config.quant == "awq" and search.refused == set() and gate.admitted == {("vllm", "awq")}
    assert any("admitted vllm/awq" in n and "95% CI -1.3 to 1.3" in n and "within the 3.0-point tolerance" in n
               for n in notes)


def test_the_same_evidence_on_100_items_is_refused_and_says_how_many_would_do():
    """Nothing changed, but 100 items only bound the loss at 3.7 points; about 125 would certify 3."""
    ref = "1" * 80 + "0" * 20
    script = Script({("vllm", "bf16"): ref, ("vllm", "awq"): ref})
    gate = _gate(100, script)
    search = _search(gate, GradingRunner(gate))

    winner, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    assert winner.config.quant == "bf16" and search.refused == {("vllm", "awq")}
    refusal = next(n for n in notes if "refused vllm/awq" in n)
    assert "95% CI -3.7 to 3.7" in refusal and "about 125 items at this rate would certify 3.0 points" in refusal
    reference = next(n for n in notes if n.startswith("accuracy gate: reference vllm/bf16"))
    assert "changed 0 of 100" in reference
    assert "cannot certify 3.0 points even for identical weights; about 125 items would" in reference


def test_the_reference_answers_twice_and_reports_its_own_noise():
    ref = "1" * 270 + "0" * 30
    again = _flip(_flip(ref, [0], "0"), [299], "1")  # one answer turned wrong, another right
    script = Script({("vllm", "bf16"): ref, ("vllm", "fp8"): ref}, second={("vllm", "bf16"): again})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    search.run(_feasible(("vllm", "fp8"), ("vllm", "bf16")))

    assert script.passes.count(("vllm", "bf16")) == 2 and script.passes.count(("vllm", "fp8")) == 1
    note = next(n for n in search.notes if n.startswith("accuracy gate: reference vllm/bf16"))
    assert "90.0% (270/300" in note and "answered 300 of 300" in note
    assert "the second pass answered 300 of 300" in note
    assert "changed 2 of 300 answers' grades (1 right then wrong, 1 wrong then right)" in note
    assert "even for identical weights" not in note  # 1 against 1 on 300 still certifies 3 points
    rec = gate.records()[0]
    assert rec.reference and (rec.self_lost, rec.self_gained) == (1, 1) and rec.outcomes_repeat == again


def test_the_wall_cap_grows_with_the_set():
    """60 s plus 2 s per item: 660 s for the default 300, 2698 s for all of GSM8K."""
    assert pass_wall_seconds(300) == 660.0 and pass_wall_seconds(1319) == 2698.0
    script = Script({("vllm", "bf16"): "1" * 300})
    gate = _gate(300, script)
    _search(gate, GradingRunner(gate)).run(_feasible(("vllm", "bf16")))
    assert script.walls == [660.0, 660.0]


def test_each_engine_reference_runs_first_whatever_order_the_planner_gives():
    grades = {k: "1" * 300 for k in [("vllm", "bf16"), ("vllm", "awq"), ("vllm", "fp8"), ("sglang", "fp8"),
                                     ("sglang", "awq")]}
    script = Script(grades)
    gate = _gate(300, script)
    runner = GradingRunner(gate)

    _search(gate, runner).run(_feasible(("vllm", "awq"), ("sglang", "awq"), ("vllm", "fp8"), ("sglang", "fp8"),
                                        ("vllm", "bf16")))

    assert [(r.config.backend, r.config.quant) for r in runner.ran[:2]] == [("sglang", "fp8"), ("vllm", "bf16")]
    assert gate.planned == {"vllm": ("vllm", "bf16"), "sglang": ("sglang", "fp8")}
    assert script.passes.count(("sglang", "fp8")) == 2 and script.passes.count(("vllm", "bf16")) == 2


def test_the_reference_survives_predictor_pruning():
    """bf16 is predicted at a tenth of awq's speed, under the 40% the predictor prunes at; without it there is
    nothing to grade awq against, so the gate keeps it."""
    class Predictor:
        def is_fitted(self, backend):
            return True

        def best_level(self, model, cfg, *args, **kwargs):
            return SimpleNamespace(tok_s=10.0 if cfg.quant == "bf16" else 100.0)

    feasible = _feasible(("vllm", "bf16"), ("vllm", "awq"))
    plain = StagedSearch(objective="throughput", runner=GradingRunner(), predictor=Predictor(),
                         models={"vllm": object()})
    plain.stage_quant(feasible)
    assert any(n.startswith("skipped vllm/bf16: predicted") for n in plain.notes)  # what the predictor does

    script = Script({("vllm", "bf16"): "1" * 300, ("vllm", "awq"): "1" * 300})
    gate = _gate(300, script)
    runner = GradingRunner(gate)
    gated = _search(gate, runner, predictor=Predictor(), models={"vllm": object()})
    gated.stage_quant(feasible)
    assert runner.ran[0].config.quant == "bf16" and gate.references == {"vllm": ("vllm", "bf16")}
    assert not any(n.startswith("skipped vllm/bf16") for n in gated.notes)


def test_a_reference_that_fails_to_launch_hands_over_to_the_next_faithful_format():
    ref = "1" * 270 + "0" * 30
    script = Script({("vllm", "fp8"): ref, ("vllm", "awq"): _flip(ref, range(6), "0")})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate, fail=[("vllm", "bf16")]))

    winner, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "fp8"), ("vllm", "awq")))

    assert gate.references == {"vllm": ("vllm", "fp8")} and winner.config.quant == "fp8"
    assert gate.unrun == {("vllm", "bf16")} and gate.refused == {("vllm", "awq")}
    assert any("vllm's planned reference vllm/bf16 did not run, so vllm/fp8 is its reference instead" in n
               and "no measure of its own noise" in n for n in notes)
    assert any("refused vllm/awq" in n and "against vllm/fp8" in n for n in notes)
    assert any(n == "accuracy gate: not graded (never ran): vllm/bf16" for n in notes)


def test_a_precision_that_ran_but_answered_nothing_is_refused_and_the_next_faithful_one_is_the_reference():
    script = Script({("vllm", "bf16"): "-" * 300, ("vllm", "fp8"): "1" * 300, ("vllm", "awq"): "1" * 300})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    winner, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "fp8"), ("vllm", "awq")))

    assert gate.refused == {("vllm", "bf16")} and gate.references == {"vllm": ("vllm", "fp8")}
    assert winner.config.quant == "awq"  # certified against fp8: nothing changed on 300 items
    assert any("refused vllm/bf16: it ran but answered none of the 300 items" in n for n in notes)
    assert any("vllm's planned reference vllm/bf16 answered none of the items, so vllm/fp8 is its reference" in n
               for n in notes)


def test_with_no_faithful_format_graded_anywhere_a_checkpoint_is_refused():
    """Fail closed: nothing to compare against means nothing cheaper is certified."""
    script = Script({("vllm", "bf16"): "-" * 300, ("vllm", "awq"): "1" * 300})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    winner, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    assert winner is None and gate.refused == {("vllm", "bf16"), ("vllm", "awq")}
    assert any("refused vllm/awq: no faithful format (bf16, fp16, fp8, Q8_0) was graded on any engine" in n
               for n in notes)


def test_the_14b_case_gates_sglang_gptq_against_vllm_fp8_across_engines():
    """On the 14B, SGLang's fp8 failed to start and its GPTQ was the fastest stage-1 baseline (701 tok/s). A
    per-engine rule would have made GPTQ SGLang's reference and admitted it ungated."""
    ref = "1" * 285 + "0" * 15
    lose = _flip(ref, range(6), "0")
    script = Script({("vllm", "fp8"): ref, ("vllm", "gptq"): lose, ("sglang", "gptq"): lose})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate, fail=[("sglang", "fp8")]))

    winner, notes = search.run(_feasible(("vllm", "fp8"), ("vllm", "gptq"), ("sglang", "fp8"), ("sglang", "gptq")))

    assert winner.config.key().startswith("vllm/fp8")
    assert gate.references == {"vllm": ("vllm", "fp8")} and gate.unrun == {("sglang", "fp8")}
    assert gate.refused == {("vllm", "gptq"), ("sglang", "gptq")}
    sg = next(r for r in gate.records() if r.backend == "sglang")
    assert sg.against == "vllm/fp8" and sg.cross_engine and not sg.admitted
    note = next(n for n in notes if n.startswith("accuracy gate: refused sglang/gptq"))
    assert "cross-engine" in note and "against vllm/fp8" in note
    assert "cross-engine" not in next(n for n in notes if n.startswith("accuracy gate: refused vllm/gptq"))


def test_a_cross_engine_checkpoint_that_changes_nothing_is_admitted_and_says_so():
    ref = "1" * 300
    script = Script({("vllm", "fp8"): ref, ("sglang", "gptq"): ref})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate, fail=[("sglang", "fp8")]))

    winner, notes = search.run(_feasible(("vllm", "fp8"), ("sglang", "fp8"), ("sglang", "gptq")))

    assert winner.config.backend == "sglang" and gate.admitted == {("sglang", "gptq")}
    assert any(n.startswith("accuracy gate: admitted sglang/gptq (cross-engine") for n in notes)


def test_llamacpp_q8_0_does_not_gate_vllm_fp8():
    """Q8_0 ranks above fp8, but it is another engine's format and template: each is its engine's reference,
    and the gap between them is noted, never refused."""
    ref = "1" * 280 + "0" * 20
    script = Script({("llamacpp-cuda", "Q8_0"): ref, ("vllm", "fp8"): _flip(ref, range(15), "0"),
                     ("vllm", "awq"): _flip(ref, range(15), "0")})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    winner, notes = search.run(_feasible(("vllm", "fp8"), ("vllm", "awq"), ("llamacpp-cuda", "Q8_0")))

    assert gate.references == {"vllm": ("vllm", "fp8"), "llamacpp-cuda": ("llamacpp-cuda", "Q8_0")}
    assert gate.refused == set() and winner.config.quant == "awq"  # same answers as vllm's own fp8
    fp8 = next(r for r in gate.records() if r.quant == "fp8")
    assert fp8.reference and fp8.admitted is None and fp8.lost == 15 and "not gated" in fp8.reason
    assert any("vllm/fp8 and llamacpp-cuda/Q8_0 are each their engine's reference, so the gap is noted, not gated"
               in n for n in notes)
    awq = next(r for r in gate.records() if r.quant == "awq")
    assert awq.against == "vllm/fp8" and not awq.cross_engine and awq.admitted


def test_the_3b_case_gates_fp8_against_bf16_on_the_same_engine():
    """On Qwen2.5-3B fp8 cost 2.4 GSM8K points against bf16: faithful formats are still gated within an engine."""
    ref = "1" * 260 + "0" * 40
    script = Script({("vllm", "bf16"): ref, ("vllm", "fp8"): _flip(ref, range(8), "0"),
                     ("vllm", "awq"): _flip(ref, range(15), "0")})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    winner, _ = search.run(_feasible(("vllm", "bf16"), ("vllm", "fp8"), ("vllm", "awq")))

    assert winner.config.quant == "bf16" and gate.refused == {("vllm", "fp8"), ("vllm", "awq")}
    fp8 = next(r for r in gate.records() if r.quant == "fp8")
    assert fp8.against == "vllm/bf16" and fp8.admitted is False


def test_a_precision_the_gate_may_refuse_cannot_push_another_engine_out(monkeypatch):
    """vLLM's awq runs at 4x bf16 and would make llama.cpp (at 70% of bf16) look hopeless; the drop rule counts
    only trials that cannot be refused, so llama.cpp's Q4_K_M is still tried."""
    ref = "1" * 280 + "0" * 20
    grades = {("vllm", "bf16"): ref, ("vllm", "awq"): _flip(ref, range(10), "0"),
              ("llamacpp-cuda", "Q8_0"): ref, ("llamacpp-cuda", "Q4_K_M"): ref}
    feasible = _feasible(("vllm", "bf16"), ("vllm", "awq"), ("llamacpp-cuda", "Q8_0"), ("llamacpp-cuda", "Q4_K_M"))
    monkeypatch.setitem(SPEED, "awq", 400.0)

    plain = StagedSearch(objective="throughput", runner=GradingRunner())
    plain.stage_quant(feasible)
    gate = _gate(300, Script(grades))
    runner = GradingRunner(gate)
    _search(gate, runner).stage_quant(feasible)
    assert any(n.startswith("skipped llamacpp-cuda/Q4_K_M") for n in plain.notes)  # the rule without a gate
    assert ("llamacpp-cuda", "Q4_K_M") in {(r.config.backend, r.config.quant) for r in runner.ran}
    assert gate.refused == {("vllm", "awq")}


def test_a_precision_that_answered_nothing_is_refused():
    script = Script({("vllm", "bf16"): "1" * 300, ("vllm", "awq"): "-" * 300})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    winner, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    assert winner.config.quant == "bf16" and ("vllm", "awq") in gate.refused
    assert any("refused vllm/awq: it ran but answered none of the 300 items" in n for n in notes)


def test_unanswered_items_drop_out_of_the_comparison():
    ref = "1" * 300
    script = Script({("vllm", "bf16"): ref, ("vllm", "awq"): _flip(ref, range(100), "-")})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    search.run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    awq = next(r for r in gate.records() if r.quant == "awq")
    assert awq.answered == 200 and awq.paired == 200 and (awq.lost, awq.gained) == (0, 0)
    assert awq.admitted and round(awq.delta_ci95[0], 1) == -1.9  # the bound for 200 items, not 300


def test_the_same_precision_on_another_engine_is_compared_but_never_refused():
    ref = "1" * 270 + "0" * 30
    script = Script({("vllm", "bf16"): ref, ("sglang", "bf16"): _flip(ref, range(20), "0")})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    search.run(_feasible(("vllm", "bf16"), ("sglang", "bf16")))

    assert gate.refused == set()
    sg = next(r for r in gate.records() if r.backend == "sglang")
    assert sg.reference and sg.admitted is None and sg.lost == 20 and "not gated" in sg.reason
    assert any("sglang/bf16 and vllm/bf16 are each their engine's reference" in n for n in search.notes)


def test_grading_time_is_not_counted_as_trial_time_for_the_budget():
    """A 300-item pass can take minutes; counted as trial time it would make the budget think every later trial
    takes as long and skip them. (It still spends the budget's wall clock: see docs/usage.md.)"""
    clock = Clock()
    script = Script({("vllm", "bf16"): "1" * 300, ("vllm", "awq"): "1" * 300}, clock=clock, seconds_per_pass=500.0)
    gate = _gate(300, script, clock=clock)
    search = _search(gate, GradingRunner(gate, clock=clock), budget_s=10_000, clock=clock)

    search.run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    assert gate.seconds_total == 1500.0 and gate.seconds_repeat == 500.0  # the reference twice, awq once
    assert search._trial_seconds and all(s == 60.0 for s in search._trial_seconds)
    assert "grading took 1500s, of which 500s was the second pass" in gate.summary()


def test_each_precision_is_graded_once_and_nothing_after_the_decision():
    script = Script({("vllm", "bf16"): "1" * 300, ("vllm", "awq"): "1" * 300})
    gate = _gate(300, script)
    runner = GradingRunner(gate)

    _search(gate, runner).run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    assert any(r.stage != "quant" for r in runner.ran)  # later stages ran, and graded nothing
    assert script.passes == [("vllm", "bf16"), ("vllm", "bf16"), ("vllm", "awq")]


def test_records_carry_per_item_outcomes_for_a_later_decision():
    ref = "1" * 290 + "0" * 10
    cand = _flip(_flip(ref, [3, 4], "0"), [295], "1")
    script = Script({("vllm", "bf16"): ref, ("vllm", "gptq"): _flip(cand, [7], "-")})
    gate = _gate(300, script)
    _search(gate, GradingRunner(gate)).run(_feasible(("vllm", "bf16"), ("vllm", "gptq")))

    by = {r.quant: r for r in gate.records()}
    assert [r.quant for r in gate.records()] == ["bf16", "gptq"]  # reference first
    assert by["bf16"].outcomes == ref and by["gptq"].outcomes == _flip(cand, [7], "-")
    assert (by["gptq"].answered, by["gptq"].correct) == (299, 288)
    assert (by["gptq"].paired, by["gptq"].lost, by["gptq"].gained) == (299, 2, 1)
    assert by["gptq"].mcnemar_p == 1.0 and by["gptq"].admitted is True and by["gptq"].against == "vllm/bf16"


def test_an_old_profile_without_accuracy_still_loads(hw_a100):
    old = {"rigtune_version": "0.2.1", "hardware_hash": "x", "hardware": hw_a100.model_dump(mode="json"),
           "model_id": "m", "objective": "throughput", "backend": "vllm", "backend_version": "1",
           "config": {"backend": "vllm", "quant": "bf16"}, "launch_args": []}
    p = Profile.model_validate(old)
    assert p.accuracy == [] and p.eval_set is None


@pytest.mark.usefixtures("no_network")
def test_a_gated_calibration_puts_the_grades_in_the_profile(hw_a100, spec, tmp_path):
    from rigtune.pipeline import SearchOptions, calibrate, prepare_and_plan, select
    from tests.test_objectives_and_search import FakeRunner

    f = tmp_path / "mine.jsonl"
    f.write_bytes("".join(json.dumps({"prompt": f"q{i}", "answer": str(i), "grader": "number"}) + "\n"
                          for i in range(400)).encode())
    script = Script({("vllm", "bf16"): "1" * 300, ("vllm", "fp8"): _flip("1" * 300, range(10), "0")})

    class Runner(FakeRunner):
        gate = None  # calibrate hands an injected runner its gate

        def run(self, cfg, stage):
            res = super().run(cfg, stage)
            if res.ok and self.gate is not None:
                self.gate.answer = script
                self.gate.record(f"fake://{cfg.backend}/{cfg.quant}", "model", cfg)
            return res

    candidates, reg = select(hw_a100, spec, force="vllm")
    plan = prepare_and_plan(hw_a100, spec, candidates, reg)
    opts = SearchOptions(eval_set=str(f), max_accuracy_loss=3.0, eval_items=300)
    p = calibrate(hw_a100, spec, "throughput", plan, reg, runner=Runner(), options=opts)

    assert p.config.quant == "bf16"  # fp8 is faster here, and lost 10 of 300
    assert p.eval_set.name == "mine" and p.eval_set.items == 300 and p.eval_set.total == 400
    assert p.options["eval"] == "mine-300" and p.options["maxloss"] == "3"
    assert [(r.quant, r.reference, r.admitted) for r in p.accuracy] == [("bf16", True, None), ("fp8", False, False)]
    assert len(p.accuracy[1].outcomes) == 300
    assert Profile.model_validate_json(p.model_dump_json()).accuracy == p.accuracy


def test_the_gate_is_built_before_any_trial_and_checks_items_against_the_context(tmp_path, caplog):
    from rigtune.evalset import EvalSetError
    from rigtune.pipeline import SearchOptions, accuracy_gate_for

    f = tmp_path / "long.jsonl"
    f.write_bytes((json.dumps({"prompt": "x" * 9000, "answer": "1", "grader": "number", "id": "long"}) + "\n"
                   + json.dumps({"prompt": "short", "answer": "2", "grader": "number"}) + "\n").encode())
    opts = SearchOptions(eval_set=str(f))
    small, large = Config(backend="vllm", quant="bf16", ctx=2048), Config(backend="vllm", quant="bf16", ctx=8192)
    with pytest.raises(EvalSetError, match="item long needs about 3512 tokens"):
        accuracy_gate_for(opts, [small])
    with caplog.at_level("WARNING"):
        gate = accuracy_gate_for(opts, [small, large])
    assert gate is not None and len(gate.items) == 2 and gate.info.sha256
    assert "more than the smallest feasible context (2048)" in caplog.text
    assert "2 items cannot certify 3.0 points even if no answer changes" in caplog.text
    assert accuracy_gate_for(SearchOptions(), [small]) is None


# --------------------------------------------------------------------------- the HTTP client


RELEASE = threading.Event()


class _Chat(BaseHTTPRequestHandler):
    """Answers "question N" with "The answer is N"; question 2 gets an HTTP 500, and question 3 is held until the
    test releases it, after the client has given up on it."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        assert self.path == "/v1/chat/completions" and body["temperature"] == 0.0 and body["stream"] is False
        n = int(body["messages"][-1]["content"].split()[-1])
        if n == 2:
            self.send_response(500)
            self.end_headers()
            return
        if n == 3:
            RELEASE.wait(30)
        data = json.dumps({"choices": [{"message": {"role": "assistant", "content": f"The answer is {n}"}}]}).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass  # the client gave up on it


def test_chat_answers_talks_to_the_chat_endpoint_and_caps_the_wall_time():
    RELEASE.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Chat)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = chat_answers(f"http://127.0.0.1:{srv.server_address[1]}", "m", _items(6), concurrency=8,
                           max_tokens=16, wall_s=2.0)
    finally:
        RELEASE.set()  # the held request can only finish after the client has returned
        srv.shutdown()
    assert out[0] == "The answer is 0" and out[5] == "The answer is 5"
    assert out[2] is None and out[3] is None  # an error and a request past the wall cap are unanswered


# --------------------------------------------------------------------------- what a refusal cost


def test_a_refusal_reports_its_speed_and_what_would_have_admitted_it():
    """awq measured 2.5x the pick in stage 1 and lost 6 of 300: its bound is -4.29 points, so 4.3 would have
    admitted it, and certifying 3 points at that rate would take about 2045 items."""
    from rigtune.calibrate.paired import certifies

    ref = "1" * 270 + "0" * 30
    script = Script({("vllm", "bf16"): ref, ("vllm", "awq"): _flip(ref, range(6), "0")})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    winner, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "awq")))

    price = next(n for n in notes if n.startswith("accuracy gate: refusing vllm/awq"))
    assert "cost at least 2.50x: its stage-1 trial measured 266 tok/s against the pick's 106" in price
    assert "it was never tuned, so that ratio is a floor" in price
    assert "--max-accuracy-loss 4.3 would have admitted it on this evidence" in price
    assert "about 2045 items at its rate of loss would certify 3.0 points" in price
    assert gate.admitting_tolerance(("vllm", "awq")) == 4.3
    assert certifies(6, 0, 300, 0.043) and not certifies(6, 0, 300, 0.042)  # 4.3 is the smallest tenth that admits


def test_a_slower_refused_precision_cost_no_speed_and_an_ungraded_one_says_so(monkeypatch):
    monkeypatch.setitem(SPEED, "fp8", 50.0)  # slower than bf16 here, and every answer wrong
    script = Script({("vllm", "bf16"): "1" * 300, ("vllm", "fp8"): "0" * 300, ("vllm", "awq"): "-" * 300})
    gate = _gate(300, script)
    search = _search(gate, GradingRunner(gate))

    _, notes = search.run(_feasible(("vllm", "bf16"), ("vllm", "fp8"), ("vllm", "awq")))

    fp8 = next(n for n in notes if n.startswith("accuracy gate: refusing vllm/fp8"))
    assert "cost no measured speed" in fp8 and "no number of items would certify 3.0" in fp8
    awq = next(n for n in notes if n.startswith("accuracy gate: refusing vllm/awq"))
    assert "it could not be graded, so no tolerance would have admitted it" in awq
