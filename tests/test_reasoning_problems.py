"""rigtune.reasoning.problems: MATH problems, their split, and how a response is graded."""

from __future__ import annotations

import json

import pytest

from rigtune.reasoning import problems as rp
from rigtune.reasoning.problems import Problem, grade, last_boxed, normalise, problems_from_rows, split_of

# Answer forms as the MATH solutions write them.
SOLUTIONS = [
    ("So the answer is $\\boxed{\\frac{\\sqrt{3}}{2}}$.", "\\frac{\\sqrt{3}}{2}"),  # nested braces
    ("Thus $x \\in \\boxed{(-\\infty, 2]}$.", "(-\\infty, 2]"),
    ("The angle is $\\boxed{30^\\circ}$.", "30^\\circ"),
    ("Hence \\boxed{\\text{(C)}}.", "\\text{(C)}"),
    ("First \\boxed{1}, then corrected to \\boxed{2}.", "2"),  # the last one counts
    ("The answer is $\\boxed 5$.", "5"),  # the bare form
    ("We get \\fbox{12}.", "12"),
]


@pytest.mark.parametrize("text, expected", SOLUTIONS)
def test_the_last_boxed_answer_is_read_with_its_braces(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("text", ["no box here", "\\boxed{unclosed", "", "\\boxed"])
def test_no_answer_is_none(text):
    assert last_boxed(text) is None


@pytest.mark.parametrize("text", ["So \\boxed{12}. (I used \\boxed as asked.)", "So \\boxed{12}. use \\boxed",
                                  "So \\boxed{12}, then \\boxed{}"])
def test_a_later_mention_does_not_hide_the_answer(text):
    assert last_boxed(text) == "12"


def test_escaped_braces_in_a_set_answer_are_kept():
    assert last_boxed("$\\boxed{\\{1,2\\}}$") == "\\{1,2\\}"


@pytest.mark.parametrize("a, b", [
    ("\\dfrac{1}{2}", "\\frac12"), ("0.5", "\\frac{1}{2}"), ("1/2", "\\frac{1}{2}"), ("\\sqrt3", "\\sqrt{3}"),
    ("30^\\circ", "30"), ("x = 5", "5"), ("10\\%", "10"), ("\\left( 1, 2 \\right)", "(1,2)"),
    ("5 \\text{ cm}", "5"), (".5", "0.5"),
])
def test_the_authors_normal_form_makes_equal_writings_equal(a, b):
    assert normalise(a) == normalise(b)


def test_the_normal_form_keeps_different_answers_apart():
    assert normalise("\\frac{1}{3}") != normalise("\\frac{1}{2}")
    assert normalise("10%") != normalise("10")  # as published: only \% is removed


@pytest.fixture
def no_math_verify(monkeypatch):
    rp._parsed.cache_clear()  # readings from another stub
    monkeypatch.setattr(rp, "_math_verify", lambda: None)


def test_a_response_is_graded_by_its_last_boxed_answer(no_math_verify):
    assert grade("... so $\\boxed{\\dfrac{1}{2}}$", "\\frac12") == (True, "\\dfrac{1}{2}", "normalised")
    assert grade("I think \\boxed{3}", "4") == (False, "3", "normalised")
    assert grade("The answer is 4.", "4") == (False, None, "no-answer")  # no box: wrong, said why
    assert grade("\\boxed{}", "4") == (False, None, "no-answer")


def _fake_math_verify(equal_pairs):
    calls = []

    def parse(text):
        calls.append(text)
        return [text]

    def verify(gold, answer):  # stands in for math-verify: equal as maths when listed, or the same text
        return gold == answer or (gold[0], answer[0]) in equal_pairs

    return calls, (parse, verify)


def test_math_verify_decides_whenever_it_can_the_reference_first(monkeypatch):
    calls, mv = _fake_math_verify({("$\\frac{1}{2}$", "$\\frac{2}{4}$")})
    monkeypatch.setattr(rp, "_math_verify", lambda: mv)
    assert grade("\\boxed{\\frac{2}{4}}", "\\frac{1}{2}") == (True, "\\frac{2}{4}", "math-verify")
    assert calls == ["$\\frac{1}{2}$", "$\\frac{2}{4}$"]  # gold first, as math-verify expects
    assert grade("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}")[2] == "math-verify"  # it decides equal text too


@pytest.mark.parametrize("answer, reference", [
    ("19", "19 \\text{ and }43"),  # a real MATH reference; the normalisation cuts after "\text{ "
    ("12 \\text{ or } 13", "12"),  # a hedge
    ("y = 4", "x = 4"),  # the normalisation drops a short left side
])
def test_what_the_authors_normal_form_wrongly_accepts_math_verify_refuses(monkeypatch, answer, reference):
    monkeypatch.setattr(rp, "_math_verify", lambda: None)
    assert grade("\\boxed{" + answer + "}", reference)[0] is True  # the authors' is_equiv, as published
    monkeypatch.setattr(rp, "_math_verify", lambda: _fake_math_verify(set())[1])
    assert grade("\\boxed{" + answer + "}", reference) == (False, answer, "math-verify")


def test_a_math_verify_failure_is_labelled_not_counted_as_its_verdict(monkeypatch):
    def broken(text):
        raise ValueError("doesn't support threaded environment")

    monkeypatch.setattr(rp, "_math_verify", lambda: (broken, lambda a, b: True))
    assert grade("\\boxed{4}", "4") == (True, "4", "math-verify-error")  # the authors' verdict, labelled
    monkeypatch.setattr(rp, "_math_verify", lambda: (lambda text: [], lambda a, b: True))
    assert grade("\\boxed{5}", "4") == (False, "5", "math-verify-error")  # read as nothing: no verdict
    monkeypatch.setattr(rp, "_math_verify", lambda: _fake_math_verify(set())[1])
    long_answer = "1+" * rp.MAX_ANSWER_CHARS + "1"
    assert grade("\\boxed{" + long_answer + "}", "4")[2] == "math-verify-error"  # never handed over


def test_math_verify_is_used_only_when_it_passes_its_self_test(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("math_verify")
    fake.parse = lambda text: [text]
    fake.verify = lambda gold, answer: False  # what a crashed timeout process looked like on Windows
    monkeypatch.setitem(sys.modules, "math_verify", fake)
    rp._math_verify.cache_clear()
    try:
        assert rp._math_verify() is None
        fake.verify = lambda gold, answer: gold == answer or {gold[0], answer[0]} == {"$\\frac{1}{2}$",
                                                                                      "$\\frac{2}{4}$"}
        rp._math_verify.cache_clear()
        assert rp._math_verify() == (fake.parse, fake.verify)
    finally:
        rp._math_verify.cache_clear()


ROWS = [
    {"problem": "Compute $1+1$.", "level": "Level 1", "type": "Algebra", "solution": "It is $\\boxed{2}$."},
    {"problem": "Find $x$ if $x^2 = 4$, $x>0$.", "level": "Level 4", "type": "Algebra",
     "solution": "So $x = \\boxed{2}$."},
    {"problem": "Simplify $\\frac{6}{8}$.", "level": "Level 5", "type": "Algebra",
     "solution": "$\\boxed{\\frac{3}{4}}$"},
    {"problem": "Broken row.", "level": "Level ?", "type": "Algebra", "solution": "$\\boxed{1}$"},
    {"problem": "No reference.", "level": "Level 5", "type": "Algebra", "solution": "It is 7."},
]


def test_rows_become_problems_at_the_chosen_levels():
    got = problems_from_rows(ROWS, "algebra")
    assert [(p.level, p.answer) for p in got] == [(4, "2"), (5, "\\frac{3}{4}")]
    assert got[0].id == rp.problem_id("Find $x$ if $x^2 = 4$, $x>0$.") and len(got[0].id) == 16
    assert [p.level for p in problems_from_rows(ROWS, "algebra", levels=(1,))] == [1]


def test_the_split_depends_on_the_problem_alone():
    many = [Problem(id=rp.problem_id(f"problem {i}"), subject="algebra", level=5, problem=f"problem {i}", answer="1")
            for i in range(2000)]
    dev = [p for p in many if split_of(p) == "dev"]
    assert 40 <= len(dev) <= 120  # about 4%
    # Pinned: a change of hash, encoding, truncation or threshold would move problems between the sets.
    assert (many[36].id, split_of(many[36])) == ("01bb2cabd87272d5", "dev")
    assert (many[0].id, split_of(many[0])) == ("20094f264602f0c1", "heldout")


def test_a_cached_set_keeps_every_level_of_its_commit_and_is_fingerprinted(tmp_path):
    every = problems_from_rows(ROWS, "algebra", levels=rp.ALL_LEVELS)
    cache = tmp_path / f"math-test-{rp.MATH_REVISION[:12]}.jsonl"
    cache.write_text("".join(json.dumps(p.to_json()) + "\n" for p in every), encoding="utf-8")
    loaded, digest = rp.load_math(cache_dir=tmp_path)
    assert [p.level for p in loaded] == sorted(p.level for p in every if p.level in (4, 5)) and len(digest) == 64
    assert [p.level for p in rp.load_math(levels=(1,), cache_dir=tmp_path)[0]] == [1]  # any levels, from one cache
    assert rp.load_math(cache_dir=tmp_path)[1] == digest  # the same set, the same fingerprint
    assert rp.load_math(levels=(5,), cache_dir=tmp_path)[1] != digest
    assert {p.id for p in rp.load_math(split="dev", cache_dir=tmp_path)[0]} | {
        p.id for p in rp.load_math(split="heldout", cache_dir=tmp_path)[0]} == {p.id for p in loaded}


def test_the_dataset_commit_is_pinned_and_in_the_fingerprint(tmp_path, monkeypatch):
    every = problems_from_rows(ROWS, "algebra", levels=rp.ALL_LEVELS)
    for rev in (rp.MATH_REVISION, "f" * 40):
        (tmp_path / f"math-test-{rev[:12]}.jsonl").write_text(
            "".join(json.dumps(p.to_json()) + "\n" for p in every), encoding="utf-8")
    assert rp.load_math(cache_dir=tmp_path)[1] != rp.load_math(revision="f" * 40, cache_dir=tmp_path)[1]
    assert len(rp.MATH_REVISION) == 40


@pytest.fixture(autouse=True)
def _fresh_parse_cache():
    rp._parsed.cache_clear()
    yield
    rp._parsed.cache_clear()
