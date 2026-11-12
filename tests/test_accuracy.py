"""Graders for labelled answers (rigtune/accuracy.py)."""

from __future__ import annotations

import pytest

from rigtune.accuracy import INSTRUCTION, chosen, final_number, grade, wilson


def test_number_grades_gsm8k_style_answers():
    assert grade("number", "3 + 4 = 7.\nThe answer is 7.", "7")
    assert grade("number", "The answer is: $1,072", "1,072")
    assert grade("number", "So 18 dollars.", "Janet makes 9 * 2 = 18.\n#### 18")  # a whole GSM8K solution
    assert not grade("number", "The answer is 8. Check: 8 - 5 = 3", "3")  # the stated answer wins
    assert not grade("number", "no numbers here", "3")
    assert grade("number", "The answer is -3.5", "-3.5")


def test_exact_ignores_only_whitespace():
    assert grade("exact", "  Paris \n", "Paris")
    assert grade("exact", "New   York", "New York")
    assert not grade("exact", "paris", "Paris")
    assert not grade("exact", "Paris.", "Paris")


def test_contains_ignores_case_and_whitespace_runs():
    assert grade("contains", "I think the capital is PARIS, France.", "paris")
    assert grade("contains", "the  Treaty of\nVersailles", "treaty of versailles")
    assert not grade("contains", "Lyon", "Paris")


def test_regex_searches_the_response():
    assert grade("regex", "The year was 1066, I believe.", r"\b1066\b")
    assert grade("regex", "YES", r"(?i)^\s*yes\b")
    assert not grade("regex", "10660", r"\b1066\b")


def test_choice_takes_the_stated_or_opening_label_and_nothing_else():
    labels = ["A", "B", "C", "D"]
    assert chosen("Let me think. B is wrong because... The answer is C.", labels) == "C"
    assert chosen("Answer: (D)", labels) == "D"
    assert chosen("(B) because the others are wrong", labels) == "B"
    assert chosen("B.", labels) == "B"
    assert chosen("A good question. Either way it is unclear.", labels) is None  # the article is not option A
    assert chosen("Because of this, the answer is a bit unclear", labels) is None
    assert grade("choice", "The answer is B", "B", labels)
    assert not grade("choice", "The answer is B", "C", labels)
    assert chosen("The answer is maybe", ["yes", "no", "maybe"]) == "maybe"


def test_an_unknown_grader_is_an_error():
    with pytest.raises(ValueError, match="unknown grader 'fuzzy'"):
        grade("fuzzy", "x", "x")


def test_the_moved_gsm8k_grader_is_the_one_the_script_uses():
    """benchmarks/task_quality.py imports these back, so the gate and the script grade identically."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "benchmarks" / "task_quality.py"
    spec = importlib.util.spec_from_file_location("task_quality_for_accuracy", path)
    tq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tq)
    assert tq.final_number is final_number and tq.wilson is wilson and tq.INSTRUCTION == INSTRUCTION
