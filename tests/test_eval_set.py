"""Labelled eval sets (rigtune/evalset.py) and the bundled GSM8K set."""

from __future__ import annotations

import importlib.util
import json
from importlib import resources
from pathlib import Path

import pytest

from rigtune.evalset import EvalSetError, eval_set_sha256, load_eval_set, parse_items

_spec = importlib.util.spec_from_file_location(
    "make_eval_set", Path(__file__).resolve().parents[1] / "benchmarks" / "make_eval_set.py")
mes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mes)


def _jsonl(*rows) -> str:
    return "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n"


def test_items_take_a_prompt_or_a_chat():
    items = parse_items(_jsonl(
        {"prompt": "2 + 3?", "answer": 5, "grader": "number", "id": 7},
        "",
        {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital of France?"}],
         "answer": "Paris", "grader": "contains", "max_tokens": 16},
        {"prompt": "Pick one", "answer": "B", "grader": "choice", "choices": ["A", "B"]},
    ))
    assert [i.messages[-1]["role"] for i in items] == ["user", "user", "user"]
    assert items[0].messages == [{"role": "user", "content": "2 + 3?"}] and items[0].answer == "5"
    assert items[0].id == "7" and items[1].max_tokens == 16 and items[2].choices == ["A", "B"]
    assert items[0].grade("The answer is 5") and not items[1].grade("Lyon") and items[2].grade("B")


@pytest.mark.parametrize("row,message", [
    ({"prompt": "q", "answer": "1"}, "line 2: no grader"),
    ({"prompt": "q", "answer": "1", "grader": "fuzzy"}, "line 2: unknown grader 'fuzzy'"),
    ({"answer": "1", "grader": "exact"}, "line 2: give exactly one of prompt or messages"),
    ({"prompt": "q", "messages": [], "answer": "1", "grader": "exact"}, "exactly one of prompt or messages"),
    ({"prompt": "q", "grader": "exact"}, "line 2: no answer"),
    ({"prompt": "q", "answer": "twelve", "grader": "number"}, "needs a numeric answer"),
    ({"prompt": "q", "answer": "(", "grader": "regex"}, "not a valid regular expression"),
    ({"prompt": "q", "answer": "E", "grader": "choice", "choices": ["A", "B"]}, "not one of the choices"),
    ({"prompt": "q", "answer": "A", "grader": "choice"}, "needs choices"),
    ({"prompt": "q", "answer": "1", "grader": "exact", "max_tokens": 0}, "max_tokens must be a positive"),
    ({"messages": [{"role": "robot", "content": "x"}], "answer": "1", "grader": "exact"}, "messages must be"),
])
def test_a_bad_item_is_refused_at_load_naming_its_line(row, message):
    good = {"prompt": "fine", "answer": "1", "grader": "number"}
    with pytest.raises(EvalSetError, match=message):
        parse_items(_jsonl(good, row), "set.jsonl")


def test_broken_json_and_an_empty_file_are_refused():
    with pytest.raises(EvalSetError, match="set.jsonl line 1: not valid JSON"):
        parse_items("{nope\n", "set.jsonl")
    with pytest.raises(EvalSetError, match="no items"):
        parse_items("\n\n", "set.jsonl")


def test_a_file_set_keeps_its_first_n_items_and_hashes_the_whole_file(tmp_path):
    f = tmp_path / "mine.jsonl"
    f.write_bytes(_jsonl(*({"prompt": f"q{i}", "answer": str(i), "grader": "number"} for i in range(10))).encode())
    s = load_eval_set(str(f), limit=4)
    assert s.name == "mine" and s.total == 10 and [i.answer for i in s.items] == ["0", "1", "2", "3"]
    assert len(load_eval_set(str(f), limit=0).items) == 10  # 0: every item
    assert len(load_eval_set(str(f), limit=50).items) == 10
    assert s.sha256 == eval_set_sha256(str(f))
    f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))  # a checkout with CRLF line endings: same set
    assert eval_set_sha256(str(f)) == s.sha256
    f.write_text(f.read_text(encoding="utf-8").replace('"q9"', '"q9 changed"'), encoding="utf-8")
    assert eval_set_sha256(str(f)) != s.sha256  # any change to the file is a different set
    with pytest.raises(EvalSetError, match="not a bundled set"):
        load_eval_set(str(tmp_path / "missing.jsonl"))
    assert eval_set_sha256(str(tmp_path / "missing.jsonl")) is None


def test_the_bundled_gsm8k_set_loads_from_the_installed_package():
    """Read through importlib.resources, as an installed wheel would serve it, never through the repo path."""
    data = resources.files("rigtune").joinpath("data", "gsm8k.jsonl")
    assert data.is_file()
    assert "Copyright (c) 2021 OpenAI" in resources.files("rigtune").joinpath("data", "gsm8k-LICENSE.txt").read_text(
        encoding="utf-8")
    s = load_eval_set("gsm8k")
    assert s.name == "gsm8k" and s.total == 1319 and len(s.items) == 1319
    assert all(i.grader == "number" and float(i.answer) == int(i.answer) for i in s.items)
    assert s.items[0].answer == "18" and s.items[0].messages[0]["content"].startswith("Janet")
    assert s.items[0].messages[0]["content"].endswith("'The answer is N'.")
    assert [i.id for i in load_eval_set("gsm8k", limit=300).items][-1] == "gsm8k-test-299"
    assert eval_set_sha256("gsm8k") == s.sha256


def test_the_bundled_file_is_what_the_script_writes():
    """make_eval_set.py's items for a GSM8K-shaped row, so a rebuilt file matches the bundled one's format."""
    row = {"question": "Janet has 3 apples and buys 1,000 more. How many?",
           "answer": "3 + 1000 = 1003\n#### 1,003"}
    from rigtune.accuracy import INSTRUCTION

    [item] = mes.to_items([row])
    assert item == {"id": "gsm8k-test-0", "prompt": f"{row['question']}\n\n{INSTRUCTION}", "answer": "1003",
                    "grader": "number"}
    assert mes.answer_text(1e6) == "1000000" and mes.answer_text(-3.0) == "-3" and mes.answer_text(0.5) == "0.5"
    parse_items(json.dumps(item))  # and the loader accepts it


def test_the_wheel_is_told_to_include_the_bundled_files():
    """rigtune/data is not a package, so only package-data puts it in the wheel (0.2.0 shipped without code
    for want of a check like this)."""
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    section = text.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
    assert '"data/*.jsonl"' in section and '"data/*.txt"' in section
