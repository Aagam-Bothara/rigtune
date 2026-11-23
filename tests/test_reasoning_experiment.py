"""rigtune.reasoning.experiment: the checks before a run, the shuffled order, and what an experiment writes."""

from __future__ import annotations

import asyncio
import json

import pytest

from rigtune.reasoning import experiment as ex
from rigtune.reasoning import methods as rm
from rigtune.reasoning import problems as rp
from rigtune.reasoning.methods import Settings
from rigtune.reasoning.problems import Problem

PROBLEMS = [Problem(id=f"{i:016x}", subject="algebra", level=5, problem=f"p{i}", answer=str(i)) for i in range(6)]


@pytest.fixture(autouse=True)
def _no_math_verify(monkeypatch):
    rp._parsed.cache_clear()
    monkeypatch.setattr(rp, "_math_verify", lambda: None)


async def correct(body):
    text = body["messages"][0]["content"].rsplit("p", 1)[1]
    return {"choices": [{"index": 0, "message": {"reasoning_content": "thought", "content": "\\boxed{" + text + "}"},
                         "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_method_specs_are_read_and_checked():
    assert ex.parse_specs("direct, self-consistency:3,self-correction,adaptive:8") == [
        ("direct", None), ("self-consistency", 3), ("self-correction", None), ("adaptive", 8)]
    for bad, msg in (("beam", "unknown method"), ("adaptive", "needs k"), ("direct:2", "takes no k"), ("", "no methods"),
                     ("self-consistency:1", "k = 1 is direct"), ("adaptive:2", "always takes 2")):
        with pytest.raises(ValueError, match=msg):
            ex.parse_specs(bad)


def test_the_context_check_counts_a_whole_critique_turn():
    long = [Problem(id="f" * 16, subject="algebra", level=5, problem="x" * 900, answer="1")]
    prompt_chars = len(rm.SOLVE.format(problem="x" * 900))
    assert ex.context_needed(long, max_tokens=4096) == int(prompt_chars / 1.5) + 2 * 4096 + 64 + 3 * 64
    need = ex.context_needed(PROBLEMS, max_tokens=4096)
    assert need > 2 * 4096
    s = Settings(model="m", max_tokens=4096)
    assert ex.check(PROBLEMS, s, "dev", 16384, [("self-correction", None)]) == []
    assert "shorter than a critique turn" in ex.check(PROBLEMS, s, "dev", 8192, [("self-correction", None)])[0]
    assert "did not say its context length" in ex.check(PROBLEMS, s, "dev", None, [("self-correction", None)])[0]
    assert ex.check(PROBLEMS, s, "dev", 8192, [("direct", None)]) == []  # only the critique turn needs two answers
    assert "shorter than an answer" in ex.check(PROBLEMS, s, "dev", 2048, [("direct", None)])[0]


def test_the_held_out_run_needs_math_verify():
    s = Settings(model="m")
    assert "needs math-verify" in ex.check(PROBLEMS, s, "heldout", 32768, [("direct", None)])[0]
    assert ex.check(PROBLEMS, s, "dev", 32768, [("direct", None)]) == []  # the dev run may go without it
    assert ex.check([], s, "dev", 32768, [("direct", None)]) == ["no problems"]


def test_methods_run_in_a_seeded_shuffled_order_and_each_is_written_as_it_ends(tmp_path):
    specs = ex.parse_specs("direct,self-consistency:3,self-correction,adaptive:4")
    logged = []
    runs = asyncio.run(ex.run_experiment(correct, Settings(model="m", keep_texts=True), PROBLEMS, specs, 8, 1,
                                         tmp_path, "set", "test server", log=logged.append))
    names = [r["name"] for r in runs]
    assert names == ["adaptive-4", "direct", "self-correction", "self-consistency-3"]  # seed 1's shuffle
    again = asyncio.run(ex.run_experiment(correct, Settings(model="m"), PROBLEMS, specs, 8, 1, tmp_path / "b", "set",
                                          "test server", log=lambda s: None))
    assert [r["name"] for r in again] == names  # the same seed, the same order
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["name"] for x in lines] == names and len(logged) == 4
    call = runs[0]["attempts"][0]["calls"][0]
    assert (call["text"], call["thinking"]) == ("\\boxed{0}", "thought")  # kept for training a verifier
    assert all(r["server"] == "test server" and r["problem_set"] == "set" for r in runs)
    off = again[0]["attempts"][0]["calls"][0]
    assert (off["text"], off["thinking"]) == (None, None)  # not kept unless asked


def test_a_whole_experiment_writes_its_manifest_runs_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "load_math", lambda **kw: (PROBLEMS, "d" * 64))

    async def context(url):
        return 32768

    async def close():
        pass

    monkeypatch.setattr(ex, "server_context", context)
    monkeypatch.setattr(rm, "http_post", lambda url: (correct, close))
    code = ex.main(["--base-url", "http://x", "--model", "m", "--methods", "direct,self-correction", "--out",
                    str(tmp_path), "--server", "vllm stock", "--sequences", "4"])
    assert code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["problem_set"] == "d" * 64 and manifest["context"] == 32768 and manifest["server"] == "vllm stock"
    assert manifest["dataset_revision"] == rp.MATH_REVISION and manifest["settings"]["thinking"] is False
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# m, dev set (6 problems), thinking off") and "| direct | 6 | 100.0% |" in report


def test_an_experiment_that_would_be_spoiled_does_not_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rp, "load_math", lambda **kw: (PROBLEMS, "d" * 64))

    async def context(url):
        return 4096

    monkeypatch.setattr(ex, "server_context", context)
    monkeypatch.setattr(rm, "http_post", lambda url: pytest.fail("nothing may be sent"))
    assert ex.main(["--base-url", "http://x", "--model", "m", "--out", str(tmp_path)]) == 2
    assert "not run: the server's context (4096)" in capsys.readouterr().err
    assert not (tmp_path / "runs.jsonl").exists()


def test_an_output_folder_with_runs_is_refused(tmp_path):
    (tmp_path / "runs.jsonl").write_text("{}\n", encoding="utf-8")
    assert "runs from another experiment would mix" in ex.check(PROBLEMS, Settings(model="m"), "dev", 32768,
                                                                 [("direct", None)], tmp_path)[0]


def test_a_batch_too_small_for_its_limit_is_warned_of():
    assert "use --sequences 1 or fewer" in ex.fill_warning(PROBLEMS, 32)
    many = PROBLEMS * 32  # 192 problems, 6 a place
    assert ex.fill_warning(many, 32) is None


def test_the_fill_warning_is_set_where_utilisation_reaches_the_floor():
    # The held-out run: 500 problems on 128 places kept direct answers 73% busy, and 125 places would have done no
    # better; 100 places (5 a place) still lands under 80%.
    held_out = [Problem(id=f"{i:016x}", subject="algebra", level=5, problem=f"p{i}", answer=str(i))
                for i in range(500)]
    assert "use --sequences 83 or fewer" in ex.fill_warning(held_out, 128)
    assert ex.fill_warning(held_out, 100) is not None
    assert ex.fill_warning(held_out, 83) is None


def test_preemptions_and_a_warm_up_batch_are_measured_around_each_run(tmp_path):
    counts = iter([0.0, 3.0, 3.0, 3.0])
    sent = []

    async def counting(body):
        sent.append(body)
        return await correct(body)

    async def preempted():
        return next(counts)

    runs = asyncio.run(ex.run_experiment(counting, Settings(model="m"), PROBLEMS, ex.parse_specs("direct,adaptive:3"),
                                         8, 0, tmp_path, "set", "srv", log=lambda s: None, preempted=preempted))
    assert [r["preemptions"] for r in runs] == [3.0, 0.0]
    assert len(sent) >= ex.WARMUP_PROBLEMS + 6  # the untimed warm-up went first


def test_top_k_is_sent_only_when_set():
    from rigtune.reasoning.methods import _body

    assert "top_k" not in _body(Settings(model="m"), [], 0, 0)
    assert _body(Settings(model="m", top_k=20), [], 0, 0)["top_k"] == 20


def test_keeping_texts_does_not_make_runs_incomparable():
    from rigtune.reasoning import report as rep

    def run(keep):
        return {"name": "x", "attempts": [{"problem_id": "a"}], "problem_set": "s", "sequences": 8, "server": "v",
                "settings": {"model": "m", "keep_texts": keep}}

    assert rep._comparable([run(True), run(False)]) is None
