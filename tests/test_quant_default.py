"""`--quant auto` leaves out 4-bit checkpoints; they are opt-in."""

from __future__ import annotations

import pytest

import rigtune.quantized as Q
from rigtune.backends import registry
from rigtune.models import Config
from rigtune.pipeline import SearchOptions, _usable, allowed_quants, prepare_and_plan


def test_auto_skips_4bit_checkpoints_unless_asked():
    supported = ["bf16", "fp8", "awq", "gptq"]
    assert allowed_quants(supported, None) == ["bf16", "fp8"]
    assert allowed_quants(supported, ["auto", "awq"]) == ["bf16", "fp8", "awq"]
    assert allowed_quants(supported, ["gptq"]) == ["gptq"]
    assert allowed_quants(supported, ["Q4_K_M"]) == []
    assert allowed_quants(["Q4_K_M", "Q8_0"], None) == ["Q4_K_M", "Q8_0"]  # llama.cpp keeps its formats
    assert allowed_quants([], None) is None  # a backend that lists nothing decides for itself


@pytest.mark.usefixtures("no_network")
def test_planner_prepares_4bit_only_on_request(hw_a100, spec, monkeypatch):
    monkeypatch.setattr(Q, "find_prequantized_repos", lambda s, methods=Q.INT4_METHODS, **kw: {
        m: Q.PrequantizedRepo(repo_id=f"org/Llama-3.2-3B-Instruct-{m.upper()}", method=m, size_bytes=2_000_000_000)
        for m in methods})
    reg = registry()
    default = prepare_and_plan(hw_a100, spec, ["vllm"], reg)
    assert set(default.prepared["vllm"].weights_bytes) == {"bf16", "fp8"}
    opted = prepare_and_plan(hw_a100, spec, ["vllm"], reg, quants=["auto", "awq"])
    assert set(opted.prepared["vllm"].weights_bytes) == {"bf16", "fp8", "awq"}
    only = prepare_and_plan(hw_a100, spec, ["vllm"], reg, quants=["gptq"])
    assert set(only.prepared["vllm"].weights_bytes) == {"gptq"}


def test_options_and_cached_profiles_follow_the_default(hw_a100, spec, prepared_vllm):
    from tests.test_cache_and_pipeline import _profile

    default, opted = SearchOptions(), SearchOptions(quants=["auto", "gptq"])
    assert default.allows("fp8") and default.allows("Q4_K_M") and not default.allows("gptq")
    assert opted.allows("gptq") and opted.allows("bf16") and not opted.allows("awq")
    assert default.key() == {} and opted.key() == {"quant": "auto+gptq"}

    base = _profile(hw_a100, spec, prepared_vllm)
    gptq = base.model_copy(update={"config": Config(backend="vllm", quant="gptq")})
    assert not _usable(gptq, default, None)  # calibrated when auto still picked 4-bit: recalibrate
    assert _usable(gptq, opted, None)
    assert _usable(base, default, None) and not _usable(base, default, "llamacpp-cuda")
    assert not _usable(None, default, None)


def test_cli_accepts_auto_inside_a_list():
    from rigtune.cli import _opts, _quant_list

    assert _quant_list("auto,awq") == "auto,awq"
    assert _opts("auto,awq", "on", "on", "on").quants == ["auto", "awq"]
    assert _opts("auto", "on", "on", "on").quants is None


def test_an_eval_set_adds_cache_keys_only_when_given(tmp_path):
    """Existing profiles keep their cache paths: the gate's options say nothing until a set is given."""
    assert SearchOptions(max_accuracy_loss=5.0, eval_items=100).key() == {}
    assert SearchOptions(eval_set="gsm8k").key() == {"eval": "gsm8k-300", "maxloss": "3"}
    mine = str(tmp_path / "mine.jsonl")
    assert SearchOptions(eval_set=mine, eval_items=0, max_accuracy_loss=2.5).key() == {"eval": "mine-all",
                                                                                          "maxloss": "2.5"}


def test_a_gated_profile_is_reused_only_on_the_same_eval_set(tmp_path, hw_a100, spec, prepared_vllm):
    from rigtune.evalset import eval_set_sha256
    from rigtune.models import EvalSetInfo
    from tests.test_cache_and_pipeline import _profile

    f = tmp_path / "mine.jsonl"
    f.write_bytes(b'{"prompt": "2 + 3?", "answer": "5", "grader": "number"}\n')
    opts = SearchOptions(eval_set=str(f))
    base = _profile(hw_a100, spec, prepared_vllm)
    gated = base.model_copy(update={"eval_set": EvalSetInfo(name="mine", path=str(f), items=1, total=1,
                                                            sha256=eval_set_sha256(str(f)), max_accuracy_loss=3.0)})
    assert _usable(gated, opts, None)
    assert not _usable(base, opts, None)  # never graded
    assert _usable(base, SearchOptions(), None)  # ungated options never look at the set
    f.write_bytes(b'{"prompt": "2 + 2?", "answer": "4", "grader": "number"}\n')
    assert not _usable(gated, opts, None)  # same name, different items: grade again
