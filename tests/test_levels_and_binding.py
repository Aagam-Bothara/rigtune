"""Settings that cannot act at the measured levels are named in the verdict, and `compare --levels` measures
at levels where they can."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from rigtune.bench import compare
from rigtune.bench.compare import ComparisonResult, ComparisonRow, with_levels
from rigtune.bench.verdict import binding_notes, score
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.models import Config, TrialMetrics
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner


def test_a_workload_is_measured_at_other_levels_under_its_own_name():
    wl, note = with_levels(get_workload("default"), [128, 64, 64])
    assert (wl.concurrencies, wl.name, wl.n_prompts) == ((64, 128), "default-c64-128", 256)
    assert note == "measured at 64, 128 users instead of the workload's own 1, 4, 8, with 256 prompts per " \
                   "level instead of 16"
    hc, note = with_levels(get_workload("high-concurrency"), [256])
    assert hc.n_prompts == 512 and hc.ttft_ceiling_ms == 1000.0  # its limits stay its own
    small = Workload(name="file-x", prompts=["q"] * 20, n_prompts=20, source="file")
    _, note = with_levels(small, [32])
    assert note.endswith("the prompt file holds 20 prompts, fewer than twice the busiest level, so that "
                         "level may not stay full")
    with pytest.raises(ValueError):
        with_levels(get_workload("default"), [0])


@pytest.mark.usefixtures("no_network")
def test_compare_at_other_levels_keeps_the_held_out_note_about_prompts_only(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())
    res = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(), levels=[16, 32])
    assert res.workload == f"{profile.workload}-c16-32" and res.workload_spec["concurrencies"] == [16, 32]
    assert res.calibrated_on is None  # the same prompts, only more users: not held out
    assert any(n.startswith("measured at 16, 32 users instead of the workload's own") for n in res.notes)


@pytest.mark.usefixtures("no_network")
def test_torch_compile_measured_above_its_batch_at_other_levels_says_so(hw_a100, spec):
    """SGLang compiles batches of up to 32 (the expert row's acts_up_to_users): at --levels 64,128 the row is
    measured outside the small batches its guide recommends it for, and the notes say so; at 16, 32 not."""
    from rigtune.bench.rows import ExtraRow

    candidates, reg = select(hw_a100, spec, force="vllm")
    planned = prepare_and_plan(hw_a100, spec, candidates, reg)
    profile = calibrate(hw_a100, spec, "throughput", planned, reg, runner=FakeRunner())

    def notes(levels):
        row = ExtraRow(config=profile.config, kind="expert", source="a guide", acts_up_to_users=32,
                       acts_note="it compiles batches of up to 32")
        res = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(), levels=levels,
                      extra={"sglang-expert-torch-compile": row})
        return [n for n in res.notes if n.startswith("sglang-expert-torch-compile is measured at up to")]

    assert notes([64, 128]) == ["sglang-expert-torch-compile is measured at up to 128 users, above 32: it "
                                "compiles batches of up to 32"]
    assert notes([16, 32]) == []


def _result(rows, levels=(1, 4, 8), prompt=63, answer=395):
    for r in rows:
        if r.label == "rigtune":
            r.metrics = TrialMetrics(prompt_tokens=prompt)
    return ComparisonResult(rigtune_version="x", hardware_hash="h", gpu="A40", cpu="c", model_id="m",
                            objective="balanced", workload="chat", ttft_ceiling_ms=500, rows=rows,
                            workload_spec={"concurrencies": list(levels), "decode_tokens": answer})


def _cfg_row(label, kind=None, **cfg):
    return ComparisonRow(label=label, runtime="vllm", config=Config(backend="vllm", quant="bf16", **cfg),
                         config_key=label, kind=kind)


def test_the_verdict_names_settings_that_cannot_act_at_the_levels_measured():
    """The 14 November case: auto_tune's pick (256 sequences, a 4096-token budget, 4096 context) measured at 1
    to 8 users with prompts of about 63 tokens, against stock vLLM with 256 sequences."""
    rows = [_cfg_row("rigtune", ctx=16384, batch=64), _cfg_row("vllm-default", ctx=32768, batch=256),
            _cfg_row("autotune-vllm", "tuner", ctx=4096, batch=256, prefill_budget=4096,
                     gpu_memory_utilization=0.98, prefix_cache=True)]
    assert binding_notes(_result(rows)) == [
        "autotune-vllm: at up to 8 users, a batched-token budget of 4096 (a step here holds at most 8 x 63 "
        "+ 8 = 512 tokens); a context of 4096 (prompts here average 63 tokens, answers at most 395) cannot "
        "act, so on these settings it measures like its engine's stock row; measure at more users "
        "(compare --levels) to see them act"]
    # The batch cap is stock's own 256: not a difference, so not named.
    busy = [_cfg_row("rigtune", ctx=16384, batch=64), _cfg_row("vllm-default", ctx=32768, batch=256),
            _cfg_row("vllm-expert-x", "expert", ctx=32768, batch=256, prefill_budget=16384)]
    assert binding_notes(_result(busy, levels=(64,), prompt=300)) == []  # 64 x 300 + 64 = 19264 > 16384
    # No stock row to compare with: each setting is judged on its own. A cap of 4 and a 256-token context
    # bind at 8 users with these prompts; a cap of 16 does not.
    binds = [_cfg_row("rigtune", ctx=16384, batch=64), _cfg_row("other-tool", "tuner", ctx=256, batch=4)]
    assert binding_notes(_result(binds, levels=(8,))) == []
    idle = [_cfg_row("rigtune", ctx=16384, batch=64), _cfg_row("other-tool", "tuner", ctx=256, batch=16)]
    assert binding_notes(_result(idle, levels=(8,))) == [
        "other-tool: at up to 8 users, a batch cap of 16 (it binds above 16 users) cannot act, so on these "
        "settings it measures like its engine's stock row; measure at more users (compare --levels) to see "
        "them act"]
    assert binding_notes(_result([_cfg_row("rigtune")], levels=())) == []


def test_binding_notes_reach_the_verdict():
    from tests.test_verdict import _row

    rig, stock, tuner = (_row("rigtune", [130.0, 131.0, 132.0]), _row("vllm-default", [100.0, 101.0, 102.0]),
                         _row("autotune-vllm", [100.0, 101.0, 102.0], kind="tuner"))
    rig.config = Config(backend="vllm", quant="bf16", ctx=16384, batch=64)
    stock.config = Config(backend="vllm", quant="bf16", ctx=32768, batch=256)
    tuner.config = Config(backend="vllm", quant="bf16", ctx=4096, batch=256, prefill_budget=4096)
    res = _result([rig, stock, tuner])
    assert any(n.startswith("autotune-vllm: at up to 8 users, a batched-token budget of 4096") for n in
               score(res).notes)


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_levels_on_the_command_line(tmp_path, monkeypatch, hw_a100):
    from rigtune.cli import app
    from tests.test_extra_rows import MODEL, _cli

    calls = _cli(monkeypatch, hw_a100)
    r = CliRunner().invoke(app, ["compare", MODEL, "--levels", "8,x"])
    assert r.exit_code == 2 and calls == []
    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--levels", "16,32", "--verdict",
                                 "--out", str(out)])
    assert r.exit_code == 0, r.output
    names = sorted(p.name for p in out.iterdir())
    assert all("__default-c16-32__balanced" in n for n in names) and len(names) == 4
