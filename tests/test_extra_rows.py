"""`rigtune compare --rows FILE`: other tools' picks, measured like every row, with their kind and source."""

from __future__ import annotations

import importlib
import json

import pytest
from typer.testing import CliRunner

from rigtune.bench import compare, to_markdown
from rigtune.bench.compare import ComparisonResult, save
from rigtune.bench.rows import ExtraRow, RowsFileError, read_rows_file
from rigtune.calibrate.workload import get_workload
from rigtune.models import Config
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_compare_repeats import DriftingRunner
from tests.test_objectives_and_search import FakeRunner

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# FakeRunner fails any configuration at 0.95 of memory or more, so 0.92 starts.
AUTOTUNE = {"backend": "vllm", "quant": "bf16", "ctx": 4096, "batch": 256, "gpu_memory_utilization": 0.92,
            "prefill_budget": 4096, "prefix_cache": True}
KEY = "vllm/bf16/ctx4096/b256/gmu0.92/pb4096/pc"


def _write(tmp_path, data, name="rows.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_a_rows_file_gives_named_configs_with_kind_and_source(tmp_path):
    path = _write(tmp_path, {"rows": [
        {"label": "autotune-vllm", "source": "auto_tune.sh, p95 TTFT <= 500 ms", "config": AUTOTUNE},
        {"label": "hand-set", "kind": "expert", "config": {"backend": "sglang", "quant": "bf16"}},
        {"label": "llm-optimizer", "skipped": "it did not start"}]})
    rows = read_rows_file(path)
    assert list(rows) == ["autotune-vllm", "hand-set", "llm-optimizer"]  # file order
    at = rows["autotune-vllm"]
    assert at.kind == "tuner" and at.source.startswith("auto_tune.sh") and at.skipped is None
    assert at.config.key() == KEY
    assert rows["hand-set"].kind == "expert"
    assert rows["llm-optimizer"].config is None and rows["llm-optimizer"].skipped == "it did not start"


@pytest.mark.parametrize("data, message", [
    ({"autotune-vllm": AUTOTUNE}, 'expected {"rows": [...]}'),
    ({"rows": [{"label": "a", "config": {**AUTOTUNE, "max_num_seqs": 64}}]},
     "unknown Config field(s) max_num_seqs"),
    ({"rows": [{"label": "a", "config": AUTOTUNE, "note": "x"}]}, "unknown key(s) note"),
    ({"rows": [{"label": "rigtune", "config": AUTOTUNE}]}, "reserved"),
    ({"rows": [{"label": "vllm-default", "config": AUTOTUNE}]}, "reserved"),
    ({"rows": [{"label": "a b", "config": AUTOTUNE}]}, "must be letters"),
    ({"rows": [{"label": "a", "config": AUTOTUNE}, {"label": "a", "config": AUTOTUNE}]}, "already used"),
    ({"rows": [{"label": "a", "kind": "stock", "config": AUTOTUNE}]}, "kind must be one of tuner, expert"),
    ({"rows": [{"label": "a"}]}, "config must be an object"),
    ({"rows": [{"label": "a", "skipped": ""}]}, "skipped must be the reason"),
])
def test_a_bad_rows_file_is_refused_with_the_reason(tmp_path, data, message):
    with pytest.raises(RowsFileError) as exc:
        read_rows_file(_write(tmp_path, data))
    assert message in str(exc.value)


def test_a_label_already_taken_elsewhere_is_refused(tmp_path):
    with pytest.raises(RowsFileError, match="already used"):
        read_rows_file(_write(tmp_path, {"rows": [{"label": "autotune-vllm", "config": AUTOTUNE}]}),
                       taken={"autotune-vllm": None})
    for label in ("vllm-expert", "vllm-expert-ngram", "sglang-expert-x"):  # the shipped expert rows' pattern
        with pytest.raises(RowsFileError, match="reserved"):
            read_rows_file(_write(tmp_path, {"rows": [{"label": label, "config": AUTOTUNE}]}))
    assert list(read_rows_file(_write(tmp_path, {"rows": [{"label": "expertise", "config": AUTOTUNE}]}))) == [
        "expertise"]


def _setup(hw, spec, objective="throughput"):
    candidates, reg = select(hw, spec, force="vllm")
    planned = prepare_and_plan(hw, spec, candidates, reg)
    return planned, reg, calibrate(hw, spec, objective, planned, reg, runner=FakeRunner())


@pytest.mark.usefixtures("no_network")
def test_added_rows_are_measured_interleaved_and_carry_their_kind_and_source(tmp_path, hw_a100, spec):
    planned, reg, profile = _setup(hw_a100, spec)
    extra = {"autotune-vllm": ExtraRow(config=Config.model_validate(AUTOTUNE), source="auto_tune.sh"),
             "nothing": ExtraRow(config=None, source="llm-optimizer", skipped="it did not start"),
             "sglang-row": ExtraRow(config=Config(backend="sglang", quant="bf16"), kind="expert",
                                    source="a guide")}
    runner = DriftingRunner()
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                     runner=runner, repeats=2, extra=extra)
    assert [s.split(":", 1)[1] for s in runner.stages] == ["rigtune", "vllm-default", "vllm-fp8-default",
                                                            "autotune-vllm"] * 2
    at = next(r for r in result.rows if r.label == "autotune-vllm")
    assert at.kind == "tuner" and at.source == "auto_tune.sh" and at.repeats == 2 and at.ok
    # `rigtune report` counts only these as defaults.
    assert [r.label for r in result.default_rows()] == ["vllm-default", "vllm-fp8-default"]
    assert [(s.label, s.kind, s.reason) for s in result.skipped] == [
        ("nothing", "tuner", "it did not start"),
        ("sglang-row", "expert", "sglang is not one of this comparison's engines")]
    assert "autotune-vllm is a tuner row: auto_tune.sh" in result.notes
    assert "nothing (tuner) was not measured: it did not start" in result.notes
    data = json.loads(save(result, tmp_path).read_text(encoding="utf-8"))
    assert [r.get("kind") for r in data["rows"]] == [None, None, None, "tuner"]
    assert "kind" not in data["rows"][0] and "source" not in data["rows"][1]  # unset: not written at all
    back = ComparisonResult.model_validate(data)
    assert back.skipped == result.skipped and back.rows[3].source == "auto_tune.sh"
    assert f"autotune-vllm ({KEY})" in to_markdown(back)


@pytest.mark.usefixtures("no_network")
def test_a_plain_config_in_extra_is_measured_as_before(hw_a100, spec):
    """benchmarks/search_baselines.py passes Configs: they stay unlabelled rows, counted with the stock
    ones, and add no note."""
    planned, reg, profile = _setup(hw_a100, spec)
    result = compare(hw_a100, spec, profile, planned.prepared, reg, runner=CompareRunner(),
                     extra={"random-0": Config.model_validate(AUTOTUNE)})
    row = result.rows[-1]
    assert row.label == "random-0" and row.kind is None
    assert "random-0" in [r.label for r in result.default_rows()]
    assert not result.skipped and not any("random-0" in n for n in result.notes)


def _cli(monkeypatch, hw):
    import rigtune.hardware as hwmod
    import rigtune.pipeline as pl

    cmp = importlib.import_module("rigtune.bench.compare")
    monkeypatch.setattr(hwmod, "probe", lambda: hw)
    calls = []

    def cached_profile(spec, objective, force_backend=None, **kwargs):
        calls.append(spec.hf_id)
        candidates, reg = select(hw, spec, force=force_backend)
        planned = prepare_and_plan(hw, spec, candidates, reg)
        return calibrate(hw, spec, objective, planned, reg, runner=FakeRunner())

    monkeypatch.setattr(pl, "resolve_profile", cached_profile)
    monkeypatch.setattr(cmp, "SubprocessTrialRunner", lambda **kwargs: CompareRunner())
    return calls


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_a_bad_rows_file_fails_before_calibrating(tmp_path, monkeypatch, hw_a100):
    from rigtune.cli import app

    calls = _cli(monkeypatch, hw_a100)
    bad = _write(tmp_path, {"rows": [{"label": "x", "config": {**AUTOTUNE, "batchsize": 1}}]})
    r = CliRunner().invoke(app, ["compare", MODEL, "--rows", str(bad)])
    assert r.exit_code == 2 and "batchsize" in r.output and calls == []
    taken = _write(tmp_path, {"rows": [{"label": "vllm-expert-ngram", "config": AUTOTUNE}]}, "taken.json")
    r = CliRunner().invoke(app, ["compare", MODEL, "--rows", str(taken)])
    assert r.exit_code == 2 and "reserved" in r.output and calls == []


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_compare_measures_rows_from_files_and_the_expert_rows(tmp_path, monkeypatch, hw_rtx4090):
    from rigtune.cli import app

    _cli(monkeypatch, hw_rtx4090)
    a = _write(tmp_path, {"rows": [{"label": "autotune-vllm", "source": "auto_tune.sh", "config": AUTOTUNE}]})
    b = _write(tmp_path, {"rows": [{"label": "llm-optimizer", "skipped": "no pick"}]}, "b.json")
    out = tmp_path / "out"
    r = CliRunner().invoke(app, ["compare", MODEL, "--backend", "vllm", "--rows", str(a), "--rows", str(b),
                                 "--expert", "--out", str(out)])
    assert r.exit_code == 0, r.output
    (path,) = out.iterdir()
    res = ComparisonResult.model_validate_json(path.read_text(encoding="utf-8"))
    kinds = {row.label: row.kind for row in res.rows}
    assert kinds == {"rigtune": None, "vllm-default": None, "vllm-fp8-default": None,
                     "vllm-expert-batched-tokens": "expert", "vllm-expert-ngram": "expert",
                     "vllm-expert-kv-fp8": "expert", "autotune-vllm": "tuner"}
    rows = {row.label: row for row in res.rows}
    assert rows["vllm-expert-batched-tokens"].config.prefill_budget == 16384
    assert rows["vllm-expert-batched-tokens"].config.ctx == rows["vllm-default"].config.ctx
    # --backend vllm: the SGLang rows are recorded as not measured, with the reason, and so is the empty pick.
    assert [(s.label, s.reason) for s in res.skipped] == [
        ("sglang-expert-ngram", "sglang is not one of this comparison's engines"),
        ("sglang-expert-eagle3", "sglang is not one of this comparison's engines"),
        ("sglang-expert-kv-fp8-e4m3", "sglang is not one of this comparison's engines"),
        ("sglang-expert-torch-compile", "sglang is not one of this comparison's engines"),
        ("llm-optimizer", "no pick")]
    assert any(n.startswith("vllm-expert-kv-fp8 changes what the model computes") for n in res.notes)


class FailsToStart(DriftingRunner):
    """kv_dtype fp8_e4m3 does not start, as SGLang's might not on Ampere."""

    def run(self, cfg, stage):
        if cfg.kv_dtype == "fp8_e4m3":
            self.stages.append(stage)
            from rigtune.models import TrialMetrics, TrialResult

            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error="RuntimeError: fp8e4nv not supported\nmore")
        return super().run(cfg, stage)


@pytest.mark.usefixtures("no_network")
def test_a_row_that_fails_to_start_is_measured_with_its_fallback_and_says_so(hw_a100, spec):
    planned, reg, profile = _setup(hw_a100, spec)
    base = Config.model_validate(AUTOTUNE)
    row = ExtraRow(config=base.model_copy(update={"kv_dtype": "fp8_e4m3"}), kind="expert", source="a page",
                   fallback=base.model_copy(update={"kv_dtype": "fp8_e5m2"}), fallback_note="x: e4m3 failed")
    runner = FailsToStart()
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                     runner=runner, repeats=2, extra={"kv-row": row})
    assert [s.split(":", 1)[1] for s in runner.stages].count("kv-row") == 3  # the failed start, then 2 runs
    kv = next(r for r in result.rows if r.label == "kv-row")
    assert kv.ok and kv.config.kv_dtype == "fp8_e5m2" and kv.repeats == 2
    assert "x: e4m3 failed (it failed with: RuntimeError: fp8e4nv not supported)" in result.notes
    # A later failure to start is that run's failure, not a switch: only the first run tries the fallback.
    runner = FailsToStart()
    late = ExtraRow(config=base.model_copy(update={"kv_dtype": "fp8_e5m2"}), kind="expert", source="a page",
                    fallback=base, fallback_note="never")
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=get_workload("default"),
                     runner=runner, extra={"kv-row": late})
    assert "never" not in " ".join(result.notes)


@pytest.mark.usefixtures("no_network")
def test_a_row_measured_above_the_users_its_setting_acts_for_says_so(hw_a100, spec):
    planned, reg, profile = _setup(hw_a100, spec)
    row = ExtraRow(config=Config.model_validate(AUTOTUNE), kind="expert", source="a page", acts_up_to_users=4,
                   acts_note="it compiles batches of up to 4")
    wl = get_workload("default")
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=wl, runner=CompareRunner(),
                     extra={"compiled": row})
    busiest = max(wl.concurrencies)
    assert busiest > 4
    want = f"compiled is measured at up to {busiest} users, above 4: it compiles batches of up to 4"
    assert want in result.notes
    row.acts_up_to_users = busiest
    result = compare(hw_a100, spec, profile, planned.prepared, reg, workload=wl, runner=CompareRunner(),
                     extra={"compiled": row})
    assert not any("above" in n for n in result.notes)
