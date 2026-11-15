"""--strict-constraints, --confirm-rounds and --min-requests: opt-in controls on how a pick is validated."""
import pytest
from typer.testing import CliRunner

import rigtune.pipeline as pipeline
from rigtune.calibrate.objectives import Constraints, strict_refusal
from rigtune.calibrate.search import StagedSearch
from rigtune.calibrate.workload import get_workload
from rigtune.cli import app
from rigtune.models import Config, TrialMetrics, TrialResult
from rigtune.pipeline import OptionRefused, SearchOptions, calibrate, prepare_and_plan, select
from tests.conftest import make_hw
from tests.test_objectives_and_search import FakeRunner


class LatencyRunner:
    """Only the `fast` variant can meet a TTFT ceiling: its TTFT is `fast_ttft`, every other config's 400 ms."""

    def __init__(self, fast_ttft: float):
        self.fast_ttft = fast_ttft
        self.ran = []

    def run(self, cfg, stage):
        self.ran.append((stage, cfg.key()))
        ttft = self.fast_ttft if cfg.extra.get("fast") else 400.0
        m = TrialMetrics(tok_s=100.0, requests=72, output_tokens=2048, concurrency=8, ttft_ms=ttft, ttft_p95_ms=ttft)
        return TrialResult(config=cfg, stage=stage, metrics=m.model_copy(update={"by_concurrency": {"8": m}}))


FAST = [("fastpath", lambda c: [c.model_copy(update={"extra": {"fast": True}})] if not c.extra.get("fast") else [])]


def _strict_search(fast_ttft: float, strict: bool = True):
    runner = LatencyRunner(fast_ttft)
    search = StagedSearch(objective="balanced", runner=runner, variant_stages=FAST, confirm_top=0,
                          constraints=Constraints(ttft_ceiling_ms=100, strict=strict))
    feasible = [Config(backend="vllm", quant="bf16", ctx=4096, batch=b, gpu_memory_utilization=0.9) for b in (16, 64)]
    winner, notes = search.run(feasible)
    return winner, notes, runner


def test_strict_mode_still_tries_the_variant_that_meets_the_target_and_serves_it():
    """Strictness applies to the final pick only: mid-search the least-violating leader still gets its variants."""
    winner, notes, runner = _strict_search(fast_ttft=50.0)
    assert any(stage == "fastpath" for stage, _ in runner.ran)
    assert winner is not None and winner.config.extra.get("fast")
    assert not any("strict constraints" in n for n in notes)


def test_strict_mode_refuses_only_when_nothing_measured_meets_the_target():
    winner, notes, runner = _strict_search(fast_ttft=150.0)
    assert any(stage == "fastpath" for stage, _ in runner.ran)  # the whole search ran first
    assert winner is None
    assert any(n.startswith("strict constraints: no measured configuration met the balanced") for n in notes)
    lenient, lenient_notes, _ = _strict_search(fast_ttft=150.0, strict=False)
    assert lenient is not None and lenient.config.extra.get("fast")  # the least-violating one, as before
    assert any("least-violating" in n for n in lenient_notes)


def test_strict_refusal_judges_the_pick_among_what_it_was_picked_from():
    m = TrialMetrics(tok_s=100, ttft_ms=600, ttft_p95_ms=700, requests=72, output_tokens=1000)
    slow = TrialResult(config=Config(backend="vllm", quant="bf16"), stage="confirm", metrics=m)
    assert strict_refusal(slow, [slow], "balanced", Constraints(ttft_ceiling_ms=100, strict=True))
    assert strict_refusal(slow, [slow], "balanced", Constraints(ttft_ceiling_ms=100)) is None  # not strict
    assert strict_refusal(slow, [slow], "balanced", Constraints(ttft_ceiling_ms=1000, strict=True)) is None


def test_the_new_options_add_cache_keys_only_when_used():
    wl = get_workload("default")
    assert SearchOptions().key("throughput", None, wl) == {}
    assert SearchOptions(confirm_rounds=3, min_requests=72).key("throughput", None, wl) == {"rounds": "3",
                                                                                           "samples": "72"}
    strict = SearchOptions().key("throughput", Constraints(ttft_ceiling_ms=500, tpot_ceiling_ms=100, strict=True), wl)
    assert set(strict) == {"limits"}  # strictness is a constraint, so it is in the constraints' fingerprint


@pytest.mark.usefixtures("no_network")
def test_requested_confirmation_rounds_are_actually_run(hw_a100, spec):
    _, reg = select(hw_a100, spec, force="vllm")
    wl = get_workload("default")
    plan = prepare_and_plan(hw_a100, spec, ["vllm"], reg, workload=wl)
    profile = calibrate(hw_a100, spec, "throughput", plan, reg, runner=FakeRunner(),
                        options=SearchOptions(confirm_rounds=3))
    keys = [r.config.key() for r in profile.calibration_table if r.stage == "confirm"]
    assert keys and all(keys.count(key) == 3 for key in set(keys))
    assert profile.options["rounds"] == "3"


def test_cli_exposes_the_validation_controls():
    runner = CliRunner()
    for command in ("bench", "recalibrate", "compare", "serve"):
        result = runner.invoke(app, [command, "--help"], terminal_width=200)
        assert result.exit_code == 0
        for flag in ("--strict-constraints", "--confirm-rounds", "--min-requests"):
            assert flag in result.output


@pytest.mark.parametrize("layout, phases, gpus, refused", [
    ("replicas", "unified", 2, True), ("tp", "unified", 2, True), ("auto", "unified", 2, True),
    ("single", "disaggregated", 2, True), ("single", "auto", 2, True),
    ("auto", "unified", 1, False), ("single", "auto", 1, False),  # these serve one engine on a one-GPU box
    ("single", "unified", 2, False),
])
def test_rounds_under_a_multi_gpu_search_are_refused_before_anything_is_calibrated(layout, phases, gpus, refused,
                                                                                   monkeypatch):
    hw = make_hw("a100")
    hw.gpus = [hw.gpus[0].model_copy(update={"index": i}) for i in range(gpus)]

    def select_(*args, **kwargs):
        raise RuntimeError("calibration started")

    monkeypatch.setattr(pipeline, "select", select_)
    expected = OptionRefused if refused else RuntimeError
    with pytest.raises(expected):
        pipeline.resolve_profile(pipeline.ModelSpec(hf_id="m"), hw=hw, layout=layout, phases=phases,
                                 recalibrate=True, options=SearchOptions(confirm_rounds=3))


@pytest.mark.parametrize("kind", ["layout", "phases"])
def test_auto_on_one_gpu_with_confirm_rounds_falls_back_to_the_single_engine_profile(kind, hw_a100, spec,
                                                                                     prepared_vllm):
    """The pre-check lets --layout auto / --phases auto through on one GPU, since they serve one engine there; the
    multi-GPU step must then fall back as before, not refuse after the single-engine calibration has run."""
    from tests.test_cache_and_pipeline import _profile

    base = _profile(hw_a100, spec, prepared_vllm)
    base.options = {"rounds": "3"}
    if kind == "layout":
        from rigtune.layout import calibrate_layout
        out = calibrate_layout(hw_a100, "balanced", base, {}, "auto")
        assert out.replicas == 1 and out.config == base.config and any("two NVIDIA GPUs" in n for n in out.notes)
    else:
        from rigtune.disagg import calibrate_disaggregated
        out = calibrate_disaggregated(hw_a100, spec, "balanced", base, {}, phases="auto")
        assert out.disagg is None and out.config == base.config


@pytest.mark.usefixtures("no_network")
def test_minimum_samples_are_recorded_without_changing_the_callers_workload(hw_a100, spec):
    wl = get_workload("default")
    _, reg = select(hw_a100, spec, force="vllm")
    plan = prepare_and_plan(hw_a100, spec, ["vllm"], reg, workload=wl)
    profile = calibrate(hw_a100, spec, "throughput", plan, reg, workload=wl, runner=FakeRunner(),
                        options=SearchOptions(min_requests=72))
    assert profile.workload_spec["n_prompts"] >= 72
    assert "requests_per_slot" not in profile.workload_spec
    assert wl.n_prompts == 16


def test_minimum_samples_cannot_be_satisfied_by_wrapping_a_small_file(tmp_path):
    from rigtune.calibrate.workload import with_minimum_requests, workload_from_file
    path = tmp_path / "small.jsonl"
    path.write_text('{"prompt":"first"}\n{"prompt":"second"}\n{"prompt":"third"}\n', encoding="utf-8")
    workload = workload_from_file(path)
    with pytest.raises(ValueError, match="pool"):
        with_minimum_requests(workload, 4)


def test_a_bad_minimum_is_refused_by_the_cli_before_the_hardware_is_probed(tmp_path, monkeypatch):
    import rigtune.hardware as hardware

    path = tmp_path / "small.jsonl"
    path.write_text('{"prompt":"first"}\n{"prompt":"second"}\n', encoding="utf-8")
    monkeypatch.setattr(hardware, "probe", lambda: (_ for _ in ()).throw(RuntimeError("probed")))
    result = CliRunner().invoke(app, ["bench", "m", "--workload-file", str(path), "--min-requests", "72"])
    assert result.exit_code == 2 and "--min-requests" in result.output


def test_minimum_samples_keep_fresh_entries_for_each_level_and_resample(tmp_path):
    from rigtune.calibrate.workload import with_minimum_requests, workload_from_file
    path = tmp_path / "large.jsonl"
    path.write_text("\n".join('{"prompt":"prompt number ' + str(i) + '"}' for i in range(500)), encoding="utf-8")
    workload = workload_from_file(path)
    adjusted = with_minimum_requests(workload, 72)
    assert adjusted.level_requests(1) == 72
    assert adjusted.n_prompts == 72
    assert workload.n_prompts != adjusted.n_prompts
