"""A custom goal this machine cannot measure, or this version cannot calibrate, is refused once the hardware is
probed and before any weights download (pipeline.check_goal_measurable); the presets never are. An accuracy_loss
limit is the accuracy gate's tolerance."""

from __future__ import annotations

import sys
import types

import pytest
import typer
from typer.testing import CliRunner

from rigtune import cli, goals, pipeline
from rigtune.calibrate.objectives import CUSTOM, Constraints
from rigtune.calibrate.workload import get_workload
from rigtune.models import GPUInfo
from rigtune.pipeline import OptionRefused, SearchOptions, check_goal_measurable
from tests.test_cli import _unboxed

runner = CliRunner()
ENERGY = Constraints(goal="min joules_per_token; tok_s>=200")


def _two_gpus(hw):
    return hw.model_copy(update={"gpus": [hw.gpus[0], GPUInfo(vendor="nvidia", name=hw.gpus[0].name, index=1,
                                                              compute_capability=(8, 0),
                                                              vram_total_bytes=hw.gpus[0].vram_total_bytes,
                                                              vram_free_bytes=hw.gpus[0].vram_free_bytes)]})


def _no_probe(hw):
    raise AssertionError("power probed")


def test_a_power_goal_is_refused_with_the_reason_power_cannot_be_read(hw_a100):
    with pytest.raises(OptionRefused, match=r"reads the GPU's power \(joules_per_token\), but no NVIDIA GPU is "
                                            r"visible; a custom goal never falls back"):
        check_goal_measurable(CUSTOM, ENERGY, SearchOptions(), hw_a100, power_probe=lambda hw: "no NVIDIA GPU is "
                                                                                             "visible")
    check_goal_measurable(CUSTOM, ENERGY, SearchOptions(), hw_a100, power_probe=lambda hw: None)
    capped = Constraints(goal="max tok_s; power_w<=250")
    with pytest.raises(OptionRefused, match=r"\(power_w\)"):
        check_goal_measurable(CUSTOM, capped, SearchOptions(), hw_a100, power_probe=lambda hw: "no readings")


def test_goals_that_read_no_power_and_every_preset_never_probe_it(hw_a100):
    check_goal_measurable(CUSTOM, Constraints(goal="max tok_s; mem_mb<=20000"), SearchOptions(), hw_a100,
                          power_probe=_no_probe)
    for preset in ("throughput", "latency", "balanced", "efficiency"):  # efficiency still falls back to tok/s
        check_goal_measurable(preset, None, SearchOptions(), _two_gpus(hw_a100), layout="tp", power_probe=_no_probe)


@pytest.mark.parametrize("layout, phases, refused", [
    ("tp", "unified", "--layout tp"), ("replicas", "unified", "--layout replicas"),
    ("single", "disaggregated", "--phases disaggregated"), ("auto", "unified", "--layout auto"),
])
def test_a_custom_goal_is_calibrated_for_one_engine_on_one_gpu_only(hw_a100, layout, phases, refused):
    with pytest.raises(OptionRefused, match=f"one engine on one GPU; {refused} is not offered"):
        check_goal_measurable(CUSTOM, ENERGY, SearchOptions(), _two_gpus(hw_a100), layout, phases,
                              power_probe=lambda hw: None)
    # On one GPU, auto serves one engine, so nothing is refused.
    check_goal_measurable(CUSTOM, ENERGY, SearchOptions(), hw_a100, "auto", "auto", power_probe=lambda hw: None)


def test_a_custom_goal_takes_no_arrival_rate(hw_a100):
    with pytest.raises(OptionRefused, match="efficiency only, not a custom goal"):
        check_goal_measurable(CUSTOM, ENERGY, SearchOptions(arrival_rate=0.5), hw_a100, power_probe=lambda hw: None)


class _NVML(types.ModuleType):
    def __init__(self, init=None, power=None):
        super().__init__("pynvml")
        self._init, self._power, self.shut = init, power, False

    def nvmlInit(self):
        if self._init:
            raise self._init

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetPowerUsage(self, handle):
        if self._power:
            raise self._power
        return 71000

    def nvmlShutdown(self):
        self.shut = True


def test_the_power_probe_reads_one_sample_and_says_why_it_cannot(hw_a100, hw_cpu, monkeypatch):
    assert pipeline.power_reading_problem(hw_cpu) == "no NVIDIA GPU is visible"
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import fails
    assert pipeline.power_reading_problem(hw_a100) == "pynvml is not installed (pip install nvidia-ml-py)"
    monkeypatch.setitem(sys.modules, "pynvml", _NVML(init=RuntimeError("driver not loaded")))
    assert pipeline.power_reading_problem(hw_a100) == "NVML did not start (driver not loaded)"
    unsupported = _NVML(power=RuntimeError("Not Supported"))
    monkeypatch.setitem(sys.modules, "pynvml", unsupported)
    assert pipeline.power_reading_problem(hw_a100) == "GPU 0 gives no power readings (Not Supported)"
    assert unsupported.shut
    fine = _NVML()
    monkeypatch.setitem(sys.modules, "pynvml", fine)
    assert pipeline.power_reading_problem(hw_a100) is None and fine.shut


def test_the_refusal_comes_before_anything_is_selected_or_downloaded(hw_a100, monkeypatch, tmp_home):
    def too_late(*args, **kwargs):
        raise AssertionError("got as far as selecting backends")

    monkeypatch.setattr(pipeline, "select", too_late)
    monkeypatch.setattr(pipeline, "prepare_and_plan", too_late)
    monkeypatch.setattr(pipeline, "power_reading_problem", lambda hw: "pynvml is not installed")
    from rigtune.models import ModelSpec

    with pytest.raises(OptionRefused, match="pynvml is not installed"):
        pipeline.resolve_profile(ModelSpec(hf_id="x/y"), CUSTOM, constraints=ENERGY, hw=hw_a100,
                                 options=SearchOptions())


def test_bench_refuses_before_the_plan_downloads_weights(hw_a100, monkeypatch):
    import rigtune.hardware

    def too_late(*args, **kwargs):
        raise AssertionError("planned before refusing")

    monkeypatch.setattr(rigtune.hardware, "probe", lambda: hw_a100)
    monkeypatch.setattr(pipeline, "select", lambda *a, **k: (["vllm"], {}))
    monkeypatch.setattr(pipeline, "prepare_and_plan", too_late)
    monkeypatch.setattr(pipeline, "power_reading_problem", lambda hw: "GPU 0 gives no power readings (Not Supported)")
    r = runner.invoke(cli.app, ["bench", "x/y", "--optimise", "joules_per_token", "--limit", "power_w<=250"])
    assert r.exit_code == 2 and "GPU 0 gives no power readings" in _unboxed(r.output)
    # The same metric in a goal the efficiency preset says is that preset, which falls back to tok/s: not refused.
    r = runner.invoke(cli.app, ["bench", "x/y", "--optimise", "joules_per_token", "--limit", "tok_s>=200"])
    assert "planned before refusing" in str(r.exception)
    r = runner.invoke(cli.app, ["recalibrate", "x/y", "--optimise", "tok_s", "--limit", "power_w<=200",
                                "--layout", "tp"])
    assert r.exit_code == 2 and "one engine on one GPU" in _unboxed(r.output)


# --------------------------------------------------------------------------- accuracy_loss


def _resolved(limits, eval_set=None, max_accuracy_loss=3.0):
    wl = get_workload("default")
    return cli._goal(None, None, limits, None, 95, None, None, wl, None, False, eval_set, max_accuracy_loss)


def test_an_accuracy_limit_is_the_gates_tolerance_with_the_same_profile():
    objective, cons, pct, loss = _resolved(["accuracy_loss<=1"], eval_set="gsm8k")
    legacy = _resolved([], eval_set="gsm8k", max_accuracy_loss=1.0)
    assert (objective, cons, pct, loss) == legacy == ("balanced", legacy[1], 95, 1.0)
    wl = get_workload("default")
    via_limit = cli._opts("auto", "on", "on", "on", ttft_percentile=pct, eval_set="gsm8k", max_accuracy_loss=loss)
    via_flag = cli._opts("auto", "on", "on", "on", eval_set="gsm8k", max_accuracy_loss=1.0)
    assert via_limit.key(objective, cons, wl) == via_flag.key(*legacy[:2], wl)
    assert via_limit.key(objective, cons, wl)["maxloss"] == "1"
    assert _resolved(["accuracy_loss<=1%"], eval_set="gsm8k")[3] == 1.0  # 1% is one point
    assert goals.DEFAULT_MAX_LOSS == SearchOptions().max_accuracy_loss


@pytest.mark.parametrize("limits, eval_set, loss, message", [
    (["accuracy_loss<=1"], None, 3.0, "add --eval-set gsm8k"),
    (["accuracy_loss<=1"], "gsm8k", 2.0, "--limit and --max-accuracy-loss"),
    (["accuracy_loss>=1"], "gsm8k", 3.0, "its limit is <="),
    (["accuracy_loss<=0"], "gsm8k", 3.0, "above 0 and below 100"),
    (["accuracy_loss<=1W"], "gsm8k", 3.0, "in accuracy points"),
])
def test_an_accuracy_limit_that_cannot_be_certified_is_refused(limits, eval_set, loss, message):
    with pytest.raises(typer.BadParameter, match=message):
        _resolved(limits, eval_set, loss)


def test_an_accuracy_limit_without_an_eval_set_is_refused_before_anything_is_probed(monkeypatch):
    import rigtune.hardware

    def probed(*args, **kwargs):
        raise AssertionError("probed")

    monkeypatch.setattr(rigtune.hardware, "probe", probed)
    monkeypatch.setattr(pipeline, "resolve_profile", probed)
    for command in ("bench", "serve"):
        r = runner.invoke(cli.app, [command, "x/y", "--limit", "accuracy_loss<=1"])
        assert r.exit_code == 2 and "add --eval-set" in _unboxed(r.output)


def test_a_custom_goal_with_an_accuracy_limit_is_said_with_the_gate():
    objective, cons, _, loss = _resolved(["accuracy_loss<=2", "mem_mb<=20000"], eval_set="gsm8k")
    assert (objective, loss, cons.goal) == (CUSTOM, 2.0, "max tok_s; mem_mb<=20000; ttft_p95_ms<=500")
    r = goals.resolve(None, "tok_s", ["accuracy_loss<=2"], ttft_ceiling=None, ttft_percentile=95, tok_s_floor=None,
                      tpot_ceiling=None, workload_ttft_ms=500.0, workload_tpot_ms=None, eval_set="gsm8k")
    assert r.line == ("goal: max tok_s = the throughput preset, with answers graded on gsm8k to lose at most 2 "
                      "points")


def test_a_cached_custom_profile_is_served_without_reading_power(hw_a100, monkeypatch):
    def probed(hw):
        raise AssertionError("read power to serve a cached profile")

    cached = object()
    monkeypatch.setattr(pipeline, "power_reading_problem", probed)
    monkeypatch.setattr(pipeline.profile_cache, "load", lambda *a, **k: cached)
    monkeypatch.setattr(pipeline, "_usable", lambda *a, **k: True)
    from rigtune.models import ModelSpec

    assert pipeline.resolve_profile(ModelSpec(hf_id="x/y"), CUSTOM, constraints=ENERGY, hw=hw_a100,
                                    options=SearchOptions()) is cached


@pytest.mark.parametrize("flags", [
    dict(),
    dict(objective="latency", ttft_ceiling=300.0),
    dict(objective="throughput", ttft_ceiling=300.0),
    dict(objective="balanced", tok_s_floor=100.0),
    dict(objective="efficiency", ttft_ceiling=800.0, arrival_rate=0.5),
    dict(objective="efficiency", tpot_ceiling=40.0, ttft_percentile=50),
])
def test_an_accuracy_limit_alone_leaves_the_preset_and_the_older_flags_as_max_accuracy_loss_does(flags):
    wl = get_workload("chat")

    def run(limits, loss):
        args = dict(objective=None, ttft_ceiling=None, ttft_percentile=95, tok_s_floor=None, tpot_ceiling=None,
                    arrival_rate=None)
        args.update(flags)
        return cli._goal(args["objective"], None, limits, args["ttft_ceiling"], args["ttft_percentile"],
                         args["tok_s_floor"], args["tpot_ceiling"], wl, args["arrival_rate"], False, "gsm8k", loss)

    via_limit, via_flag = run(["accuracy_loss<=1"], 3.0), run([], 1.0)
    assert via_limit == via_flag
    keys = [cli._opts("auto", "on", "on", "on", ttft_percentile=r[2], eval_set="gsm8k", max_accuracy_loss=r[3],
                      arrival_rate=flags.get("arrival_rate")).key(r[0], r[1], wl) for r in (via_limit, via_flag)]
    assert keys[0] == keys[1] and keys[0]["maxloss"] == "1"


def test_an_accuracy_limit_alone_reaches_calibration_with_efficiency_at_a_rate(monkeypatch):
    import rigtune.hardware

    def probed(*args, **kwargs):
        raise AssertionError("probed: the options were accepted")

    monkeypatch.setattr(rigtune.hardware, "probe", probed)
    base = ["bench", "m", "--objective", "efficiency", "--arrival-rate", "0.5", "--ttft-ceiling", "800",
            "--eval-set", "gsm8k"]
    for extra in (["--max-accuracy-loss", "1"], ["--limit", "accuracy_loss<=1"]):
        r = runner.invoke(cli.app, base + extra)
        assert "probed: the options were accepted" in str(r.exception), r.output
