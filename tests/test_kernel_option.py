"""--kernel K=V: one named kernel setting tried on each tuned engine's leader, as a full trial.

Without the option nothing is different: the same options key, the same trials, the same pick. With it, the setting
is launched only when the engine probe confirmed it and no rule says it changes nothing; it is kept only when it
beats the leader on the objective's own score, and a tie keeps the plain configuration."""

from __future__ import annotations

import re

import pytest

import rigtune.backends.vllm as vl
from rigtune.calibrate.objectives import Constraints
from rigtune.models import OBJECTIVES, Config, TrialMetrics, TrialResult
from rigtune.pipeline import SearchOptions, calibrate, prepare_and_plan, resolve_profile, select
from tests.kernel_facts import vllm_facts
from tests.test_objectives_and_search import FakeRunner

SETTING = "attn=TRITON_ATTN"


class Kerneled(FakeRunner):
    """The fake runner, with a config that carries kernel settings `gain` times as fast and `ttft` ms slower to its
    first token."""

    def __init__(self, gain: float = 1.0, ttft: float = 0.0):
        super().__init__()
        self.gain, self.ttft = gain, ttft

    def run(self, cfg: Config, stage: str) -> TrialResult:
        res = super().run(cfg, stage)
        if cfg.kernels is None or not res.launched:
            return res
        m = res.metrics
        scaled = m.model_copy(update={"tok_s": m.tok_s * self.gain, "tpot_ms": m.tpot_ms / self.gain,
                                      "ttft_ms": m.ttft_ms + self.ttft})
        return TrialResult(config=cfg, stage=stage, metrics=scaled)


def _calibrate(hw, spec, runner, kernel=SETTING, objective="throughput", constraints=None, facts=True, monkeypatch=None):
    if facts:
        monkeypatch.setattr(vl, "_engine_facts", lambda: vllm_facts())
    candidates, reg = select(hw, spec, force="vllm")
    plan = prepare_and_plan(hw, spec, candidates, reg)
    return calibrate(hw, spec, objective, plan, reg, runner=runner, constraints=constraints,
                     options=SearchOptions(kernel=kernel))


def _kernel_calls(runner) -> list:
    return [key for key in runner.calls if "kern=" in key]


def test_default_options_key_unchanged():
    for objective in OBJECTIVES:
        plain = SearchOptions().key(objective)
        assert "kernel" not in plain and plain == SearchOptions(kernel=None).key(objective)
        with_kernel = SearchOptions(kernel=SETTING).key(objective)
        assert {k: v for k, v in with_kernel.items() if k != "kernel"} == plain  # one entry more, nothing else


def test_option_key_is_hash_and_filename_safe():
    value = SearchOptions(kernel="cgsizes=1+2+4,pass.fuse_norm_quant=1,attn=TRITON_ATTN").key("balanced")["kernel"]
    assert re.fullmatch(r"[0-9a-f]{16}", value)  # none of < > : " / \ | ? * and short, whatever the setting holds


def test_same_setting_any_order_same_key():
    a = SearchOptions(kernel="cg=FULL,attn=TRITON_ATTN").key("balanced")
    b = SearchOptions(kernel=" attn=TRITON_ATTN , cg=FULL").key("balanced")
    assert a == b and a != SearchOptions(kernel="cg=FULL").key("balanced")


@pytest.mark.usefixtures("no_network")
def test_named_setting_reaches_a_launch(hw_a100, spec, monkeypatch):
    runner = Kerneled()
    profile = _calibrate(hw_a100, spec, runner, monkeypatch=monkeypatch)
    assert len(_kernel_calls(runner)) >= 1 and all(k.endswith("/kern=attn=TRITON_ATTN") for k in _kernel_calls(runner))
    rows = [r for r in profile.calibration_table if r.stage == "kernels"]
    assert len(rows) == 1 and rows[0].config.kernels == SETTING  # an ordinary row of the trade-off table


@pytest.mark.usefixtures("no_network")
def test_variant_tried_once_per_tuned_engine(hw_a100, spec, monkeypatch):
    runner = Kerneled()
    profile = _calibrate(hw_a100, spec, runner, monkeypatch=monkeypatch)
    staged = [r for r in profile.calibration_table if r.stage == "kernels"]
    assert len(staged) == 1  # one engine tuned, one trial of the setting, on that engine's leader
    leader_key = staged[0].config.model_copy(update={"kernels": None}).key()
    assert leader_key in runner.calls[:runner.calls.index(staged[0].config.key())]


@pytest.mark.usefixtures("no_network")
def test_without_the_option_no_trial_and_the_same_pick(hw_a100, spec, monkeypatch):
    plain_runner, runner = FakeRunner(), Kerneled()
    plain = _calibrate(hw_a100, spec, plain_runner, kernel=None, monkeypatch=monkeypatch)
    tied = _calibrate(hw_a100, spec, runner, monkeypatch=monkeypatch)
    assert _kernel_calls(plain_runner) == [] and not any("kernel" in n for n in plain.notes)
    assert [k for k in runner.calls if "kern=" not in k] == plain_runner.calls  # the same trials, plus the setting's
    assert tied.config.key() == plain.config.key()


@pytest.mark.usefixtures("no_network")
def test_tie_keeps_the_plain_config(hw_a100, spec, monkeypatch):
    profile = _calibrate(hw_a100, spec, Kerneled(gain=1.0), monkeypatch=monkeypatch)
    assert profile.config.kernels is None  # as fast, one strategy more: the plain configuration is served


@pytest.mark.usefixtures("no_network")
def test_a_measured_win_is_adopted(hw_a100, spec, monkeypatch):
    profile = _calibrate(hw_a100, spec, Kerneled(gain=1.10), monkeypatch=monkeypatch)
    assert profile.config.kernels == SETTING and profile.config.key().endswith("/kern=attn=TRITON_ATTN")


@pytest.mark.usefixtures("no_network")
def test_win_on_tok_s_that_breaks_a_limit_is_not_adopted(hw_a100, spec, monkeypatch):
    """Faster, and past the first-token ceiling: the objective's own limits judge it, not tok/s alone."""
    profile = _calibrate(hw_a100, spec, Kerneled(gain=1.5, ttft=60_000.0), objective="balanced",
                         constraints=Constraints(), monkeypatch=monkeypatch)
    assert profile.config.kernels is None


@pytest.mark.usefixtures("no_network")
def test_objective_ranks_by_its_own_metric(hw_a100, spec, monkeypatch):
    """Under `latency` a setting that is much faster to the first token wins without any gain in tok/s."""
    profile = _calibrate(hw_a100, spec, Kerneled(gain=1.0, ttft=-50.0), objective="latency", monkeypatch=monkeypatch)
    assert profile.config.kernels == SETTING


@pytest.mark.usefixtures("no_network")
def test_refused_setting_note_not_launch(hw_a100, spec, monkeypatch):
    runner = Kerneled(gain=2.0)
    profile = _calibrate(hw_a100, spec, runner, facts=False)  # the probe is off in tests: nothing was asked
    assert _kernel_calls(runner) == [] and profile.config.kernels is None
    (note,) = [n for n in profile.notes if n.startswith("kernel setting")]
    assert note.startswith("kernel setting attn=TRITON_ATTN not tried on vllm: ") and "not asked" in note


@pytest.mark.usefixtures("no_network")
def test_equivalent_setting_is_not_launched_and_the_note_names_the_rule(hw_a100, spec, monkeypatch):
    hw_a100.backends["vllm"].version = "0.33.0"
    runner = Kerneled(gain=2.0)
    profile = _calibrate(hw_a100, spec, runner, kernel="O=3", monkeypatch=monkeypatch)
    assert _kernel_calls(runner) == []
    (note,) = [n for n in profile.notes if n.startswith("kernel setting")]
    assert "O=3 is vLLM's default optimisation level" in note and "no change to measure" in note


def test_not_tried_under_layout_or_phases_and_says_so(hw_a100, spec):
    for kw in ({"layout": "replicas"}, {"phases": "disaggregated"}):
        with pytest.raises(RuntimeError, match="--kernel tries one setting on a single engine"):
            resolve_profile(spec, hw=hw_a100, options=SearchOptions(kernel=SETTING), **kw)


def test_metrics_helper_is_only_for_kernel_configs():
    cfg = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    plain, fast = FakeRunner().run(cfg, "s").metrics, Kerneled(gain=3.0).run(cfg, "s").metrics
    assert isinstance(plain, TrialMetrics) and plain.tok_s == fast.tok_s
