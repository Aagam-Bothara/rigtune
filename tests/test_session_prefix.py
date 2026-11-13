"""A session resends its history, so prefix-cache settings are tuned and ablated for it as for a shared prefix."""

from __future__ import annotations

import pytest

from rigtune.backends import get_backend
from rigtune.bench.ablation import strategy_variants
from rigtune.calibrate.objectives import Constraints
from rigtune.calibrate.search import StagedSearch
from rigtune.calibrate.workload import get_workload, workload_from_file
from rigtune.models import Config
from tests.test_session_replay import _file
from tests.test_strategies import FakeRunner

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture
def sessions(tmp_path):
    return workload_from_file(_file(tmp_path), get_workload("chat"))


def test_a_session_workload_reuses_a_prefix(sessions):
    assert sessions.reuses_prefix and get_workload("chat-system").reuses_prefix
    assert not get_workload("chat").reuses_prefix


def _llamacpp(hw_gtx1080, spec, monkeypatch, workload):
    import rigtune.backends.llamacpp as lc
    from rigtune.pipeline import prepare_and_plan, select

    def fake_materialize(self, m, quants):
        for q in quants:
            m.gguf_paths[q] = f"/models/{q}.gguf"
        return m

    monkeypatch.setattr(lc.LlamaCppBackend, "materialize", fake_materialize)
    candidates, reg = select(hw_gtx1080, spec, force="llamacpp-cuda", workload=workload)
    return reg, prepare_and_plan(hw_gtx1080, spec, candidates, reg, materialize=True, workload=workload)


def test_calibration_runs_the_prefix_stage_for_sessions(hw_gtx1080, spec, monkeypatch, sessions):
    from rigtune.pipeline import calibrate

    reg, plan = _llamacpp(hw_gtx1080, spec, monkeypatch, sessions)
    profile = calibrate(hw_gtx1080, spec, "throughput", plan, reg, workload=sessions, runner=FakeRunner())
    prefix = [r for r in profile.calibration_table if r.stage == "prefix"]
    assert prefix and {k for r in prefix for k in r.config.extra} & {"cache_reuse", "kv_unified"}


def test_the_explore_space_holds_prefix_settings_for_sessions(hw_gtx1080, spec, monkeypatch, sessions):
    from rigtune.pipeline import search_space

    reg, plan = _llamacpp(hw_gtx1080, spec, monkeypatch, sessions)
    space = search_space(hw_gtx1080, plan, reg, sessions)
    assert any("cache_reuse" in c.extra or "kv_unified" in c.extra for c in space)
    flat = search_space(hw_gtx1080, plan, reg, get_workload("chat"))
    assert not any("cache_reuse" in c.extra or "kv_unified" in c.extra for c in flat)


def test_the_ablation_measures_prefix_caching_for_sessions(hw_gtx1080, prepared_llamacpp, sessions):
    be = get_backend("llamacpp-cuda")
    pick = Config(backend="llamacpp-cuda", quant="Q4_K_M", ctx=4096, batch=4, n_gpu_layers=29, n_batch=512)
    labels = [v.label for v in strategy_variants(pick, be, hw_gtx1080, prepared_llamacpp, sessions)
              if v.strategy == "prefix"]
    assert labels == ["+prefix:cache_reuse", "+prefix:kv_unified"]


def test_the_predictor_does_not_prune_a_session_workload(sessions):
    class Fitted:
        def is_fitted(self, backend):
            return True

        def best_level(self, *args, **kwargs):  # would price every history token as uncached prefill
            raise AssertionError("asked to predict a session workload")

    cfg = Config(backend="vllm", quant="bf16")
    search = StagedSearch(objective="throughput", runner=FakeRunner(), constraints=Constraints(),
                          predictor=Fitted(), models={"vllm": object()}, workload=sessions)
    assert search._predicted_tok_s(cfg) is None
