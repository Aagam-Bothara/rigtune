"""`compare --bare-stock`: vLLM and SGLang launched with the model and port alone, beside the stock rows."""

from __future__ import annotations

import pytest

from rigtune.bench import compare
from rigtune.bench.answers import reference_label
from rigtune.bench.references import BARE_KEY, BareStock, bare_configs
from rigtune.bench.verdict import kind_of
from rigtune.models import ModelSpec
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_compare import CompareRunner
from tests.test_objectives_and_search import FakeRunner


@pytest.mark.usefixtures("no_network")
def test_a_bare_engine_is_launched_with_the_model_and_port_alone(hw_rtx4090, spec, monkeypatch):
    import rigtune.backends.vllm as vl
    import rigtune.hardware as hwmod

    candidates, reg = select(hw_rtx4090, spec)
    planned = prepare_and_plan(hw_rtx4090, spec, candidates, reg)
    configs = bare_configs(planned.prepared)
    assert list(configs) == ["vllm-bare", "sglang-bare"]  # not llama.cpp: no bare launch for it
    monkeypatch.setattr(vl, "_serve_cli", lambda: "/venv/bin/vllm")
    vllm = BareStock(reg["vllm"]).launch_spec(configs["vllm-bare"], planned.prepared["vllm"], 8123)
    assert vllm.args == ["/venv/bin/vllm", "serve", spec.hf_id, "--host", "127.0.0.1", "--port", "8123"]
    assert vllm.env == {}
    monkeypatch.setattr(hwmod, "sglang_python", lambda: "/sgl/bin/python")
    sgl = BareStock(reg["sglang"]).launch_spec(configs["sglang-bare"], planned.prepared["sglang"], 8124)
    assert sgl.args == ["/sgl/bin/python", "-m", "sglang.launch_server", "--model-path", spec.hf_id,
                        "--host", "127.0.0.1", "--port", "8124"]
    pinned = planned.prepared["vllm"].model_copy(update={"spec": ModelSpec(hf_id=spec.hf_id, revision="abc")})
    pinned_args = BareStock(reg["vllm"]).launch_spec(configs["vllm-bare"], pinned, 1).args
    assert pinned_args[-2:] == ["--revision", "abc"]
    wrapped = BareStock(reg["vllm"])
    assert wrapped.version(hw_rtx4090) == reg["vllm"].version(hw_rtx4090) and wrapped.name == "vllm-bare"
    hooks = wrapped.workload_hooks(hw_rtx4090, planned.prepared["vllm"])
    assert hooks.model_name == spec.hf_id  # the name the engine serves a path under, when nothing renames it


class BareAware(CompareRunner):
    """The fake engine's speed table is keyed by precision; a bare row has none, so it is measured as bf16."""

    def run(self, cfg, stage):
        if cfg.backend.endswith("-bare"):
            cfg = cfg.model_copy(update={"quant": "bf16", "ctx": 4096, "batch": 64})
        return super().run(cfg, stage)


@pytest.mark.usefixtures("no_network")
def test_compare_measures_bare_rows_as_stock_beside_the_default_rows(hw_rtx4090, spec):
    candidates, reg = select(hw_rtx4090, spec, force="vllm")
    planned = prepare_and_plan(hw_rtx4090, spec, candidates, reg)
    profile = calibrate(hw_rtx4090, spec, "throughput", planned, reg, runner=FakeRunner())
    plain = compare(hw_rtx4090, spec, profile, planned.prepared, reg, runner=CompareRunner())
    assert not any(r.label.endswith("-bare") for r in plain.rows)
    res = compare(hw_rtx4090, spec, profile, planned.prepared, reg, runner=BareAware(), bare_stock=True,
                  include=["vllm-default"])
    rows = {r.label: r for r in res.rows}
    assert list(rows) == ["rigtune", "vllm-default", "vllm-bare"]  # --include does not filter bare rows
    bare = rows["vllm-bare"]
    assert (bare.kind, bare.config_key, bare.runtime, bare.ok) == ("stock", BARE_KEY, "vllm-bare", True)
    assert bare.launch_args[-4:] == ["--host", "127.0.0.1", "--port", "0"]
    assert [r.label for r in res.default_rows()] == ["vllm-default", "vllm-bare"]
    assert kind_of(bare) == "stock" and reference_label("vllm-bare") == "vllm-default"
