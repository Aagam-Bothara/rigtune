import json

import pytest

from rigtune import speculative as sp
from rigtune.backends import vllm as vl
from rigtune.backends.facts import EngineFacts
from rigtune.calibrate.search import StagedSearch
from rigtune.models import Config
from tests.test_spec_mtp import with_head, vcfg


@pytest.fixture
def capable(monkeypatch):
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 33))
    monkeypatch.setattr(vl, "_mtp_archs", lambda: sp.MTP_ARCHS)
    facts = EngineFacts(engine="vllm", version="0.33.0", spec_fields=["draft_lm_head_quantization"])
    monkeypatch.setattr(vl, "_engine_facts", lambda: facts)
    return facts


def test_key_unchanged_when_unset():
    cfg = Config(backend="vllm", quant="bf16", spec_decode="mtp:1")
    assert cfg.key() == "vllm/bf16/ctx4096/b64/sd=mtp:1"
    assert "spec_head_quant" not in cfg.model_dump()
    assert cfg.model_copy(update={"spec_head_quant": "fp8"}).key().endswith("/shq=fp8")


def test_offered_only_for_mtp_on_0_33_with_the_field(capable, hw_rtx4090, monkeypatch):
    be, pm, cfg = vl.VllmBackend(), with_head(), vcfg(batch=8)
    out = be.spec_variants(cfg, pm, hw_rtx4090, True)
    assert [(c.spec_decode, c.spec_head_quant) for c in out if c.spec_head_quant] == [("mtp:1", "fp8")]
    assert not any(c.spec_head_quant for c in be.spec_variants(cfg, pm, hw_rtx4090))
    capable.spec_fields = None
    assert not any(c.spec_head_quant for c in be.spec_variants(cfg, pm, hw_rtx4090, True))
    capable.spec_fields = ["draft_lm_head_quantization"]
    monkeypatch.setattr(vl, "_vllm_version", lambda: (0, 29))
    assert not any(c.spec_head_quant for c in be.spec_variants(cfg, pm, hw_rtx4090, True))


def test_not_on_ampere(capable, hw_a100):
    assert not any(c.spec_head_quant for c in vl.VllmBackend().spec_variants(vcfg(), with_head(), hw_a100, True))


def test_v1_override_withholds_head_quantization(capable, hw_rtx4090, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    be, pm, cfg = vl.VllmBackend(), with_head(), vcfg(batch=8)
    assert not any(c.spec_head_quant for c in be.spec_variants(cfg, pm, hw_rtx4090, True))
    assert any("Model Runner V2" in note for note in be.spec_notes(cfg, pm, hw_rtx4090, True))
    with pytest.raises(ValueError, match="Model Runner V2"):
        be.launch_spec(cfg.model_copy(update={"spec_decode": "mtp:1", "spec_head_quant": "fp8"}), pm, 8000)


def test_json_carries_fp8(capable):
    cfg = vcfg(spec_decode="mtp:1", spec_head_quant="fp8")
    args = vl.VllmBackend().launch_spec(cfg, with_head(), 8000).args
    assert json.loads(args[args.index("--speculative-config") + 1])["draft_lm_head_quantization"] == "fp8"


def test_assistant_not_offered_with_head_quant():
    with pytest.raises(ValueError):
        sp.vllm_config("assist:org/assistant:1", head_quant="fp8")


def test_switching_proposer_clears_head_quant(capable, hw_rtx4090):
    cfg = vcfg(spec_decode="mtp:1", spec_head_quant="fp8")
    variants = vl.VllmBackend().spec_variants(cfg, with_head(), hw_rtx4090, True)
    assert all(c.spec_head_quant is None for c in variants if not sp.own_head(c.spec_decode))
    assert StagedSearch._apply(cfg, {"spec_decode": "ngram:4"}).spec_head_quant is None


def test_head_quant_delta_carries_the_mtp_method():
    base = vcfg(spec_decode="mtp:1")
    delta = StagedSearch._delta(base, base.model_copy(update={"spec_head_quant": "fp8"}))
    assert delta == {"spec_decode": "mtp:1", "spec_head_quant": "fp8"}
