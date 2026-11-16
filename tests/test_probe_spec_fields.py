import sys

from rigtune.backends import facts as ef
from tests import engine_stubs


def test_probe_reports_fields_from_a_fake_package(tmp_path, monkeypatch):
    root = engine_stubs.build("vllm", tmp_path / "stub")
    path = root / "vllm/config/speculative.py"
    with path.open("a") as fh:
        fh.write("\nfrom dataclasses import dataclass\n@dataclass\nclass SpeculativeConfig:\n"
                 "    method: str = 'mtp'\n    draft_lm_head_quantization: str = None\n")
    monkeypatch.setenv("PYTHONPATH", str(root))
    f = ef.run_probe("vllm", sys.executable)
    assert f.spec_fields == ["draft_lm_head_quantization", "method"]


def test_old_kept_answer_reads_unanswered():
    f = ef.EngineFacts.model_validate({"engine": "vllm", "version": "0.33.0"})
    assert f.spec_fields is None and not f.answered("spec_fields")


def test_empty_list_is_an_answer_only_with_provenance_probe():
    assert ef.EngineFacts(engine="vllm", spec_fields=[], provenance="probe").answered("spec_fields")
    assert not ef.EngineFacts(engine="vllm", spec_fields=[], provenance="fixture").answered("spec_fields")
    parsed = ef.parse("vllm", 'RIGTUNE-PROBE {"key":"spec_fields","value":[]}', "no answer")
    assert parsed.spec_fields == [] and parsed.answered("spec_fields")
