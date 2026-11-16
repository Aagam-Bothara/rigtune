"""A profile calibrated without verified drafts is saved with the keys it had before they existed."""
import json

import pytest

from rigtune.models import PreparedModel, Profile
from rigtune.pipeline import calibrate, prepare_and_plan, select
from tests.test_objectives_and_search import FakeRunner

# The keys of a saved prepared model and of a saved configuration before the verified-drafts options.
PREPARED = {"spec", "backend", "arch", "hf_path", "gguf_paths", "weights_bytes", "hf_paths", "draft_paths",
            "spec_repos", "spec_bytes"}
CONFIG = {"backend", "quant", "ctx", "batch", "gpu_memory_utilization", "n_gpu_layers", "n_batch", "prefill_budget",
          "kv_dtype", "prefix_cache", "spec_decode", "tp", "kv_tier", "extra", "text_only", "gmu_before_spec",
          "power_limit_w", "sm_clock_mhz"}
VERIFIED_ONLY = ("spec_revisions", "spec_pairs", "drafts_mode", "spec_paths")


@pytest.mark.usefixtures("no_network")
def test_table_mode_profile_serializes_as_before(hw_a100, spec):
    candidates, reg = select(hw_a100, spec, force="vllm")
    profile = calibrate(hw_a100, spec, "throughput", prepare_and_plan(hw_a100, spec, candidates, reg), reg,
                        runner=FakeRunner())
    text = profile.model_dump_json(indent=2)
    data = json.loads(text)
    assert set(data["prepared"]) == PREPARED and set(data["config"]) == CONFIG
    assert all(set(pm) == PREPARED for pm in data["prepared_all"].values())
    assert all(set(trial["config"]) == CONFIG for trial in data["calibration_table"])
    assert not any(f'"{name}"' in text for name in VERIFIED_ONLY)
    assert set(profile.prepared.model_dump()) == set(data["prepared"])  # dict and JSON dumps agree
    again = Profile.model_validate_json(text)
    assert again.prepared.drafts_mode == "table" and again.prepared.spec_pairs == []
    assert again.model_dump_json(indent=2) == text


def test_verified_fields_are_saved_once_set_and_load_back(prepared_vllm):
    assert set(prepared_vllm.model_dump()) == PREPARED
    pm = prepared_vllm.model_copy(deep=True)
    pm.drafts_mode = "verified"
    pm.spec_revisions = {"org/draft": "b" * 40}
    pm.spec_pairs = [{"kind": "draft", "repo": "org/draft"}]
    pm.spec_paths = {"local/draft-12345678": "/home/private-person/draft"}
    data = pm.model_dump()
    assert set(data) == PREPARED | {"spec_revisions", "spec_pairs", "drafts_mode"}
    loaded = PreparedModel.model_validate_json(pm.model_dump_json())
    assert (loaded.drafts_mode, loaded.spec_revisions, loaded.spec_pairs) == (
        "verified", pm.spec_revisions, pm.spec_pairs)
    assert loaded.spec_paths == {}
