import pytest
from typer.main import get_command

from rigtune import speculative
from rigtune.cli import app
from rigtune.models import PreparedModel
from rigtune.pairing import local_alias
from rigtune.pipeline import SearchOptions, prepare_and_plan


def test_fingerprint_unchanged_by_default_options():
    assert SearchOptions().key() == {}  # dcfab53 literal
    assert SearchOptions(drafts="table", hub="on").key() == {}


def test_verified_and_draft_enter_the_fingerprint():
    assert SearchOptions(drafts="verified").key()["drafts"] == "verified"
    assert SearchOptions(draft="org/draft").key()["draft"] != SearchOptions(draft="org/other").key()["draft"]


def test_hub_off_does_not():
    assert SearchOptions(hub="off").key() == {}


def test_profile_without_new_fields_loads(prepared_vllm):
    pm = PreparedModel.model_validate(prepared_vllm.model_dump())
    assert pm.spec_revisions == {} and pm.spec_paths == {}
    pm.spec_paths["local/d-12345678"] = "C:/private-person/d"
    assert "private-person" not in pm.model_dump_json()


@pytest.mark.parametrize("path", [r"C:\m\d", "/m/d"])
def test_local_draft_spec_has_no_path(path):
    alias = local_alias(path, "12345678abc")
    assert alias == "local/d-12345678"
    assert speculative.parse(speculative.draft(alias, 4)) == ("draft", alias, 4)


def test_candidates_cover_old_rows_and_assistants():
    assert speculative.candidates_for("Qwen/Qwen3-8B") == [("draft", "Qwen/Qwen3-0.6B")]
    assert speculative.candidates_for("google/gemma-4-12B-it") == [("assistant", "google/gemma-4-12B-it-assistant")]
    assert speculative.candidates_for("Qwen/Qwen3-8B", "me/mine")[0] == ("draft", "me/mine")


def test_table_mode_makes_no_lookup(no_network, hw_a100, spec, monkeypatch):
    from rigtune.backends.vllm import VllmBackend
    def forbidden(*a, **kw):
        raise AssertionError("table mode reached pairing")
    monkeypatch.setattr("rigtune.pairing.read_files", forbidden)
    plan = prepare_and_plan(hw_a100, spec, ["vllm"], {"vllm": VllmBackend()})
    assert plan.feasible["vllm"]


@pytest.mark.parametrize("command", ["plan", "bench", "recalibrate", "compare", "serve"])
def test_cli_exposes_draft_options(command):
    params = {p.name for p in get_command(app).commands[command].params}
    assert {"drafts", "draft", "hub"} <= params


def test_new_launch_goldens_only():
    import json
    from pathlib import Path
    root = Path(__file__).parent / "golden/drafts"
    for spec, expected in json.loads((root / "vllm-configs.json").read_text()).items():
        assert speculative.vllm_config(spec, "b" * 40) == expected
    for spec, expected in json.loads((root / "sglang-flags.json").read_text()).items():
        assert speculative.sglang_config(spec, draft_revision="b" * 40, draft_quantization="unquant") == expected
