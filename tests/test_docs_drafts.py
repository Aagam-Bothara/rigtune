import json

from tests.test_next_session_plan import nsp


def test_session_draft_check_requires_the_actual_method_to_pass(tmp_path):
    path = tmp_path / "vllm.json"
    path.write_text(json.dumps([{"details": {"C06-ngram": {"status": "pass"}}}]))
    assert nsp.check_drafts(tmp_path, "assist") == 1
    path.write_text(json.dumps([{"details": {"C06-assist": {"status": "pass"}}}]))
    assert nsp.check_drafts(tmp_path, "assist") == 0
    path.write_text(json.dumps([{"details": {"C06-assist": {"status": "fail"}}}]))
    assert nsp.check_drafts(tmp_path, "assist") == 1


def test_usage_lists_the_new_options():
    from pathlib import Path
    from typer.main import get_command
    from rigtune.cli import app
    usage = Path("docs/usage.md").read_text(encoding="utf-8")
    names = {name for p in get_command(app).commands["compare"].params for name in p.opts}
    assert {"--drafts", "--draft", "--hub"} <= names
    assert all(f"`{name}" in usage for name in ("--drafts", "--draft", "--hub"))
