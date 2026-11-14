"""`rigtune compare --expert`: stock rows, each with one setting an engine's own documentation recommends."""

from __future__ import annotations

import pytest

from rigtune.bench.references import reference_configs
from rigtune.bench.rows import ENGINE_DEFAULTS, expert_data, expert_rows
from rigtune.models import Config, GiB, ModelSpec
from rigtune.pipeline import prepare_and_plan, select
from tests.conftest import make_hw

# The four releases whose docs and source the rows were checked against.
READ_AT = {"vllm": {"v0.29.0", "v0.33.0"}, "sglang": {"v0.5.19", "v0.5.23"}}
LABELS = ["vllm-expert-batched-tokens", "vllm-expert-ngram", "vllm-expert-kv-fp8", "sglang-expert-ngram",
          "sglang-expert-eagle3", "sglang-expert-kv-fp8-e4m3", "sglang-expert-torch-compile"]


def test_every_setting_quotes_its_page_at_a_release_it_was_read_at():
    data = expert_data()
    assert [r["label"] for r in data["rows"]] == LABELS
    for row in data["rows"]:
        assert row["from"] == f"{row['backend']}-default"
        assert row["label"].startswith(f"{row['backend']}-expert-")
        assert isinstance(row["changes_numerics"], bool)
        assert set(row.get("requires", ())) <= {"cuda", "kv_dtype_offered", "eagle3_head"}
        for s in row["settings"]:
            name = s["field"].split(":", 1)[1] if s["field"].startswith("extra:") else s["field"]
            assert s["field"].startswith("extra:") or name in Config.model_fields
            assert name not in ("backend", "quant") and s["flag"] and s["cites"]
            for c in s["cites"]:
                assert c["quote"] and c["page"] and c["url"].startswith("https://github.com/")
                assert c["read_at"] and set(c["read_at"]) <= READ_AT[row["backend"]]
                assert any(f"/{tag}/" in c["url"] for tag in c["read_at"])
            if "applies_when_default_at_most" in s:
                assert (row["backend"], s["field"]) in ENGINE_DEFAULTS
            for k in ("on_cards_without_it", "fallback", "acts_up_to_users"):  # each quoted the same way
                for c in s.get(k, {}).get("cites", []):
                    assert c["quote"] and set(c["read_at"]) <= READ_AT[row["backend"]] and s[k]["why"]
    for c in data["considered"]:
        assert set(c["read_at"]) <= READ_AT[c["engine"]] and c["why_not"] and c["advice"]
    # changes_numerics marks exactly the rows that quantize the KV cache
    assert {r["label"] for r in data["rows"] if r["changes_numerics"]} == {"vllm-expert-kv-fp8",
                                                                         "sglang-expert-kv-fp8-e4m3"}


def _rows(hw, spec, force=None):
    candidates, reg = select(hw, spec, force=force)
    planned = prepare_and_plan(hw, spec, candidates, reg)
    return expert_rows(hw, planned.prepared, reg), reg, planned


@pytest.mark.usefixtures("no_network")
def test_on_a_4090_each_row_is_stock_with_its_one_documented_setting(hw_rtx4090, spec):
    rows, reg, planned = _rows(hw_rtx4090, spec)
    stock = reference_configs(hw_rtx4090, planned.prepared, reg)
    v, s = stock["vllm-default"], stock["sglang-default"]
    ngram = '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
    want = {
        "vllm-expert-batched-tokens": v.model_copy(update={"prefill_budget": 16384}),
        "vllm-expert-ngram": v.model_copy(update={"extra": {"speculative_config": ngram}}),
        "vllm-expert-kv-fp8": v.model_copy(update={"kv_dtype": "fp8"}),
        "sglang-expert-ngram": s.model_copy(update={"extra": {"speculative_algorithm": "NGRAM"}}),
        "sglang-expert-kv-fp8-e4m3": s.model_copy(update={"kv_dtype": "fp8_e4m3"}),
        "sglang-expert-torch-compile": s.model_copy(update={"extra": {"enable_torch_compile": True}}),
    }
    for label, cfg in want.items():
        assert rows[label].skipped is None and rows[label].config == cfg, label
        assert rows[label].kind == "expert"
        assert rows[label].source.startswith(f"{label.split('-')[0]}-default plus")
    # Llama 3.2 3B has no EAGLE-3 head rigtune knows of.
    assert rows["sglang-expert-eagle3"].skipped.startswith(
        "rigtune lists no EAGLE-3 head for meta-llama/Llama-3.2")
    assert rows["vllm-expert-kv-fp8"].changes_numerics and not rows["vllm-expert-ngram"].changes_numerics
    # The flag each setting names is the one rigtune's backend launches for it.
    data = {r["label"]: r for r in expert_data()["rows"]}
    for label, cfg in want.items():
        args = " ".join(reg[cfg.backend].launch_spec(cfg, planned.prepared[cfg.backend], 8000).args)
        for setting in data[label]["settings"]:
            assert setting["flag"].replace("'", "") in args, (label, setting["flag"], args)
    # The fixtures' engines report version 1.0, not one the pages were read at.
    assert rows["vllm-expert-ngram"].notes == [
        "vllm-expert-ngram: the page was read at vllm 0.29.0 and 0.33.0; this is 1.0"]


@pytest.mark.usefixtures("no_network")
def test_the_eagle3_row_uses_the_models_head_when_rigtune_knows_one(hw_rtx4090):
    rows, _, _ = _rows(hw_rtx4090, ModelSpec(hf_id="meta-llama/Llama-3.1-8B-Instruct"), force="sglang")
    row = rows["sglang-expert-eagle3"]
    assert row.skipped is None
    assert row.config.extra == {"speculative_algorithm": "EAGLE3",
                                "speculative_draft_model_path": "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"}


@pytest.mark.usefixtures("no_network")
def test_a_row_is_skipped_where_its_page_or_the_engine_says_it_does_not_apply(spec):
    a100 = make_hw("a100")  # 80 GB, but vLLM's own source says large budgets hurt an A100
    rows, _, _ = _rows(a100, spec, force="vllm")
    assert rows["vllm-expert-batched-tokens"].skipped.startswith(
        "not on an A100: vLLM's own source: \"Setting large `max_num_batched_tokens` for A100 reduces "
        "throughput")
    # Ampere without FlashInfer: rigtune runs no fp8 KV cache on vLLM there.
    assert rows["vllm-expert-kv-fp8"].skipped.startswith(
        "rigtune does not run vllm's fp8 KV cache on this card (compute capability 8.0; it runs: none)")
    big = make_hw("a100")
    big.gpus[0] = big.gpus[0].model_copy(update={"name": "NVIDIA B200", "vram_total_bytes": 180 * GiB,
                                                 "compute_capability": (10, 0)})
    rows, _, _ = _rows(big, spec, force="vllm")
    assert rows["vllm-expert-batched-tokens"].skipped == (
        "vllm's own default on this card, prefill_budget 16384, already meets \"For optimal throughput, we "
        "recommend setting `max_num_batched_tokens > 8192` especially for smaller models on large GPUs.\", "
        "so the row would not be what the page recommends")
    assert rows["vllm-expert-kv-fp8"].skipped is None
    h100 = make_hw("a100")  # 80 GB and not an A100: vLLM's default is 8192, not above the page's 8192
    h100.gpus[0] = h100.gpus[0].model_copy(update={"name": "NVIDIA H100 80GB HBM3"})
    rows, _, _ = _rows(h100, spec, force="vllm")
    assert rows["vllm-expert-batched-tokens"].config.prefill_budget == 16384


@pytest.mark.usefixtures("no_network")
def test_a_row_is_skipped_when_its_engine_is_not_compared_or_known_to_fail(hw_rtx4090, spec, monkeypatch):
    rows, _, _ = _rows(hw_rtx4090, spec, force="sglang")
    assert {r.skipped for label, r in rows.items() if label.startswith("vllm")} == {
        "vllm is not one of this comparison's engines"}
    candidates, reg = select(hw_rtx4090, spec, force="vllm")
    monkeypatch.setattr(type(reg["vllm"]), "known_failure", lambda self, cfg: "a reason")
    rows = expert_rows(hw_rtx4090, prepare_and_plan(hw_rtx4090, spec, candidates, reg).prepared, reg)
    assert rows["vllm-expert-ngram"].skipped == "rigtune expects it to fail: a reason"


def test_rows_files_are_checked_without_reading_the_expert_data(tmp_path, monkeypatch):
    import rigtune.bench.rows as rows_mod
    from rigtune.bench.rows import RowsFileError, read_rows_file

    monkeypatch.setattr(rows_mod, "expert_data", lambda: pytest.fail("read the expert data"))
    path = tmp_path / "r.json"
    path.write_text('{"rows": [{"label": "vllm-expert-ngram", "config": {"backend": "vllm", '
                    '"quant": "bf16"}}]}', encoding="utf-8")
    with pytest.raises(RowsFileError, match="reserved"):
        read_rows_file(path)


def _a40():
    hw = make_hw("a100")
    hw.gpus[0] = hw.gpus[0].model_copy(update={"name": "NVIDIA A40", "vram_total_bytes": 48 * GiB,
                                               "compute_capability": (8, 6)})
    return hw


@pytest.mark.usefixtures("no_network")
def test_on_an_a40_vllm_has_the_fp8_cache_it_runs_there_and_sglang_its_rows(spec, monkeypatch):
    """Ampere: rigtune runs vLLM's fp8 cache as fp8_e5m2 through FlashInfer (when FlashInfer is installed),
    the format the same page lists beside fp8; SGLang keeps the page's recommended fp8_e4m3, with fp8_e5m2
    to fall back on should it not start, and torch.compile says how many users it compiles for."""
    hw = _a40()
    candidates, reg = select(hw, spec)
    monkeypatch.setattr(type(reg["vllm"]), "kv_dtypes", lambda self, hw: ["fp8_e5m2"])  # FlashInfer found
    planned = prepare_and_plan(hw, spec, candidates, reg)
    rows = expert_rows(hw, planned.prepared, reg)
    stock = reference_configs(hw, planned.prepared, reg)
    kv = rows["vllm-expert-kv-fp8"]
    assert kv.skipped is None
    assert kv.config == stock["vllm-default"].model_copy(update={"kv_dtype": "fp8_e5m2"})
    assert kv.notes[0].startswith(
        "vllm-expert-kv-fp8: this card does not have vllm's fp8 KV cache as rigtune runs it, so the row uses "
        "fp8_e5m2: `--kv-cache-dtype fp8_e5m2`: \"`kv_cache_dtype=\"fp8_e5m2\"`: Supported on CUDA 11.8+\"")
    assert "--kv-cache-dtype fp8_e5m2" in " ".join(
        reg["vllm"].launch_spec(kv.config, planned.prepared["vllm"], 8000).args)
    s = stock["sglang-default"]
    ngram = s.model_copy(update={"extra": {"speculative_algorithm": "NGRAM"}})
    assert rows["sglang-expert-ngram"].config == ngram
    e4m3 = rows["sglang-expert-kv-fp8-e4m3"]
    assert e4m3.skipped is None and e4m3.config.kv_dtype == "fp8_e4m3"
    assert e4m3.fallback == e4m3.config.model_copy(update={"kv_dtype": "fp8_e5m2"})
    assert e4m3.fallback_note.startswith(
        "sglang-expert-kv-fp8-e4m3: fp8_e4m3 failed to start, so the row was measured with fp8_e5m2")
    tc = rows["sglang-expert-torch-compile"]
    assert tc.skipped is None and tc.acts_up_to_users == 32 and "--torch-compile-max-bs" in tc.acts_note
    assert rows["sglang-expert-eagle3"].skipped.startswith("rigtune lists no EAGLE-3 head")
    # Without FlashInfer the vLLM row is skipped, as on the A100 above.
    monkeypatch.setattr(type(reg["vllm"]), "kv_dtypes", lambda self, hw: [])
    assert expert_rows(hw, planned.prepared, reg)["vllm-expert-kv-fp8"].skipped.startswith(
        "rigtune does not run vllm's fp8 KV cache on this card")
