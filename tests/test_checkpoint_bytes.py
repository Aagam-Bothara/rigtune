"""A checkpoint's weights are sized by the files the engines load, not by its parameter count.

The Hub's parameter count counts elements: an FP8 checkpoint holds one byte a parameter, and the Hub counts AWQ's
packed int4 values unpacked. The file listings below are the Hub's for openai/gpt-oss-20b (6cee5e81) and
Qwen/Qwen3-8B-FP8 (220b46e3), 21 Nov 2026.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import huggingface_hub
import pytest

from rigtune.hfconfig import INDEX, arch_from_config, load_arch, loaded_bytes, local_checkpoint_bytes
from rigtune.models import ModelSpec
from rigtune.quantized import hf_weight_options
from tests.conftest import LLAMA_3B_CONFIG

GPT_OSS = {"model-00000-of-00002.safetensors": 4792272488, "model-00001-of-00002.safetensors": 4798702184,
           "model-00002-of-00002.safetensors": 4170342232, "original/model.safetensors": 13761300984}
GPT_OSS_INDEX = sorted(n for n in GPT_OSS if "/" not in n)
QWEN3_FP8 = {"model-00001-of-00002.safetensors": 4953240624, "model-00002-of-00002.safetensors": 4483388160}
QWEN3_FP8_PARAMS = 8190735360  # the Hub's count


def test_the_index_leaves_out_a_second_copy_of_the_weights():
    assert loaded_bytes(GPT_OSS, GPT_OSS_INDEX) == 13761316904  # not 27.5 GB with original/


def test_without_an_index_only_top_level_files_count():
    assert loaded_bytes(GPT_OSS, None) == 13761316904
    assert loaded_bytes({"model.safetensors": 988097824}, None) == 988097824


@pytest.mark.parametrize("sizes", [{"model-00001-of-00002.safetensors": 4953240624},  # a shard not listed
                                   {**QWEN3_FP8, "model-00002-of-00002.safetensors": None},  # no size given
                                   {}])
def test_a_missing_file_or_size_is_unknown_not_a_partial_sum(sizes):
    assert loaded_bytes(sizes, sorted(QWEN3_FP8)) is None


def test_a_local_directory_is_sized_by_its_index(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "b.safetensors").write_bytes(b"x" * 20)
    (tmp_path / "consolidated.safetensors").write_bytes(b"x" * 30)  # Mistral's second copy beside the shards
    (tmp_path / INDEX).write_text(json.dumps({"weight_map": {"w1": "a.safetensors", "w2": "b.safetensors"}}))
    assert local_checkpoint_bytes(tmp_path) == 30
    (tmp_path / "b.safetensors").unlink()  # a shard still downloading
    assert local_checkpoint_bytes(tmp_path) is None


def _hub(monkeypatch, sizes, params, index_files=None, tmp_path=None):
    siblings = [SimpleNamespace(rfilename=n, size=s) for n, s in sizes.items()]
    if index_files is not None:
        siblings.append(SimpleNamespace(rfilename=INDEX, size=1000))
        path = tmp_path / INDEX
        path.write_text(json.dumps({"weight_map": {f"t{i}": f for i, f in enumerate(index_files)}}))
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))
    info = SimpleNamespace(sha="0" * 40, siblings=siblings, safetensors=SimpleNamespace(total=params))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: SimpleNamespace(model_info=lambda *a, **k: info))
    monkeypatch.setattr("rigtune.hfconfig.fetch_config", lambda spec, token=None: LLAMA_3B_CONFIG)


def test_load_arch_reads_both_from_the_hub(monkeypatch, tmp_path):
    _hub(monkeypatch, QWEN3_FP8, QWEN3_FP8_PARAMS, sorted(QWEN3_FP8), tmp_path)
    arch = load_arch(ModelSpec(hf_id="Qwen/Qwen3-8B-FP8"))
    assert (arch.num_params, arch.checkpoint_bytes) == (QWEN3_FP8_PARAMS, 9436628784)


def test_a_gated_index_leaves_the_size_unknown(monkeypatch, tmp_path):
    _hub(monkeypatch, QWEN3_FP8, QWEN3_FP8_PARAMS, sorted(QWEN3_FP8), tmp_path)

    def refused(*a, **k):
        raise PermissionError("gated")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refused)
    arch = load_arch(ModelSpec(hf_id="Qwen/Qwen3-8B-FP8"))
    assert (arch.num_params, arch.checkpoint_bytes) == (QWEN3_FP8_PARAMS, None)


def test_the_size_is_never_saved_in_a_profile():
    arch = arch_from_config(LLAMA_3B_CONFIG)
    arch.checkpoint_bytes = 123
    assert "checkpoint_bytes" not in arch.model_dump() and "checkpoint_bytes" not in arch.model_dump_json()


def _gpt_oss_info():
    siblings = [SimpleNamespace(rfilename=n, size=s, lfs=None) for n, s in GPT_OSS.items()]
    siblings.append(SimpleNamespace(rfilename=INDEX, size=36355, lfs=None))
    return SimpleNamespace(sha="b" * 40, gated=False, siblings=siblings, safetensors=SimpleNamespace(total=1))


def _gpt_oss_files(tmp_path):
    """A downloader for the gpt-oss listing: its index, and a minimal config."""
    def download(repo, filename, **kw):
        f = tmp_path / filename
        body = {"weight_map": {f"t{i}": n for i, n in enumerate(GPT_OSS_INDEX)}} if filename == INDEX else {}
        f.write_text(json.dumps(body))
        return str(f)
    return download


def test_a_draft_or_head_repository_is_sized_by_its_index(monkeypatch, tmp_path):
    from rigtune import speculative
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: SimpleNamespace(
        model_info=lambda *a, **k: _gpt_oss_info()))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _gpt_oss_files(tmp_path))
    assert speculative.repo_bytes("openai/gpt-oss-20b") == 13761316904


def test_a_prequantized_repository_is_sized_by_its_index(monkeypatch, tmp_path):
    from rigtune import quantized
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _gpt_oss_files(tmp_path))
    api = SimpleNamespace(model_info=lambda *a, **k: _gpt_oss_info())
    assert quantized._repo_size(api, "openai/gpt-oss-20b") == 13761316904


def test_a_pair_is_sized_by_its_index_and_an_unreadable_index_is_unread(tmp_path):
    from rigtune import pairing
    files = pairing.read_files("openai/gpt-oss-20b", info_fetcher=lambda *a, **k: _gpt_oss_info(),
                               file_fetcher=_gpt_oss_files(tmp_path))
    assert files.size_bytes == 13761316904

    def no_index(repo, filename, **kw):
        if filename == INDEX:
            raise PermissionError("gated")
        return _gpt_oss_files(tmp_path)(repo, filename, **kw)
    files = pairing.read_files("openai/gpt-oss-20b", info_fetcher=lambda *a, **k: _gpt_oss_info(),
                               file_fetcher=no_index)
    assert files.size_bytes is None and files.sha == "b" * 40  # the read stands; only its size is unread


def _weights(params, stored, wanted=("bf16", "fp8")):
    return hf_weight_options(ModelSpec(hf_id="m"), params, list(wanted),
                             lambda p: 1 if p == "fp8" else 2, stored)[0]


def test_an_fp8_checkpoint_is_planned_at_its_files_size():
    assert _weights(QWEN3_FP8_PARAMS, 9436628784) == {"bf16": 9436628784, "fp8": QWEN3_FP8_PARAMS}


def test_a_bf16_checkpoint_keeps_two_bytes_a_parameter():
    """Qwen/Qwen3.5-9B: 9,653,104,368 parameters in 19,306,216,416 bytes of tensors plus the files' headers."""
    assert _weights(9653104368, 19306216416 + 120000)["bf16"] == 2 * 9653104368


def test_a_float32_checkpoint_is_capped_at_two_bytes_a_parameter():
    assert _weights(1000, 4000)["bf16"] == 2000


def test_without_a_size_or_a_count():
    assert _weights(1000, None) == {"bf16": 2000, "fp8": 1000}  # as before
    assert _weights(0, 5000, ("bf16",)) == {"bf16": 5000}  # no count: the files say
