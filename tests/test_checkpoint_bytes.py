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


# The two EAGLE3 heads rigtune lists ship a pytorch_model.bin only (listings read 21 Nov 2026).
LMSYS_HEAD = {".gitattributes": 1519, "README.md": 31, "config.json": 599, "pytorch_model.bin": 849768795}
ANGELSLIM_HEAD = {".gitattributes": 1519, "License_AngelSlim_model_and_dataset.txt": 42641, "README.md": 42865,
                  "config.json": 763, "pytorch_model.bin": 799493246}


def test_a_checkpoint_of_bin_files_only_is_sized_from_them_as_the_engines_load_them():
    from rigtune.hfconfig import listed_bytes

    def no_index():
        raise AssertionError("a .bin checkpoint has no safetensors index to read")
    assert listed_bytes(LMSYS_HEAD, no_index) == 849768795  # was None: SGLang never offered the head
    assert listed_bytes(ANGELSLIM_HEAD, no_index) == 799493246
    shards = {"pytorch_model-00001-of-00002.bin": 10, "pytorch_model-00002-of-00002.bin": 20,
              "training_args.bin": 5, "optimizer.bin": 7, "sub/pytorch_model.bin": 9}  # training files: not loaded
    assert listed_bytes(shards, no_index) == 30
    # Safetensors beside .bin files win, as both engines take the first pattern that matches.
    assert listed_bytes({**LMSYS_HEAD, "model.safetensors": 849000000}, no_index) == 849000000


def test_some_bin_shards_of_several_are_unknown_not_a_partial_sum(tmp_path):
    from rigtune.hfconfig import listed_bytes

    assert listed_bytes({"pytorch_model-00001-of-00002.bin": 10}, lambda: None) is None
    (tmp_path / "pytorch_model-00001-of-00002.bin").write_bytes(b"x" * 10)
    assert local_checkpoint_bytes(tmp_path) is None  # a shard still downloading
    (tmp_path / "pytorch_model-00002-of-00002.bin").write_bytes(b"x" * 20)
    (tmp_path / "training_args.bin").write_bytes(b"x" * 3)
    assert local_checkpoint_bytes(tmp_path) == 30


def test_whole_shard_sets_are_recognised_however_they_are_numbered():
    from rigtune.hfconfig import bin_complete

    assert bin_complete([f"m-{i}-of-10.bin" for i in range(1, 11)])  # unpadded
    assert bin_complete([f"pytorch_model-{i:05d}-of-00003.bin" for i in range(1, 4)] + ["pytorch_model.bin"])
    assert not bin_complete([f"m-{i}-of-10.bin" for i in range(1, 10)])
    assert not bin_complete(["a.bin"], index_files=["a.bin", "b.bin"])  # its index names a file still downloading


def test_a_local_bin_checkpoint_is_whole_only_when_its_index_s_files_are_there(tmp_path):
    (tmp_path / "weights-a.bin").write_bytes(b"x" * 10)
    (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps(
        {"weight_map": {"w1": "weights-a.bin", "w2": "weights-b.bin"}}))
    assert local_checkpoint_bytes(tmp_path) is None
    (tmp_path / "weights-b.bin").write_bytes(b"x" * 20)
    (tmp_path / "x_training_args.bin").write_bytes(b"x" * 5)  # the engines drop names that end so, too
    assert local_checkpoint_bytes(tmp_path) == 30
    (tmp_path / "model.safetensors").write_bytes(b"x" * 7)  # safetensors win, as in both engines
    assert local_checkpoint_bytes(tmp_path) == 7


def test_a_local_bin_checkpoint_never_counts_fewer_parameters_than_its_config_says(monkeypatch, tmp_path):
    """.bin files carry no header to count from: fp16 files under a float32 config would halve a count taken from their
    size over the config's dtype, and the planned weights with it."""
    from rigtune.hfconfig import arch_from_config

    estimate = arch_from_config(LLAMA_3B_CONFIG).num_params
    (tmp_path / "config.json").write_text(json.dumps({**LLAMA_3B_CONFIG, "torch_dtype": "float32"}))
    (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 1000)
    arch = load_arch(ModelSpec(hf_id=str(tmp_path)))
    assert arch.checkpoint_bytes == 1000 and arch.num_params == estimate


def test_a_local_bin_draft_is_identified_by_its_weights(tmp_path):
    from rigtune.pairing import read_files

    (tmp_path / "config.json").write_text(json.dumps(LLAMA_3B_CONFIG))
    (tmp_path / "pytorch_model.bin").write_bytes(b"a" * 100)
    first = read_files(str(tmp_path))
    assert first.size_bytes == 100
    (tmp_path / "pytorch_model.bin").write_bytes(b"b" * 100)  # edited in place
    assert read_files(str(tmp_path)).sha != first.sha


def test_an_eagle3_head_shipped_as_a_bin_is_sized_for_sglang(monkeypatch):
    from rigtune import speculative

    info = SimpleNamespace(sha="2" * 40,
                           siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in LMSYS_HEAD.items()])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: SimpleNamespace(model_info=lambda *a, **k: info))
    assert speculative.repo_bytes("lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B") == 849768795


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


def _safetensors(path, tensors):
    """A real safetensors file: {name: (dtype, shape)}, each tensor zero-filled."""
    import math
    import struct

    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def _local(tmp_path, dtype="BF16"):
    """A local checkpoint: 64 parameters of body and a 16-parameter MTP head, which the config's estimate omits."""
    (tmp_path / "config.json").write_text(json.dumps({**LLAMA_3B_CONFIG, "torch_dtype": dtype.lower()}))
    _safetensors(tmp_path / "model.safetensors", {"model.layers.0.w": (dtype, [8, 8]), "mtp.layers.0.w": (dtype, [4, 4])})
    return ModelSpec(hf_id=str(tmp_path))


def test_a_local_checkpoint_is_counted_from_its_files_mtp_head_included(tmp_path):
    arch = load_arch(_local(tmp_path))
    assert arch.num_params == 80 and arch.checkpoint_bytes == (tmp_path / "model.safetensors").stat().st_size


def test_offline_a_cached_snapshot_is_read_as_a_local_checkpoint_is(monkeypatch, tmp_path):
    # The Hub not answering (HF_HUB_OFFLINE): its count and listing are gone, the cached snapshot is what the engines
    # load. Counted from its headers like a local directory, not from the config's estimate (3 billion here).
    from huggingface_hub import constants

    monkeypatch.setattr("rigtune.hfconfig.hub_weights", lambda spec, token=None: (None, None, {}))
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--org--model"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text("c" * 40)
    snap = repo / "snapshots" / ("c" * 40)
    snap.mkdir(parents=True)
    _local(snap)
    monkeypatch.setattr("rigtune.hfconfig.fetch_config",
                        lambda spec, token=None: json.loads((snap / "config.json").read_text()))
    arch = load_arch(ModelSpec(hf_id="org/model"))
    assert arch.num_params == 80 and arch.checkpoint_bytes == (snap / "model.safetensors").stat().st_size
    # Nothing cached: the config's estimate, as before.
    monkeypatch.setattr("rigtune.hfconfig.fetch_config", lambda spec, token=None: LLAMA_3B_CONFIG)
    other = load_arch(ModelSpec(hf_id="org/other"))
    assert other.checkpoint_bytes is None and other.num_params > 10**9


def test_when_the_headers_cannot_be_counted_the_size_over_the_dtype_stands_in(tmp_path):
    spec = _local(tmp_path)
    (tmp_path / "extra.safetensors").write_bytes(b"not a safetensors file")  # the strict count gives up
    arch = load_arch(spec)
    assert arch.num_params == arch.checkpoint_bytes // 2 > 80


@pytest.mark.parametrize("files_read, added", [(True, 0), (False, 512)])
def test_vllm_adds_a_local_mtp_head_only_when_the_files_were_not_read(tmp_path, files_read, added):
    from rigtune.backends.vllm import vllm_mtp_extra_weights
    from rigtune.models import Config, PreparedModel

    arch = load_arch(_local(tmp_path))
    if not files_read:
        arch.checkpoint_bytes = None
    model = PreparedModel(spec=ModelSpec(hf_id=str(tmp_path)), backend="vllm", arch=arch,
                          spec_repos={"mtp": str(tmp_path)}, spec_bytes={"mtp": 512})
    assert vllm_mtp_extra_weights(Config(backend="vllm", quant="bf16", spec_decode="mtp:1"), model) == added


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
