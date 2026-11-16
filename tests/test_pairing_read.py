import importlib
import json
from types import SimpleNamespace as NS

import pytest

from rigtune import pairing as p


def metadata(tokenizer=True):
    files = [NS(rfilename="config.json", size=100, lfs=None),
             NS(rfilename="model.safetensors", size=200, lfs=None)]
    if tokenizer:
        files.append(NS(rfilename="tokenizer.json", size=50, lfs=NS(sha256="a" * 64)))
    return NS(sha="b" * 40, gated=False, siblings=files, safetensors=NS(total=100))


def downloader(tmp_path):
    def download(repo, filename, **kw):
        assert kw["revision"] == "b" * 40
        f = tmp_path / filename
        f.write_text(json.dumps({"vocab_size": 2, "torch_dtype": "bfloat16"}
                               if filename == "config.json" else {}))
        return str(f)
    return download


@pytest.mark.parametrize("code,why", [(401, "gated"), (403, "gated"), (404, "not_found"), (429, "rate_limited")])
def test_http_failures_are_honest_and_not_retried(tmp_path, code, why):
    calls = []
    class Failure(Exception):
        response = NS(status_code=code)
    def info(*a, **kw):
        calls.append(1)
        raise Failure("private path must not escape")
    with pytest.raises(p.HubLookupError) as exc:
        p.read_files("org/model", info_fetcher=info, file_fetcher=downloader(tmp_path))
    assert exc.value.why == why
    assert len(calls) == 1
    assert "private path" not in str(exc.value)


def test_fetcher_exception_is_not_a_refusal(tmp_path):
    def info(*a, **kw):
        raise RuntimeError("surprise")
    with pytest.raises(p.HubLookupError, match="error"):
        p.read_files("org/model", info_fetcher=info, file_fetcher=downloader(tmp_path))


def test_listing_without_tokenizer_file_is_unread_not_equal(tmp_path):
    f = p.read_files("org/model", info_fetcher=lambda *a, **kw: metadata(False),
                     file_fetcher=downloader(tmp_path))
    assert f.tokenizer_sha256 is None and f.token_ids is None
    assert f.num_params == 100 and f.sha == "b" * 40


def test_tokenizer_downloaded_only_when_hashes_differ(tmp_path):
    a = p.PairFiles(repo="org/a", sha="b" * 40, tokenizer_sha256="a" * 64, tokenizer_bytes=50)
    b = a.model_copy(update={"repo": "org/b"})
    def forbidden(*a, **kw):
        raise AssertionError("unnecessary tokenizer download")
    p.read_tokenizers(a, b, file_fetcher=forbidden)
    assert a.token_ids is None
    a.tokenizer_sha256 = b.tokenizer_sha256 = None
    tok = tmp_path / "tokenizer.json"
    tok.write_text(json.dumps({"model": {"vocab": {"a": 0}}, "added_tokens": [{"content": "extra", "id": 1}]}))
    p.read_tokenizers(a, b, file_fetcher=lambda *a, **kw: str(tok))
    assert a.token_ids == b.token_ids == {"a": 0, "extra": 1}


def test_large_or_unlisted_tokenizer_not_downloaded():
    a = p.PairFiles(repo="org/a", sha="a" * 40, tokenizer_bytes=64_000_001)
    b = p.PairFiles(repo="org/b", sha="b" * 40)
    def forbidden(*a, **kw):
        raise AssertionError("oversized download")
    p.read_tokenizers(a, b, file_fetcher=forbidden)
    assert a.token_ids is None and b.token_ids is None


def test_local_directory_reads_the_same_fields(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 2, "torch_dtype": "bfloat16"}))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": 0, "b": 1}}}))
    (tmp_path / "model.safetensors").write_bytes(b"0123456789")
    f = p.read_files(str(tmp_path))
    assert f.config["vocab_size"] == 2 and f.size_bytes == 10
    assert f.token_ids == {"a": 0, "b": 1}
    assert f.repo.startswith("local/") and str(tmp_path) not in f.model_dump_json()
    before = f.sha
    (tmp_path / "model.safetensors").write_bytes(b"9876543210")
    assert p.read_files(str(tmp_path)).sha != before


@pytest.mark.parametrize("indexed", [True, False])
@pytest.mark.parametrize("source", ["local", "cache"])
def test_some_shards_of_several_leave_the_size_unknown_not_too_small(tmp_path, source, indexed):
    root = tmp_path / "models--org--draft" / "snapshots" / ("b" * 40)
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"vocab_size": 2, "torch_dtype": "bfloat16"}))
    (root / "model-00001-of-00003.safetensors").write_bytes(b"0123456789")
    if indexed:
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": 30},
            "weight_map": {f"w{i}": f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)}}))

    def read():
        if source == "local":
            return p.read_files(str(root))
        return p.read_files("org/draft", "cache", file_fetcher=lambda *a, **kw: str(root / "config.json"))

    partial = read()
    assert partial.size_bytes is None and partial.num_params is None
    # Nothing to plan memory from: the pair is not offered on a third of its weights.
    b = pair_files(partial)
    assert p.check_pair(b.model_copy(update={"repo": "org/target", "size_bytes": 80, "num_params": 40}), b,
                        "draft", 4096).status == "unknown"
    for i in (2, 3):
        (root / f"model-0000{i}-of-00003.safetensors").write_bytes(b"0123456789")
    assert read().size_bytes == 30


def pair_files(read):
    """What was read, with the tokenizer and architecture facts a complete check needs."""
    return read.model_copy(update={"config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                                              "vocab_size": 2, "max_position_embeddings": 8192,
                                              "torch_dtype": "bfloat16"}, "tokenizer_sha256": "a" * 64})


def test_no_network_at_import(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("network at import")
    monkeypatch.setattr("socket.socket", forbidden)
    importlib.reload(p)


def write_weights(path, counts=(5, 7)):
    """A valid small safetensors file, with mixed BF16 and F32 tensors."""
    import struct
    offset, header = 0, {}
    for i, count in enumerate(counts):
        width, dtype = (2, "BF16") if i % 2 == 0 else (4, "F32")
        end = offset + count * width
        header[f"weight{i}"] = {"dtype": dtype, "shape": [count], "data_offsets": [offset, end]}
        offset = end
    data = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(data)) + data + b"\0" * offset)


def test_local_counts_tensor_elements_not_file_bytes(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    write_weights(tmp_path / "model.safetensors")
    assert p.read_files(str(tmp_path)).num_params == 12


def test_incomplete_sharded_weights_leave_parameter_count_unread(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    write_weights(tmp_path / "model-00001-of-00002.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 2000},
        "weight_map": {"weight0": "model-00001-of-00002.safetensors", "weight1": "model-00002-of-00002.safetensors"}}))
    assert p.read_files(str(tmp_path)).num_params is None


def test_malformed_weight_header_is_not_a_known_parameter_count(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    (tmp_path / "model.safetensors").write_bytes(b"not a tensor file")
    assert p.read_files(str(tmp_path)).num_params is None
