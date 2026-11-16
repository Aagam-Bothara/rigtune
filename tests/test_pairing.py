"""A vocabulary-size check alone admits incompatible embedding rows."""
import pytest

from rigtune import pairing as p


def pair(**updates):
    data = dict(repo="org/draft", sha="b" * 40, config={
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": 262144, "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16"}, tokenizer_sha256="a" * 64,
        size_bytes=2_000_000, num_params=1_000_000)
    data.update(updates)
    return p.PairFiles(**data)


def target(**updates):
    return pair(**dict(dict(repo="org/target", num_params=8_000_000), **updates))


def test_equal_hash_is_ok_and_names_the_hash():
    v = p.check_pair(target(), pair(), "draft", 4096)
    assert v.status == "ok"
    assert v.evidence["target_tokenizer_sha256"] == "a" * 64


def test_same_size_one_id_differs_is_refused():
    a = target(tokenizer_sha256="a" * 64, token_ids={"x": 0, "𡉺": 249788})
    b = pair(tokenizer_sha256="b" * 64, token_ids={"x": 0, "<glitch_pruned_249788>": 249788})
    assert "ids_differ" in p.check_pair(a, b, "draft", 4096).reasons


def test_missing_vocab_size_is_unknown_not_32000():
    a, b = target(), pair()
    a.config.pop("vocab_size")
    b.config.pop("vocab_size")
    assert p.check_pair(a, b, "draft", 4096).status == "unknown"


def test_equal_sizes_unread_maps_is_unknown():
    assert p.check_pair(target(tokenizer_sha256=None), pair(), "draft", 4096).status == "unknown"


@pytest.mark.parametrize("config", [None, {}])
def test_nothing_read_is_unknown_never_ok(config):
    assert p.check_pair(p.PairFiles(config=config), p.PairFiles(config=config), "draft", 1).status == "unknown"


def test_short_context_draft_refused():
    assert "context_too_short" in p.check_pair(target(), pair(), "draft", 16384).reasons


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_qwen3_5_sibling_refused_as_rewritten_head(engine):
    b = pair()
    b.config.update(architectures=["Qwen3_5ForConditionalGeneration"], model_type="qwen3_5")
    assert "rewritten_as_head" in p.check_pair(target(), b, "draft", 4096, engine=engine).reasons


@pytest.mark.parametrize("name", ["org/eagle-draft", "org/draft-eagle3", "org/dflash", "org/dspark"])
def test_name_that_changes_vllm_method_refused(name):
    assert "method_changing_name" in p.check_pair(target(), pair(repo=name), "draft", 4096).reasons


def test_over_cost_bound_refused():
    assert "over_cost_bound" in p.check_pair(target(), pair(num_params=2_000_001), "draft", 4096).reasons


def test_template_difference_reported_not_refused():
    v = p.check_pair(target(template_sha256="a"), pair(template_sha256="b"), "draft", 4096)
    assert v.status == "ok"
    assert v.evidence["template"] == "differs"


def test_vocab_in_text_config_is_read():
    a, b = target(), pair()
    a.config = {"text_config": a.config}
    b.config = {"text_config": b.config}
    assert p.check_pair(a, b, "draft", 4096).status == "ok"


def test_empty_token_maps_are_not_positive_evidence():
    assert p.check_pair(target(tokenizer_sha256=None, token_ids={}),
                        pair(tokenizer_sha256=None, token_ids={}), "draft", 4096).status == "unknown"
