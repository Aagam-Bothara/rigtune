"""DFlash and DSpark drafters on SGLang (C7): offered where SGLang 0.5.23's flags are, launched at the k vLLM drafts,
planned with their weights and pool inside the fraction and their logits buffer outside it."""

from __future__ import annotations

import pytest

import rigtune.backends.sglang as sg
from rigtune import headsearch, speculative
from rigtune.memory import estimate
from rigtune.models import Config
from tests.test_block_drafters import DFLASH, DSPARK, _config, _qwen3_8b

# SGLang 0.5.23's arg_groups/fields/spec.py, the lines that declare the two flags (70, 78).
FIELDS_0_5_23 = '''
    speculative_dflash_block_size: A[
        Optional[int], Field(cli_flags=["--speculative-dflash-block-size"])]
    speculative_dspark_block_size: A[
        Optional[int], Field(cli_flags=["--speculative-dspark-block-size"])]
'''
# 0.5.19 named both algorithms, and its server_args.py declared both flags (they came with the drafters), but its draft
# memory was never read.
SERVER_ARGS_0_5_19 = ('help="Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK"\n'
                      "    speculative_dflash_block_size: Optional[int] = None\n"
                      "    speculative_dspark_block_size: Optional[int] = None\n")


def test_the_flags_offer_the_drafters_only_where_their_memory_was_read():
    assert sg.block_drafters_in(FIELDS_0_5_23) == {"dflash", "dspark"}
    assert sg.block_drafters_in(SERVER_ARGS_0_5_19, draft_pool_read=False) == frozenset()
    assert sg.block_drafters_in("", draft_pool_read=True) == frozenset()


def test_the_fields_directory_is_read(tmp_path):
    pkg = tmp_path / "sglang"
    (pkg / "srt" / "arg_groups" / "fields").mkdir(parents=True)
    (pkg / "srt" / "server_args.py").write_text("")
    (pkg / "srt" / "arg_groups" / "fields" / "spec.py").write_text(FIELDS_0_5_23)
    assert sg.block_drafters_in(sg._server_args_source(pkg)) == {"dflash", "dspark"}


@pytest.mark.parametrize("version, expected", [("0.5.19", frozenset()), ("0.5.23", {"dflash", "dspark"})])
def test_each_release_s_source_decides(tmp_path, monkeypatch, version, expected):
    from tests import engine_trees

    pkg = engine_trees.build("sglang", version, tmp_path)
    if version == "0.5.19":  # the flags it has, without the draft pool rigtune read
        (pkg / "srt" / "server_args.py").write_text(SERVER_ARGS_0_5_19)
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    monkeypatch.setattr(sg, "_sglang_package_dir", lambda python: pkg)
    sg.sglang_block_drafters.cache_clear()
    try:
        assert sg.sglang_block_drafters() == expected
    finally:
        sg.sglang_block_drafters.cache_clear()


# DFlash's sampling rehearsal as each source has it: none at 0.5.19; 0.5.23's fresh matrix (dflash_worker_v2.py lines
# 799, 825); SGLang PR #52548's rehearsal on the shared logits buffer, which falls back to a fresh matrix under a
# deferred TP gather or DP attention, so it counts as 0.5.23's.
WORKER_0_5_19 = "class DFlashWorkerV2(BaseSpecWorker):\n    def forward_batch_generation(self, batch):\n        ...\n"
WORKER_0_5_23 = ("    def prewarm_sampling(self) -> SamplingPrewarmResult:\n"
                 "        target_logits = torch.zeros(\n"
                 "            (bs * block_size, vocab_size), dtype=torch.float32, device=device\n")
WORKER_PR_52548 = WORKER_0_5_23 + "        shared_output = self._verify_logits_shared_output(rows)\n"


def test_the_rehearsal_s_matrices_are_read_from_the_worker_source():
    assert sg.rehearsal_matrices_in(WORKER_0_5_19) == 0
    assert sg.rehearsal_matrices_in(WORKER_0_5_23) == 4
    assert sg.rehearsal_matrices_in(WORKER_PR_52548) == 4
    assert sg.rehearsal_matrices_in(None) == 0


@pytest.mark.parametrize("source, expected", [(WORKER_0_5_23, 4), (None, 0)])
def test_the_installed_worker_is_read(tmp_path, monkeypatch, source, expected):
    pkg = tmp_path / "sglang"
    (pkg / "srt" / "speculative").mkdir(parents=True)
    if source is not None:
        (pkg / "srt" / "speculative" / "dflash_worker_v2.py").write_text(source)
    monkeypatch.setattr(sg, "sglang_python", lambda: "python")
    monkeypatch.setattr(sg, "_sglang_package_dir", lambda python: pkg)
    sg.sglang_rehearsal_matrices.cache_clear()
    try:
        assert sg.sglang_rehearsal_matrices() == expected
    finally:
        sg.sglang_rehearsal_matrices.cache_clear()


def _model():
    pm = _qwen3_8b().model_copy(update={"backend": "sglang"})
    pm.spec_repos.update({"dflash": DFLASH, "dspark": DSPARK})
    pm.spec_bytes.update({DFLASH: 2_100_000_000, DSPARK: 2_200_000_000})
    pm.spec_revisions.update({DFLASH: "d" * 40, DSPARK: "e" * 40})
    for repo, kind in ((DFLASH, "dflash"), (DSPARK, "dspark")):
        k, lookahead = speculative.drafter_tokens(_config(repo), kind)
        pm.spec_facts[repo] = {"k": k, "lookahead": lookahead,
                               "kv_elements": speculative.drafter_kv_elements(_config(repo)), "needs_v2": False}
    return pm


def _cfg(**kw):
    return Config(**{"backend": "sglang", "quant": "bf16", "ctx": 4096, "batch": 64, "gpu_memory_utilization": 0.93,
                     **kw})


def _flags(args):
    i = args.index("--speculative-algorithm")
    return args[i:]


def test_each_launches_at_the_tokens_it_drafts_unquantized_at_its_checked_commit():
    be, pm = sg.SglangBackend(), _model()
    dflash = be.launch_spec(_cfg(quant="fp8", spec_decode=f"dflash:{DFLASH}:15"), pm, 1)
    assert _flags(dflash.args)[:10] == [
        "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", DFLASH,
        "--speculative-num-draft-tokens", "16",  # the last accepted token and 15 drafts: its block of 16
        "--speculative-draft-model-revision", "d" * 40, "--speculative-draft-model-quantization", "unquant"]
    dspark = be.launch_spec(_cfg(spec_decode=f"dspark:{DSPARK}:7"), pm, 1)
    assert _flags(dspark.args)[:6] == [
        "--speculative-algorithm", "DSPARK", "--speculative-draft-model-path", DSPARK,
        "--speculative-dspark-block-size", "7"]  # gamma; SGLang drafts gamma + 1 tokens a step
    assert "--speculative-num-steps" not in dspark.args  # SGLang forces one step and top-k 1
    # SGLang checks a drafter's context against the target's, as it does a head's (C1).
    assert dflash.env.get(sg.LONGER_CONTEXT_VAR) == "1" and dspark.env.get(sg.LONGER_CONTEXT_VAR) == "1"


@pytest.fixture
def sglang_offers(monkeypatch):
    monkeypatch.setattr(sg, "sglang_spec_methods", lambda: frozenset({"ngram", "eagle3", "draft"}))
    monkeypatch.setattr(sg, "sglang_block_drafters", lambda: frozenset({"dflash", "dspark"}))
    # The rehearsal an installed SGLang would add is tested on its own; a test machine with SGLang must not see it.
    monkeypatch.setattr(sg, "sglang_rehearsal_matrices", lambda: 0)


def _offered(cfg, pm, hw):
    return {c.spec_decode: c for c in sg.SglangBackend().spec_variants(cfg, pm, hw, True)}


def test_both_are_offered_at_the_k_vllm_drafts(tmp_home, hw_a100, sglang_offers):
    got = _offered(_cfg(), _model(), hw_a100)
    assert {f"dflash:{DFLASH}:15", f"dspark:{DSPARK}:7"} <= set(got)
    assert sg.SglangBackend().spec_variants(_cfg(), _model(), hw_a100) == []  # --speculative on: as before


def test_the_fraction_leaves_the_logits_buffer_room_not_the_weights(tmp_home, hw_a100, hw_rtx4090, sglang_offers):
    be, pm = sg.SglangBackend(), _model()
    spec = f"dflash:{DFLASH}:15"
    assert sg.sglang_logits_bytes(spec, pm, 48) == 48 * 16 * 151936 * 4  # 467 MB at SGLang's default 48 requests
    vram = hw_rtx4090.gpu.vram_total_bytes
    outside = sg.sglang_logits_bytes(spec, pm, 64) + be.memory_model(hw_rtx4090).runtime_workspace + \
        sg.MIN_SAFETY_MARGIN
    assert be.spec_fraction(spec, pm, hw_rtx4090, 64) == int((1 - outside / vram) * 100) / 100
    assert be.spec_fraction(spec, pm, hw_rtx4090, 256) < be.spec_fraction(spec, pm, hw_rtx4090, 16)
    # The drafter's 2.1 GB are planned inside the fraction, not asked for outside it, as an EAGLE3 head's are.
    weights_outside = int((1 - (2_100_000_000 + outside) / vram) * 100) / 100
    assert be.spec_fraction(spec, pm, hw_rtx4090, 64) > weights_outside + 0.05
    # A buffer that cannot be sized leaves no fraction known to fit.
    assert be.spec_fraction(spec, pm, hw_rtx4090, None) == 0.0
    unknown_vocab = pm.model_copy(update={"arch": pm.arch.model_copy(update={"vocab_size": None})})
    assert be.spec_fraction(spec, unknown_vocab, hw_rtx4090, 64) == 0.0


def test_its_weights_and_pool_are_planned_inside_the_fraction(tmp_home, hw_a100, hw_rtx4090, sglang_offers):
    be, pm = sg.SglangBackend(), _model()
    mm = be.memory_model(hw_a100)
    plain, drafting = (estimate(hw_a100, pm, _cfg(spec_decode=s), mm) for s in (None, f"dflash:{DFLASH}:15"))
    assert drafting.weights - plain.weights == 2_100_000_000
    assert drafting.kv_cache > plain.kv_cache  # its own pool, a slot per target token
    # Qwen3-8B in bf16 at 4k x 16 fits a 24 GB card at 0.93; with the drafter's 2.1 GB and pool, at the 0.88 its
    # logits buffer leaves, it does not, so it is not offered there.
    small = _cfg(batch=16)
    assert estimate(hw_rtx4090, pm, small, be.memory_model(hw_rtx4090)).feasible
    assert f"dflash:{DFLASH}:15" not in _offered(small, pm, hw_rtx4090)


def test_not_offered_where_this_sglang_lacks_the_flags_or_a_second_model(tmp_home, hw_a100, monkeypatch):
    monkeypatch.setattr(sg, "sglang_spec_methods", lambda: frozenset({"ngram", "eagle3", "draft"}))
    monkeypatch.setattr(sg, "sglang_block_drafters", lambda: frozenset())
    assert not any(s.startswith(("dflash", "dspark")) for s in _offered(_cfg(), _model(), hw_a100))
    monkeypatch.setattr(sg, "sglang_block_drafters", lambda: frozenset({"dflash", "dspark"}))
    monkeypatch.setattr(sg, "sglang_spec_methods", lambda: frozenset({"ngram"}))  # second models not read here
    assert not any(s.startswith(("dflash", "dspark")) for s in _offered(_cfg(), _model(), hw_a100))


@pytest.fixture
def found_for_sglang(monkeypatch, sglang_offers):
    def kept(target_id, revision, seeds=(), **kw):
        cands = [headsearch.Candidate(repo=r, downloads=1, config=_config(r), files={"model.safetensors": size},
                                      size_bytes=size, sha=sha)
                 for r, size, sha in ((DFLASH, 2_100_000_000, "d" * 40), (DSPARK, 2_200_000_000, "e" * 40))]
        return headsearch.Search(target=target_id, revision=revision, day="test", aliases=[target_id],
                                 candidates=cands), []

    monkeypatch.setattr(headsearch, "kept_or_search", kept)
    monkeypatch.setattr(speculative, "draft_for", lambda hf_id: None)
    monkeypatch.setattr(sg.SglangBackend, "_own_head_notes", lambda self, prepared: [])


def test_sglang_records_the_drafters_it_loads(found_for_sglang):
    pm = _qwen3_8b().model_copy(update={"backend": "sglang"})
    notes = sg.SglangBackend().find_second_models(pm)
    assert pm.spec_repos == {"dflash": DFLASH, "dspark": DSPARK}
    assert pm.spec_facts[DFLASH]["k"] == 15 and pm.spec_facts[DSPARK]["k"] == 7
    assert pm.spec_revisions == {DFLASH: "d" * 40, DSPARK: "e" * 40}
    assert any(n.startswith(f"sglang: dflash drafter {DFLASH} found by a Hub search") for n in notes)


def test_a_drafter_this_sglang_refused_for_the_model_gives_way(found_for_sglang, hw_a100, monkeypatch):
    monkeypatch.setattr(sg.SglangBackend, "release", lambda self, hw: "sglang 0.5.23")
    headsearch.refuse(DSPARK, "e" * 40, "sglang 0.5.23", "requires markov_rank > 0", "Qwen/Qwen3-8B")
    pm = _qwen3_8b().model_copy(update={"backend": "sglang"})
    notes = sg.SglangBackend().find_second_models(pm, hw_a100)
    assert pm.spec_repos == {"dflash": DFLASH}
    assert any("not offered: sglang 0.5.23 refused to start with it" in n for n in notes)


def test_a_drafter_of_unknown_size_is_not_recorded(found_for_sglang, monkeypatch):
    real = headsearch.kept_or_search

    def unsized(*a, **kw):
        found, notes = real(*a, **kw)
        for c in found.candidates:
            c.size_bytes = None
        return found, notes

    monkeypatch.setattr(headsearch, "kept_or_search", unsized)
    pm = _qwen3_8b().model_copy(update={"backend": "sglang"})
    notes = sg.SglangBackend().find_second_models(pm)
    assert pm.spec_repos == {} and any("its size could not be read" in n for n in notes)


def test_dflash_s_fraction_leaves_its_start_up_rehearsal_room(tmp_home, hw_a100, sglang_offers, monkeypatch):
    """On an A40 or A6000 (46,068 MiB) at 64 requests the cap falls from 0.93 to 0.88 with the rehearsal's 4 matrices
    (2.37 GiB logged on 0.5.23). SGLang main still failed at 0.88 there and started at 0.841: the rest is graph memory
    beyond the planned workspace, which a start's own log measures."""
    be, pm = sg.SglangBackend(), _model()
    gpu = hw_a100.gpus[0].model_copy(update={"vram_total_bytes": 46068 << 20})
    hw = hw_a100.model_copy(update={"gpus": [gpu]})
    dflash, dspark = f"dflash:{DFLASH}:15", f"dspark:{DSPARK}:7"
    monkeypatch.setattr(sg, "sglang_rehearsal_matrices", lambda: 0)
    before = be.spec_fraction(dflash, pm, hw, 64)
    dspark_before = be.spec_fraction(dspark, pm, hw, 64)
    monkeypatch.setattr(sg, "sglang_rehearsal_matrices", lambda: 4)
    assert sg.sglang_rehearsal_bytes(dflash, pm, 64) == 4 * 64 * 16 * 151936 * 4  # 2.32 GiB
    assert (before, be.spec_fraction(dflash, pm, hw, 64)) == (0.93, 0.88)
    # DSpark has no rehearsal; a spec without a drafter neither.
    assert sg.sglang_rehearsal_bytes(dspark, pm, 64) == 0 and sg.sglang_rehearsal_bytes(None, pm, 64) == 0
    assert be.spec_fraction(dspark, pm, hw, 64) == dspark_before


def test_a_larger_batch_cannot_keep_a_fraction_the_buffer_no_longer_fits(tmp_home, hw_rtx4090, sglang_offers):
    """The KV stage steps the batch up at the leader's fraction: it may not keep one the drafter's logits buffer, which
    grows with the batch, no longer leaves room for (nor may a combination: pipeline.combination_fits passes the
    batch the same way)."""
    from rigtune.pipeline import kv_variants_fn

    be, pm = sg.SglangBackend(), _model()
    spec = f"dflash:{DFLASH}:15"
    at16 = be.spec_fraction(spec, pm, hw_rtx4090, 16)
    assert at16 > be.spec_fraction(spec, pm, hw_rtx4090, 64)
    leader = _cfg(quant="fp8", batch=16, gpu_memory_utilization=at16, spec_decode=spec)

    class Plan:
        prepared = {"sglang": pm}
        errors = {}

    assert [v for v in kv_variants_fn(hw_rtx4090, {"sglang": be}, Plan())(leader) if v.batch > 16] == []
