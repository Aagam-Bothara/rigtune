"""Speculative decoding on SGLang: which methods the installed release has, read from its source."""

from __future__ import annotations

import subprocess
import sys

import pytest

import rigtune.backends.sglang as sg

# The lines that matter, as each release's argument parser has them.
SERVER_ARGS_0_5_2 = '''
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE"],
            help="Speculative algorithm.",
        )
'''
SERVER_ARGS_0_5_3 = '''
    speculative_ngram_min_match_window_size: int = 1
    speculative_ngram_max_bfs_breadth: int = 10
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"],
        )
'''
SERVER_ARGS_0_5_19 = '''
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            help="Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK",
        )
'''
SPECULATIVE_HOOK_0_5_19 = '''
    if server_args.speculative_algorithm == "NGRAM":
        server_args.speculative_eagle_topk = server_args.speculative_ngram_max_bfs_breadth
'''
SERVER_ARGS_0_4 = '''choices=["EAGLE", "EAGLE3", "NEXTN"]'''


def fake_sglang(root, server_args: str, hooks: str = ""):
    srt = root / "sglang" / "srt"
    srt.mkdir(parents=True)
    (root / "sglang" / "__init__.py").write_text("raise ImportError('a scan must not import sglang')\n")
    (srt / "server_args.py").write_text(server_args)
    if hooks:
        (srt / "arg_groups").mkdir()
        (srt / "arg_groups" / "speculative_hook.py").write_text(hooks)
    return root / "sglang"


@pytest.fixture
def fresh():
    sg.sglang_spec_methods.cache_clear()
    yield
    sg.sglang_spec_methods.cache_clear()


@pytest.mark.parametrize("server_args, hooks, expected", [
    (SERVER_ARGS_0_4, "", set()),  # before 0.5.2: not checked, so nothing
    (SERVER_ARGS_0_5_2, "", set()),  # a second model only where 0.5.19's behaviour was read
    (SERVER_ARGS_0_5_3, "", {"ngram"}),
    (SERVER_ARGS_0_5_19, SPECULATIVE_HOOK_0_5_19, {"eagle3", "draft", "ngram"}),
])
def test_the_methods_come_from_the_installed_source(monkeypatch, tmp_path, fresh, server_args, hooks, expected):
    fake_sglang(tmp_path, server_args, hooks)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    monkeypatch.setattr(sg, "sglang_python", lambda: sys.executable)
    assert sg.sglang_spec_methods() == expected
    assert "sglang" not in sys.modules  # found and read, never imported


def test_sglang_in_its_own_environment_is_located_by_its_python(monkeypatch, tmp_path, fresh):
    pkg = fake_sglang(tmp_path, SERVER_ARGS_0_5_19, SPECULATIVE_HOOK_0_5_19)
    calls = []

    def run(args, **kw):
        calls.append((args, kw.get("timeout")))
        return subprocess.CompletedProcess(args, 0, stdout=f"{pkg}\n", stderr="")

    monkeypatch.setattr(sg, "sglang_python", lambda: "/envs/sgl/bin/python")
    monkeypatch.setattr(sg.subprocess, "run", run)
    assert sg.sglang_spec_methods() == {"eagle3", "draft", "ngram"}
    assert calls[0][0][0] == "/envs/sgl/bin/python" and calls[0][1] == 60


def test_no_sglang_offers_nothing(monkeypatch, fresh):
    monkeypatch.setattr(sg, "sglang_python", lambda: None)
    assert sg.sglang_spec_methods() == frozenset()
    sg.sglang_spec_methods.cache_clear()
    monkeypatch.setattr(sg, "sglang_python", lambda: "/envs/sgl/bin/python")
    monkeypatch.setattr(sg.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 1, "", "boom"))
    assert sg.sglang_spec_methods() == frozenset()


# ---- offering it under --speculative all: n-gram, the EAGLE3 head and the draft model, with room for the second

HEAD, DRAFT = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B", "unsloth/Llama-3.2-1B-Instruct"
HEAD_BYTES, DRAFT_BYTES = 850_000_000, 2_471_645_608  # the head as listed on the Hub; Llama 3.2 1B's safetensors


@pytest.fixture
def llama8b():
    from rigtune.models import ArchInfo, ModelSpec, PreparedModel

    arch = ArchInfo(architecture="LlamaForCausalLM", num_layers=32, hidden_size=4096, num_attention_heads=32,
                    num_kv_heads=8, head_dim=128, vocab_size=128256, max_position_embeddings=131072,
                    num_params=8_030_261_248)
    return PreparedModel(spec=ModelSpec(hf_id="unsloth/Meta-Llama-3.1-8B-Instruct"), backend="sglang", arch=arch,
                         hf_path="unsloth/Meta-Llama-3.1-8B-Instruct",
                         weights_bytes={"bf16": 16_060_522_496, "fp8": 8_030_261_248},
                         spec_repos={"eagle3": HEAD, "draft": DRAFT}, spec_bytes={HEAD: HEAD_BYTES, DRAFT: DRAFT_BYTES})


@pytest.fixture
def offers(monkeypatch):
    def set_methods(*methods):
        monkeypatch.setattr(sg, "sglang_spec_methods", lambda: frozenset(methods))
    set_methods("ngram", "eagle3", "draft")
    return set_methods


def _sgl(**kw):
    from rigtune.models import Config

    return Config(**{"backend": "sglang", "quant": "fp8", "ctx": 8192, "batch": 16, "gpu_memory_utilization": 0.93,
                     **kw})


def _offered(cfg, model, hw=None):
    return [(c.spec_decode, c.gpu_memory_utilization) for c in sg.SglangBackend().spec_variants(cfg, model, hw, True)]


def test_nothing_is_offered_unless_asked_for(tmp_home, hw_rtx4090, llama8b, offers):
    assert sg.SglangBackend().spec_variants(_sgl(), llama8b, hw_rtx4090) == []  # --speculative on: as before


def test_variants_leave_the_second_model_room_on_a_24_gb_card(tmp_home, hw_rtx4090, hw_a100, llama8b, offers):
    assert _offered(_sgl(), llama8b, hw_rtx4090) == [
        ("ngram:12", 0.93), (f"eagle3:{HEAD}:3", 0.86), (f"draft:{DRAFT}:3", 0.79)]
    assert {g for _, g in _offered(_sgl(), llama8b, hw_a100)} == {0.93}  # 80 GB leaves room at 0.93
    assert _offered(_sgl(), llama8b) == [("ngram:12", 0.93)]  # without the machine, nothing with a second model


def test_variants_of_a_draft_leader_start_from_the_fraction_before_it(tmp_home, hw_rtx4090, llama8b, offers):
    leader = sg.SglangBackend().spec_variants(_sgl(), llama8b, hw_rtx4090, True)[2]
    assert (leader.gpu_memory_utilization, leader.gmu_before_spec) == (0.79, 0.93)
    assert _offered(leader, llama8b, hw_rtx4090) == [("ngram:12", 0.93), (f"eagle3:{HEAD}:3", 0.86)]


def test_a_variant_that_no_longer_fits_at_the_lower_fraction_is_dropped(tmp_home, hw_rtx4090, llama8b, offers):
    # bf16 at 4k fits at 0.93 and 0.86, not at the 0.79 the 1B draft leaves
    assert [s for s, _ in _offered(_sgl(quant="bf16", ctx=4096), llama8b, hw_rtx4090)] == [
        "ngram:12", f"eagle3:{HEAD}:3"]


def test_only_what_the_release_has_is_offered(tmp_home, hw_rtx4090, llama8b, offers):
    offers("ngram")  # 0.5.3 to before 0.5.19: n-gram only
    assert _offered(_sgl(), llama8b, hw_rtx4090) == [("ngram:12", 0.93)]
    offers()  # no SGLang, or one older than 0.5.2
    assert _offered(_sgl(), llama8b, hw_rtx4090) == []
    offers("ngram", "eagle3", "draft")
    bare = llama8b.model_copy(update={"spec_repos": {}, "spec_bytes": {}})  # no head or draft found
    assert _offered(_sgl(), bare, hw_rtx4090) == [("ngram:12", 0.93)]


def _flags(args):
    i = args.index("--speculative-algorithm")
    return args[i:i + (4 if args[i + 1] == "NGRAM" else 10)]


def test_launch_flags_for_each_method(llama8b):
    be = sg.SglangBackend()
    assert _flags(be.launch_spec(_sgl(spec_decode="ngram:12"), llama8b, 1).args) == [
        "--speculative-algorithm", "NGRAM", "--speculative-num-draft-tokens", "12"]
    assert _flags(be.launch_spec(_sgl(spec_decode=f"eagle3:{HEAD}:3"), llama8b, 1).args) == [
        "--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path", HEAD, "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]
    assert _flags(be.launch_spec(_sgl(spec_decode=f"draft:{DRAFT}:3"), llama8b, 1).args) == [
        "--speculative-algorithm", "STANDALONE", "--speculative-draft-model-path", DRAFT, "--speculative-num-steps",
        "3", "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"]
    assert "--dtype" not in be.launch_spec(_sgl(spec_decode=f"eagle3:{HEAD}:3"), llama8b, 1).args
    assert "--speculative-algorithm" not in be.launch_spec(_sgl(), llama8b, 1).args


def test_a_combination_cannot_raise_the_fraction_back_or_load_a_model_of_unknown_size(tmp_home, hw_rtx4090,
                                                                                         hw_a100, llama8b):
    from rigtune.backends import registry
    from rigtune.pipeline import PlanResult, combination_fits

    reg = registry()
    for hw, fits in ((hw_rtx4090, False), (hw_a100, True)):
        plan = PlanResult(hw=hw, spec=llama8b.spec, candidates=["sglang"], prepared={"sglang": llama8b})
        ok = combination_fits(hw, reg, plan)
        assert ok(_sgl(spec_decode=f"eagle3:{HEAD}:3")) is fits  # 0.93 with the head
        assert ok(_sgl(spec_decode=f"eagle3:{HEAD}:3", gpu_memory_utilization=0.86))
        assert ok(_sgl(spec_decode="ngram:12"))
        assert not ok(_sgl(spec_decode="eagle3:someone/else:3", gpu_memory_utilization=0.5))  # size unknown
    be = sg.SglangBackend()
    assert be.spec_fraction("ngram:12", llama8b, hw_rtx4090) is None  # no second model: no limit
    assert be.spec_fraction("eagle3:someone/else:3", llama8b, hw_rtx4090) == 0.0  # unknown size: no room known


def test_a_start_up_failure_with_a_second_model_steps_down_from_its_own_fraction(tmp_home, hw_rtx4090, llama8b):
    from rigtune.backends import registry
    from rigtune.calibrate.search import StagedSearch
    from rigtune.models import TrialMetrics, TrialResult
    from rigtune.pipeline import PlanResult, retry_floor_fn

    plan = PlanResult(hw=hw_rtx4090, spec=llama8b.spec, candidates=["sglang"], prepared={"sglang": llama8b})
    floor = retry_floor_fn(hw_rtx4090, registry(), plan)
    draft = _sgl(spec_decode=f"draft:{DRAFT}:3", gpu_memory_utilization=0.79, gmu_before_spec=0.93)
    assert floor(draft) == 0.69 and floor(_sgl()) is None and floor(_sgl(spec_decode="ngram:12")) is None
    oom = TrialResult(config=draft, stage="spec", metrics=TrialMetrics(), launched=False,
                      error="out of memory at start-up; failed to start (rc=1)")
    assert StagedSearch(objective="throughput", runner=None)._headroom_retry(draft, oom) is None  # fixed 0.85 floor
    s = StagedSearch(objective="throughput", runner=None, retry_floor=floor)
    assert s._headroom_retry(draft, oom).gpu_memory_utilization == 0.74


def test_without_the_speculation_an_ablation_gives_the_memory_back(tmp_home, hw_rtx4090, llama8b):
    from rigtune.bench.ablation import strategy_variants
    from rigtune.calibrate.workload import get_workload

    pick = _sgl(spec_decode=f"draft:{DRAFT}:3", gpu_memory_utilization=0.79, gmu_before_spec=0.93)
    out = strategy_variants(pick, sg.SglangBackend(), hw_rtx4090, llama8b, get_workload("chat"))
    (off,) = [v.config for v in out if v.label == "-spec"]
    assert (off.spec_decode, off.gpu_memory_utilization, off.gmu_before_spec) == (None, 0.93, None)


def test_second_models_are_looked_up_only_under_all_and_failures_are_noted(monkeypatch, tmp_home, hw_rtx4090,
                                                                           llama8b, offers, no_network):
    import rigtune.speculative as speculative
    from rigtune.backends import registry
    from rigtune.pipeline import prepare_and_plan

    monkeypatch.setattr(sg, "load_arch", lambda spec, token=None: llama8b.arch)
    monkeypatch.setattr(sg, "hf_weight_options", lambda *a, **k: (dict(llama8b.weights_bytes), {}))
    monkeypatch.setattr(speculative, "find_eagle3_head", lambda spec, arch: (HEAD, HEAD_BYTES))
    monkeypatch.setattr(speculative, "repo_bytes", {DRAFT: DRAFT_BYTES}.get)
    reg = registry()
    plan = prepare_and_plan(hw_rtx4090, llama8b.spec, ["sglang"], reg, speculative_all=True)
    pm = plan.prepared["sglang"]
    assert pm.spec_repos == {"eagle3": HEAD, "draft": DRAFT}
    assert pm.spec_bytes == {HEAD: HEAD_BYTES, DRAFT: DRAFT_BYTES} and plan.notes == []

    def offline(*a, **k):
        raise speculative.HubLookupError("could not read: no network")

    monkeypatch.setattr(speculative, "find_eagle3_head", offline)
    monkeypatch.setattr(speculative, "repo_bytes", offline)
    plan = prepare_and_plan(hw_rtx4090, llama8b.spec, ["sglang"], reg, speculative_all=True)
    assert plan.prepared["sglang"].spec_repos == {}
    assert len(plan.notes) == 2 and all("the Hub could not be asked" in n for n in plan.notes)
    monkeypatch.setattr(speculative, "find_eagle3_head", lambda *a, **k: pytest.fail("asked the Hub"))
    monkeypatch.setattr(speculative, "repo_bytes", lambda *a, **k: pytest.fail("asked the Hub"))
    assert prepare_and_plan(hw_rtx4090, llama8b.spec, ["sglang"], reg).notes == []  # --speculative on: no lookup


# What calibration measured on this fixture before --speculative all existed (rigtune 0.2.1), with an SGLang here
# that has every speculative method: the default search must still measure exactly this, in this order.
DEFAULT_TRIALS = [
    ('quant', 'vllm/bf16/ctx4096/b64/gmu0.95'),
    ('quant', 'vllm/bf16/ctx4096/b64/gmu0.90'),
    ('quant', 'vllm/fp8/ctx4096/b64/gmu0.95'),
    ('quant', 'vllm/fp8/ctx4096/b64/gmu0.90'),
    ('quant', 'sglang/bf16/ctx4096/b64/gmu0.93'),
    ('quant', 'sglang/fp8/ctx4096/b64/gmu0.93'),
    ('quant', 'llamacpp-cuda/Q4_K_M/ctx4096/b4/ngl29/nb512'),
    ('memory', 'sglang/fp8/ctx8192/b64/gmu0.93'),
    ('memory', 'vllm/fp8/ctx8192/b64/gmu0.95'),
    ('memory', 'vllm/fp8/ctx2048/b64/gmu0.95'),
    ('batch', 'sglang/fp8/ctx8192/b16/gmu0.93'),
    ('batch', 'vllm/fp8/ctx4096/b16/gmu0.90'),
    ('spec', 'vllm/fp8/ctx4096/b64/gmu0.90/sd=ngram:4'),
    ('prefill', 'vllm/fp8/ctx4096/b64/gmu0.90/pb2048'),
    ('prefill', 'vllm/fp8/ctx4096/b64/gmu0.90/pb8192'),
    ('prefill', 'vllm/fp8/ctx4096/b64/gmu0.90/pb16384'),
    ('kv', 'vllm/fp8/ctx4096/b64/gmu0.90/kvfp8'),
    ('kv', 'vllm/fp8/ctx4096/b256/gmu0.90/kvfp8'),
    ('combine', 'vllm/fp8/ctx4096/b256/gmu0.90/pb2048/kvfp8'),
    ('combine', 'vllm/fp8/ctx4096/b256/gmu0.90/pb8192/kvfp8'),
    ('combine', 'vllm/fp8/ctx4096/b64/gmu0.90/pb2048/kvfp8'),
    ('combine', 'vllm/fp8/ctx4096/b64/gmu0.90/pb8192/kvfp8'),
    ('prefill', 'vllm/fp8/ctx4096/b64/gmu0.90/pb2048/sd=ngram:4'),
    ('prefill', 'vllm/fp8/ctx4096/b64/gmu0.90/pb8192/sd=ngram:4'),
    ('prefill', 'vllm/fp8/ctx4096/b64/gmu0.90/pb16384/sd=ngram:4'),
    ('kv', 'vllm/fp8/ctx4096/b64/gmu0.90/kvfp8/sd=ngram:4'),
    ('kv', 'vllm/fp8/ctx4096/b256/gmu0.90/kvfp8/sd=ngram:4'),
    ('combine', 'vllm/fp8/ctx4096/b256/gmu0.90/pb2048/kvfp8/sd=ngram:4'),
    ('combine', 'vllm/fp8/ctx4096/b256/gmu0.90/pb8192/kvfp8/sd=ngram:4'),
    ('combine', 'vllm/fp8/ctx4096/b64/gmu0.90/pb2048/kvfp8/sd=ngram:4'),
    ('combine', 'vllm/fp8/ctx4096/b64/gmu0.90/pb8192/kvfp8/sd=ngram:4'),
    ('confirm', 'vllm/fp8/ctx4096/b256/gmu0.90/kvfp8'),
    ('confirm', 'vllm/fp8/ctx4096/b256/gmu0.90/pb2048/kvfp8'),
    ('confirm', 'vllm/fp8/ctx4096/b256/gmu0.90/pb8192/kvfp8'),
]


def test_the_default_search_measures_what_it_did_before(hw_rtx4090, spec, no_network, tmp_home, offers):
    from rigtune.pipeline import SearchOptions, calibrate, prepare_and_plan, select
    from tests.test_objectives_and_search import FakeRunner

    candidates, reg = select(hw_rtx4090, spec)
    plan = prepare_and_plan(hw_rtx4090, spec, candidates, reg)
    profile = calibrate(hw_rtx4090, spec, "throughput", plan, reg, runner=FakeRunner())
    assert [(t.stage, t.config.key()) for t in profile.calibration_table] == DEFAULT_TRIALS
    assert profile.options == {}
    # the opt-in does change it, and says so in the profile's options
    opts = SearchOptions(speculative_all=True)
    plan = prepare_and_plan(hw_rtx4090, spec, candidates, reg, speculative_all=True)
    profile = calibrate(hw_rtx4090, spec, "throughput", plan, reg, runner=FakeRunner(), options=opts)
    assert [(t.stage, t.config.key()) for t in profile.calibration_table] != DEFAULT_TRIALS
    assert profile.options == {"spec": "all"}


def test_speculative_all_is_a_value_of_the_existing_option():
    import typer

    from rigtune.cli import _on_off_all, _opts

    assert _opts("auto", "on", "all", "on").speculative_all and _opts("auto", "on", "all", "on").speculative
    assert not _opts("auto", "on", "on", "on").speculative_all
    assert _opts("auto", "on", "off", "on").key() == {"spec": "off"}
    with pytest.raises(typer.BadParameter):
        _on_off_all("some")
