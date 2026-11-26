"""A seed (rigtune.seed) in calibration: built into a Config here, measured first, improved from, and reported."""

from __future__ import annotations

from typing import List

import pytest

from rigtune.backends import get_backend
from rigtune.backends.vllm import vllm_default_max_num_seqs
from rigtune.calibrate.search import StagedSearch
from rigtune.memory import plan
from rigtune.models import Config, PreparedModel
from rigtune.seed import parse_command, seed_config
from tests.test_objectives_and_search import FakeRunner

GiB = 1 << 30


def _gpu(hw, name, gib, cc):
    gpu = hw.gpus[0].model_copy(update={"name": name, "vram_total_bytes": int(gib * GiB),
                                        "vram_free_bytes": int((gib - 1) * GiB), "compute_capability": cc})
    return hw.model_copy(update={"gpus": [gpu]})


@pytest.mark.parametrize("name, gib, cc, seqs", [
    ("NVIDIA H100 PCIe", 79.18, (9, 0), 1024),  # session 2's card: the recipe ran 1,024 requests at once
    ("NVIDIA H100 NVL", 93.6, (9, 0), 1024),
    ("NVIDIA A100-SXM4-80GB", 80, (8, 0), 256),  # vLLM keeps the A100 on the small tier
    ("NVIDIA A40", 44.4, (8, 6), 256),
    ("NVIDIA B200", 179, (10, 0), 1024),
    ("NVIDIA GB10", 119, (12, 1), 256),  # DGX Spark: the memory, not the compute
])
def test_vllm_s_default_batch_is_its_own_per_gpu_rule(hw_a100, name, gib, cc, seqs):
    assert vllm_default_max_num_seqs(_gpu(hw_a100, name, gib, cc)) == seqs


def test_a_seed_leaves_to_the_engine_what_it_does_not_say_and_says_so(hw_a100, prepared_vllm):
    h100 = _gpu(hw_a100, "NVIDIA H100 PCIe", 79.18, (9, 0))
    seed = parse_command("vllm serve meta-llama/Llama-3.2-3B-Instruct --reasoning-parser x "
                         "--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,"
                         "\"prompt_lookup_min\":2,\"prompt_lookup_max\":4}'")
    cfg, notes = seed_config(seed, h100, prepared_vllm, get_backend("vllm").precisions(h100))
    assert (cfg.ctx, cfg.batch, cfg.gpu_memory_utilization) == (prepared_vllm.arch.max_position_embeddings, 1024, 0.90)
    assert (cfg.quant, cfg.spec_decode, cfg.extra) == ("bf16", "ngram:4", {"reasoning_parser": "x"})
    assert any("vLLM's default on this GPU, 1,024" in n for n in notes)
    # What the command says is kept as said.
    said = parse_command("vllm serve m --max-num-seqs 32 --max-model-len 8192 --gpu-memory-utilization 0.8 "
                         "--quantization fp8")
    cfg, notes = seed_config(said, h100, prepared_vllm, ["bf16", "fp8"])
    assert (cfg.batch, cfg.ctx, cfg.gpu_memory_utilization, cfg.quant) == (32, 8192, 0.8, "fp8") and notes == []


def test_seed_for_refuses_another_model_or_an_engine_not_calibrated(hw_a100, prepared_vllm):
    from rigtune.pipeline import seed_for

    class Plan:
        prepared = {"vllm": prepared_vllm}

    reg = {"vllm": get_backend("vllm")}
    spec = prepared_vllm.spec
    cfg, notes = seed_for(hw_a100, spec, Plan(), reg, f"vllm serve {spec.hf_id} --max-num-seqs 16")
    assert cfg.batch == 16 and any("seed measured as" in n for n in notes)
    assert seed_for(hw_a100, spec, Plan(), reg, None) == (None, [])
    for command, why in [("vllm serve other/model", "this calibration is of"),
                         (f"python -m sglang.launch_server --model-path {spec.hf_id}", "does not"),
                         (f"vllm serve {spec.hf_id} --revision abc", "revision")]:
        with pytest.raises(ValueError, match=why):
            seed_for(hw_a100, spec, Plan(), reg, command)


def _feasible(hw, prepared: PreparedModel) -> List[Config]:
    be = get_backend("vllm")
    return [c for c, _ in plan(hw, prepared, be.candidate_configs(hw, prepared), be.memory_model(hw))]


def test_the_seed_is_measured_after_stage_1_and_wins_when_nothing_beats_it(hw_a100, prepared_vllm):
    # FakeRunner's speed grows with the batch: a seed at batch 512, outside the grid (16/64/256), beats the grid.
    seed = Config(backend="vllm", quant="fp8", ctx=4096, batch=512, gpu_memory_utilization=0.9)
    runner = FakeRunner()
    search = StagedSearch(objective="throughput", runner=runner, seed=seed)
    winner, notes = search.run(_feasible(hw_a100, prepared_vllm))
    # Right after the precision stage (whose own trials an accuracy gate grades), before any other stage.
    first_seed = next(i for i, r in enumerate(search.results) if r.stage == "seed")
    assert {r.stage for r in search.results[:first_seed]} == {"quant"}
    assert runner.calls[first_seed] == seed.key()
    assert winner.config.key() == seed.key()
    assert any("kept as the pick" in n for n in notes)
    # The grid's own stages ran as without a seed: the memory stage was not anchored to the seed's batch.
    assert any(r.stage == "memory" and r.config.batch != 512 for r in search.results)


def test_a_pick_better_than_the_seed_says_by_how_much(hw_a100, prepared_vllm):
    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=16, gpu_memory_utilization=0.9)
    search = StagedSearch(objective="throughput", runner=FakeRunner(), seed=seed)
    winner, notes = search.run(_feasible(hw_a100, prepared_vllm))
    assert winner.config.key() != seed.key()
    note = next(n for n in notes if n.startswith("seed "))
    assert seed.key() in note and winner.config.key() in note and "+" in note


def test_a_seed_that_does_not_start_leaves_the_rest_to_pick_and_says_why(hw_a100, prepared_vllm):
    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=1024, gpu_memory_utilization=0.95)  # FakeRunner: OOM
    search = StagedSearch(objective="throughput", runner=FakeRunner(), seed=seed)
    winner, notes = search.run(_feasible(hw_a100, prepared_vllm))
    assert winner is not None and winner.config.key() != seed.key()
    assert any("did not run" in n and "CUDA OOM" in n for n in notes)


def test_a_seed_alone_can_be_picked_when_the_plan_left_nothing_feasible():
    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9)
    search = StagedSearch(objective="throughput", runner=FakeRunner(), seed=seed)
    winner, _ = search.run([])
    assert winner is not None and winner.config.key() == seed.key()


def test_a_seeded_profile_is_cached_under_its_own_name():
    from rigtune.pipeline import SearchOptions

    assert "seed" not in SearchOptions().key()
    a = SearchOptions(seed="vllm serve m --max-num-seqs 16").key()["seed"]
    assert a != SearchOptions(seed="vllm serve m --max-num-seqs 32").key()["seed"]


@pytest.mark.parametrize("same", [
    "vllm serve m   --max-num-seqs 16 --enforce-eager",
    "vllm serve m --enforce-eager --max_num_seqs=16",
    "python -m vllm.entrypoints.openai.api_server --model m --max-num-seqs 16 --enforce-eager",
    "vllm serve m --enforce-eager \\\n  --max-num-seqs 16",
])
def test_one_launch_spelled_another_way_is_the_same_cached_profile(same):
    from rigtune.pipeline import SearchOptions

    base = SearchOptions(seed="vllm serve m --max-num-seqs 16 --enforce-eager").key()["seed"]
    assert SearchOptions(seed=same).key()["seed"] == base
    # A pass-through flag's value still tells two launches apart, as does another engine.
    assert SearchOptions(seed="vllm serve m --max-num-seqs 16").key()["seed"] != base
    assert SearchOptions(seed="sglang serve --model-path m --max-running-requests 16").key()["seed"] != base


class _Ranked(FakeRunner):
    """FakeRunner whose speed is set per batch, so the seed's rank is chosen; start-up OOM at memory 0.95."""

    def __init__(self, speeds):
        super().__init__()
        self.speeds = speeds

    def run(self, cfg, stage):
        from rigtune.models import TrialMetrics, TrialResult

        self.calls.append(cfg.key())
        if cfg.gpu_memory_utilization is not None and cfg.gpu_memory_utilization >= 0.95:
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error="out of memory at start-up; CUDA out of memory")
        speed = self.speeds.get(cfg.batch, 100)
        m = TrialMetrics(tok_s=speed, ttft_ms=100, tpot_ms=1000 / speed, requests=48, output_tokens=48 * 128)
        return TrialResult(config=cfg, stage=stage, metrics=m)


def test_the_seed_is_re_measured_with_the_shortlist_whatever_its_rank(hw_a100, prepared_vllm):
    # The seed (batch 512) ranks below three grid configurations inside the confirm band: it is re-measured anyway,
    # so the choice on the fresh runs is never made without it.
    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=512, gpu_memory_utilization=0.9)
    runner = _Ranked({16: 1040, 64: 1030, 256: 1020, 512: 1000})
    search = StagedSearch(objective="throughput", runner=runner, seed=seed, confirm_top=3)
    winner, notes = search.run(_feasible(hw_a100, prepared_vllm))
    confirmed = [r for r in search.results if r.stage == "confirm"]
    assert any(r.config.key() == seed.key() for r in confirmed)
    assert any(n.startswith("seed ") for n in notes)


def test_a_seed_that_starts_only_after_a_retry_is_named_as_it_ran(hw_a100, prepared_vllm):
    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=512, gpu_memory_utilization=0.95)
    search = StagedSearch(objective="throughput", runner=_Ranked({512: 5000}), seed=seed)
    winner, notes = search.run(_feasible(hw_a100, prepared_vllm))
    ran = winner.config.key()
    assert ran != seed.key() and winner.config.batch == 512  # the retry, at less memory, won
    assert any(f"started only as {ran}" in n for n in search.notes + notes)
    assert any("kept as the pick" in n for n in notes)


def test_the_leader_engine_is_represented_by_the_grid_not_the_seed(hw_a100, prepared_vllm):
    # A seed carrying speculation must not stand for what its engine can do when other engines are judged.
    seed = Config(backend="vllm", quant="bf16", ctx=4096, batch=64, gpu_memory_utilization=0.9, spec_decode="ngram:4")
    search = StagedSearch(objective="throughput", runner=FakeRunner(), seed=seed)
    search.run(_feasible(hw_a100, prepared_vllm))
    lead_base = next((r.config for r in search.results if r.config.backend == "vllm" and r.stage != "seed"))
    assert lead_base.spec_decode is None


def test_a_seed_in_a_precision_quant_excludes_is_refused(hw_a100, prepared_vllm):
    from rigtune.pipeline import SearchOptions, seed_for

    class Plan:
        prepared = {"vllm": prepared_vllm}

    command = f"vllm serve {prepared_vllm.spec.hf_id} --quantization fp8"
    with pytest.raises(ValueError, match="--quant bf16 excludes"):
        seed_for(hw_a100, prepared_vllm.spec, Plan(), {"vllm": get_backend("vllm")}, command,
                 SearchOptions(quants=["bf16"]))


def test_sglang_s_seed_states_the_defaults_rigtune_launches(hw_a100, prepared_vllm):
    seed = parse_command(f"python -m sglang.launch_server --model-path {prepared_vllm.spec.hf_id}")
    cfg, notes = seed_config(seed, hw_a100, prepared_vllm, ["bf16"])
    assert (cfg.backend, cfg.batch, cfg.gpu_memory_utilization) == ("sglang", 256, 0.88)
    assert any("max-running-requests" in n for n in notes) and any("mem-fraction-static" in n for n in notes)


# ---------------------------------------------------------------- --seed recipe (S4)


def _inkling(hw_a100, prepared_vllm, gpu, gib):
    from tests.test_inkling2 import _prepared

    return _gpu(hw_a100, gpu, gib, (9, 0)), _prepared()


@pytest.mark.parametrize("gpu, gib, ctx", [
    ("NVIDIA H100 PCIe", 79.18, 262144),  # the recipe's H100 line
    ("NVIDIA H200", 139.8, 1048576),  # elsewhere the command names no length: the model's own
])
def test_the_shipped_recipe_seeds_a_calibration_for_its_gpu(hw_a100, prepared_vllm, gpu, gib, ctx):
    from rigtune.pipeline import seed_for

    hw, pm = _inkling(hw_a100, prepared_vllm, gpu, gib)

    class Plan:
        prepared = {"vllm": pm}

    cfg, notes = seed_for(hw, pm.spec, Plan(), {"vllm": get_backend("vllm")}, "recipe")
    assert (cfg.ctx, cfg.spec_decode, cfg.quant, cfg.batch) == (ctx, "dspark:7", "ckpt-nvfp4", 1024)
    assert cfg.extra == {"reasoning_parser": "inkling"}
    assert notes[0].startswith("seed: the recipe, vLLM's recipe recipes/models/thinkingmachines/Inkling-2-Mini.yaml")


def test_a_model_with_no_shipped_recipe_asks_for_the_command(hw_a100, prepared_vllm):
    from rigtune.pipeline import seed_for

    class Plan:
        prepared = {"vllm": prepared_vllm}

    with pytest.raises(ValueError, match="ships no recipe"):
        seed_for(hw_a100, prepared_vllm.spec, Plan(), {"vllm": get_backend("vllm")}, "recipe")


def test_every_shipped_recipe_parses_and_names_its_model():
    import json
    from importlib import resources

    data = json.loads(resources.files("rigtune").joinpath("data", "recipes.json").read_text(encoding="utf-8"))
    for r in data["recipes"]:
        seed = parse_command(r["command"])
        assert (seed.model, seed.engine) == (r["model"], r["engine"]) and r["source"]


def test_a_recipe_seeded_profile_is_not_served_once_the_shipped_recipes_change(monkeypatch):
    import rigtune.pipeline as pl

    before = pl.SearchOptions(seed="recipe").key()["seed"]

    class Changed:
        def joinpath(self, *parts):
            return self

        def read_text(self, encoding=None):
            return '{"recipes": []}'

    import importlib.resources

    monkeypatch.setattr(importlib.resources, "files", lambda pkg: Changed())
    assert pl.SearchOptions(seed="recipe").key()["seed"] != before


@pytest.mark.parametrize("gpu, line", [
    ("NVIDIA H100 NVL", "gpus"), ("NVIDIA H100 80GB HBM3", "gpus"), ("NVIDIA GB10", "gpus"),
    ("NVIDIA H200", None), ("NVIDIA GH200 480GB", None),
    ("NVIDIA H100 80GB HBM3 MIG 1g.10gb", None),  # a slice of an H100 is not an H100 for the recipe's length
])
def test_the_recipe_line_follows_the_gpu_name(gpu, line):
    from rigtune.seed import recipe_for

    command, source = recipe_for("thinkingmachines/Inkling-2-Mini", ["vllm"], gpu)
    assert ("--max-model-len 262144" in command) == (line == "gpus")
    assert recipe_for("thinkingmachines/Inkling-2-Mini", ["sglang"], gpu) is None  # no SGLang recipe shipped


def test_the_cli_takes_the_word_recipe_and_reads_any_other_seed_now():
    import typer

    from rigtune.cli import _seed

    assert _seed("recipe") == "recipe" and _seed(None) is None
    with pytest.raises(typer.BadParameter):
        _seed("llama-server -m x.gguf")


@pytest.mark.usefixtures("no_network", "tmp_home")
def test_a_profile_s_notes_are_printed_as_text_not_markup(spec, monkeypatch):
    # A seed's note quotes its command, which may hold brackets rich would read as markup ("[/x]" raises).
    import io

    from rich.console import Console

    from rigtune import cli
    from rigtune.pipeline import calibrate, prepare_and_plan, select
    from tests.test_phases import two_gpus

    hw = two_gpus()
    candidates, reg = select(hw, spec, force="vllm")
    profile = calibrate(hw, spec, "throughput", prepare_and_plan(hw, spec, candidates, reg), reg, runner=FakeRunner())
    profile.notes = ["seed: the recipe: vllm serve m --x '[/x]' --y [cuda]"]
    out = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=out, width=100_000))
    cli._print_profile(profile)
    assert "--x '[/x]' --y [cuda]" in out.getvalue()
