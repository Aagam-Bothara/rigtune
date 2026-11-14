"""A multimodal model whose multimodal profiling fails at vLLM start-up is retried once, text-only.

The log lines are from `vllm serve google/gemma-4-12B-it` on an A40 with vLLM 0.29.0 and transformers 5.23.0, a
mismatched pair (later vLLM releases pin transformers below 5.21). Launches that start are
unchanged: tests/test_text_only_off.py holds their launch arguments and trial sequences to the tree before.
"""

from __future__ import annotations

from typing import List

import pytest

from rigtune.backends import registry
from rigtune.backends.base import BaseBackend, LlmtraceHooks
from rigtune.backends.vllm import VllmBackend
from rigtune.calibrate.search import MM_PROFILING_FAILED, SubprocessTrialRunner, mentions_oom, mm_profiling_failure
from rigtune.calibrate.workload import Workload, get_workload
from rigtune.memory import plan
from rigtune.models import ArchInfo, Config, ModelSpec, TrialMetrics, TrialResult
from rigtune.pipeline import TEXT_ONLY_NOTE, PlanResult, SearchOptions, calibrate
from tests.conftest import make_hw
from tests.test_hybrid_memory import gemma4, qwen25
from tests.test_objectives_and_search import FakeRunner

VIDEO = "ValueError: Found 2048 <|video|> tokens in the text but no videos were passed."
APPLY = ("ValueError: Failed to apply Gemma4UnifiedProcessor on data={'text': '00:00 <|image><|video|>...'} with "
         "kwargs={'truncation': False, 'return_tensors': 'pt'}")
ENGINE = "RuntimeError: Engine core initialization failed. See root cause above."
MM_LOG = "\n".join(["INFO profiling", "Traceback (most recent call last):", VIDEO, "", "The above exception ...",
                    APPLY, *["  frame"] * 30, ENGINE])
OOM_LOG = "\n".join(["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", ENGINE])
LOAD_LOG = "\n".join(["ValueError: Model architectures ['FooForCausalLM'] are not supported for now.", ENGINE])


# --------------------------------------------------------------------------- classifier


def test_classifier_reads_vllm_multimodal_profiling_failure():
    assert mm_profiling_failure(MM_LOG) == VIDEO  # the root cause, first
    assert mm_profiling_failure(APPLY) == APPLY
    long = APPLY.replace("...", "x" * 200)
    assert mm_profiling_failure(long) == long[:157] + "..."
    for kind in ("images", "audios"):
        assert mm_profiling_failure(f"Found 4 tokens in the text but no {kind} were passed.")
    assert not mentions_oom(MM_LOG)


def test_classifier_ignores_oom_and_ordinary_load_failures():
    assert mm_profiling_failure(OOM_LOG) is None and mentions_oom(OOM_LOG)
    assert mm_profiling_failure(LOAD_LOG) is None
    assert mm_profiling_failure("Failed to apply chat template") is None


class _Dead:
    pid = 1

    def __init__(self, log: str):
        self.log = log

    def wait_ready(self, timeout: float) -> bool:
        return False

    def tail_log(self, n: int = 50) -> str:
        return "\n".join(self.log.splitlines()[-n:])

    def returncode(self) -> int:
        return 1

    def stop(self) -> None:
        pass


class _DeadBackend(BaseBackend):
    name = "vllm"

    def __init__(self, log: str):
        self.log = log

    def launch(self, cfg, model, port, log_path=None):
        return _Dead(self.log)

    def workload_hooks(self, hw, model) -> LlmtraceHooks:
        return LlmtraceHooks(gpu_ids=[])


def _runner_error(log: str) -> TrialResult:
    runner = SubprocessTrialRunner(backends={"vllm": _DeadBackend(log)}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    return runner.run(Config(backend="vllm", quant="bf16", gpu_memory_utilization=0.9), "quant")


def test_runner_names_the_cause_even_above_the_tail():
    res = _runner_error(MM_LOG)
    assert not res.launched
    assert res.error.startswith(f"{MM_PROFILING_FAILED} ({VIDEO}); failed to start (rc=1)\n")
    assert VIDEO not in res.error.split("\n", 1)[1]  # 30 frames pushed it out of the 20-line tail
    assert _runner_error(OOM_LOG).error.startswith("out of memory at start-up; failed to start")
    assert _runner_error(LOAD_LOG).error.startswith("failed to start (rc=1)")


# --------------------------------------------------------------------------- the retry


class MMRunner(FakeRunner):
    """The fake engine of tests/test_objectives_and_search.py, whose multimodal profiling fails unless text-only."""

    def run(self, cfg: Config, stage: str) -> TrialResult:
        if not cfg.text_only:
            self.calls.append(cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error=f"{MM_PROFILING_FAILED} ({VIDEO}); failed to start (rc=1)\n{ENGINE}")
        return super().run(cfg, stage)


@pytest.fixture
def lmo(monkeypatch):
    """The vLLM that would run has --language-model-only; returns a setter to take it away."""
    import rigtune.backends.vllm as vl

    monkeypatch.setattr(vl, "_language_model_only_flag", lambda: True)
    return lambda on: monkeypatch.setattr(vl, "_language_model_only_flag", lambda: on)


def _calibrate(pm, runner, **opts):
    hw = make_hw("rtx4090")
    hw.backends["vllm"].version = "0.29.0"
    reg = registry()
    be = reg["vllm"]
    feasible = plan(hw, pm, be.candidate_configs(hw, pm), be.memory_model(hw))
    result = PlanResult(hw=hw, spec=ModelSpec(hf_id=pm.spec.hf_id), candidates=["vllm"], prepared={"vllm": pm},
                        feasible={"vllm": feasible})
    return calibrate(hw, result.spec, "balanced", result, reg, workload=get_workload("chat"), runner=runner,
                     options=SearchOptions(**opts))


@pytest.mark.usefixtures("tmp_home")
def test_multimodal_failure_is_retried_once_text_only_and_the_pick_keeps_it(lmo):
    runner = MMRunner()
    profile = _calibrate(gemma4(), runner)
    table = profile.calibration_table
    failed = [r for r in table if not r.config.text_only]
    assert len(failed) == 1 and len(runner.calls) - sum(1 for k in runner.calls if k.endswith("textonly")) == 1
    # The retry is the failed config text-only (the fake engine then fails its gmu 0.95 for memory, as it does any).
    assert table[0] is failed[0] and table[1].config == failed[0].config.model_copy(update={"text_only": True})
    # Every later trial launched text-only, the confirm stage's and the combinations' included.
    stages = {r.stage for r in table[1:]}
    assert {"confirm", "combine"} <= stages and all(r.config.text_only for r in table[1:])
    assert profile.config.text_only and profile.launch_args[-1] == "--language-model-only"
    assert profile.config.key().endswith("/textonly")
    note = ("google/gemma-4-12B-it could not start on vLLM 0.29.0 with its multimodal profiling "
            f"({VIDEO}), as when the engine and its transformers mismatch; retried text-only (--language-model-only)")
    assert profile.notes.count(note) == 1 and TEXT_ONLY_NOTE in profile.notes


@pytest.mark.usefixtures("tmp_home")
def test_no_retry_for_a_text_only_architecture(lmo):
    runner = MMRunner()
    with pytest.raises(RuntimeError) as err:
        _calibrate(qwen25(), runner)
    assert not any(k.endswith("textonly") for k in runner.calls)
    assert "not retried text-only: Qwen2ForCausalLM is not a multimodal architecture" in str(err.value)


@pytest.mark.usefixtures("tmp_home")
def test_no_retry_without_the_flag(lmo):
    lmo(False)
    runner = MMRunner()
    with pytest.raises(RuntimeError) as err:
        _calibrate(gemma4(), runner)
    assert not any(k.endswith("textonly") for k in runner.calls)
    assert ("not retried text-only: the vLLM that would run has no --language-model-only (read from its source)"
            in str(err.value))


@pytest.mark.usefixtures("tmp_home")
def test_an_ordinary_start_up_failure_is_not_retried(lmo):
    class Failing(FakeRunner):
        def run(self, cfg, stage):
            self.calls.append(cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error=f"failed to start (rc=1)\n{LOAD_LOG}")

    runner = Failing()
    with pytest.raises(RuntimeError):
        _calibrate(gemma4(), runner)
    assert not any(k.endswith("textonly") for k in runner.calls)


def test_feature_check_reads_the_engine_source(tmp_path, monkeypatch):
    import rigtune.backends.vllm as vl

    (tmp_path / "engine").mkdir()
    monkeypatch.setattr(vl, "_vllm_package", lambda: tmp_path)
    for text, want in (("    language_model_only: bool = MultiModalConfig.language_model_only\n", True),
                       ("class EngineArgs:\n    model: str\n", False)):
        (tmp_path / "engine" / "arg_utils.py").write_text(text, encoding="utf-8")
        vl._language_model_only_flag.cache_clear()
        assert vl._language_model_only_flag() is want
        assert (VllmBackend().text_only_refusal() is None) is want
    monkeypatch.setattr(vl, "_vllm_package", lambda: None)
    vl._language_model_only_flag.cache_clear()
    assert vl._language_model_only_flag() is False
    vl._language_model_only_flag.cache_clear()
    assert registry()["sglang"].text_only_refusal() == "rigtune has no text-only launch for sglang"


def test_multimodal_architectures():
    def arch(name: str, **kw) -> ArchInfo:
        return ArchInfo(architecture=name, num_layers=1, hidden_size=8, num_attention_heads=1, num_kv_heads=1,
                        head_dim=8, **kw)

    assert gemma4().arch.multimodal_arch and arch("Gemma3ForConditionalGeneration").multimodal_arch
    assert arch("InternVLChatModel", has_mm_config=True).multimodal_arch
    assert not qwen25().arch.multimodal_arch and not arch("LlamaForCausalLM").multimodal_arch
    from rigtune.hfconfig import arch_from_config

    assert arch_from_config({"architectures": ["X"], "vision_config": {"depth": 2}}).has_mm_config
    assert arch_from_config({"architectures": ["X"], "audio_config": {}}).has_mm_config
    assert not arch_from_config({"architectures": ["X"], "text_config": {}}).has_mm_config


# --------------------------------------------------------------------------- key and launch


def test_key_unchanged_unless_text_only():
    cfg = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9, power_limit_w=250)
    assert cfg.key() == "vllm/bf16/ctx8192/b64/gmu0.90/pl250"
    assert cfg.model_copy(update={"text_only": False}).key() == cfg.key()
    assert cfg.model_copy(update={"text_only": True}).key() == "vllm/bf16/ctx8192/b64/gmu0.90/textonly/pl250"
    assert Config.model_validate({"backend": "vllm", "quant": "bf16"}).text_only is False  # a profile saved before


def _launches(cfg: Config) -> List[List[str]]:
    be, pm = VllmBackend(), gemma4()
    return [be.launch_spec(cfg, pm, 8000).args, be.replica_launch_spec(cfg, pm, 8000, 1).args]


def test_serve_launch_carries_the_flag(monkeypatch):
    import rigtune.backends.vllm as vl
    from rigtune.serve.supervisor import Supervisor

    monkeypatch.setattr(vl, "_serve_cli", lambda: None)
    cfg = Config(backend="vllm", quant="bf16", ctx=8192, batch=64, gpu_memory_utilization=0.9)
    plain, text = _launches(cfg), _launches(cfg.model_copy(update={"text_only": True}))
    for a, b in zip(plain, text):
        assert "--language-model-only" not in a and b == a + ["--language-model-only"]

    seen = []

    class Captured(Exception):
        pass

    def launch(self, c, model, port, log_path=None):
        seen.append(self.launch_spec(c, model, port).args)
        raise Captured

    monkeypatch.setattr(VllmBackend, "launch", launch)
    sup = Supervisor(VllmBackend(), cfg.model_copy(update={"text_only": True}), gemma4(), port=8000)
    with pytest.raises(Captured):
        sup._launch()
    assert seen and seen[0][-1] == "--language-model-only"
