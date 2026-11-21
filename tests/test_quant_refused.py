"""An engine that refuses a checkpoint's quantization on this GPU is believed, once, for the whole precision.

Both engines resolve the checkpoint's method and check the card against it at start-up, compressed-tensors per
layer's scheme; rigtune does not guess that check ahead of time, where a wrong guess would refuse a model that runs.
The lines are the engines' format strings filled in (cited in rigtune/calibrate/search.py beside _QUANT_GPU).
"""

from __future__ import annotations

import pytest

from rigtune.calibrate.search import QUANT_REFUSED, StagedSearch, failure_kind, quant_refusal, startup_cause
from rigtune.models import Config, TrialMetrics, TrialResult
from tests.test_objectives_and_search import FakeRunner, _feasible_a100
from tests.test_text_only import ENGINE, _runner_error

VLLM_METHOD = ("ValueError: The quantization method modelopt_fp4 is not supported for the current GPU. Minimum "
               "capability: 100. Current capability: 89.")
VLLM_SCHEME = ("RuntimeError: Quantization scheme is not supported for the current GPU. Required capability: 89. "
               "Current capability: 80.")
SGLANG_SCHEME = ("RuntimeError: ('Quantization scheme is not supported for ', 'the current GPU. Min capability: "
                 "100. ', 'Current capability: 90.')")
SGLANG_MOE = ("ValueError: The quantization method moe_wna16 + awq is not supported for the current GPU. Minimum "
              "capability: 75. Current capability: 70.")
DTYPE = ("ValueError: torch.bfloat16 is not supported for quantization method awq. Supported dtypes: "
         "[torch.float16]")


@pytest.mark.parametrize("line, said", [
    (VLLM_METHOD, "modelopt_fp4 needs compute capability 10.0, this GPU has 8.9"),
    (VLLM_SCHEME, "its quantization scheme needs compute capability 8.9, this GPU has 8.0"),
    (SGLANG_SCHEME, "its quantization scheme needs compute capability 10.0, this GPU has 9.0"),
    (SGLANG_MOE, "moe_wna16 + awq needs compute capability 7.5, this GPU has 7.0"),
    (DTYPE, DTYPE[len("ValueError: "):]),
])
def test_each_engines_refusal_is_read(line, said):
    assert quant_refusal(f"INFO loading\n{line}\n{ENGINE}") == said


@pytest.mark.parametrize("line", [
    "WARNING fp8 quantization is not fully optimized yet. The speed can be slower than non-quantized models.",
    "INFO Minimum capability: 80",  # no refusal around it
    "ValueError: Quantization method specified in the model config (fp8) does not match the quantization method "
    "specified in the `quantization` argument (awq).",
])
def test_other_lines_are_not_a_refusal(line):
    assert quant_refusal(line) is None


def test_the_runner_names_it_and_its_kind():
    res = _runner_error(f"INFO loading\n{VLLM_METHOD}\n" + "  frame\n" * 30 + ENGINE)
    assert failure_kind(res) == "quant_refused" and not res.launched
    assert startup_cause(res.error) == (QUANT_REFUSED, "modelopt_fp4 needs compute capability 10.0, this GPU has 8.9")


class _Refusing(FakeRunner):
    """fp8 is refused on this GPU in every config; bf16 runs."""

    def run(self, cfg: Config, stage: str) -> TrialResult:
        if cfg.quant != "fp8":
            return super().run(cfg, stage)
        self.calls.append(cfg.key())
        why = "fp8 needs compute capability 8.9, this GPU has 8.0"
        return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                           error=f"{QUANT_REFUSED} ({why}); failed to start (rc=1)\n", failure_kind="quant_refused")


def test_a_refused_precision_is_launched_once_and_the_search_goes_on(hw_a100, prepared_vllm):
    runner = _Refusing()
    search = StagedSearch(objective="throughput", runner=runner)
    winner, notes = search.run(_feasible_a100(hw_a100, prepared_vllm))
    assert winner is not None and winner.config.quant == "bf16"
    assert sum(1 for key in runner.calls if "/fp8/" in key) == 1  # refused once, never launched again
    assert ("vllm: the engine refused fp8 on this GPU (fp8 needs compute capability 8.9, this GPU has 8.0); "
            "it is not launched again in this calibration") in notes


def test_when_the_only_precision_is_refused_the_reason_is_what_calibration_reports(hw_a100, prepared_vllm):
    runner = _Refusing()
    search = StagedSearch(objective="throughput", runner=runner)
    only_fp8 = [c for c in _feasible_a100(hw_a100, prepared_vllm) if c.quant == "fp8"]
    winner, notes = search.run(only_fp8)
    assert winner is None and runner.calls and all("/fp8/" in key for key in runner.calls)
    assert len(runner.calls) == 1
    assert any("needs compute capability 8.9, this GPU has 8.0" in n for n in notes)


def test_a_refusal_in_a_trial_with_a_second_model_blocks_nothing(hw_a100, prepared_vllm):
    """A draft or an fp8 MTP head carries its own quantization: the refusal may be that model's."""
    runner = _Refusing()
    search = StagedSearch(objective="throughput", runner=runner)
    cfg = Config(backend="vllm", quant="fp8", ctx=4096, batch=16, gpu_memory_utilization=0.9, spec_decode="mtp:1")
    search._run(cfg, "spec")
    assert not search._quant_blocked
