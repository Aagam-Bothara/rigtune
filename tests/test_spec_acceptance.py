"""Speculative decoding's acceptance, recorded on a trial that drafted (models.SpecAcceptance).

Read from vLLM's /metrics counters either side of the workload, else from the engine's log lines written while it
ran: never from what a gate or quality probe sends the same engine afterwards. The log lines are the A100 session's
(18 Nov 2026, vLLM 0.33.0 and SGLang 0.5.23, Qwen3.5-9B).
"""

from __future__ import annotations

import json

import pytest

import rigtune.calibrate.search as search_mod
from rigtune.calibrate.search import SubprocessTrialRunner
from rigtune.calibrate.workload import Workload
from rigtune.memlog import acceptance_from_counters, parse_acceptance
from rigtune.models import Config, SpecAcceptance, TrialMetrics
from tests.test_failure_kind import _Live, _LiveBackend

VLLM_MTP3 = ("(APIServer pid=18891) INFO 11-18 05:58:42 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: "
             "3.01, Accepted throughput: 573.25 tokens/s, Drafted throughput: 854.72 tokens/s, Accepted: 6501 tokens, "
             "Drafted: 9693 tokens, Per-position acceptance rate: 0.844, 0.659, 0.509, Avg Draft acceptance rate: 67.1%")
VLLM_SUFFIX = [
    ("(APIServer pid=15863) INFO 11-18 05:50:51 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 1.95, "
     "Accepted throughput: 149.41 tokens/s, Drafted throughput: 324.66 tokens/s, Accepted: 1700 tokens, Drafted: "
     "3694 tokens, Per-position acceptance rate: 0.290, 0.141"),
    ("(APIServer pid=15863) INFO 11-18 05:51:01 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: 1.88, "
     "Accepted throughput: 242.47 tokens/s, Drafted throughput: 590.23 tokens/s, Accepted: 2425 tokens, Drafted: "
     "5903 tokens, Per-position acceptance rate: 0.297, 0.150"),
]
SGLANG = [
    ("[2026-11-18 05:39:39] Decode batch, #running-req: 8, #full token: 835, full token usage: 0.00, mamba num: 26, "
     "mamba usage: 0.10, accept len: 2.71, accept rate: 0.57, cuda graph: True, gen throughput (token/s): 84.23, "
     "#queue-req: 0"),
    ("[2026-11-18 05:39:40] Decode batch, #running-req: 8, #full token: 1738, full token usage: 0.00, mamba num: 26, "
     "mamba usage: 0.10, accept len: 2.82, accept rate: 0.61, cuda graph: True, gen throughput (token/s): 1111.14, "
     "#queue-req: 0"),
]


def test_vllm_log_one_line():
    got = parse_acceptance("vllm", VLLM_MTP3)
    assert got == {"rate": 0.6707, "mean_length": 3.01, "accepted_tokens": 6501, "drafted_tokens": 9693,
                   "lines": 1, "source": "vllm_log"}


def test_vllm_log_lines_are_summed():
    got = parse_acceptance("vllm", "\n".join(VLLM_SUFFIX))
    assert (got["accepted_tokens"], got["drafted_tokens"], got["lines"]) == (4125, 9597, 2)
    assert got["rate"] == round(4125 / 9597, 4) and got["mean_length"] == 1.915


def test_sglang_log_rates_are_averaged():
    got = parse_acceptance("sglang", "\n".join(SGLANG))
    assert got == {"rate": 0.59, "mean_length": 2.765, "lines": 2, "source": "sglang_log"}


@pytest.mark.parametrize("backend,text", [
    ("vllm", ""), ("vllm", "INFO Engine 000: Avg prompt throughput: 837.2 tokens/s"), ("sglang", ""),
    ("llamacpp-cuda", VLLM_MTP3),  # an engine whose lines are not known
    ("vllm", VLLM_MTP3.replace("Drafted: 9693", "Drafted: 0")),  # nothing drafted: no rate
])
def test_nothing_logged_is_no_figure(backend, text):
    assert parse_acceptance(backend, text) is None


def test_counters_give_vllm_exact_mean_length():
    """vLLM's own definition (v1/spec_decode/metrics.py): mean acceptance length = 1 + accepted / drafts."""
    got = acceptance_from_counters(drafts=3231, drafted=9693, accepted=6501)
    assert got == {"rate": 0.6707, "mean_length": 3.012, "accepted_tokens": 6501, "drafted_tokens": 9693,
                   "source": "vllm_metrics"}
    assert acceptance_from_counters(0, 0, 0) is None


def test_unset_acceptance_is_left_out_of_the_file():
    plain = TrialMetrics(tok_s=10, requests=4, output_tokens=40)
    assert "spec_acceptance" not in plain.model_dump(mode="json")
    assert "spec_acceptance" not in json.loads(plain.model_dump_json())
    drafted = plain.model_copy(update={"spec_acceptance": SpecAcceptance(**parse_acceptance("vllm", VLLM_MTP3))})
    back = TrialMetrics.model_validate(drafted.model_dump(mode="json"))
    assert back.spec_acceptance.rate == 0.6707 and back.spec_acceptance.source == "vllm_log"


class _Logging(_Live):
    """An engine whose log gains SpecDecoding lines while the workload runs, and a gate's afterwards."""

    def __init__(self):
        self.text = "INFO starting\n"

    def log_size(self) -> int:
        return len(self.text.encode())

    def log_since(self, offset: int) -> str:
        return self.text.encode()[offset:].decode()


class _LoggingBackend(_LiveBackend):
    def __init__(self, proc):
        super().__init__("")
        self.proc = proc

    def launch(self, cfg, model, port, log_path=None):
        return self.proc


def _runner_run(cfg, monkeypatch, before_lines=("",), during=(VLLM_MTP3,)):
    proc = _Logging()
    proc.text += "\n".join(before_lines) + "\n"

    def workload(*a, **k):
        proc.text += "\n".join(during) + "\n"
        return TrialMetrics(tok_s=100, requests=8, output_tokens=800)

    monkeypatch.setattr(search_mod, "run_trial", workload)
    monkeypatch.setattr(search_mod, "spec_counters", lambda *a, **k: None)  # no /metrics: the log is read
    runner = SubprocessTrialRunner(backends={cfg.backend: _LoggingBackend(proc)}, models={cfg.backend: None},
                                   hw=None, workload=Workload())
    return runner.run(cfg, "spec")


def test_a_drafting_trial_reads_only_the_lines_its_workload_wrote(monkeypatch):
    earlier = VLLM_SUFFIX[0]  # a line from before the workload began: not this workload's
    res = _runner_run(Config(backend="vllm", quant="bf16", spec_decode="mtp:3"), monkeypatch, before_lines=(earlier,))
    assert res.ok and res.metrics.spec_acceptance.drafted_tokens == 9693
    assert res.metrics.spec_acceptance.source == "vllm_log"


def test_sglang_trial_reads_its_decode_lines(monkeypatch):
    res = _runner_run(Config(backend="sglang", quant="bf16", spec_decode="mtp:3"), monkeypatch, during=SGLANG)
    assert res.metrics.spec_acceptance.rate == 0.59 and res.metrics.spec_acceptance.source == "sglang_log"


def test_a_trial_without_speculation_records_none(monkeypatch):
    res = _runner_run(Config(backend="vllm", quant="bf16"), monkeypatch)
    assert res.metrics.spec_acceptance is None
    assert "spec_acceptance" not in res.model_dump(mode="json")["metrics"]


def test_vllm_counters_are_preferred_over_the_log(monkeypatch):
    readings = iter([(100.0, 300.0, 200.0), (3331.0, 9993.0, 6701.0)])
    proc = _Logging()

    def workload(*a, **k):
        proc.text += VLLM_MTP3 + "\n"
        return TrialMetrics(tok_s=100, requests=8, output_tokens=800)

    monkeypatch.setattr(search_mod, "run_trial", workload)
    monkeypatch.setattr(search_mod, "spec_counters", lambda *a, **k: next(readings))
    runner = SubprocessTrialRunner(backends={"vllm": _LoggingBackend(proc)}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    got = runner.run(Config(backend="vllm", quant="bf16", spec_decode="mtp:3"), "spec").metrics.spec_acceptance
    assert (got.source, got.drafted_tokens, got.accepted_tokens, got.mean_length) == ("vllm_metrics", 9693, 6501, 3.012)


def test_a_stand_in_process_without_a_log_still_measures(monkeypatch):
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: TrialMetrics(tok_s=100, requests=8, output_tokens=800))
    monkeypatch.setattr(search_mod, "spec_counters", lambda *a, **k: None)
    runner = SubprocessTrialRunner(backends={"vllm": _LiveBackend("")}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    res = runner.run(Config(backend="vllm", quant="bf16", spec_decode="mtp:3"), "spec")
    assert res.ok and res.metrics.spec_acceptance is None
