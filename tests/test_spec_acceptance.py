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
                   "lines": 1, "source": "vllm_log", "per_position": [0.844, 0.659, 0.509], "full_drafts": True}
    assert 1 + sum(got["per_position"]) == pytest.approx(got["mean_length"], abs=0.005)  # the same steps, counted alike


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


def _line(mean, accepted, drafted, rates):
    """A vLLM 0.33 SpecDecoding line (v1/spec_decode/metrics.py lines 120-136)."""
    return (f"INFO [metrics.py:120] SpecDecoding metrics: Mean acceptance length: {mean:.2f}, Accepted throughput: "
            f"1.00 tokens/s, Drafted throughput: 1.00 tokens/s, Accepted: {accepted} tokens, Drafted: {drafted} tokens, "
            f"Per-position acceptance rate: {', '.join(f'{r:.3f}' for r in rates)}, Avg Draft acceptance rate: 50.0%")


def test_lines_are_pooled_by_their_drafts():
    """Each line's drafts are its accepted tokens over the sum of its rates: 3,231 and 1,000 here."""
    lines = [VLLM_MTP3, _line(2.80, 1800, 3000, [0.8, 0.6, 0.4])]
    got = parse_acceptance("vllm", "\n".join(lines))
    assert got["per_position"] == pytest.approx([(0.844 * 3231.1 + 0.8 * 1000) / 4231.1,
                                                 (0.659 * 3231.1 + 0.6 * 1000) / 4231.1,
                                                 (0.509 * 3231.1 + 0.4 * 1000) / 4231.1], abs=1e-4)
    assert got["full_drafts"] is True


def test_a_line_whose_rates_do_not_add_up_to_its_mean_is_not_used():
    """1 + the sum of a line's rates is its mean acceptance length: the suffix lines above, cut short when copied
    (suffix decoding prints one rate per position up to 24), do not add up, and give no curve."""
    assert "per_position" not in parse_acceptance("vllm", "\n".join(VLLM_SUFFIX))


def test_lines_with_different_numbers_of_positions_give_none():
    mixed = VLLM_MTP3 + "\n" + _line(1.50, 500, 1000, [0.5])
    assert "per_position" not in parse_acceptance("vllm", mixed)


def test_short_drafts_are_flagged():
    """3,231 drafts of 3 tokens are 9,693 tokens; fewer drafted means some drafts were short, so the deeper rates
    understate a full draft's and the curve must not be read as one."""
    got = parse_acceptance("vllm", _line(3.01, 6501, 8000, [0.844, 0.659, 0.509]))
    assert got["full_drafts"] is False


def test_per_position_counters_over_the_drafts():
    got = acceptance_from_counters(3231, 9693, 6501, per_position=[2727, 2129, 1645])
    assert got["per_position"] == [0.844, 0.6589, 0.5091] and got["full_drafts"] is True
    assert "per_position" not in acceptance_from_counters(3231, 9693, 6501)
    assert acceptance_from_counters(3231, 8000, 6501, per_position=[2727, 2129, 1645])["full_drafts"] is False


def test_counts_that_do_not_add_up_to_the_accepted_tokens_are_not_a_curve():
    """A restart between the readings: 50 drafts, 140 accepted at position 0 alone, a rate of 2.8."""
    assert "per_position" not in acceptance_from_counters(50, 150, 140, per_position=[130, 10])
    # Adding up but rising with the position: misread positions.
    assert "per_position" not in acceptance_from_counters(100, 300, 150, per_position=[50, 100])


def test_the_counters_are_read_by_position_and_summed_over_engines(monkeypatch):
    import httpx

    import rigtune.calibrate.measure as measure

    text = "\n".join([
        'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 3231.0',
        'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 9693.0',
        'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 6501.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 2000.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 2700.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",model_name="m",position="0"} 27.0',
    ])
    monkeypatch.setattr(httpx, "get", lambda *a, **k: type("R", (), {"text": text})())
    hooks = type("H", (), {"metrics_path": "/metrics"})()
    assert measure.spec_counters("http://x", hooks) == (3231.0, 9693.0, 6501.0, 2727.0, 2000.0)
    gappy = text.replace('position="1"', 'position="2"')
    monkeypatch.setattr(httpx, "get", lambda *a, **k: type("R", (), {"text": gappy})())
    assert measure.spec_counters("http://x", hooks) == (3231.0, 9693.0, 6501.0)  # a gap: no positions


def test_a_trial_records_the_per_position_rates_from_the_counters(monkeypatch):
    readings = iter([(100.0, 300.0, 200.0, 90.0, 70.0, 40.0), (3331.0, 9993.0, 6701.0, 2817.0, 2199.0, 1685.0)])
    proc = _Logging()
    monkeypatch.setattr(search_mod, "run_trial", lambda *a, **k: TrialMetrics(tok_s=100, requests=8, output_tokens=800))
    monkeypatch.setattr(search_mod, "spec_counters", lambda *a, **k: next(readings))
    runner = SubprocessTrialRunner(backends={"vllm": _LoggingBackend(proc)}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    got = runner.run(Config(backend="vllm", quant="bf16", spec_decode="mtp:3"), "spec").metrics.spec_acceptance
    assert got.per_position == [0.844, 0.6589, 0.5091]
    assert "per_position" in got.model_dump()


def test_a_line_still_being_written_is_not_read(monkeypatch):
    proc = _Logging()

    def workload(*a, **k):
        proc.text += VLLM_MTP3 + "\n" + VLLM_MTP3[:150]  # the second line cut mid-write
        return TrialMetrics(tok_s=100, requests=8, output_tokens=800)

    monkeypatch.setattr(search_mod, "run_trial", workload)
    monkeypatch.setattr(search_mod, "spec_counters", lambda *a, **k: None)
    runner = SubprocessTrialRunner(backends={"vllm": _LoggingBackend(proc)}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    got = runner.run(Config(backend="vllm", quant="bf16", spec_decode="mtp:3"), "spec").metrics.spec_acceptance
    assert (got.lines, got.per_position) == (1, [0.844, 0.659, 0.509])


def test_counters_without_positions_take_the_curve_from_the_log(monkeypatch):
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
    assert (got.source, got.per_position, got.full_drafts) == ("vllm_metrics", [0.844, 0.659, 0.509], True)


def test_a_record_from_before_the_curve_loads():
    from rigtune.models import SpecAcceptance

    old = SpecAcceptance.model_validate({"rate": 0.67, "mean_length": 3.01, "source": "vllm_log"})
    assert (old.per_position, old.full_drafts) == (None, None)


def test_a_weak_drafter_s_full_drafts_are_not_taken_for_short_ones():
    """1,000 drafts of 3 tokens at low acceptance: the rates' rounding moves the draft count most when their sum is
    small, which a fixed tolerance read as short drafts."""
    got = parse_acceptance("vllm", _line(1.30, 300, 3000, [0.150, 0.100, 0.049]))
    assert got["full_drafts"] is True


def test_a_log_curve_unlike_the_counters_is_not_taken(monkeypatch):
    readings = iter([(100.0, 300.0, 200.0), (1100.0, 3300.0, 1200.0)])  # mean 2.0 by the counters
    proc = _Logging()

    def workload(*a, **k):
        proc.text += VLLM_MTP3 + "\n"  # mean 3.01 by the log: other steps
        return TrialMetrics(tok_s=100, requests=8, output_tokens=800)

    monkeypatch.setattr(search_mod, "run_trial", workload)
    monkeypatch.setattr(search_mod, "spec_counters", lambda *a, **k: next(readings))
    runner = SubprocessTrialRunner(backends={"vllm": _LoggingBackend(proc)}, models={"vllm": None}, hw=None,
                                   workload=Workload())
    got = runner.run(Config(backend="vllm", quant="bf16", spec_decode="mtp:3"), "spec").metrics.spec_acceptance
    assert (got.mean_length, got.per_position) == (2.0, None)
