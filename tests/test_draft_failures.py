import pytest

from rigtune import pairing as p
from rigtune.calibrate.search import StagedSearch, SubprocessTrialRunner, failure_kind
from rigtune.calibrate.workload import Workload
from rigtune.models import Config, TrialMetrics, TrialResult
from tests.test_text_only import _DeadBackend
from tests.test_pairing import pair, target


VLLM = "Target and draft model should have the same vocabulary size"
SGLANG = "STANDALONE speculative decoding requires the draft model to share the same vocabulary as the target model"


def run(log, backend="vllm", model=None, hw=None):
    cfg = Config(backend=backend, quant="bf16", spec_decode="draft:org/draft:4", gpu_memory_utilization=.95)
    runner = SubprocessTrialRunner(backends={backend: _DeadBackend(log)}, models={backend: model},
                                   hw=hw, workload=Workload())
    return runner.run(cfg, "spec")


@pytest.mark.parametrize("log,engine", [(VLLM, "vllm"), (SGLANG, "sglang"),
                                       (VLLM + "\nCUDA out of memory", "vllm")])
def test_vocabulary_refusal_precedes_oom(log, engine):
    res = run(log, engine)
    assert res.failure_kind == "draft_refused"
    assert StagedSearch(objective="throughput", runner=None)._headroom_retry(res.config, res) is None


def test_free_memory_precheck_on_a_draft_trial_is_not_oom():
    res = run("Free memory on device cuda:0 is less than desired GPU memory utilization.\nCUDA out of memory")
    assert res.failure_kind == "startup_other"
    assert not res.error.startswith("out of memory at start-up")


def test_gated_draft_download_is_draft_unavailable():
    assert run("401 Client Error: Cannot access gated repo org/draft").failure_kind == "draft_unavailable"
    assert run("401 Client Error: Cannot access gated repo org/target").failure_kind == "startup_other"


# What every vLLM draft launch prints, whatever happens next: the second repository is always in the log.
SPEC_LINE = ("INFO 11-16 10:00:01 [config.py] speculative_config=SpeculativeConfig(method='draft_model', "
             "model='org/draft', num_spec_tokens=4)\n")
SGLANG_LINE = "[2026-11-16 10:00:01] server_args=ServerArgs(speculative_draft_model_path='org/draft')\n"
OOM = ("(EngineCore_DP0 pid={pid}) ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {mib} MiB. "
       "GPU 0 has a total capacity of 23.53 GiB\n")
# Worded by huggingface_hub (hf_raise_for_status) and transformers (cached_file). Both engines fetch a second
# repository through them, so a failed download reads the same in a vLLM log and an SGLang log.
ACCESS = [
    ("huggingface_hub.errors.GatedRepoError: 401 Client Error. (Request ID: Root=1-0000)\n\n"
     "Cannot access gated repo for url https://huggingface.co/{repo}/resolve/main/config.json.\n"
     "Access to model {repo} is restricted. You must have access to it and be authenticated to access it.\n"),
    ("OSError: You are trying to access a gated repo.\n"
     "Make sure to have access to it at https://huggingface.co/{repo}.\n"),
    ("huggingface_hub.errors.HfHubHTTPError: (Request ID: Root=1-0000)\n\n403 Forbidden: Please enable access to "
     "public gated repositories in your fine-grained token settings to view this repository..\n"
     "Cannot access content at: https://huggingface.co/{repo}/resolve/main/config.json.\n"),
    ("huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. (Request ID: Root=1-0000)\n\n"
     "Repository Not Found for url: https://huggingface.co/api/models/{repo}/revision/main.\n"
     "Please make sure you specified the correct `repo_id` and `repo_type`.\n"),
    ("OSError: {repo} is not a local folder and is not a valid model identifier listed on "
     "'https://huggingface.co/models'\n"),
]


@pytest.mark.parametrize("pid,mib", [("14040", "512.00"), ("2210", "404.00"), ("401", "403.00")])
def test_oom_with_incidental_numbers_stays_oom(pid, mib):
    """A pid or an allocation size holding 401, 403 or 404, beside the repository every draft launch prints."""
    res = run(SPEC_LINE + OOM.format(pid=pid, mib=mib))
    assert res.failure_kind == "oom_startup"
    assert res.error.startswith("out of memory at start-up")


@pytest.mark.parametrize("engine,prefix", [("vllm", SPEC_LINE), ("sglang", SGLANG_LINE)])
@pytest.mark.parametrize("text", ACCESS)
def test_real_access_errors_naming_the_draft_are_draft_unavailable(engine, prefix, text):
    assert run(prefix + text.format(repo="org/draft"), engine).failure_kind == "draft_unavailable"
    # The same failure for the target, or for a repository whose name only starts the same, is not the draft's.
    assert run(prefix + text.format(repo="org/target"), engine).failure_kind == "startup_other"
    assert run(prefix + text.format(repo="org/draft-v2"), engine).failure_kind == "startup_other"


def test_access_error_beside_oom_text_is_left_to_the_oom_rule():
    res = run(SPEC_LINE + ACCESS[0].format(repo="org/draft") + OOM.format(pid="7", mib="512.00"))
    assert res.failure_kind == "oom_startup"


def test_refusal_of_an_ok_pair_marks_the_record_refuted(prepared_vllm, hw_a100):
    a, b = target(), pair()
    path = p.save_record(a, b, "draft", p.check_pair(a, b, "draft", 4096))
    prepared_vllm.spec_pairs = [{"kind": "draft", "repo": "org/draft", "record": path.name}]
    res = run(VLLM, model=prepared_vllm, hw=hw_a100)
    assert "vllm 1.0" in p.load_record(path).refuted_by
    assert "rigtune's check was wrong" in res.error


class _FailingPair:
    def __init__(self, kind="draft_refused"):
        self.kind, self.launches = kind, 0

    def run(self, cfg, stage):
        if cfg.backend == "vllm" and "org/bad" in cfg.spec_decode:
            self.launches += 1
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                               error=f"failed to start (rc=1)\n{VLLM}", failure_kind=self.kind)
        return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(tok_s=100, requests=20, output_tokens=100))


@pytest.mark.parametrize("kind", ["draft_refused", "draft_unavailable", "oom_startup", "startup_other"])
def test_default_options_block_no_pair(kind):
    """Table mode launches what it launched before these kinds existed: no failure skips a later candidate."""
    runner = _FailingPair(kind)
    search = StagedSearch(objective="throughput", runner=runner)
    cfg = Config(backend="vllm", quant="bf16", spec_decode="draft:org/bad:4")
    for batch in (64, 128, 256):
        assert search._run(cfg.model_copy(update={"batch": batch}), "spec").failure_kind == kind
    assert runner.launches == 3 and not search._pair_blocked
    assert not any("blocked" in note for note in search.notes)


def test_failed_candidate_blocks_only_itself():
    search = StagedSearch(objective="throughput", runner=_FailingPair(), block_failed_pairs=True)
    cfg = Config(backend="vllm", quant="bf16", spec_decode="draft:org/bad:4")
    search._run(cfg, "spec")
    assert search._run(cfg.model_copy(update={"batch": 128}), "spec").failure_kind == "skipped_blocked"
    assert search._run(cfg.model_copy(update={"spec_decode": "draft:org/good:4"}), "spec").ok
    assert search._run(cfg.model_copy(update={"backend": "sglang"}), "spec").ok


def test_old_result_without_kind_classifies_as_before():
    cfg = Config(backend="vllm", quant="bf16")
    res = TrialResult(config=cfg, stage="quant", launched=False, error="failed to start", metrics=TrialMetrics())
    assert failure_kind(res) == "startup_other"
