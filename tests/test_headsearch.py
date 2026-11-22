"""rigtune.headsearch: drafters found on a Hub that answers with what the real one held on 21 Nov 2026."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rigtune import headsearch, speculative
from rigtune.headsearch import READ_QUOTA, core_name, kept_or_search, name_consistent
from rigtune.models import ArchInfo, ModelSpec

pytestmark = pytest.mark.real_head_search


@pytest.fixture(autouse=True)
def _hub_on(monkeypatch):
    """The suite sets HF_HUB_OFFLINE, which the search honours: these tests turn it off and pass a fake Hub."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)


@pytest.fixture(autouse=True)
def _empty_hf_cache(monkeypatch):
    """Offline, the search falls back to the listed heads in the local Hugging Face cache: never this machine's."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)

FIXTURES = Path(__file__).parent / "fixtures" / "heads"
# Real sizes of the two listed heads' only weight file (pytorch_model.bin); every other fixture's weights are sized
# 1000 bytes a file, which the search reads from the listing like any other.
SIZES = {"lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B": 849768795, "AngelSlim/Qwen3-8B_eagle3": 799493246}


def _fixture(repo: str) -> dict:
    return json.loads((FIXTURES / (repo.replace("/", "__") + ".json")).read_text(encoding="utf-8"))


def _target(layers: int, hidden: int, vocab: int, arch: str = "unknown") -> ArchInfo:
    return ArchInfo(architecture=arch, num_layers=layers, hidden_size=hidden, num_attention_heads=32, num_kv_heads=8,
                    head_dim=128, vocab_size=vocab)


QWEN3_8B = _target(36, 4096, 151936)
LLAMA31_8B = _target(32, 4096, 128256)
GPT_OSS_20B = _target(24, 2880, 201088, "GptOssForCausalLM")


class FakeHub:
    """A Hub whose listings are given per query, whose repositories are the fixtures, and which counts its calls."""

    def __init__(self, tmp_path, listings, base_models=None, gated=(), downloads=None, fail=()):
        self.tmp, self.listings, self.base_models = tmp_path, listings, base_models or {}
        self.gated, self.downloads, self.fail = set(gated), downloads or {}, set(fail)
        self.calls = []

    def _listed(self, repo):
        cfg = _fixture(repo)["config"]
        return SimpleNamespace(id=repo, downloads=self.downloads.get(repo, 100), gated="auto" if repo in self.gated
                               else False, config={"architectures": cfg.get("architectures")},
                               card_data={"base_model": self.base_models.get(repo)})

    def list_models(self, search=None, filter=None):
        key = search or filter
        self.calls.append(("list", key))
        if key in self.fail:
            raise OSError("rate limited")
        return [self._listed(r) for r in self.listings.get(key, [])]

    def model_info(self, repo, revision=None):
        self.calls.append(("info", repo))
        if repo in self.fail:
            raise OSError("no network")
        if not (FIXTURES / (repo.replace("/", "__") + ".json")).exists():  # the target itself
            return SimpleNamespace(sha="t" * 40, siblings=[], card_data={})
        data = _fixture(repo)
        size = SIZES.get(repo)
        siblings = [SimpleNamespace(rfilename=n, size=size or 1000 if n.endswith((".bin", ".safetensors")) else 10)
                    for n in data["files"]]
        return SimpleNamespace(sha=data["revision"], siblings=siblings, card_data={})

    def download(self, repo, filename, revision=None):
        self.calls.append(("download", repo))
        path = self.tmp / f"{repo.replace('/', '__')}-{filename}"
        path.write_text(json.dumps(_fixture(repo)["config"]), encoding="utf-8")
        return str(path)


QWEN3_8B_LISTINGS = {
    "Qwen3-8B eagle": ["AngelSlim/Qwen3-8B_eagle3", "RedHatAI/Qwen3-8B-speculator.eagle3",
                       "nm-testing/Speculator-Qwen3-8B-Eagle3", "Tengyunw/qwen3_30b_moe_eagle3",
                       "AvitoTech/avibe-eagle"],
    "Qwen3-8B dflash": ["z-lab/Qwen3-8B-DFlash-b16"],
    "Qwen3-8B dspark": ["deepseek-ai/dspark_qwen3_8b_block7"],
    "Qwen3-8B speculator": ["RedHatAI/Qwen3-8B-speculator.eagle3"],
}


def test_the_names_a_drafter_carries_for_its_target():
    assert core_name("Qwen/Qwen3-30B-A3B-Instruct-2507") == "Qwen3-30B-A3B"
    assert core_name("unsloth/Meta-Llama-3.1-8B-Instruct") == "Llama-3.1-8B"
    good = [("Qwen/Qwen3-8B", "deepseek-ai/dspark_qwen3_8b_block7"),
            ("unsloth/Meta-Llama-3.1-8B-Instruct", "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"),
            ("Qwen/Qwen3-30B-A3B-Instruct-2507", "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"),
            ("Qwen/Qwen3-8B-FP8", "RedHatAI/Qwen3-8B-FP8-dynamic-speculator.eagle3"),
            ("google/gemma-4-12B-it", "z-lab/gemma4-12B-it-DFlash")]
    # Traps seen on the Hub: another variant, another size, a quantized copy, another release, a newer family.
    bad = [("Qwen/Qwen3-8B", "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3"),
           ("Qwen/Qwen3-8B", "Tengyunw/qwen3_30b_moe_eagle3"),
           ("Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-FP8-dynamic-speculator.eagle3"),
           ("Qwen/Qwen3-8B", "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized-w4a16"),
           ("Qwen/Qwen3-8B", "x/Qwen3.5-8B-eagle3"),
           ("Qwen/Qwen3-30B-A3B", "z-lab/Qwen3-Coder-30B-A3B-DFlash"),
           ("Qwen/Qwen3-30B-A3B", "nvidia/Qwen3-30B-A3B-Thinking-2507-Eagle3"),
           ("Qwen/Qwen3-30B-A3B-Instruct-2507", "RedHatAI/Qwen3-30B-A3B-speculator.eagle3"),
           ("openai/gpt-oss-20b", "lmsys/EAGLE3-gpt-oss-120b-bf16"),
           ("unsloth/Meta-Llama-3.1-8B-Instruct", "x/Llama-3-8B-eagle3")]
    assert all(name_consistent(r, t) for t, r in good)
    assert not any(name_consistent(r, t) for t, r in bad)


def test_a_listed_head_comes_first_and_both_engines_get_it(tmp_path):
    hub = FakeHub(tmp_path, QWEN3_8B_LISTINGS)
    spec = ModelSpec(hf_id="Qwen/Qwen3-8B")
    for engine in ("vllm", "sglang"):
        assert speculative.find_eagle3_head(spec, QWEN3_8B, engine=engine, hub=hub, hub_off=False) == (
            "AngelSlim/Qwen3-8B_eagle3", 799493246)  # sized from its pytorch_model.bin
    found, _ = kept_or_search("Qwen/Qwen3-8B", None, hub_off=True)
    read = [c.repo for c in found.candidates if c.config is not None]
    assert "Tengyunw/qwen3_30b_moe_eagle3" not in read  # dropped by its name before anything was read
    # The rest are read and judged per engine: vLLM also loads the speculators head and the plain-Llama head.
    vllm = [c.repo for c in headsearch.heads_for(found, QWEN3_8B, "vllm")]
    assert vllm[0] == "AngelSlim/Qwen3-8B_eagle3"
    assert {"RedHatAI/Qwen3-8B-speculator.eagle3", "nm-testing/Speculator-Qwen3-8B-Eagle3"} <= set(vllm)
    assert [c.repo for c in headsearch.heads_for(found, QWEN3_8B, "sglang")] == ["AngelSlim/Qwen3-8B_eagle3"]
    assert "AvitoTech/avibe-eagle" not in vllm  # a 36-layer "head" with its own vocabulary
    dflash = [c.repo for c in headsearch.heads_for(found, QWEN3_8B, "vllm", "dflash")]
    assert dflash == ["z-lab/Qwen3-8B-DFlash-b16"]  # found and kept for the DFlash launch to come


def test_a_head_nobody_listed_is_found_and_said_so(tmp_path):
    hub = FakeHub(tmp_path, {"gpt-oss-20b eagle": ["zhuyksir/EAGLE3-gpt-oss-20b-bf16"]})
    spec = ModelSpec(hf_id="openai/gpt-oss-20b")
    found = speculative.find_eagle3_head(spec, GPT_OSS_20B, engine="sglang", hub=hub, hub_off=False)
    assert found == ("zhuyksir/EAGLE3-gpt-oss-20b-bf16", 1000)
    assert speculative.found_head_notes("sglang", spec, GPT_OSS_20B, found[0]) == [
        "sglang: EAGLE3 head zhuyksir/EAGLE3-gpt-oss-20b-bf16 found by a Hub search; its config fits openai/gpt-oss-20b "
        "on sglang (rigtune.heads); it has not been run by rigtune before"]
    assert speculative.found_head_notes("sglang", ModelSpec(hf_id="Qwen/Qwen3-8B"), QWEN3_8B,
                                        "AngelSlim/Qwen3-8B_eagle3") == []
    # Its layers sit in eagle_config only, which vLLM 0.33's default runner never reads: not offered there.
    assert speculative.find_eagle3_head(spec, GPT_OSS_20B, engine="vllm", hub=hub, hub_off=False) is None


def test_downloads_rank_the_short_list_and_a_base_model_tie_breaks_equal_counts(tmp_path):
    """On the real Hub, ranking base_model children first spent Qwen3-8B's reads on experiments with a few hundred
    downloads and never read RedHatAI's head, downloaded 51,787 times."""
    listings = {"Qwen3-8B eagle": ["RedHatAI/Qwen3-8B-speculator.eagle3", "nm-testing/Speculator-Qwen3-8B-Eagle3",
                                   "AvitoTech/avibe-eagle"],
                "base_model:Qwen/Qwen3-8B": ["nm-testing/Speculator-Qwen3-8B-Eagle3", "AvitoTech/avibe-eagle"]}
    ties = {"nm-testing/Speculator-Qwen3-8B-Eagle3": "Qwen/Qwen3-8B", "AvitoTech/avibe-eagle": "Qwen/Qwen3-8B"}
    hub = FakeHub(tmp_path, listings, base_models=ties, downloads={"RedHatAI/Qwen3-8B-speculator.eagle3": 51787,
                                                                   "nm-testing/Speculator-Qwen3-8B-Eagle3": 9,
                                                                   "AvitoTech/avibe-eagle": 9})
    found = headsearch.search("Qwen/Qwen3-8B", None, seeds=["AngelSlim/Qwen3-8B_eagle3"], hub=hub)
    assert [c.repo for c in found.candidates][:2] == ["AngelSlim/Qwen3-8B_eagle3",  # listed by rigtune: first
                                                      "RedHatAI/Qwen3-8B-speculator.eagle3"]
    level = FakeHub(tmp_path, listings, base_models={"nm-testing/Speculator-Qwen3-8B-Eagle3": "Qwen/Qwen3-8B"})
    order = [c.repo for c in headsearch.search("Qwen/Qwen3-8B", None, hub=level).candidates]
    assert order.index("nm-testing/Speculator-Qwen3-8B-Eagle3") < order.index("RedHatAI/Qwen3-8B-speculator.eagle3")


def test_gated_repositories_are_skipped(tmp_path):
    listings = {"Qwen3-8B eagle": ["RedHatAI/Qwen3-8B-speculator.eagle3", "nm-testing/Speculator-Qwen3-8B-Eagle3"]}
    gated = FakeHub(tmp_path, listings, gated={"RedHatAI/Qwen3-8B-speculator.eagle3"})
    assert [c.repo for c in headsearch.search("Qwen/Qwen3-8B", None, hub=gated).candidates] == [
        "nm-testing/Speculator-Qwen3-8B-Eagle3"]


def test_each_kind_of_drafter_has_its_own_read_quota(tmp_path, monkeypatch):
    """gemma-4-12B-it's eight most downloaded drafters were seven DFlash and DSpark ones: one shared budget left none
    for EAGLE3."""
    repos = sorted(p.stem.replace("__", "/") for p in FIXTURES.glob("*.json"))
    monkeypatch.setattr(headsearch, "name_consistent", lambda repo, target: True)  # all twenty short-listed
    hub = FakeHub(tmp_path, {"Qwen3-8B eagle": repos})
    found = headsearch.search("Qwen/Qwen3-8B", None, hub=hub)
    read = [c for c in found.candidates if c.config is not None]
    assert sum(1 for kind, _ in hub.calls if kind == "download") == len(read) == sum(READ_QUOTA.values())
    kinds = [headsearch.guessed_kind(c.repo, c.config.get("architectures")) for c in read]
    assert {k: kinds.count(k) for k in READ_QUOTA} == READ_QUOTA
    lists = sum(1 for kind, _ in hub.calls if kind == "list")
    assert lists == len(headsearch.SEARCH_WORDS) + 1  # one base_model page for the target, which has none of its own


def test_a_search_is_kept_for_the_day_and_read_offline(tmp_path):
    hub = FakeHub(tmp_path, QWEN3_8B_LISTINGS)
    kept_or_search("Qwen/Qwen3-8B", "a" * 40, hub=hub, today="2026-11-21")
    calls = len(hub.calls)
    kept_or_search("Qwen/Qwen3-8B", "a" * 40, hub=hub, today="2026-11-21")
    assert len(hub.calls) == calls  # the same day: nothing asked
    kept_or_search("Qwen/Qwen3-8B", "a" * 40, hub=hub, today="2026-11-22")
    assert len(hub.calls) > calls  # the next day: searched again
    found, notes = kept_or_search("Qwen/Qwen3-8B", "a" * 40, hub_off=True, today="2026-12-25")
    assert found.day == "2026-11-22" and "the Hub is off" in notes[0]
    assert kept_or_search("Qwen/Qwen3-8B", "b" * 40, hub_off=True) == (None, [
        "no drafter search kept for Qwen/Qwen3-8B; the Hub is off, so none was made"])  # another commit


def test_a_hub_that_cannot_be_asked_is_told_apart_from_no_head_fitting(tmp_path):
    spec = ModelSpec(hf_id="Qwen/Qwen3-8B")
    everything = {"Qwen/Qwen3-8B", *(f"Qwen3-8B {w}" for w in headsearch.SEARCH_WORDS), "base_model:Qwen/Qwen3-8B"}
    with pytest.raises(speculative.HubLookupError, match="could not search the Hub"):
        speculative.find_eagle3_head(spec, QWEN3_8B, engine="vllm", hub=FakeHub(tmp_path, {}, fail=everything),
                                     hub_off=False)
    with pytest.raises(speculative.HubLookupError, match="no drafter search kept"):
        speculative.find_eagle3_head(spec, QWEN3_8B, engine="vllm", hub_off=True)
    # A listing that fails is noted, the rest still searched, and the search not reused: a head found is used, and
    # none found is "could not ask", not "none fits" (gpt-oss-20b has no listed head; zhuyksir's fits SGLang only).
    gpt_oss = ModelSpec(hf_id="openai/gpt-oss-20b")
    hub = FakeHub(tmp_path, {"gpt-oss-20b eagle": ["zhuyksir/EAGLE3-gpt-oss-20b-bf16"]}, fail={"gpt-oss-20b dflash"})
    assert speculative.find_eagle3_head(gpt_oss, GPT_OSS_20B, engine="sglang", hub=hub, hub_off=False)[0] == (
        "zhuyksir/EAGLE3-gpt-oss-20b-bf16")
    with pytest.raises(speculative.HubLookupError, match="incomplete.*gpt-oss-20b dflash"):
        speculative.find_eagle3_head(gpt_oss, GPT_OSS_20B, engine="vllm", hub=hub, hub_off=False)
    assert sum(1 for kind, key in hub.calls if key == "gpt-oss-20b eagle") == 2  # searched again, not reused
    # Complete, and nothing fits vLLM: None.
    whole = FakeHub(tmp_path, {"gpt-oss-20b eagle": ["zhuyksir/EAGLE3-gpt-oss-20b-bf16"]})
    assert speculative.find_eagle3_head(gpt_oss, GPT_OSS_20B, engine="vllm", hub=whole, hub_off=False) is None


def test_hf_hub_offline_keeps_the_search_off_the_hub(tmp_path, monkeypatch):
    hub = FakeHub(tmp_path, QWEN3_8B_LISTINGS)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(speculative.HubLookupError):
        speculative.find_eagle3_head(ModelSpec(hf_id="Qwen/Qwen3-8B"), QWEN3_8B, engine="vllm", hub=hub,
                                     hub_off=False)
    assert hub.calls == []


def test_offline_with_nothing_kept_the_listed_heads_come_from_the_local_cache(tmp_path, monkeypatch):
    """As before the search existed: a listed head whose config is in the Hugging Face cache is offered offline."""
    import huggingface_hub

    snapshot = tmp_path / "snapshots" / ("9" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps(_fixture("AngelSlim/Qwen3-8B_eagle3")["config"]))
    (snapshot / "pytorch_model.bin").write_bytes(b"x" * 123)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, filename, **kw: str(snapshot / filename) if repo.startswith("AngelSlim") else None)
    spec = ModelSpec(hf_id="Qwen/Qwen3-8B")
    assert speculative.find_eagle3_head(spec, QWEN3_8B, engine="sglang", hub_off=True) == (
        "AngelSlim/Qwen3-8B_eagle3", 123)


def test_what_cannot_be_read_is_skipped_never_fatal(tmp_path, monkeypatch):
    spec = ModelSpec(hf_id="Qwen/Qwen3-8B")
    hub = FakeHub(tmp_path, QWEN3_8B_LISTINGS)
    path = headsearch._cache_path("Qwen/Qwen3-8B", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")  # a kept file of another shape
    assert speculative.find_eagle3_head(spec, QWEN3_8B, engine="vllm", hub=hub, hub_off=False)[0] == (
        "AngelSlim/Qwen3-8B_eagle3")
    # A config that is not an object, and one whose eagle_config is a string, are not judged.
    found, _ = kept_or_search("Qwen/Qwen3-8B", None, hub_off=True)
    found.candidates[0].config = ["not", "an", "object"]
    found.candidates[1].config = {"architectures": ["LlamaForCausalLMEagle3"], "eagle_config": "oops",
                                  "eagle_aux_hidden_state_layer_ids": None, "num_hidden_layers": 1}
    assert isinstance(headsearch.heads_for(found, QWEN3_8B, "sglang"), list)
    # A listing whose config is not an object is skipped like one without a drafter's architecture.
    odd = SimpleNamespace(id="x/Qwen3-8B-eagle3", downloads=1, gated=False, config="?", card_data=None)
    bare = FakeHub(tmp_path, {})
    bare.list_models = lambda **kw: [odd]
    assert headsearch.search("Qwen/Qwen3-8B", None, hub=bare).candidates == []


def test_the_engines_drafting_with_different_heads_is_said(tmp_path, monkeypatch):
    """Each engine gets its own best head. Without AngelSlim listed, vLLM's best for Qwen3-8B is RedHatAI's speculators
    head, downloaded 51,787 times, which SGLang cannot load; SGLang's is AngelSlim's."""
    monkeypatch.setattr(speculative, "eagle3_heads_for", lambda hf_id: ())
    listings = {"Qwen3-8B eagle": ["RedHatAI/Qwen3-8B-speculator.eagle3", "AngelSlim/Qwen3-8B_eagle3"]}
    hub = FakeHub(tmp_path, listings, downloads={"RedHatAI/Qwen3-8B-speculator.eagle3": 51787,
                                                 "AngelSlim/Qwen3-8B_eagle3": 5109})
    spec = ModelSpec(hf_id="Qwen/Qwen3-8B")
    vllm = speculative.find_eagle3_head(spec, QWEN3_8B, engine="vllm", hub=hub, hub_off=False)[0]
    sglang = speculative.find_eagle3_head(spec, QWEN3_8B, engine="sglang", hub=hub, hub_off=False)[0]
    assert (vllm, sglang) == ("RedHatAI/Qwen3-8B-speculator.eagle3", "AngelSlim/Qwen3-8B_eagle3")
    assert speculative.found_head_notes("vllm", spec, QWEN3_8B, vllm)[-1] == (
        "vllm: drafts with EAGLE3 head RedHatAI/Qwen3-8B-speculator.eagle3; sglang's best head for this model is "
        "AngelSlim/Qwen3-8B_eagle3, which vllm does not load or ranks lower")


def test_a_local_directory_is_never_searched_for(tmp_path):
    hub = FakeHub(tmp_path, QWEN3_8B_LISTINGS)
    assert speculative.find_eagle3_head(ModelSpec(hf_id=str(tmp_path)), QWEN3_8B, engine="vllm", hub=hub,
                                        hub_off=False) is None
    assert hub.calls == []


def test_listed_heads_are_still_the_seeds():
    assert speculative.eagle3_heads_for("unsloth/Meta-Llama-3.1-8B-Instruct") == (
        "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",)
    assert speculative.eagle3_heads_for("meta-llama/Llama-3.3-70B-Instruct") == ()


def _http(status: int) -> Exception:
    """A Hub HTTP failure as transient() reads it: its response's status code."""
    exc = OSError(f"HTTP {status}")
    exc.response = SimpleNamespace(status_code=status)
    return exc


def test_a_repository_s_own_answer_completes_a_search_and_a_passing_failure_does_not(tmp_path):
    """inference-optimization/ctest-Qwen3.5-9B-sliding-window-speculator.dflash has no config.json: had that left the
    search incomplete, every search for Qwen3.5-9B would be redone on every call and end in "could not ask"."""
    from huggingface_hub import errors

    class Broken(FakeHub):
        def __init__(self, *a, error=None, **k):
            super().__init__(*a, **k)
            self.error = error

        def download(self, repo, filename, revision=None):
            if repo == "nm-testing/Speculator-Qwen3-8B-Eagle3":
                raise self.error
            return super().download(repo, filename, revision)

    listings = {"Qwen3-8B eagle": ["AngelSlim/Qwen3-8B_eagle3", "nm-testing/Speculator-Qwen3-8B-Eagle3"]}
    missing = Broken(tmp_path, listings, error=errors.EntryNotFoundError("config.json not found"))
    assert headsearch.search("Qwen/Qwen3-8B", None, hub=missing).complete is True
    limited = Broken(tmp_path, listings, error=_http(429))
    assert headsearch.search("Qwen/Qwen3-8B", None, hub=limited).complete is False
    refused = Broken(tmp_path, listings, error=_http(404))
    assert headsearch.search("Qwen/Qwen3-8B", None, hub=refused).complete is True
    assert headsearch.transient(ConnectionError("reset")) and headsearch.transient(TimeoutError())
    assert not headsearch.transient(ValueError("config.json is not an object"))
