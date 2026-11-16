import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rigtune import pairing as p
from tests.test_pairing import pair, target


def save():
    a, b = target(), pair()
    return p.save_record(a, b, "draft", p.check_pair(a, b, "draft", 4096))


def test_round_trip():
    path = save()
    r = p.load_record(path)
    assert r.verdict.status == "ok"
    assert r.target.repo == "org/target" and r.second.sha == "b" * 40


def test_unknown_is_never_kept():
    assert p.save_record(target(), pair(), "draft", p.PairVerdict(status="unknown")) is None


def test_file_name_is_short_hex():
    path = save()
    assert len(path.name) < 40 and int(path.stem, 16) > 0


@pytest.mark.parametrize("data", ["", "{", "{}", "[]"])
def test_corrupt_or_empty_record_reads_as_absent(tmp_path, data):
    path = tmp_path / "bad.json"
    path.write_text(data)
    assert p.load_record(path) is None


def test_two_writers_leave_one_whole_record():
    with ThreadPoolExecutor(2) as pool:
        paths = list(pool.map(lambda _: save(), range(2)))
    assert paths[0] == paths[1] and p.load_record(paths[0]).verdict.status == "ok"
    assert list(paths[0].parent.glob("*.tmp")) == []


def test_replace_retries_when_the_file_is_held(monkeypatch):
    real = p.os.replace
    count = []
    def held_once(src, dest):
        count.append(src)
        if len(count) == 1:
            raise PermissionError("held")
        return real(src, dest)
    monkeypatch.setattr(p.os, "replace", held_once)
    assert p.load_record(save()).verdict.status == "ok"
    assert len(count) == 2


def test_record_has_no_local_path_or_user_name(tmp_path):
    private = tmp_path / "private-person" / "draft"
    private.mkdir(parents=True)
    (private / "config.json").write_text(json.dumps({"vocab_size": 2}))
    b = p.read_files(str(private))
    path = p.save_record(target(), b, "draft", p.PairVerdict(status="refused", reasons=["vocab_differs"]))
    text = path.read_text()
    assert "private-person" not in text and str(tmp_path) not in text


def test_nothing_is_written_under_the_real_home(tmp_path, monkeypatch):
    home = tmp_path / "untouched"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    save()
    assert not list(home.iterdir())
