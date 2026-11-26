"""The drop-in prefix: `rigtune [options] vllm serve MODEL [flags]` is `rigtune serve` seeded with that command."""

import shlex
from pathlib import Path

import pytest

from rigtune import cli
from rigtune.cli import drop_in_argv
from rigtune.seed import parse_command, server_flags

QWEN = "Qwen/Qwen3-8B"


def test_a_vllm_command_becomes_a_seeded_serve_on_its_own_port():
    env = {}
    out = drop_in_argv(["vllm", "serve", QWEN, "--max-model-len", "8192", "--port", "9000"], env)
    assert out == ["serve", QWEN, "--seed", f"vllm serve {QWEN} --max-model-len 8192", "--backend", "vllm",
                   "--port", "9000", "--host", "0.0.0.0"]  # vLLM listens on every interface with no --host
    assert env == {}


@pytest.mark.parametrize("command, port, host", [
    (["vllm", "serve", QWEN], "8000", "0.0.0.0"),  # each engine's own defaults, so clients do not change
    (["python", "-m", "sglang.launch_server", "--model-path", QWEN], "30000", "127.0.0.1"),
    (["python3", "-m", "sglang.launch_server", "--model-path", QWEN, "--port=30001"], "30001", "127.0.0.1"),
    (["/usr/bin/python3.12", "-m", "vllm.entrypoints.openai.api_server", "--model", QWEN], "8000", "0.0.0.0"),
    (["sglang", "serve", "--model-path", QWEN, "--host", "0.0.0.0"], "30000", "0.0.0.0"),
    (["vllm", "serve", QWEN, "--host", "127.0.0.1", "--port", "1", "--port", "2"], "2", "127.0.0.1"),  # last wins
])
def test_each_server_command_form_listens_where_it_would_have(command, port, host):
    out = drop_in_argv(command, {})
    assert out[:2] == ["serve", QWEN]
    assert out[out.index("--port") + 1] == port
    assert out[out.index("--host") + 1] == host
    engine = "sglang" if "sglang" in " ".join(command) else "vllm"
    assert out[out.index("--backend") + 1] == engine
    seed = out[out.index("--seed") + 1]
    assert "--port" not in seed and "--host" not in seed  # a new port is not a new calibration
    assert parse_command(seed).engine == engine


def test_rigtune_options_before_the_command_are_serve_s_and_verbose_stays_global():
    out = drop_in_argv(["-v", "--budget", "20m", "--workload", "long", "--", "vllm", "serve", QWEN], {})
    assert out[0] == "-v" and out[1:3] == ["serve", QWEN]
    assert out[-4:] == ["--budget", "20m", "--workload", "long"]  # last: one left without a value takes none of ours
    assert "--" not in out
    assert drop_in_argv(["--quant", "vllm", "serve", QWEN], {})[-1] == "--quant"


def test_environment_variables_after_rigtune_are_refused_with_where_they_go():
    with pytest.raises(ValueError, match="set them before it") as exc:
        drop_in_argv(["CUDA_VISIBLE_DEVICES=1", "vllm", "serve", QWEN], {})
    assert "=1" not in str(exc.value)


@pytest.mark.parametrize("port", ["", "0", "65536", "-1", "x"])
def test_a_port_that_is_not_one_is_refused(port):
    with pytest.raises(ValueError, match="not a port number"):
        drop_in_argv(["vllm", "serve", QWEN, f"--port={port}"], {})


def test_help_in_the_command_shows_serve_s_help():
    assert drop_in_argv(["vllm", "serve", QWEN, "--help"], {}) == ["serve", "--help"]


def test_a_renamed_model_is_refused_up_front_rather_than_404_its_clients():
    with pytest.raises(ValueError, match="served-model-name"):
        drop_in_argv(["vllm", "serve", QWEN, "--served-model-name", "qwen"], {})


def test_the_command_s_hub_token_reaches_the_download_and_not_the_seed():
    env = {}
    out = drop_in_argv(["vllm", "serve", QWEN, "--hf-token", "hf_secret"], env)
    assert env == {"HF_TOKEN": "hf_secret"} and not any("hf_secret" in t for t in out)
    assert out == drop_in_argv(["vllm", "serve", QWEN], {})  # a new token is not a new calibration
    with pytest.raises(ValueError) as exc:
        drop_in_argv(["vllm", "serve", QWEN, "--hf-token", "hf_a"], {"HF_TOKEN": "hf_b"})
    assert "hf_" not in str(exc.value)


@pytest.mark.parametrize("argv", [
    [], ["--help"], ["serve", QWEN], ["version"],
    ["calibrate", QWEN, "--seed", f"vllm serve {QWEN}"],  # a command word first: rigtune's own command
    ["serve", "vllm", "serve", QWEN],
    ["vllm"], ["vllm", "--help"],
])
def test_rigtune_s_own_commands_are_left_alone(argv):
    assert drop_in_argv(argv, {}) is None


@pytest.mark.parametrize("before", [["--port", "1"], ["--seed", "recipe"], ["--backend", "sglang"], ["--host=x"],
                                    ["--profile", "p.json"]])
def test_what_the_command_says_itself_is_refused_before_it(before):
    with pytest.raises(ValueError, match="the command itself says that"):
        drop_in_argv([*before, "vllm", "serve", QWEN], {})


def test_the_command_s_api_key_becomes_rigtune_s_and_never_enters_argv():
    env = {}
    out = drop_in_argv(["vllm", "serve", QWEN, "--api-key", "sk-secret-1", "--max-num-seqs", "64"], env)
    assert env == {"RIGTUNE_API_KEY": "sk-secret-1"}
    assert not any("sk-secret-1" in t for t in out)
    env = {"RIGTUNE_API_KEY": "sk-secret-1"}  # the same key already set: fine
    assert drop_in_argv(["vllm", "serve", QWEN, "--api-key=sk-secret-1"], env) is not None
    with pytest.raises(ValueError) as exc:
        drop_in_argv(["vllm", "serve", QWEN, "--api-key", "sk-secret-2"], {"RIGTUNE_API_KEY": "sk-other"})
    assert "sk-" not in str(exc.value)
    with pytest.raises(ValueError, match="one way"):
        drop_in_argv(["--api-key-env", "MY_KEY", "vllm", "serve", QWEN, "--api-key", "k"], {})


def test_a_key_change_is_not_a_new_calibration():
    a = drop_in_argv(["vllm", "serve", QWEN, "--api-key", "one", "--port", "1"], {})
    b = drop_in_argv(["vllm", "serve", QWEN, "--api-key", "two", "--port", "2"], {})
    assert a[a.index("--seed") + 1] == b[b.index("--seed") + 1]


def test_a_command_rigtune_cannot_read_fails_without_echoing_it():
    env = {}
    with pytest.raises(ValueError) as exc:
        drop_in_argv(["vllm", "serve", "--hf-token", "hf_abc", "--max-model-len"], env)
    assert "hf_abc" not in str(exc.value) and env == {}
    with pytest.raises(ValueError, match="names no model"):
        drop_in_argv(["vllm", "serve", "--max-model-len", "8192"], {})


def test_server_flags_keeps_every_other_token_in_order():
    tokens = ["vllm", "serve", QWEN, "--port", "9", "--speculative-config", '{"method": "mtp"}', "--host=h",
              "--enforce-eager"]
    engine, kept, found = server_flags(tokens)
    assert engine == "vllm" and found == {"port": "9", "host": "h"}
    assert kept == ["vllm", "serve", QWEN, "--speculative-config", '{"method": "mtp"}', "--enforce-eager"]
    assert shlex.split(shlex.join(kept)) == kept


def test_main_runs_the_drop_in_as_serve_and_reports_a_refusal(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(type(cli.app), "__call__", lambda self, args=None, prog_name=None: seen.append(args))
    monkeypatch.delenv("RIGTUNE_API_KEY", raising=False)
    cli.main(["vllm", "serve", QWEN])
    cli.main(["version"])
    assert seen[0][:2] == ["serve", QWEN] and seen[1] == ["version"]
    with pytest.raises(SystemExit) as exc:
        cli.main(["--port", "1", "vllm", "serve", QWEN])
    assert exc.value.code == 2 and "the command itself says that" in capsys.readouterr().err


def test_the_installed_command_is_main():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'rigtune = "rigtune.cli:main"' in text
