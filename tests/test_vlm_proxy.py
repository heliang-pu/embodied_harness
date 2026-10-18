"""`data annotate --vlm-api openai`: a local endpoint that adds the key and speaks the OpenAI API."""

import base64
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from conftest import RUN_CONFIG
from embodied_harness.cli import main
from embodied_harness.data import annotate
from embodied_harness.data.vlm_proxy import (
    OpenAIProxy,
    UnsupportedContent,
    chat_to_responses,
    codex_prompt,
    responses_to_chat,
)
from fake_openai import FakeChatServer
from lerobot_fixture import write_dataset
from test_annotate import FAKE_ANNOTATE, TEMPLATES

IMAGE = "data:image/png;base64,iVBORw0KGgo="
KEY = "sk-test-not-a-real-key"
CHAT = {  # what lerobot-annotate sends
    "model": "gpt-6-sol",
    "messages": [
        {"role": "system", "content": "You segment robot videos."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Segment this episode."},
                {"type": "image_url", "image_url": {"url": IMAGE}},
            ],
        },
    ],
    "max_tokens": 512,
    "temperature": 0.2,
}


def responses_reply(text='{"subtasks": []}', status="completed"):
    return {
        "id": "resp_1",
        "model": "gpt-6-sol",
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 300,
            "output_tokens_details": {"reasoning_tokens": 250},
            "input_tokens_details": {"cached_tokens": 100},
        },
    }


def post(url, body):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def upstream():
    servers = []

    def start(reply):
        server = FakeChatServer(reply)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def proxy_for(server, monkeypatch, **kwargs):
    monkeypatch.setenv("TEST_OPENAI_KEY", KEY)
    return OpenAIProxy(
        model="gpt-6-sol",
        upstream=server.base_url,
        api_key_env="TEST_OPENAI_KEY",
        backoff_s=0.01,
        use_env_proxy=False,
        **kwargs,
    )


def test_chat_requests_become_responses_requests():
    request = chat_to_responses(CHAT, reasoning_effort="low", min_output_tokens=4096)
    assert request["model"] == "gpt-6-sol" and request["store"] is False
    assert request["input"][0] == {
        "role": "system",
        "content": [{"type": "input_text", "text": "You segment robot videos."}],
    }
    assert request["input"][1]["content"][1] == {"type": "input_image", "image_url": IMAGE, "detail": "auto"}
    assert request["max_output_tokens"] == 4096  # 512 would leave nothing after the reasoning
    assert request["reasoning"] == {"effort": "low"} and "temperature" not in request
    schema = {
        "type": "json_schema",
        "json_schema": {"name": "plan", "schema": {"type": "object"}, "strict": True},
    }
    formatted = chat_to_responses(
        {**CHAT, "response_format": schema}, reasoning_effort=None, min_output_tokens=0
    )
    assert formatted["text"]["format"]["name"] == "plan" and "reasoning" not in formatted


def test_responses_replies_read_like_chat_completions():
    reply = responses_to_chat(responses_reply('{"label": "grasp"}'), "gpt-6-sol")
    assert reply["choices"][0]["message"]["content"] == '{"label": "grasp"}'
    assert reply["choices"][0]["finish_reason"] == "stop"
    assert reply["usage"] == {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}
    assert (
        responses_to_chat(responses_reply("", status="incomplete"), "m")["choices"][0]["finish_reason"]
        == "length"
    )


def test_the_proxy_adds_the_key_and_translates(upstream, monkeypatch):
    server = upstream(lambda body, headers: (200, responses_reply('{"subtasks": [1]}')))
    with proxy_for(server, monkeypatch) as proxy:
        status, reply = post(proxy.base_url, CHAT)
        assert status == 200 and reply["choices"][0]["message"]["content"] == '{"subtasks": [1]}'
        assert server.paths == ["/v1/responses"]
        assert server.headers[0]["authorization"] == f"Bearer {KEY}"  # the client's "EMPTY" is replaced
        sent = server.requests[0]
        assert sent["max_output_tokens"] == 4096 and "temperature" not in sent and "messages" not in sent
        summary = proxy.usage.summary("gpt-6-sol")
    assert (
        "1 request(s) to gpt-6-sol: 1200 input token(s) (100 cached), 300 output token(s) (250 reasoning)"
        in summary
    )
    assert "about $0.01 at list price" in summary  # 1200 x $2/M + 300 x $10/M


def test_rate_limits_are_retried(upstream, monkeypatch):
    answers = iter([(429, {"error": {"message": "slow down"}}), (200, responses_reply())])
    server = upstream(lambda body, headers: next(answers))
    with proxy_for(server, monkeypatch) as proxy:
        status, _ = post(proxy.base_url, CHAT)
    assert status == 200 and proxy.usage.retries == 1 and len(server.requests) == 2


def test_errors_and_video_come_back_as_errors(upstream, monkeypatch):
    server = upstream(lambda body, headers: (400, {"error": {"message": "bad model"}}))
    with proxy_for(server, monkeypatch) as proxy:
        status, reply = post(proxy.base_url, CHAT)
        assert status == 400 and reply["error"]["message"] == "bad model"
        video = {
            **CHAT,
            "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "x"}}]}],
        }
        status, reply = post(proxy.base_url, video)
        assert status == 400 and "send frames as images" in reply["error"]["message"]
    assert proxy.usage.failures == 2 and len(server.requests) == 1  # the video never left the machine


def test_the_chat_endpoint_for_reasoning_models(upstream, monkeypatch):
    chat_reply = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}}], "usage": {}}
    server = upstream(lambda body, headers: (200, chat_reply))
    with proxy_for(server, monkeypatch, endpoint="chat") as proxy:
        status, _ = post(proxy.base_url, {**CHAT, "chat_template_kwargs": {"enable_thinking": False}})
    sent = server.requests[0]
    assert status == 200 and server.paths == ["/v1/chat/completions"]
    assert sent["max_completion_tokens"] == 4096 and "max_tokens" not in sent and "temperature" not in sent
    assert "chat_template_kwargs" not in sent and sent["reasoning_effort"] == "low"


def test_no_key_no_proxy(monkeypatch):
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_OPENAI_KEY is not set"):
        OpenAIProxy(model="gpt-6-sol", api_key_env="TEST_OPENAI_KEY")


# --- data annotate --vlm-api openai ---------------------------------------------------------------

CALLS_THE_VLM = r"""
import json, sys, urllib.request
base = next(arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--vlm.api_base="))
body = {"model": "gpt-6-sol", "max_tokens": 64, "temperature": 0.2,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "label"}]}]}
request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
reply = json.loads(opener.open(request, timeout=10).read())
assert reply["choices"][0]["message"]["content"] == '{"subtasks": []}', reply
"""


def test_annotate_through_the_openai_api(tmp_path, upstream, monkeypatch, capsys):
    server = upstream(lambda body, headers: (200, responses_reply()))
    script = tmp_path / "fake_lerobot_annotate.py"
    script.write_text(CALLS_THE_VLM + FAKE_ANNOTATE, encoding="utf-8")
    monkeypatch.setattr(annotate, "packaged_prompt", lambda name: TEMPLATES[name])
    monkeypatch.setattr(annotate, "annotate_executable", lambda: [sys.executable, str(script)])
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    dataset, out = write_dataset(tmp_path / "dataset"), tmp_path / "annotated"
    argv = ["data", "annotate", str(RUN_CONFIG), str(dataset), "--out", str(out), "--vlm-api", "openai"]
    argv += [
        "--vlm-model",
        "gpt-6-sol",
        "--vlm-upstream",
        server.base_url,
        "--episodes",
        "0,1",
        "--concurrency",
        "4",
    ]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "1 request(s) to gpt-6-sol" in printed and server.headers[0]["authorization"] == f"Bearer {KEY}"
    record = json.loads((out / ".embodied_harness" / "annotate.json").read_text())
    command = " ".join(record["command"])
    assert "--vlm.api_base=http://127.0.0.1:" in command and "--vlm.auto_serve=false" in command
    assert "--only_episodes=[0,1]" in command and "--vlm.client_concurrency=4" in command
    assert KEY not in command and KEY not in (out / ".embodied_harness" / "usage.json").read_text()
    usage = json.loads((out / ".embodied_harness" / "usage.json").read_text())
    assert (
        usage["requests"] == 1
        and usage["input_tokens"] == 1200
        and usage["cost_usd"] == pytest.approx(0.0054)
    )


def test_annotate_needs_a_model_and_a_key(tmp_path, monkeypatch, capsys):
    dataset = write_dataset(tmp_path / "dataset")
    base = [
        "data",
        "annotate",
        str(RUN_CONFIG),
        str(dataset),
        "--out",
        str(tmp_path / "a"),
        "--vlm-api",
        "openai",
    ]
    assert main(base) == 2
    assert "needs --vlm-model" in capsys.readouterr().err
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(annotate, "packaged_prompt", lambda name: TEMPLATES[name])
    assert main([*base, "--vlm-model", "gpt-6-sol"]) == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err
    assert main([*base, "--vlm-model", "gpt-6-sol", "--dry-run"]) == 0  # nothing is sent, no key needed
    assert "--vlm.api_base=http://127.0.0.1:<port>/v1" in capsys.readouterr().out


CODEX_CONFIG = """
model_provider = "custom"
model = "gpt-6-astra"

[model_providers.custom]
name = "custom"
wire_api = "responses"
requires_openai_auth = false
base_url = "https://relay.example/v1"
"""


def test_the_codex_provider_is_read_from_its_config(tmp_path, monkeypatch):
    from embodied_harness.data.vlm_proxy import codex_provider

    (tmp_path / "config.toml").write_text(CODEX_CONFIG, encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert codex_provider() == {
        "provider": "custom",
        "base_url": "https://relay.example/v1",
        "endpoint": "responses",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-6-astra",
    }
    (tmp_path / "config.toml").write_text('model = "gpt-6-sol"\n', encoding="utf-8")
    assert codex_provider()["base_url"] == "https://api.openai.com/v1"  # the built-in openai provider
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="no Codex config"):
        codex_provider()


def test_annotate_through_the_codex_provider(tmp_path, monkeypatch, capsys):
    home = tmp_path / "codex"
    home.mkdir()
    (home / "config.toml").write_text(CODEX_CONFIG, encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.setattr(annotate, "packaged_prompt", lambda name: TEMPLATES[name])
    dataset = write_dataset(tmp_path / "dataset")
    argv = [
        "data",
        "annotate",
        str(RUN_CONFIG),
        str(dataset),
        "--out",
        str(tmp_path / "a"),
        "--vlm-api",
        "codex",
    ]
    assert main([*argv, "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "VLM: gpt-6-astra via codex exec (the Codex CLI's own provider and login)" in printed
    assert "--vlm.model_id=gpt-6-astra" in printed


# --- answering through `codex exec` ------------------------------------------------------------------


class FakeCodex:
    """Stands in for the `codex` binary: lists features, and answers exec runs from the prompt."""

    def __init__(self, answer='{"subtasks": []}', exit_code=0):
        self.answer, self.exit_code = answer, exit_code
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        if command[1:3] == ["features", "list"]:
            stdout = "apps  stable  true\nshell_tool  stable  true\n"
            stdout += "memories  experimental  false\nold  removed  true\n"
            return subprocess.CompletedProcess(command, 0, stdout, "")
        out = Path(command[command.index("-o") + 1])
        images = [command[i + 1] for i, part in enumerate(command) if part == "-i"]
        self.seen = {"prompt": command[-1], "images": [Path(p).read_bytes() for p in images]}
        if self.exit_code == 0:
            out.write_text(self.answer, encoding="utf-8")
        usage = {
            "input_tokens": 12000,
            "cached_input_tokens": 3000,
            "output_tokens": 40,
            "reasoning_output_tokens": 20,
        }
        events = json.dumps({"type": "turn.completed", "usage": usage})
        return subprocess.CompletedProcess(command, self.exit_code, events + "\n", "boom\n")


PNG = b"\x89PNG\r\n\x1a\nfake"


def test_codex_prompts_keep_the_image_order():
    image = "data:image/png;base64," + base64.b64encode(PNG).decode()
    messages = [
        {"role": "system", "content": "Segment the video."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Sheet A:"},
                {"type": "image_url", "image_url": {"url": image}},
                {"type": "text", "text": "Sheet B:"},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        },
    ]
    prompt, images = codex_prompt(messages)
    assert prompt.startswith("Segment the video.\nSheet A:\n[image 1]\nSheet B:\n[image 2]")
    assert "2 image(s) are attached" in prompt and images == [(PNG, "png"), (PNG, "png")]
    with pytest.raises(UnsupportedContent):
        codex_prompt(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]
        )


def test_the_proxy_answers_through_codex_exec(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # Codex brings its own login
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/codex")
    fake = FakeCodex('{"subtasks": [{"label": "grasp"}]}')
    image = "data:image/png;base64," + base64.b64encode(PNG).decode()
    body = {
        **CHAT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Label it."},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ],
    }
    with OpenAIProxy(model="gpt-6-astra", endpoint="codex", codex_runner=fake) as proxy:
        status, reply = post(proxy.base_url, body)
    assert status == 200 and reply["choices"][0]["message"]["content"] == '{"subtasks": [{"label": "grasp"}]}'
    run = fake.commands[-1]
    assert run[1:3] == ["exec", "--skip-git-repo-check"] and run[4:6] == ["-s", "read-only"]
    assert run[run.index("-m") + 1] == "gpt-6-astra" and "--ephemeral" in run and run[-2] == "--"
    disabled = [run[i + 1] for i, part in enumerate(run) if part == "--disable"]
    assert disabled == ["apps", "shell_tool"]  # only features this Codex has enabled
    assert fake.seen["images"] == [PNG] and "[image 1]" in fake.seen["prompt"]
    assert (
        proxy.usage.input_tokens == 12000
        and proxy.usage.cached_tokens == 3000
        and proxy.usage.reasoning_tokens == 20
    )


def test_a_failed_codex_run_is_an_error(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/codex")
    fake = FakeCodex(exit_code=1)
    with OpenAIProxy(model="gpt-6-astra", endpoint="codex", codex_runner=fake) as proxy:
        status, reply = post(proxy.base_url, CHAT)
    assert status == 502 and "codex exec failed (1): boom" in reply["error"]["message"]
    assert proxy.usage.failures == 1
