"""A local OpenAI-compatible endpoint in front of the OpenAI API, for `data annotate --vlm-api openai`.

lerobot-annotate talks to its VLM through the Chat Completions API of a (usually local) server and only
takes an API key on its command line. This proxy lets it use OpenAI's hosted models instead:

- it listens on 127.0.0.1 only and adds the key from an environment variable (`OPENAI_API_KEY` by default)
  to the upstream request, so the key never appears in a command line, a log or the run record;
- it turns each Chat Completions request into what the model needs. The current models (GPT-6 family) are
  reasoning models served through the Responses API: `endpoint="responses"` rewrites the messages into
  `input` items, `max_tokens` into `max_output_tokens` (raised to `min_output_tokens`, since reasoning
  tokens count against it) and drops `temperature`, which reasoning models reject. `endpoint="chat"`
  keeps Chat Completions and only renames `max_tokens` to `max_completion_tokens`;
- it retries rate limits (429) and server errors with backoff, honouring Retry-After;
- it adds up the tokens, so the run can report what it cost.

The upstream request goes through the HTTP(S) proxy of the environment when one is set (the API is often
only reachable that way); lerobot-annotate reaches this proxy directly on localhost.

`endpoint="codex"` answers through the Codex CLI instead (`codex exec`, one run per request): whatever
provider and login Codex is set up with do the talking, so no key passes through here at all. Each run
gets the request's images as attachments, a short annotator system prompt instead of the coding agent's,
a read-only sandbox in an empty directory and none of the optional tools, and reports its tokens.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger("embodied_harness.vlm_proxy")

OPENAI_API = "https://api.openai.com/v1"
ENDPOINTS = ("responses", "chat", "codex")
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)
# USD per million input / output tokens, from OpenAI's model page (2026-09). Only for an estimate.
PRICES = {"gpt-6-astra": (10.0, 50.0), "gpt-6-sol": (2.0, 10.0), "gpt-6-luna": (0.1, 0.5)}


CODEX_INSTRUCTIONS = (
    "You label robot videos for a dataset. Answer exactly in the format the user asks for. "
    "Never run commands or use tools.\n"
)
# Optional Codex features whose tools and context an annotation call does not need (disabled when present).
CODEX_DISABLE = (
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "personality",
    "plugin_sharing",
    "plugins",
    "remote_plugin",
    "shell_tool",
    "skill_mcp_dependency_install",
    "tool_suggest",
    "unified_exec",
    "workspace_dependencies",
)
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}


class UnsupportedContent(ValueError):
    pass


def codex_prompt(messages: list[Mapping[str, Any]]) -> tuple[str, list[tuple[bytes, str]]]:
    """A chat request as one Codex prompt plus its images, in order. Each image leaves an `[image N]` mark
    where it stood, so text that refers to it still reads right."""
    texts: list[str] = []
    images: list[tuple[bytes, str]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content or []:
            kind = part.get("type")
            if kind == "text":
                texts.append(part.get("text", ""))
            elif kind == "image_url":
                image = part.get("image_url")
                url = str(image.get("url") if isinstance(image, Mapping) else image)
                header, _, data = url.partition(",")
                if not header.startswith("data:") or ";base64" not in header:
                    raise UnsupportedContent("only data: URL images can be passed to Codex")
                mime = header[len("data:") :].split(";")[0]
                images.append((base64.b64decode(data), IMAGE_TYPES.get(mime, "png")))
                texts.append(f"[image {len(images)}]")
            else:
                raise UnsupportedContent(
                    f"content part {kind!r} cannot be passed to Codex; send frames as images"
                )
    prompt = "\n".join(texts)
    if images:
        prompt += f"\n\n({len(images)} image(s) are attached, in the order of the [image N] marks.)"
    return prompt, images


class CodexExec:
    """Answer chat requests with `codex exec`, one run per request."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str | None = "low",
        timeout_s: float = 900.0,
        executable: str | None = None,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        found = executable or shutil.which("codex")
        if not found:
            raise ValueError("the Codex CLI (`codex`) is not on PATH")
        self.model = model
        self._codex = found
        self._effort = reasoning_effort
        self._timeout = timeout_s
        self._run = runner
        self._work = Path(tempfile.mkdtemp(prefix="embodied_harness_codex_"))
        self._instructions = self._work / "instructions.txt"
        self._instructions.write_text(CODEX_INSTRUCTIONS, encoding="utf-8")
        self._disable = [name for name in CODEX_DISABLE if name in self._enabled_features()]

    def _enabled_features(self) -> set[str]:
        try:
            listed = self._run([self._codex, "features", "list"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return set()
        enabled = set()
        for line in (listed.stdout or "").splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[-1] == "true" and fields[-2] != "removed":
                enabled.add(fields[0])
        return enabled

    def command(self, run_dir: Path, images: list[Path], prompt: str) -> list[str]:
        command = [self._codex, "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only"]
        command += ["-C", str(run_dir), "-m", self.model]
        if self._effort:
            command += ["-c", f"model_reasoning_effort={self._effort}"]
        command += ["-c", f'model_instructions_file="{self._instructions}"']
        for name in self._disable:
            command += ["--disable", name]
        for path in images:
            command += ["-i", str(path)]
        return [*command, "--json", "-o", str(run_dir / "answer.txt"), "--", prompt]

    def complete(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """(HTTP status, Chat Completions reply, token usage)."""
        prompt, images = codex_prompt(list(body.get("messages", [])))
        run_dir = Path(tempfile.mkdtemp(prefix="call_", dir=self._work))
        try:
            paths = []
            for index, (data, extension) in enumerate(images, 1):
                path = run_dir / f"image_{index:02d}.{extension}"
                path.write_bytes(data)
                paths.append(path)
            try:
                done = self._run(
                    self.command(run_dir, paths, prompt),
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    stdin=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                return 504, {"error": {"message": f"codex exec took longer than {self._timeout:g}s"}}, {}
            answer = run_dir / "answer.txt"
            if done.returncode != 0 or not answer.is_file():
                tail = (done.stderr or "").strip().splitlines()[-3:]
                return (
                    502,
                    {"error": {"message": f"codex exec failed ({done.returncode}): {' | '.join(tail)}"}},
                    {},
                )
            usage: dict[str, Any] = {}
            for line in (done.stdout or "").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "turn.completed":
                    usage = dict(event.get("usage") or {})
            text = answer.read_text(encoding="utf-8").strip()
            reply = responses_to_chat(
                {
                    "status": "completed",
                    "model": self.model,
                    "output_text": text,
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                    },
                },
                self.model,
            )
            return (
                200,
                reply,
                {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "output_tokens_details": {"reasoning_tokens": usage.get("reasoning_output_tokens", 0)},
                    "input_tokens_details": {"cached_tokens": usage.get("cached_input_tokens", 0)},
                },
            )
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def close(self) -> None:
        shutil.rmtree(self._work, ignore_errors=True)


def codex_provider(config_path: Path | None = None) -> dict[str, str | None]:
    """The model provider the Codex CLI is set up with (`$CODEX_HOME/config.toml`, else ~/.codex/config.toml):
    its base URL, wire API (responses or chat), the environment variable with its key, and the default model.
    `data annotate --vlm-api codex` labels through the same service Codex uses."""
    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise ValueError(
                "reading the Codex config needs Python 3.11+ (tomllib) or the tomli package"
            ) from exc
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    path = config_path or home / "config.toml"
    if not path.is_file():
        raise ValueError(f"no Codex config at {path}; use --vlm-api openai with --vlm-upstream instead")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("model_provider", "openai"))
    if name == "openai":
        base_url, wire, env_key = OPENAI_API, "responses", "OPENAI_API_KEY"
    else:
        spec = (data.get("model_providers") or {}).get(name)
        if not spec or not spec.get("base_url"):
            raise ValueError(f"{path}: model provider {name!r} has no base_url")
        base_url, wire = str(spec["base_url"]), str(spec.get("wire_api", "chat"))
        env_key = str(spec.get("env_key", "OPENAI_API_KEY"))
    return {
        "provider": name,
        "base_url": base_url,
        "endpoint": "responses" if wire == "responses" else "chat",
        "api_key_env": env_key,
        "model": data.get("model"),
    }


def _parts(role: str, content: Any) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    parts = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": text_type, "text": part.get("text", "")})
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, Mapping) else image
            detail = image.get("detail", "auto") if isinstance(image, Mapping) else "auto"
            parts.append({"type": "input_image", "image_url": url, "detail": detail})
        else:
            raise UnsupportedContent(
                f"content part {kind!r} is not supported by the OpenAI API; send frames as images"
            )
    return parts


def chat_to_responses(
    body: Mapping[str, Any], *, reasoning_effort: str | None, min_output_tokens: int
) -> dict:
    """A Chat Completions request as a Responses API request."""
    request: dict[str, Any] = {
        "model": body["model"],
        "input": [
            {"role": message["role"], "content": _parts(message["role"], message.get("content"))}
            for message in body.get("messages", [])
        ],
        "store": False,  # do not keep the dataset's images on OpenAI's side
    }
    limit = body.get("max_completion_tokens") or body.get("max_tokens") or 0
    request["max_output_tokens"] = max(int(limit), min_output_tokens)
    effort = body.get("reasoning_effort") or reasoning_effort
    if effort:
        request["reasoning"] = {"effort": effort}
    response_format = body.get("response_format") or {}
    if response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema") or {}
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema.get("name", "answer"),
                "schema": schema.get("schema", {}),
                "strict": bool(schema.get("strict", False)),
            }
        }
    elif response_format.get("type") == "json_object":
        request["text"] = {"format": {"type": "json_object"}}
    return request


def responses_to_chat(response: Mapping[str, Any], model: str) -> dict[str, Any]:
    """A Responses API reply in the Chat Completions shape lerobot-annotate reads."""
    text = response.get("output_text")
    if not isinstance(text, str):
        text = "".join(
            part.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
    usage = response.get("usage") or {}
    prompt, completion = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    return {
        "id": response.get("id", "response"),
        "object": "chat.completion",
        "created": int(response.get("created_at", time.time())),
        "model": response.get("model", model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if response.get("status") == "incomplete" else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def chat_for_reasoning_model(
    body: Mapping[str, Any], *, reasoning_effort: str | None, min_output_tokens: int
) -> dict:
    """A Chat Completions request a reasoning model accepts."""
    request = {key: value for key, value in body.items() if key not in ("max_tokens", "temperature", "top_p")}
    limit = body.get("max_completion_tokens") or body.get("max_tokens") or 0
    request["max_completion_tokens"] = max(int(limit), min_output_tokens)
    effort = body.get("reasoning_effort") or reasoning_effort
    if effort:
        request["reasoning_effort"] = effort
    for key in ("chat_template_kwargs", "mm_processor_kwargs"):  # vLLM-only fields
        request.pop(key, None)
    return request


@dataclass
class Usage:
    requests: int = 0
    failures: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Mapping[str, Any]) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            self.output_tokens += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
            details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
            self.reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)
            cached = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
            self.cached_tokens += int(cached.get("cached_tokens", 0) or 0)

    def to_dict(self) -> dict[str, int]:
        with self._lock:
            names = ("requests", "failures", "retries", "input_tokens", "output_tokens")
            return {name: getattr(self, name) for name in (*names, "reasoning_tokens", "cached_tokens")}

    def count(self, what: str) -> None:
        with self._lock:
            setattr(self, what, getattr(self, what) + 1)

    def cost(self, model: str) -> float | None:
        price = PRICES.get(model)
        if price is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1e6

    def summary(self, model: str) -> str:
        cost = self.cost(model)
        estimate = f", about ${cost:.2f} at list price" if cost is not None else ""
        return (
            f"{self.requests} request(s) to {model}: {self.input_tokens} input token(s) "
            f"({self.cached_tokens} cached), {self.output_tokens} output token(s) "
            f"({self.reasoning_tokens} reasoning){estimate}; {self.retries} retr(ies), {self.failures} failed"
        )


class OpenAIProxy:
    """Serve on 127.0.0.1 until closed; use as a context manager. `base_url` is what the client points at."""

    def __init__(
        self,
        *,
        model: str,
        upstream: str = OPENAI_API,
        endpoint: str = "responses",
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str | None = "low",
        min_output_tokens: int = 4096,
        max_retries: int = 6,
        timeout_s: float = 300.0,
        backoff_s: float = 2.0,
        use_env_proxy: bool = True,
        codex_runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        if endpoint not in ENDPOINTS:
            raise ValueError(f"endpoint must be one of {ENDPOINTS}")
        self._codex: CodexExec | None = None
        key = ""
        if endpoint == "codex":  # Codex brings its own provider and login
            self._codex = CodexExec(model=model, reasoning_effort=reasoning_effort, runner=codex_runner)
        else:
            key = os.environ.get(api_key_env) or ""
            if not key:
                raise ValueError(
                    f"{api_key_env} is not set; export the API key in the shell that runs this command "
                    "(it is never taken on the command line)"
                )
        self.model = model
        self.usage = Usage()
        self._key = key
        self._upstream = upstream.rstrip("/")
        self._endpoint = endpoint
        self._effort = reasoning_effort
        self._min_output = min_output_tokens
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._backoff = backoff_s
        # The environment's HTTP(S) proxy applies upstream: the API is often only reachable through it.
        handlers = [] if use_env_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._codex is not None:
            self._codex.close()

    def __enter__(self) -> OpenAIProxy:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- one request ------------------------------------------------------------------------------

    def complete(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """Answer one Chat Completions request through the upstream API: (HTTP status, JSON body)."""
        body = {**body, "model": body.get("model") or self.model}
        if self._codex is not None:
            try:
                status, reply, usage = self._codex.complete(body)
            except UnsupportedContent as exc:
                self.usage.count("failures")
                return 400, {"error": {"message": str(exc), "type": "invalid_request_error"}}
            if status != 200:
                self.usage.count("failures")
                return status, reply
            self.usage.add(usage)
            return 200, reply
        try:
            if self._endpoint == "responses":
                path, request = (
                    "/responses",
                    chat_to_responses(
                        body, reasoning_effort=self._effort, min_output_tokens=self._min_output
                    ),
                )
            else:
                path, request = (
                    "/chat/completions",
                    chat_for_reasoning_model(
                        body, reasoning_effort=self._effort, min_output_tokens=self._min_output
                    ),
                )
        except UnsupportedContent as exc:
            self.usage.count("failures")
            return 400, {"error": {"message": str(exc), "type": "invalid_request_error"}}
        status, reply = self._post(path, request)
        if status != 200:
            self.usage.count("failures")
            return status, reply
        self.usage.add(reply.get("usage") or {})
        if self._endpoint == "responses":
            return 200, responses_to_chat(reply, str(body["model"]))
        return 200, reply

    def _post(self, path: str, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(self._upstream + path, data=data, headers=headers, method="POST")
            wait = self._backoff * 2**attempt
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    return 200, json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")
                if exc.code not in RETRY_STATUS or attempt == self._max_retries:
                    try:
                        return exc.code, json.loads(detail)
                    except json.JSONDecodeError:
                        return exc.code, {"error": {"message": detail[:500]}}
                after = exc.headers.get("Retry-After") if exc.headers else None
                if after and after.replace(".", "", 1).isdigit():
                    wait = float(after)
                logger.warning(
                    "upstream HTTP %d; retry %d/%d in %.1fs", exc.code, attempt + 1, self._max_retries, wait
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self._max_retries:
                    return 502, {"error": {"message": f"upstream unreachable: {exc}"}}
                logger.warning(
                    "upstream unreachable (%s); retry %d/%d in %.1fs",
                    exc,
                    attempt + 1,
                    self._max_retries,
                    wait,
                )
            self.usage.count("retries")
            time.sleep(min(wait, 60.0))
        return 502, {"error": {"message": "no attempt left"}}

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, payload: Mapping[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": proxy.model, "object": "model"}]})
                else:
                    self._send(404, {"error": {"message": f"unknown path {self.path}"}})

            def do_POST(self) -> None:
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(
                        404, {"error": {"message": f"only /v1/chat/completions is served, not {self.path}"}}
                    )
                    return
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._send(400, {"error": {"message": "the request body is not JSON"}})
                    return
                self._send(*proxy.complete(body))

            def log_message(self, *args: Any) -> None:
                pass

        return Handler
