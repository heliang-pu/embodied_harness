"""A tiny OpenAI-compatible chat server standing in for vLLM in tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Reply = Callable[[dict[str, Any], dict[str, str]], tuple[int, dict[str, Any]]]


def chat_reply(content: str) -> tuple[int, dict[str, Any]]:
    body = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return 200, body


class FakeChatServer:
    def __init__(self, reply: Reply) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {key.lower(): value for key, value in self.headers.items()}
                owner.requests.append(body)
                owner.headers.append(headers)
                owner.paths.append(self.path)
                status, payload = reply(body, headers)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def user_text(body: dict[str, Any]) -> str:
    """The text parts of the user message, joined."""
    content = body["messages"][1]["content"]
    return "".join(part["text"] for part in content if part["type"] == "text")


def image_urls(body: dict[str, Any]) -> list[str]:
    content = body["messages"][1]["content"]
    return [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
