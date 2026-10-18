"""A small client for OpenAI-compatible chat endpoints (vLLM), shared by the VLM planner backends."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class ChatClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        api_key_env: str | None = None,
        use_env_proxy: bool = False,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout_s
        self._api_key_env = api_key_env
        # Local proxies (clash and the like) must not sit between the robot and its planner.
        handlers = [] if use_env_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self._api_key_env) if self._api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"planner server returned HTTP {exc.code}: {detail}") from None

    @staticmethod
    def content(reply: dict[str, Any]) -> str:
        try:
            return reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"unexpected reply from the planner server: {str(reply)[:300]}") from None

    @staticmethod
    def reasoning(reply: dict[str, Any]) -> str | None:
        """What a thinking model reasoned, when the server splits it out (vLLM's --reasoning-parser)."""
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        return message.get("reasoning_content") or message.get("reasoning") or None


def split_reasoning(content: str) -> tuple[str, str | None]:
    """Separate a `<think>...</think>` block left in the content (a server without a reasoning parser)
    from the answer after it. Returns (answer, reasoning or None)."""
    end = content.find("</think>")
    if end == -1:
        return content, None
    thought = content[:end].strip()
    thought = thought.removeprefix("<think>").strip()  # the template may have opened the block itself
    return content[end + len("</think>") :].strip(), thought
