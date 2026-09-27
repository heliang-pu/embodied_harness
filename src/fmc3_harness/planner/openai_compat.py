"""Planner backed by an OpenAI-compatible chat endpoint: vLLM serving the fine-tuned Qwen3-VL-2B.

Each call renders the prompt with the shared renderer, encodes every configured camera,
constrains the reply with the decision schema and returns the raw text. Parsing, checking,
resampling and timeouts stay in the supervisor. The images actually sent are returned as
attachments so the trace keeps them and a replay can resend them byte for byte.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..config.schema import PlannerConfig, TaskSpec
from ..skills.registry import SkillRegistry
from .base import PlannerContext, PlannerOutput
from .images import data_url, encode_image, file_extension
from .render import (
    RENDER_VERSION,
    RenderConfig,
    RenderedPrompt,
    render_prompt,
    skill_signatures,
    to_openai_messages,
)
from .schema import decision_schema

STRUCTURED_OUTPUT_MODES = ("response_format", "guided_json", "none")


@dataclass(frozen=True)
class PlannerRequest:
    """What one planner call shows the model: the rendered prompt and each camera's encoded image."""

    prompt: RenderedPrompt
    images: dict[str, bytes]  # by camera, in the configured camera order


class OpenAICompatPlanner:
    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        cameras: Sequence[str],
        proprio_keys: Sequence[str] | None = None,
        proprio_digits: int = 3,
        image_size: Sequence[int] | None = None,
        image_format: str = "jpeg",
        jpeg_quality: int = 90,
        temperature: float = 0.0,
        resample_temperature: float = 0.7,
        max_tokens: int = 256,
        request_timeout_s: float = 30.0,
        structured_output: str = "response_format",
        api_key_env: str | None = None,
        use_env_proxy: bool = False,
    ) -> None:
        if not cameras:
            raise ValueError("cameras must list at least one camera, in the order the planner was trained on")
        if image_size is not None and len(image_size) != 2:
            raise ValueError("image_size must be [width, height]")
        if structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(f"structured_output must be one of {STRUCTURED_OUTPUT_MODES}")
        file_extension(image_format)  # rejects unknown formats early
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._cameras = tuple(cameras)
        self._proprio_keys = tuple(proprio_keys) if proprio_keys is not None else None
        self._proprio_digits = proprio_digits
        self._image_size = tuple(image_size) if image_size is not None else None
        self._image_format = image_format
        self._jpeg_quality = jpeg_quality
        self._temperature = temperature
        self._resample_temperature = resample_temperature
        self._max_tokens = max_tokens
        self._timeout = request_timeout_s
        self._structured = structured_output
        self._api_key_env = api_key_env
        # Local proxies (clash and the like) must not sit between the robot and its planner.
        handlers = [] if use_env_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)
        self._signatures: list[str] | None = None
        self._schema: dict[str, Any] | None = None
        self._render: RenderConfig | None = None

    @property
    def cameras(self) -> tuple[str, ...]:
        return self._cameras

    @property
    def proprio_keys(self) -> tuple[str, ...] | None:
        return self._proprio_keys

    @property
    def image_extension(self) -> str:
        return file_extension(self._image_format)

    def settings(self) -> dict[str, Any]:
        """Everything that shapes what the model sees. Training data must be made with the same values."""
        return {
            "render": RENDER_VERSION,
            "cameras": list(self._cameras),
            "image_size": list(self._image_size) if self._image_size is not None else None,
            "image_format": self._image_format,
            "jpeg_quality": self._jpeg_quality,
            "proprio_keys": list(self._proprio_keys) if self._proprio_keys is not None else None,
            "proprio_digits": self._proprio_digits,
            "memory_max_chars": self._render.memory_max_chars if self._render is not None else None,
        }

    def bind(self, *, registry: SkillRegistry, task: TaskSpec, planner_config: PlannerConfig) -> None:
        """Called by the runner before the first plan(): the skill list and schema come from the configs."""
        self._signatures = skill_signatures(registry, task.skills)
        self._schema = decision_schema(registry, task.skills, planner_config.memory_max_chars)
        self._render = RenderConfig(
            cameras=self._cameras,
            proprio_keys=self._proprio_keys,
            proprio_digits=self._proprio_digits,
            memory_max_chars=planner_config.memory_max_chars,
        )

    def render_request(self, context: PlannerContext) -> PlannerRequest:
        """Render the prompt and encode the images. The data export calls this too, so a training
        sample and a live call go through exactly the same code."""
        if self._render is None or self._signatures is None:
            raise RuntimeError("bind() must be called before plan()")
        prompt = render_prompt(
            task=context.task,
            signatures=self._signatures,
            images=context.images,
            proprio=context.proprio,
            memory=context.memory,
            config=self._render,
        )
        images = {
            camera: encode_image(
                context.images[camera],
                size=self._image_size,
                fmt=self._image_format,
                quality=self._jpeg_quality,
            )
            for camera in self._cameras
        }
        return PlannerRequest(prompt, images)

    def plan(self, context: PlannerContext) -> PlannerOutput:
        if self._schema is None:
            raise RuntimeError("bind() must be called before plan()")
        request = self.render_request(context)
        prompt = request.prompt
        extension = self.image_extension
        attachments = {f"{camera}.{extension}": data for camera, data in request.images.items()}

        def encode(camera: str, _image: Any) -> str:
            return data_url(request.images[camera], self._image_format)

        temperature = self._temperature if context.resample == 0 else self._resample_temperature
        body: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(prompt, encode),
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if self._structured == "response_format":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "planner_decision", "schema": self._schema, "strict": True},
            }
        elif self._structured == "guided_json":
            body["guided_json"] = self._schema
        reply = self._post(body)
        try:
            content = reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"unexpected reply from the planner server: {str(reply)[:300]}") from None
        meta = {
            "planner": self.name,
            "model": self._model,
            "render": prompt.version,
            "temperature": temperature,
            "prompt": prompt.text(),
            "usage": reply.get("usage"),
        }
        return PlannerOutput(raw=content, meta=meta, attachments=attachments)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self._api_key_env) if self._api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self._url, data=data, headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"planner server returned HTTP {exc.code}: {detail}") from None
