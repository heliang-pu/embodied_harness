"""Planner backed by an OpenAI-compatible chat endpoint: vLLM serving the fine-tuned Qwen3-VL-2B.

Each call renders the prompt with the shared renderer, encodes every configured camera,
constrains the reply with the decision schema and returns the raw text. Parsing, checking,
resampling and timeouts stay in the supervisor. The images actually sent are returned as
attachments so the trace keeps them and a replay can resend them byte for byte.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config.expressions import ExpressionError
from ..config.schema import PlannerConfig, TaskSpec
from ..config.skills import SkillRegistry
from ..core.types import SkillCall
from .base import PlannerContext, PlannerOutput
from .http import ChatClient, split_reasoning
from .images import data_url, encode_image, file_extension
from .render import (
    RENDER_VERSION,
    RenderConfig,
    RenderedPrompt,
    render_prompt,
    render_system,
    skill_signatures,
    to_openai_messages,
)
from .schema import decision_schema

if TYPE_CHECKING:
    from ..config.loader import Bundle

STRUCTURED_OUTPUT_MODES = ("response_format", "guided_json", "none")
RESERVED_BODY_KEYS = ("model", "messages", "response_format", "guided_json")


def lessons_digest(lessons: Sequence[str]) -> str | None:
    """A short fingerprint of the task lessons, so an export and a live planner can be compared."""
    if not lessons:
        return None
    return hashlib.sha256("\n".join(lessons).encode("utf-8")).hexdigest()[:12]


def system_digest(
    task: TaskSpec, signatures: Sequence[str], memory_max_chars: int, *, allow_done: bool
) -> str:
    """A short fingerprint of the system prompt without its goal: rules, lessons, skills and the completion
    sentence. Every exported sample records its goal, so `data eval` compares this alone."""
    blank = task.model_copy(update={"goal": ""})
    text = render_system(blank, signatures, memory_max_chars, allow_done=allow_done)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def goal_digest(task: TaskSpec) -> str | None:
    """A short fingerprint of the configured goal; None when every episode brings its own (goal_source
    episode)."""
    if task.goal_source == "episode":
        return None
    return hashlib.sha256(task.goal.encode("utf-8")).hexdigest()[:12]


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
        show_history: bool = False,
        show_state: bool = False,
        state_predicates: Sequence[str] | None = None,
        show_progress: bool = False,
        progress_fields: Sequence[str] | None = None,
        show_rejections: bool = False,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        """extra_body: more request fields, e.g. {chat_template_kwargs: {enable_thinking: true}} for a
        thinking model (give it a larger max_tokens). The model, messages and output schema are set
        here and cannot be overridden."""
        if not cameras:
            raise ValueError("cameras must list at least one camera, in the order the planner was trained on")
        if image_size is not None and len(image_size) != 2:
            raise ValueError("image_size must be [width, height]")
        if structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(f"structured_output must be one of {STRUCTURED_OUTPUT_MODES}")
        reserved = [key for key in extra_body or {} if key in RESERVED_BODY_KEYS]
        if reserved:
            raise ValueError(f"extra_body cannot set {', '.join(reserved)}; the planner sets those itself")
        file_extension(image_format)  # rejects unknown formats early
        self._client = ChatClient(
            base_url, timeout_s=request_timeout_s, api_key_env=api_key_env, use_env_proxy=use_env_proxy
        )
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
        self._structured = structured_output
        self._show_history = show_history
        self._show_state = show_state
        self._state_predicates = tuple(state_predicates) if state_predicates is not None else None
        self._show_progress = show_progress
        self._progress_fields = tuple(progress_fields) if progress_fields is not None else None
        self._show_rejections = show_rejections
        self._extra_body = dict(extra_body or {})
        self._signatures: list[str] | None = None
        self._schema: dict[str, Any] | None = None
        self._render: RenderConfig | None = None
        self._registry: SkillRegistry | None = None
        self._lessons: str | None = None
        self._system: str | None = None
        self._goal: str | None = None
        self._call: str | None = None

    @classmethod
    def check_config(cls, bundle: Bundle) -> list[str]:
        """Validate-time checks (factory.check_bundle). allow_done belongs to the planner section, where
        the schema, the prompt and the export read it; among the params it would be ignored."""
        if "allow_done" in bundle.run.planner.params:
            return ["planner.params.allow_done: not a parameter of openai_compat; set planner.allow_done"]
        return []

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
        """Everything that shapes what the model sees. Training data must be made with the same values.

        Each option reads off (False, None or absent) by default, so a setting an older export did not
        record compares equal: `no_done` (planner.allow_done false), and `call` only when periodic.
        `system` fingerprints the rendered system prompt without its goal (rules, lessons, skills,
        completion), and `goal` the configured goal."""
        settings = {
            "render": RENDER_VERSION,
            "cameras": list(self._cameras),
            "image_size": list(self._image_size) if self._image_size is not None else None,
            "image_format": self._image_format,
            "jpeg_quality": self._jpeg_quality,
            "proprio_keys": list(self._proprio_keys) if self._proprio_keys is not None else None,
            "proprio_digits": self._proprio_digits,
            "memory_max_chars": self._render.memory_max_chars if self._render is not None else None,
            "show_history": self._show_history,
            "show_state": self._show_state,
            "state_predicates": list(self._state_predicates) if self._state_predicates is not None else None,
            "show_progress": self._show_progress,
            "progress_fields": list(self._progress_fields) if self._progress_fields is not None else None,
            "show_rejections": self._show_rejections,
            "lessons": self._lessons,
            "extra_body": dict(self._extra_body) or None,
            "no_done": not self._render.allow_done if self._render is not None else None,
            "system": self._system,
            "goal": self._goal,
        }
        if self._call == "periodic":
            settings["call"] = "periodic"
        return settings

    def bind(self, *, registry: SkillRegistry, task: TaskSpec, planner_config: PlannerConfig) -> None:
        """Called by the runner before the first plan(): the skill list and schema come from the configs."""
        unknown = [name for name in self._state_predicates or () if name not in registry.predicate_names]
        if unknown:
            raise ValueError(
                f"state_predicates: {', '.join(unknown)} not defined; the skills file has "
                f"{', '.join(registry.predicate_names) or 'no predicates'}"
            )
        missing = [name for name in self._progress_fields or () if name not in task.ledger]
        if missing:
            fields = ", ".join(task.ledger) or "none"
            raise ValueError(f"progress_fields: the ledger has no {', '.join(missing)}; it has {fields}")
        self._registry = registry
        self._lessons = lessons_digest(task.lessons)
        self._signatures = skill_signatures(registry, task.skills)
        self._system = system_digest(
            task, self._signatures, planner_config.memory_max_chars, allow_done=planner_config.allow_done
        )
        self._goal = goal_digest(task)
        self._call = planner_config.call
        self._schema = decision_schema(
            registry, task.skills, planner_config.memory_max_chars, allow_done=planner_config.allow_done
        )
        self._render = RenderConfig(
            cameras=self._cameras,
            proprio_keys=self._proprio_keys,
            proprio_digits=self._proprio_digits,
            memory_max_chars=planner_config.memory_max_chars,
            show_history=self._show_history,
            show_state=self._show_state,
            state_predicates=self._state_predicates,
            show_progress=self._show_progress,
            progress_fields=self._progress_fields,
            show_rejections=self._show_rejections,
            allow_done=planner_config.allow_done,
        )

    def render_request(self, context: PlannerContext) -> PlannerRequest:
        """Render the prompt and encode the images. The data export calls this too, so a training
        sample and a live call go through exactly the same code."""
        if self._render is None or self._signatures is None or self._registry is None:
            raise RuntimeError("bind() must be called before plan()")
        predicates = None
        if self._show_state:
            try:
                predicates = self._registry.predicate_values(context.proprio, self._state_predicates)
            except ExpressionError as exc:
                raise ValueError(
                    f"show_state: cannot compute the predicates from this proprio: {exc}"
                ) from exc
        prompt = render_prompt(
            task=context.task,
            signatures=self._signatures,
            images=context.images,
            proprio=context.proprio,
            memory=context.memory,
            config=self._render,
            history=[SkillCall(str(c["skill"]), dict(c.get("args") or {})).key() for c in context.history],
            reason=context.reason.value,
            predicates=predicates,
            last=context.last,
            ledger=context.ledger,
            rejected=context.rejected,
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
        body.update(self._extra_body)
        reply = self._client.post(body)
        content, inline = split_reasoning(ChatClient.content(reply))
        reasoning = ChatClient.reasoning(reply) or inline
        meta = {
            "planner": self.name,
            "model": self._model,
            "render": prompt.version,
            "temperature": temperature,
            "prompt": prompt.text(),
            "usage": reply.get("usage"),
        }
        if reasoning is not None:
            meta["reasoning"] = reasoning
        return PlannerOutput(raw=content, meta=meta, attachments=attachments)
