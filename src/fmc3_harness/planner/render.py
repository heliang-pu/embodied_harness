"""Prompt renderer shared by the harness and the planner's training-data scripts (design doc §5).

The fine-tuned planner only works if it sees at deployment exactly what it saw in training:
the same wording, the same camera order and resolution, the same proprio and memory format.
Training scripts should import `render_prompt` and `render_target` from here rather than
re-implementing them. Any change to the output of this module needs a new RENDER_VERSION,
and the training data must be regenerated with it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Union

from ..config.schema import TaskSpec
from ..skills.registry import SkillRegistry

RENDER_VERSION = "render_v1"
EMPTY_MEMORY = "（空）"
IMAGE_PLACEHOLDER = "<image>"


@dataclass(frozen=True)
class RenderConfig:
    cameras: tuple[str, ...]  # fixed order; every call and every training sample uses it
    proprio_keys: tuple[str, ...] | None = None  # None means every key, sorted by name
    proprio_digits: int = 3
    memory_max_chars: int = 200


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    camera: str
    image: Any  # whatever the env produced; the caller decides how to encode it


Part = Union[TextPart, ImagePart]  # noqa: UP007 -- a runtime alias, must work on Python 3.10


@dataclass(frozen=True)
class RenderedPrompt:
    version: str
    system: str
    parts: tuple[Part, ...]

    def text(self) -> str:
        """The prompt with images shown as <image:camera>, for traces and debugging."""
        body = "".join(p.text if isinstance(p, TextPart) else f"<image:{p.camera}>" for p in self.parts)
        return f"[system]\n{self.system}\n[user]\n{body}"


def skill_signatures(registry: SkillRegistry, skills: Sequence[str]) -> list[str]:
    """One line per skill, e.g. `place(bin: A|B|C)`."""
    lines = []
    for name in skills:
        spec = registry.skill(name).spec
        if spec.args:
            args = ", ".join(f"{arg}: {'|'.join(arg_spec.values)}" for arg, arg_spec in spec.args.items())
            lines.append(f"{name}({args})")
        else:
            lines.append(name)
    return lines


def render_system(task: TaskSpec, signatures: Sequence[str], memory_max_chars: int) -> str:
    lines = [
        "你是机器人的任务规划器。根据任务、各相机画面、本体状态和你上一轮写下的记忆，决定下一步做什么，并改写记忆。",
        f"任务：{task.goal}",
    ]
    if task.rules:
        lines.append("规则：")
        lines += [f"- {rule}" for rule in task.rules]
    lines.append("可用技能：")
    lines += [f"- {signature}" for signature in signatures]
    lines.append(
        '只输出一个 JSON 对象：{"skill": 技能名, "args": {参数}, "memory": 新记忆}，'
        '任务完成时输出 {"done": true, "memory": 新记忆}。'
        f"记忆不超过 {memory_max_chars} 字，只保留对后续有用的信息。"
    )
    return "\n".join(lines)


def format_proprio(proprio: Mapping[str, Any], keys: Sequence[str] | None, digits: int) -> str:
    names = list(keys) if keys is not None else sorted(proprio)
    items = []
    for name in names:
        if name not in proprio:
            raise KeyError(f"proprio has no {name!r}; it provides {', '.join(sorted(proprio)) or 'nothing'}")
        value = proprio[name]
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, float):
            text = f"{value:.{digits}f}"
        else:
            text = str(value)
        items.append(f"{name}={text}")
    return ", ".join(items)


def render_prompt(
    *,
    task: TaskSpec,
    signatures: Sequence[str],
    images: Mapping[str, Any],
    proprio: Mapping[str, Any],
    memory: str,
    config: RenderConfig,
) -> RenderedPrompt:
    missing = [camera for camera in config.cameras if camera not in images]
    if missing:
        available = ", ".join(sorted(images)) or "none"
        raise KeyError(f"missing camera image(s) {', '.join(missing)}; got {available}")
    parts: list[Part] = []
    for camera in config.cameras:
        parts += [TextPart(f"[{camera}] "), ImagePart(camera, images[camera]), TextPart("\n")]
    parts.append(
        TextPart(f"本体状态：{format_proprio(proprio, config.proprio_keys, config.proprio_digits)}\n")
    )
    parts.append(TextPart(f"记忆：{memory or EMPTY_MEMORY}"))
    system = render_system(task, signatures, config.memory_max_chars)
    return RenderedPrompt(RENDER_VERSION, system, tuple(parts))


def render_target(decision: Mapping[str, Any]) -> str:
    """The assistant turn of a training sample, in the exact form the harness parses."""
    if decision.get("done"):
        ordered: dict[str, Any] = {"done": True, "memory": decision["memory"]}
    else:
        ordered = {"skill": decision["skill"]}
        if decision.get("args"):
            ordered["args"] = dict(decision["args"])
        ordered["memory"] = decision["memory"]
    return json.dumps(ordered, ensure_ascii=False)


def to_openai_messages(prompt: RenderedPrompt, encode: Callable[[str, Any], str]) -> list[dict[str, Any]]:
    """Chat messages for an OpenAI-compatible endpoint. `encode(camera, image)` returns an image URL."""
    content: list[dict[str, Any]] = []
    for part in prompt.parts:
        if isinstance(part, TextPart):
            if content and content[-1]["type"] == "text":
                content[-1]["text"] += part.text
            else:
                content.append({"type": "text", "text": part.text})
        else:
            content.append({"type": "image_url", "image_url": {"url": encode(part.camera, part.image)}})
    return [{"role": "system", "content": prompt.system}, {"role": "user", "content": content}]


def to_sft_record(prompt: RenderedPrompt, target: str, images: Sequence[str]) -> dict[str, Any]:
    """A training sample in the messages + images layout that ms-swift and LLaMA-Factory read.

    Each image becomes an `<image>` placeholder where to_openai_messages puts the image part,
    so the Qwen3-VL chat template produces the same token sequence in training as vLLM
    produces at deployment. `images` are file paths in camera order.
    """
    count = sum(isinstance(part, ImagePart) for part in prompt.parts)
    if count != len(images):
        raise ValueError(f"the prompt has {count} image(s) but {len(images)} path(s) were given")
    user = "".join(part.text if isinstance(part, TextPart) else IMAGE_PLACEHOLDER for part in prompt.parts)
    return {
        "messages": [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ],
        "images": list(images),
    }
