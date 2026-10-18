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
from ..config.skills import SkillRegistry

RENDER_VERSION = "render_v1"
EMPTY_MEMORY = "（空）"
NO_STEPS = "（无）"
IMAGE_PLACEHOLDER = "<image>"
REASON_TEXTS = {
    "start": "任务开始",
    "skill_done": "上一步已完成",
    "skill_failed": "上一步失败了",
    "check_failed": "你说任务完成了，但完成条件还不满足",
    "human_resume": "人工处理后让你继续",
    "resume": "从中断处继续",
    "periodic": "执行中的定期检查",
}
STATUS_TEXTS = {"success": "成功", "failure": "失败", "preempted": "被打断", "skipped": "人工完成"}


@dataclass(frozen=True)
class RenderConfig:
    cameras: tuple[str, ...]  # fixed order; every call and every training sample uses it
    proprio_keys: tuple[str, ...] | None = None  # None means every key, sorted by name
    proprio_digits: int = 3
    memory_max_chars: int = 200
    # The options below are off in render_v1's default layout. Each one adds lines, so a planner must
    # be trained with the same options it runs with (they are part of the planner's settings).
    # Also list the calls dispatched so far and why the planner is asked now. A planner that has not
    # been fine-tuned needs it to know where the task stands.
    show_history: bool = False
    # The skills file's predicates as text (e.g. holding=true) and how the last call ended: the
    # structured state a VLM reads more reliably than the images.
    show_state: bool = False
    state_predicates: tuple[str, ...] | None = None  # None means every predicate, in file order
    # Task progress from the ledger, e.g. sorted: A=1, B=0, C=0.
    show_progress: bool = False
    progress_fields: tuple[str, ...] | None = None  # None means every ledger field
    # When a resample follows a refused answer, say why it was refused. Only for planners that are not
    # fine-tuned: training data has no refused answers.
    show_rejections: bool = False
    allow_done: bool = True


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


def render_system(
    task: TaskSpec, signatures: Sequence[str], memory_max_chars: int, *, allow_done: bool = True
) -> str:
    lines = [
        "你是机器人的任务规划器。根据任务、各相机画面、本体状态和你上一轮写下的记忆，决定下一步做什么，并改写记忆。",
        f"任务：{task.goal}",
    ]
    if task.rules:
        lines.append("规则：")
        lines += [f"- {rule}" for rule in task.rules]
    if task.lessons:
        lines.append("经验（以往运行的总结）：")
        lines += [f"- {lesson}" for lesson in task.lessons]
    lines.append("可用技能：")
    lines += [f"- {signature}" for signature in signatures]
    completion = (
        '任务完成时输出 {"done": true, "memory": 新记忆}。'
        if allow_done
        else "任务完成由仿真器终止信号决定；始终选择下一个技能，不要输出 done。"
    )
    lines.append(
        '只输出一个 JSON 对象：{"skill": 技能名, "args": {参数}, "memory": 新记忆}，'
        f"{completion}记忆不超过 {memory_max_chars} 字，只保留对后续有用的信息。"
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


def format_last(last: Mapping[str, Any] | None) -> str:
    """How the last main call ended, e.g. `pick 失败（empty_grasp）`."""
    if not last:
        return NO_STEPS
    status = str(last.get("status"))
    text = f"{last.get('call')} {STATUS_TEXTS.get(status, status)}"
    if status == "failure" and last.get("reason"):
        text += f"（{last['reason']}）"
    return text


def format_progress(
    ledger: Mapping[str, Any], fields: Sequence[str] | None, expected_total: int | None
) -> str:
    """The ledger as text, e.g. `sorted: A=1, B=0, C=0（共 1）；目标总数 3`."""
    items = []
    for name in list(fields) if fields is not None else list(ledger):
        if name not in ledger:
            raise KeyError(f"the ledger has no field {name!r}; it has {', '.join(ledger) or 'none'}")
        value = ledger[name]
        if isinstance(value, Mapping):  # a counter
            counts = ", ".join(f"{key}={count}" for key, count in value.items())
            items.append(f"{name}: {counts}（共 {sum(value.values())}）")
        elif isinstance(value, bool):
            items.append(f"{name}={'true' if value else 'false'}")
        else:
            items.append(f"{name}={value}")
    if expected_total is not None:
        items.append(f"目标总数 {expected_total}")
    return "；".join(items) or NO_STEPS


def render_prompt(
    *,
    task: TaskSpec,
    signatures: Sequence[str],
    images: Mapping[str, Any],
    proprio: Mapping[str, Any],
    memory: str,
    config: RenderConfig,
    history: Sequence[str] = (),
    reason: str | None = None,
    predicates: Mapping[str, bool] | None = None,
    last: Mapping[str, Any] | None = None,
    ledger: Mapping[str, Any] | None = None,
    rejected: Sequence[str] = (),
) -> RenderedPrompt:
    """`history` holds the calls dispatched so far as SkillCall keys, `reason` why the planner is asked;
    both appear only with config.show_history. `predicates` (the values to show, in order) and `last`
    appear with config.show_state, `ledger` with config.show_progress, `rejected` (why the previous
    answer was refused) with config.show_rejections."""
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
    if config.show_state:
        if predicates is None:
            raise ValueError("show_state needs the predicate values")
        state = ", ".join(f"{name}={'true' if value else 'false'}" for name, value in predicates.items())
        parts.append(TextPart(f"当前状态：{state or NO_STEPS}\n上一步：{format_last(last)}\n"))
    if config.show_progress:
        if ledger is None:
            raise ValueError("show_progress needs the ledger, and this context has none")
        progress = format_progress(ledger, config.progress_fields, task.expected_total)
        parts.append(TextPart(f"进度：{progress}\n"))
    if config.show_history:
        steps = "；".join(f"{index}. {call}" for index, call in enumerate(history, 1)) or NO_STEPS
        parts.append(TextPart(f"已执行的步骤：{steps}\n"))
        if reason is not None:
            parts.append(TextPart(f"这次调用的原因：{REASON_TEXTS.get(reason, reason)}\n"))
    if config.show_rejections and rejected:
        parts.append(TextPart(f"上一次的回答没有通过检查：{'；'.join(rejected)}。请换一个回答。\n"))
    parts.append(TextPart(f"记忆：{memory or EMPTY_MEMORY}"))
    system = render_system(task, signatures, config.memory_max_chars, allow_done=config.allow_done)
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
