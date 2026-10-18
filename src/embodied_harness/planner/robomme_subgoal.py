"""RoboMME's fine-tuned Qwen3-VL subgoal predictor (symbolic memory, SimpleSG) as a harness planner.

Reproduces `Qwen3VLModel` from robomme_policy_learning: the same system prompt, the task goal,
the numbered history of the subgoals predicted so far ("1. ...; 2. ..."), one front-camera image
sent losslessly, greedy decoding with at most 128 new tokens. The model runs in vLLM with its
LoRA adapter (`model` names the adapter).

It answers with subgoal text such as "pick up the red cube for the second time". That text is
mapped onto the skill whose single enum argument contains it (one `subgoal` skill, or e.g. pick /
place / press_button), so the executor still only ever receives texts from its prompt library. Use it with
`planner.call: periodic` and `every_steps: 16`: RoboMME asks it before every 16-step chunk.
"""

from __future__ import annotations

import json
import time

from ..config.schema import PlannerConfig, TaskSpec
from ..config.skills import SkillRegistry
from .base import PlannerContext, PlannerOutput
from .http import ChatClient
from .images import data_url, encode_image, file_extension

SYSTEM_PROMPT = (
    "You are a helpful assistant to help guide the robot to complete the task "
    "by predicting a sequence of language subgoals"
)
FIRST_TURN = "This is the initial turn for prediction\n"
HISTORY_LEAD = "The history of previous predicted language subgoals are: "
QUESTION = "What's the next language subgoal based on current observation?"


def wrap_history(subgoals: list[str]) -> str:
    return "; ".join(f"{index + 1}. {subgoal}" for index, subgoal in enumerate(subgoals))


class RoboMMESubgoalPlanner:
    name = "robomme_subgoal"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        camera: str = "front",
        skill: str | None = None,
        image_format: str = "png",
        max_tokens: int = 128,
        temperature: float = 0.0,
        resample_temperature: float = 0.7,
        request_timeout_s: float = 60.0,
        api_key_env: str | None = None,
        use_env_proxy: bool = False,
    ) -> None:
        file_extension(image_format)
        self._client = ChatClient(
            base_url, timeout_s=request_timeout_s, api_key_env=api_key_env, use_env_proxy=use_env_proxy
        )
        self._model = model
        self._camera = camera
        self._skill = skill  # None: every task skill with exactly one enum argument
        self._image_format = image_format
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._resample_temperature = resample_temperature
        self._subgoals: dict[str, tuple[str, str, str]] | None = None  # text -> (skill, arg, value)

    @property
    def cameras(self) -> tuple[str, ...]:
        return (self._camera,)

    def bind(self, *, registry: SkillRegistry, task: TaskSpec, planner_config: PlannerConfig) -> None:
        if self._skill is not None and self._skill not in task.skills:
            raise ValueError(f"the task has no skill {self._skill!r} for subgoals")
        subgoals: dict[str, tuple[str, str, str]] = {}
        for name in [self._skill] if self._skill else task.skills:
            args = registry.skill(name).spec.args
            if len(args) != 1:
                if self._skill:
                    raise ValueError(
                        f"skill {name!r} must have exactly one enum argument holding the subgoal"
                    )
                continue
            ((arg, spec),) = args.items()
            for value in spec.values:
                subgoals[value.strip().lower()] = (name, arg, value)
        if not subgoals:
            raise ValueError("no task skill has exactly one enum argument to hold subgoal texts")
        self._subgoals = subgoals

    def history(self, context: PlannerContext) -> list[str]:
        skills = {skill for skill, _, _ in (self._subgoals or {}).values()}
        return [
            str(next(iter(call["args"].values())))
            for call in context.history
            if call.get("skill") in skills and len(call.get("args", {})) == 1
        ]

    def user_text(self, context: PlannerContext) -> tuple[str, str]:
        """The user turn around the image: what comes before it and after it."""
        history = self.history(context)
        lead = f"The task goal is: {context.task.goal}\n"
        lead += FIRST_TURN if not history else f"{HISTORY_LEAD}{wrap_history(history)}\n"
        return lead, QUESTION

    def plan(self, context: PlannerContext) -> PlannerOutput:
        if self._subgoals is None:
            raise RuntimeError("bind() must be called before plan()")
        if self._camera not in context.images:
            raise KeyError(f"no {self._camera!r} image; got {', '.join(sorted(context.images)) or 'none'}")
        image = encode_image(context.images[self._camera], fmt=self._image_format)
        before, after = self.user_text(context)
        temperature = self._temperature if context.resample == 0 else self._resample_temperature
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": before},
                        {"type": "image_url", "image_url": {"url": data_url(image, self._image_format)}},
                        {"type": "text", "text": after},
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        started = time.perf_counter()
        reply = self._client.post(body)
        text = ChatClient.content(reply).strip()
        skill, arg, _ = next(iter(self._subgoals.values()))
        skill, arg, subgoal = self._subgoals.get(text.lower(), (skill, arg, text))  # unknown: rejected later
        decision = {"skill": skill, "args": {arg: subgoal}, "memory": ""}
        meta = {
            "planner": self.name,
            "model": self._model,
            "prompt": f"[system]\n{SYSTEM_PROMPT}\n[user]\n{before}<image:{self._camera}>{after}",
            "response": text,
            "temperature": temperature,
            "usage": reply.get("usage"),
        }
        return PlannerOutput(
            raw=json.dumps(decision, ensure_ascii=False),
            latency_s=time.perf_counter() - started,
            meta=meta,
            attachments={f"{self._camera}.{file_extension(self._image_format)}": image},
        )
