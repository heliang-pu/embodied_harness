"""A LeRobot processor step that makes a policy train on the subtask active at each frame.

One dataset serves the policy and the planner: its `task` column stays the episode instruction (the
planner's goal) and `language_persistent` holds the subtask and memory rows. During training this step
replaces `task` with the subtask active at the frame, so the policy learns the same prompts the harness
sends it (the prompt-library texts, after `data relabel --in-place`). At inference there are no language
rows and the step changes nothing: the prompt the harness sends is the task. Both sides see
`Task: <library text>`.

It goes into a checkpoint's `policy_preprocessor.json` before the step that builds the prompt;
`embodied-harness policy subtask-checkpoint` does that. Needs lerobot, and embodied_harness installed in
the same environment, since LeRobot imports the step by its class path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.processor import ProcessorStep, TransitionKey

from .language import SUBTASK_STYLE, subtask_prompts


def _stamps(value: Any, count: int) -> list[float]:
    if value is None:
        raise ValueError("the batch has language rows but no timestamp; the subtask cannot be looked up")
    if hasattr(value, "reshape"):  # torch tensor or numpy array
        value = value.reshape(-1).tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    values = [float(v) for v in value]
    if len(values) != count:
        raise ValueError(f"{len(values)} timestamp(s) for {count} sample(s)")
    return values


@dataclass
class ActiveSubtaskAsTaskStep(ProcessorStep):
    style: str = SUBTASK_STYLE

    def __call__(self, transition: Any) -> Any:
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        rows = complementary.get("language_persistent")
        tasks = complementary.get("task")
        if not rows or tasks is None:
            return transition  # inference, or an unannotated dataset: keep the task
        batched = isinstance(tasks, (list, tuple))
        task_list = list(tasks) if batched else [tasks]
        row_list = list(rows) if batched else [rows]
        prompts = subtask_prompts(
            task_list, row_list, _stamps(complementary.get("timestamp"), len(task_list)), self.style
        )
        complementary["task"] = prompts if batched else prompts[0]
        updated = transition.copy()
        updated[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return updated

    def get_config(self) -> dict[str, Any]:
        return {"style": self.style}

    def transform_features(self, features: Any) -> Any:
        return features
