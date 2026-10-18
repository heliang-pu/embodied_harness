"""Prepare a LeRobot policy checkpoint to be fine-tuned on subtask prompts (no lerobot needed).

The checkpoint is copied (weights linked) and `ActiveSubtaskAsTaskStep` is inserted into its
`policy_preprocessor.json` before the step that turns `task` into the prompt. Fine-tune from the copy
with `lerobot-train --policy.path=<copy> --dataset.root=<the annotated dataset>`; the checkpoints the
training saves keep the step.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

PREPROCESSOR = "policy_preprocessor.json"
STEP_CLASS = "embodied_harness.policy.integrations.lerobot.ActiveSubtaskAsTaskStep"
# The first step of a LeRobot preprocessor that reads `task` to build the prompt: pi05 first formats
# "Task: ..., State: ...", other policies (pi0, smolvla, ...) tokenize `task` directly.
PROMPT_STEPS = ("pi05_prepare_state_tokenizer_processor_step", "tokenizer_processor")
LINKED_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")  # large weight files are linked, not copied


def add_subtask_step(pipeline: dict[str, Any], style: str = "subtask") -> bool:
    """Insert the step before the prompt-building step. False if it is there already."""
    steps = pipeline.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{PREPROCESSOR} has no list of steps")
    for step in steps:
        if step.get("class") == STEP_CLASS:
            return False
    for position, step in enumerate(steps):
        if step.get("registry_name") in PROMPT_STEPS:
            steps.insert(position, {"class": STEP_CLASS, "config": {"style": style}})
            return True
    names = [step.get("registry_name") or step.get("class") for step in steps]
    raise ValueError(f"no prompt step ({', '.join(PROMPT_STEPS)}) in the preprocessor; its steps are {names}")


def prepare_checkpoint(
    source: Path, out: Path, *, style: str = "subtask", copy_weights: bool = False
) -> dict[str, Any]:
    source, out = source.resolve(), out.resolve()
    if not (source / PREPROCESSOR).is_file():
        raise FileNotFoundError(
            f"{source / PREPROCESSOR} not found; is {source} a LeRobot policy checkpoint?"
        )
    if out == source or source in out.parents:
        raise ValueError("write the prepared checkpoint outside the source checkpoint")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        destination = out / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in LINKED_SUFFIXES and not copy_weights:
            destination.symlink_to(path)
            linked += 1
        else:
            shutil.copy2(path, destination)
            copied += 1
    pipeline = json.loads((out / PREPROCESSOR).read_text(encoding="utf-8"))
    add_subtask_step(pipeline, style)
    (out / PREPROCESSOR).write_text(json.dumps(pipeline, indent=4) + "\n", encoding="utf-8")
    return {
        "linked": linked,
        "copied": copied,
        "steps": [s.get("registry_name") or s.get("class") for s in pipeline["steps"]],
    }
