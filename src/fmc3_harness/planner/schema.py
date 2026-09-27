"""JSON schema of the planner's output, generated from the skill vocabulary (design doc §5).

vLLM enforces it through structured outputs, so every reply parses, names one of the task's
skills and uses only argument values that exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..skills.registry import SkillRegistry


def decision_schema(registry: SkillRegistry, skills: Sequence[str], memory_max_chars: int) -> dict[str, Any]:
    memory = {"type": "string", "maxLength": memory_max_chars}
    variants: list[dict[str, Any]] = []
    for name in skills:
        spec = registry.skill(name).spec
        properties: dict[str, Any] = {"skill": {"type": "string", "enum": [name]}}
        required = ["skill"]
        if spec.args:
            properties["args"] = {
                "type": "object",
                "properties": {
                    arg: {"type": "string", "enum": list(a.values)} for arg, a in spec.args.items()
                },
                "required": list(spec.args),
                "additionalProperties": False,
            }
            required.append("args")
        properties["memory"] = memory
        required.append("memory")
        variants.append(
            {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
        )
    variants.append(
        {
            "type": "object",
            "properties": {"done": {"type": "boolean", "enum": [True]}, "memory": memory},
            "required": ["done", "memory"],
            "additionalProperties": False,
        }
    )
    return {"anyOf": variants}
