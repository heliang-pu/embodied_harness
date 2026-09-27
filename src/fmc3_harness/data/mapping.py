"""How a dataset maps onto the planner's inputs: which camera is which, and proprio from the state vector."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config.errors import ConfigError
from ..config.loader import read_yaml
from ..skills.expressions import Expression, ExpressionError, compile_expression


class DataMapping(BaseModel):
    """Example:

    state_key: observation.state
    cameras: {head: observation.images.head, wrist: observation.images.wrist}
    proprio: {gripper_width: "state[7]", gripper_closed: "state[7] < 0.01"}
    """

    model_config = ConfigDict(extra="forbid")

    state_key: str = "observation.state"
    cameras: dict[str, str] = Field(default_factory=dict)  # planner camera -> dataset feature key
    proprio: dict[str, str] = Field(default_factory=dict)  # proprio name -> expression over `state`


def load_data_mapping(path: Path) -> DataMapping:
    try:
        return DataMapping.model_validate(read_yaml(path) or {})
    except Exception as exc:
        raise ConfigError([f"{path}: {exc}"]) from exc


class ProprioMapper:
    """Computes named proprio values from a raw state vector, e.g. gripper_width = state[7]."""

    def __init__(self, expressions: dict[str, str]) -> None:
        problems = []
        self._expressions: dict[str, Expression] = {}
        for name, source in expressions.items():
            try:
                expr = compile_expression(source)
            except ExpressionError as exc:
                problems.append(f"proprio.{name}: {exc}")
                continue
            unknown = expr.names - {"state"}
            if unknown:
                problems.append(
                    f"proprio.{name}: only `state` can be used, found {', '.join(sorted(unknown))}"
                )
            self._expressions[name] = expr
        if problems:
            raise ConfigError(problems)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._expressions)

    def __call__(self, state: Sequence[float]) -> dict[str, Any]:
        namespace = {"state": list(state)}
        return {name: _plain(expr(namespace)) for name, expr in self._expressions.items()}


def _plain(value: Any) -> Any:
    """numpy scalars become Python bool/int/float, which the renderer formats like live proprio."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    return item() if callable(item) else value
