"""The ledger: structured task state that only verified skill outcomes may change (design doc §9).

The planner never reads or writes it. The harness uses it to decide when the task is done,
for budgets and for failure attribution.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..config.schema import LedgerField


class LedgerError(ValueError):
    pass


@dataclass(frozen=True)
class LedgerOp:
    """One change to the ledger. Traces store these, so replaying them rebuilds the ledger."""

    op: Literal["set", "add"]
    field: str
    key: str | None = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "field": self.field, "key": self.key, "value": self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LedgerOp:
        return cls(op=data["op"], field=data["field"], key=data.get("key"), value=data.get("value"))


def _initial_value(spec: LedgerField) -> Any:
    if spec.type == "counter":
        counts = dict.fromkeys(spec.keys or [], 0)
        if isinstance(spec.init, Mapping):
            counts.update(spec.init)
        return counts
    if spec.init is not None:
        return spec.init
    return {"int": 0, "bool": False}.get(spec.type)


class Ledger:
    def __init__(self, fields: Mapping[str, LedgerField]) -> None:
        self._fields = dict(fields)
        self._values = {name: _initial_value(spec) for name, spec in self._fields.items()}

    @classmethod
    def replay(cls, fields: Mapping[str, LedgerField], ops: Iterable[LedgerOp]) -> Ledger:
        ledger = cls(fields)
        for op in ops:
            ledger.apply(op)
        return ledger

    def apply(self, op: LedgerOp) -> None:
        spec = self._fields.get(op.field)
        if spec is None:
            raise LedgerError(f"unknown ledger field {op.field!r}")
        if spec.type == "counter":
            key = None if op.key is None else str(op.key)
            if key not in (spec.keys or []):
                raise LedgerError(f"{op.field}[{op.key!r}] is not one of {spec.keys}")
            counts = self._values[op.field]
            counts[key] = self._number(op, counts[key])
            return
        if op.key is not None:
            raise LedgerError(f"{op.field} is a {spec.type} field and takes no key")
        if spec.type == "int":
            self._values[op.field] = self._number(op, self._values[op.field])
            return
        if op.op != "set":
            raise LedgerError(f"{op.field} is a {spec.type} field and can only be set")
        if spec.type == "enum" and op.value not in (spec.values or []):
            raise LedgerError(f"{op.field}={op.value!r} is not one of {spec.values}")
        if spec.type == "bool" and not isinstance(op.value, bool):
            raise LedgerError(f"{op.field} needs true or false, got {op.value!r}")
        self._values[op.field] = op.value

    def view(self) -> dict[str, Any]:
        """A copy for expressions and traces; changing it does not change the ledger."""
        return copy.deepcopy(self._values)

    @staticmethod
    def _number(op: LedgerOp, current: int) -> int:
        if isinstance(op.value, bool) or not isinstance(op.value, int):
            raise LedgerError(f"{op.field} needs an integer, got {op.value!r}")
        return current + op.value if op.op == "add" else op.value
