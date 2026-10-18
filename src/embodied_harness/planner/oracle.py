"""Oracle planner for the mock sorting world.

It reads privileged simulator state instead of images, so it only exists to test the harness
and to give an upper bound for the rest of the loop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .base import PlannerContext, PlannerOutput


class MockSortingOracle:
    name = "mock_oracle"

    def __init__(self, size_to_bin: Mapping[str, str] | None = None, flip_barcode_down: bool = True) -> None:
        self._size_to_bin = dict(size_to_bin or {"small": "A", "medium": "B", "large": "C"})
        self._flip = flip_barcode_down

    def plan(self, context: PlannerContext) -> PlannerOutput:
        info = context.info
        if "packages" not in info:
            raise ValueError("the mock oracle needs the mock world's privileged state in context.info")
        held = info.get("held")
        bins = info.get("bins", {})
        on_table = info.get("on_table", 0)
        decision: dict[str, Any]
        if held is not None:
            if self._flip and not held["barcode_up"]:
                decision = {"skill": "flip"}
            else:
                decision = {"skill": "place", "args": {"bin": self._size_to_bin[held["size"]]}}
        elif on_table:
            decision = {"skill": "pick"}
        else:
            decision = {"done": True}
        placed = "、".join(f"{name} {count} 个" for name, count in sorted(bins.items()))
        decision["memory"] = f"已放 {placed}；桌上还剩 {on_table} 个"
        return PlannerOutput(raw=json.dumps(decision, ensure_ascii=False), meta={"planner": self.name})
