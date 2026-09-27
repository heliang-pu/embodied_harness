"""The environment contract. Concrete envs also expose `proprio_keys(params)` as a classmethod.

Optional: `snapshot() -> dict | None` and `restore(state)` let an episode resume from its trace
with the world restored too. Only simulated envs can offer this; the real world keeps its own
state.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..core.types import Observation


class Env(Protocol):
    def reset(self, seed: int | None = None) -> Observation: ...

    def observe(self) -> Observation: ...

    def act(self, action: Any) -> None: ...

    def hold(self) -> None:
        """Keep the current pose; called whenever no skill is running."""
        ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
