"""The environment contract. Concrete envs also expose `proprio_keys(params)` as a classmethod.

Optional: `snapshot() -> dict | None` and `restore(state)` let an episode resume from its trace
with the world restored too. Only simulated envs can offer this; the real world keeps its own
state.

Optional: `instruction() -> str | None` gives the episode's own instruction after reset, for
tasks whose goal varies per episode (`task.goal_source: episode`).

An env may end an episode itself by returning an observation with `terminated=True`.

`act` and `hold` may return the action actually sent to the robot (a numeric vector, after clipping or
after turning a primitive into joints), the way LeRobot's `Robot.send_action` does. Per-tick capture
(`trace.frames`) records it, so runs can become training data; None means "not known".
"""

from __future__ import annotations

from typing import Any, Protocol

from ..core.types import Observation


class Env(Protocol):
    def reset(self, seed: int | None = None) -> Observation: ...

    def observe(self) -> Observation: ...

    def act(self, action: Any) -> Any: ...

    def hold(self) -> Any:
        """Keep the current pose; called whenever no skill is running."""
        ...

    def stop(self) -> None: ...

    def close(self) -> None: ...
