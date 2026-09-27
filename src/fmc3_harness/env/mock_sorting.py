"""A symbolic parcel-sorting world for exercising the harness without a robot or a model.

Actions are prompt ids (from the mock VLA) or primitive names (from the scripted executor).
Each prompt takes a fixed number of steps, then resolves to an outcome drawn from
`fault_probs` or taken from `scripted_outcomes`. The world only exposes what a real robot
would, gripper width and closed state, plus privileged ground truth in `info` for the oracle.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import Observation

SIZES = ("small", "medium", "large")
BINS = ("A", "B", "C")
GRIPPER_OPEN = 0.09
HELD_WIDTH = {"small": 0.03, "medium": 0.05, "large": 0.07}
OUTCOMES = {
    "pick_next": ("success", "empty_grasp", "timeout"),
    "flip": ("success", "dropped", "timeout"),
    "place_bin_a": ("success", "timeout"),
    "place_bin_b": ("success", "timeout"),
    "place_bin_c": ("success", "timeout"),
}
PRIMITIVES = ("retract",)
DEFAULT_STEPS = {"pick_next": 30, "flip": 20, "place_bin_a": 25, "place_bin_b": 25, "place_bin_c": 25}


class MockSortingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_packages: int = Field(10, ge=0)
    barcode_down_prob: float = Field(0.3, ge=0, le=1)
    steps: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_STEPS))
    # prompt id -> {outcome: probability}, e.g. {"pick_next": {"empty_grasp": 0.1}}
    fault_probs: dict[str, dict[str, float]] = Field(default_factory=dict)
    # prompt id -> outcomes forced in order, consumed one per attempt before any random draw
    scripted_outcomes: dict[str, list[str]] = Field(default_factory=dict)
    seed: int | None = None
    # Synthetic camera images (needs numpy). They only exercise the image path end to end; a
    # planner trained on real images will not plan well from them.
    cameras: list[str] = Field(default_factory=list)
    image_size: tuple[int, int] = (96, 72)  # width, height


@dataclass
class Package:
    id: str
    size: str
    barcode_up: bool
    location: str  # "table", "gripper" or a bin name


def _check_outcomes(table: Mapping[str, Any], field: str) -> None:
    for prompt_id, outcomes in table.items():
        if prompt_id not in OUTCOMES:
            raise ValueError(f"{field}: unknown prompt {prompt_id!r}; known: {', '.join(OUTCOMES)}")
        for outcome in outcomes:
            if outcome not in OUTCOMES[prompt_id]:
                raise ValueError(f"{field}.{prompt_id}: {outcome!r} is not one of {OUTCOMES[prompt_id]}")


class MockSortingWorld:
    PROPRIO_KEYS = ("gripper_width", "gripper_closed")

    @classmethod
    def proprio_keys(cls, params: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        return cls.PROPRIO_KEYS

    @classmethod
    def camera_keys(cls, params: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        return tuple((params or {}).get("cameras", ()))

    def __init__(self, params: Mapping[str, Any] | MockSortingParams | None = None) -> None:
        self.params = (
            params if isinstance(params, MockSortingParams) else MockSortingParams(**dict(params or {}))
        )
        _check_outcomes(self.params.fault_probs, "fault_probs")
        _check_outcomes(self.params.scripted_outcomes, "scripted_outcomes")
        self._rng = random.Random(self.params.seed)
        self._scripted: dict[str, list[str]] = {}
        self.packages: list[Package] = []
        self.gripper_width = GRIPPER_OPEN
        self.gripper_closed = False
        self.hold_calls = 0
        self._active: tuple[str, Any] | None = None
        self._progress = 0
        self._outcome = "success"
        self._resolved = False

    # --- env contract ------------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None or self.params.seed is not None:
            self._rng = random.Random(seed if seed is not None else self.params.seed)
        self._scripted = {key: list(values) for key, values in self.params.scripted_outcomes.items()}
        self.packages = [
            Package(
                id=f"p{index + 1:02d}",
                size=self._rng.choice(SIZES),
                barcode_up=self._rng.random() >= self.params.barcode_down_prob,
                location="table",
            )
            for index in range(self.params.num_packages)
        ]
        self.gripper_width, self.gripper_closed = GRIPPER_OPEN, False
        self._active = None
        return self.observe()

    def observe(self) -> Observation:
        held = self.held
        bins = dict.fromkeys(BINS, 0)
        for package in self.packages:
            if package.location in bins:
                bins[package.location] += 1
        return Observation(
            images={camera: self._render(camera) for camera in self.params.cameras},
            proprio={"gripper_width": self.gripper_width, "gripper_closed": self.gripper_closed},
            info={
                "packages": [asdict(p) for p in self.packages],
                "held": asdict(held) if held else None,
                "bins": bins,
                "on_table": sum(1 for p in self.packages if p.location == "table"),
            },
        )

    def act(self, action: Mapping[str, Any] | None) -> None:
        if not action:
            return
        key = action.get("prompt") or action.get("primitive")
        attempt = (key, action.get("version"))
        if attempt != self._active:
            self._start(attempt)
        self._progress += 1
        if self._resolved:
            return
        if key in PRIMITIVES or self._progress >= self.params.steps.get(key, 20):
            self._resolved = True
            self._resolve(key, self._outcome)

    def hold(self) -> None:
        self.hold_calls += 1

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        version, internal, gauss = self._rng.getstate()
        return {
            "packages": [asdict(p) for p in self.packages],
            "gripper_width": self.gripper_width,
            "gripper_closed": self.gripper_closed,
            "scripted": {key: list(values) for key, values in self._scripted.items()},
            "rng": [version, list(internal), gauss],
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        self.packages = [Package(**p) for p in state["packages"]]
        self.gripper_width = state["gripper_width"]
        self.gripper_closed = state["gripper_closed"]
        self._scripted = {key: list(values) for key, values in state["scripted"].items()}
        version, internal, gauss = state["rng"]
        self._rng.setstate((version, tuple(internal), gauss))
        self._active = None

    # --- simulation --------------------------------------------------------------------------

    @property
    def held(self) -> Package | None:
        return next((p for p in self.packages if p.location == "gripper"), None)

    def _render(self, camera: str) -> Any:
        """One column per package: grey on the table, white in the gripper, green in a bin."""
        import numpy as np

        width, height = self.params.image_size
        image = np.full((height, width, 3), 40, dtype=np.uint8)
        image[: height // 4] = sum(camera.encode()) % 200 + 30  # a band that differs per camera
        colours = {"table": (150, 150, 150), "gripper": (250, 250, 250)}
        step = max(1, width // max(1, len(self.packages)))
        for index, package in enumerate(self.packages):
            left = index * step
            image[height // 2 :, left : left + max(1, step - 1)] = colours.get(
                package.location, (60, 180, 60)
            )
        return image

    def _start(self, attempt: tuple[str, Any]) -> None:
        key = attempt[0]
        self._active = attempt
        self._progress = 0
        self._resolved = False
        self._outcome = "success" if key in PRIMITIVES else self._draw(key)
        # A real pick starts by opening the gripper, so an empty closed gripper left over
        # from an earlier failure does not read as an empty grasp.
        if key == "pick_next" and self.held is None:
            self.gripper_width, self.gripper_closed = GRIPPER_OPEN, False

    def _draw(self, key: str) -> str:
        queue = self._scripted.get(key)
        if queue:
            return queue.pop(0)
        roll = self._rng.random()
        cumulative = 0.0
        for outcome, probability in self.params.fault_probs.get(key, {}).items():
            cumulative += probability
            if roll < cumulative:
                return outcome
        return "success"

    def _resolve(self, key: str, outcome: str) -> None:
        held = self.held
        if key == "retract":
            if held is None:
                self.gripper_width, self.gripper_closed = GRIPPER_OPEN, False
            return
        if outcome == "timeout":
            return  # nothing happens; the verifier's timeout catches it
        if key == "pick_next":
            on_table = [p for p in self.packages if p.location == "table"]
            if held is None and on_table and outcome == "success":
                on_table[0].location = "gripper"
                self.gripper_width, self.gripper_closed = HELD_WIDTH[on_table[0].size], True
            elif held is None:
                self.gripper_width, self.gripper_closed = 0.0, True  # closed on nothing
            return
        if key == "flip":
            if held is None:
                return
            if outcome == "dropped":
                held.location = "table"
                self.gripper_width, self.gripper_closed = 0.0, True
            else:
                held.barcode_up = not held.barcode_up
            return
        if key.startswith("place_bin_"):
            if held is not None:
                held.location = key.removeprefix("place_bin_").upper()
            self.gripper_width, self.gripper_closed = GRIPPER_OPEN, False
