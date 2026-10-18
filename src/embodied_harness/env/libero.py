"""LIBERO (robosuite / MuJoCo, one Franka Panda) as a harness env, with long tasks made of several tasks of a
suite that share one scene. Needs LIBERO: `hf-libero` (the package LeRobot uses) or the original repo. On
Linux it renders headless with MUJOCO_GL=egl unless that variable is already set.

`goals` lists tasks of `suite`, e.g. three of LIBERO-Goal's ten, which all start from the same kitchen. The
episode asks for all of them in that order; its instruction joins their LIBERO instructions with ", then ".
Every goal must have the same scene as `init_from` (the first goal unless set): the same objects, fixtures
and regions, so each goal's BDDL condition can be checked in the one simulator. Seed k starts from the k-th
of the fixed initial states LIBERO ships for `init_from`, as LIBERO's own evaluation does. The env ends the
episode once every goal's condition holds at the same time (success) or after `max_steps` steps.

The observation is what LIBERO policies are trained on (OpenPI's pi05_libero, LeRobot's LIBERO checkpoints):
the third-person and wrist cameras turned upright (MuJoCo renders them upside down) and an 8-value `state`,
the end effector's position and axis-angle orientation and the two finger joints. The same values are named
for verifiers and the planner (`eef_x` ... `gripper_right`), plus `gripper_width` (about 0.08 open, 0
closed). Actions are LIBERO's 7 values: an end-effector pose change in [-1, 1] (1 is 5 cm, or 0.5 rad) and
the gripper (-1 opens, 1 closes).

Privileged, in `Observation.info` only: `goals` (each goal, whether its condition holds now and the step it
first held), which the `libero_oracle` planner reads to measure a policy on its own; `success`; `status`;
and `score`, the goals satisfied at the end, reached at some point, and reached in order from the first
(CALVIN's chain length). The harness records `success` and `score` once, when the episode ends.

One harness primitive: `go_home` opens the gripper where the arm is, then moves the end effector back to the
pose it had when the episode started, at most _HOME_STEP_M and _HOME_STEP_RAD per step. Every LIBERO
demonstration starts from there.

`hold()` keeps the pose the arm had when holding began (a small correction towards it each step: a zero
delta would let it sink a little further every step) and repeats the last gripper command.

`snapshot()` / `restore()` save MuJoCo's state (LIBERO's flattened time, qpos and qvel), the gripper
controller's integrator and this adapter's own state, so `run --resume` continues in the same world and
`probe` can rerun a call from where it started.
"""

from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.types import Observation

CAMERAS = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}
STATE_KEYS = ("eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "gripper_left", "gripper_right")
PRIMITIVES = ("go_home",)
# Steps LIBERO's evaluations (OpenPI's and LeRobot's) give one task of each suite, at 20 Hz.
SUITE_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
_POS_SCALE = 0.05  # robosuite's OSC_POSE controller: an action of 1 moves the end effector 5 cm ...
_ROT_SCALE = 0.5  # ... or turns it 0.5 rad
_HOME_STEP_M = 0.01  # end-effector travel per step while going home (0.2 m/s at 20 Hz)
_HOME_STEP_RAD = 0.05
_RELEASE_STEPS = 8  # go_home first opens the gripper in place; it takes about 8 steps to open fully
_GRIPPER_OPEN = -1.0
_NO_OP = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _GRIPPER_OPEN)


def quat_to_axisangle(quat: Sequence[float]) -> np.ndarray:
    """robosuite's (x, y, z, w) quaternion as an axis-angle vector, the way LIBERO's training data has it."""
    x, y, z, w = (float(v) for v in quat)
    w = min(1.0, max(-1.0, w))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return np.array([x, y, z]) * 2.0 * math.acos(w) / den


def _quat_to_mat(quat: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mat_to_axisangle(matrix: np.ndarray) -> np.ndarray:
    angle = math.acos(min(1.0, max(-1.0, (float(np.trace(matrix)) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:  # a half turn: the axis is the column of R + I with the largest norm
        columns = matrix + np.eye(3)
        axis = columns[:, int(np.argmax(np.linalg.norm(columns, axis=0)))]
        norm = float(np.linalg.norm(axis))
    return axis / norm * angle


def scene_of(bddl: str) -> str:
    """A BDDL problem without what differs between the tasks of one scene (its instruction, objects of
    interest and goal): tasks with the same result start from the same world."""
    text = re.sub(r"\(:language[^\n]*\n", "", bddl)
    text = re.sub(r"\(:obj_of_interest.*?\)\s*\n", "", text, flags=re.S)
    goal = text.find("(:goal")
    return (text if goal < 0 else text[:goal]).strip()


def _ensure_libero_config() -> None:
    """LIBERO asks on stdin where to keep its files when its config file is missing, which would block a run.
    Write what it would pick itself: the installed package's own BDDL and init-state files."""
    directory = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    path = directory / "config.yaml"
    if path.exists():
        return
    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise ImportError("LIBERO is not installed; see docs/libero.md (pip install hf-libero)")
    if spec.origin is not None:
        package = Path(spec.origin).parent
    else:  # the original repo installed as a namespace package
        package = Path(next(iter(spec.submodule_search_locations or [])))
    root = package / "libero"
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(root),
                "bddl_files": str(root / "bddl_files"),
                "init_states": str(root / "init_files"),
                "datasets": str(root.parent / "datasets"),
                "assets": str(root / "assets"),
            }
        ),
        encoding="utf-8",
    )


class LiberoEnv:
    simulated = True  # a simulator: snapshot()/restore() bring the whole world back on a resume

    def __init__(self, params: dict[str, Any]) -> None:
        self._suite_name = str(params.get("suite", "libero_goal"))
        goals = params.get("goals")
        if isinstance(goals, str):
            goals = [goals]
        if not isinstance(goals, Sequence) or not goals or not all(isinstance(g, str) for g in goals):
            raise ValueError("goals must list at least one task name of the suite")
        if len(set(goals)) != len(goals):
            raise ValueError("goals lists a task more than once")
        self._goal_names = [str(goal) for goal in goals]
        self._init_from = str(params.get("init_from", self._goal_names[0]))
        per_goal = int(params.get("steps_per_goal", SUITE_STEPS.get(self._suite_name, 300)))
        self._max_steps = int(params.get("max_steps", per_goal * len(self._goal_names)))
        self._image_size = int(params.get("image_size", 256))
        self._settle_steps = int(params.get("settle_steps", 10))
        if per_goal < 1 or self._max_steps < 1:
            raise ValueError("steps_per_goal and max_steps must be at least 1")
        if self._image_size < 16:
            raise ValueError("image_size must be at least 16 pixels")
        if self._settle_steps < 0:
            raise ValueError("settle_steps must not be negative")
        self._env: Any = None
        self._init_states: Any = None
        self._goal_states: list[list[list[str]]] = []  # per goal: its BDDL conditions
        self._languages: list[str] = []
        self._raw: dict[str, Any] = {}
        self._done: list[bool] = [False] * len(self._goal_names)
        self._first_at: list[int | None] = [None] * len(self._goal_names)
        self._conditions_done = 0
        self._steps = 0
        self._success = False
        self._terminated = False
        self._gripper = _GRIPPER_OPEN  # the last gripper command, repeated while holding
        self._home: tuple[np.ndarray, np.ndarray] | None = None  # end-effector position and rotation
        self._hold_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._primitive_version: Any = None
        self._primitive_steps = 0

    @classmethod
    def proprio_keys(cls, params: dict[str, Any]) -> list[str]:
        return [*STATE_KEYS, "gripper_width", "state"]

    @classmethod
    def camera_keys(cls, params: dict[str, Any]) -> list[str]:
        return list(CAMERAS)

    # --- episode -------------------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        if self._env is None:
            self._build()
        index = (seed or 0) % len(self._init_states)
        self._env.seed(seed or 0)
        self._env.reset()
        raw = self._env.set_init_state(self._init_states[index])
        for _ in range(self._settle_steps):  # let the objects settle, as LIBERO's evaluations do
            raw, *_ = self._env.step(list(_NO_OP))
        self._raw = raw
        self._steps, self._success, self._terminated = 0, False, False
        self._gripper = _GRIPPER_OPEN
        self._hold_pose = None
        self._primitive_version, self._primitive_steps = None, 0
        self._first_at = [None] * len(self._goal_names)
        self._home = self._pose()
        self._check_goals()
        return self.observe()

    def instruction(self) -> str:
        """The episode's instruction: the goals' LIBERO instructions in order."""
        if not self._languages:
            return ", then ".join(name.replace("_", " ") for name in self._goal_names)
        return ", then ".join(self._languages)

    def observe(self) -> Observation:
        images = {
            camera: np.ascontiguousarray(np.asarray(self._raw[key])[::-1, ::-1])
            for camera, key in CAMERAS.items()
        }
        state = self._state()
        proprio: dict[str, Any] = {key: float(value) for key, value in zip(STATE_KEYS, state, strict=True)}
        proprio["gripper_width"] = float(state[6] - state[7])
        proprio["state"] = state
        goals = [
            {"goal": name, "done": done, "first_at": first}
            for name, done, first in zip(self._goal_names, self._done, self._first_at, strict=True)
        ]
        status = "success" if self._success else ("timeout" if self._terminated else "running")
        return Observation(
            images=images,
            proprio=proprio,
            info={
                "success": self._success,
                "status": status,
                "steps": self._steps,
                "goals": goals,
                "score": self._score(),
            },
            terminated=self._terminated,
        )

    def act(self, action: Any) -> np.ndarray | None:
        """Step the sim with LIBERO's 7 values, or one step of a primitive; returns the action sent."""
        if self._env is None or self._terminated:
            return None
        self._hold_pose = None
        if isinstance(action, Mapping):  # a harness primitive from the scripted executor
            action = self._primitive(str(action.get("primitive")), action.get("version"))
        else:
            self._primitive_version = None
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (7,):
            raise ValueError(f"LIBERO takes 7 action values (pose change and gripper), got {action.shape[0]}")
        return self._step(action)

    def hold(self) -> np.ndarray | None:
        """Keep the pose the arm had when holding began and repeat the last gripper command."""
        if self._env is None or self._terminated:
            return None
        if self._hold_pose is None:
            self._hold_pose = self._pose()
        move = self._toward(*self._hold_pose, _HOME_STEP_M, _HOME_STEP_RAD)
        return self._step(np.concatenate([move, [self._gripper]]))

    def stop(self) -> None:
        pass

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    # --- resume and probe ----------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any] | None:
        """The world for env_snapshot.json (resume) and trace.snapshots (probe), as plain JSON values."""
        if self._env is None or self._home is None:
            return None
        return {
            "suite": self._suite_name,
            "goals": list(self._goal_names),
            "mujoco": np.asarray(self._env.get_sim_state(), dtype=np.float64).tolist(),
            "gripper_integrator": np.asarray(self._gripper_model().current_action, dtype=np.float64).tolist(),
            "steps": self._steps,
            "first_at": list(self._first_at),
            "success": self._success,
            "terminated": self._terminated,
            "gripper": self._gripper,
            "home": [self._home[0].tolist(), self._home[1].tolist()],
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Put the sim back where snapshot() was taken; reset() must have built the sim first."""
        if self._env is None:
            raise RuntimeError("reset() the env before restoring a snapshot")
        if state.get("suite") != self._suite_name or list(state.get("goals") or []) != self._goal_names:
            raise ValueError(
                f"the snapshot is of {state.get('suite')!r} goals {state.get('goals')!r}, this env runs "
                f"{self._suite_name!r} goals {self._goal_names!r}"
            )
        mujoco_state = np.asarray(state["mujoco"], dtype=np.float64)
        expected = np.asarray(self._env.get_sim_state()).shape
        if mujoco_state.shape != expected:
            raise ValueError(
                f"the snapshot holds {mujoco_state.size} MuJoCo state values, this sim has {expected[0]}"
            )
        self._raw = self._env.regenerate_obs_from_state(mujoco_state)
        self._gripper_model().current_action = np.asarray(state["gripper_integrator"], dtype=np.float64)
        self._steps = int(state["steps"])
        self._first_at = [None if at is None else int(at) for at in state["first_at"]]
        self._success = bool(state["success"])
        self._terminated = bool(state["terminated"])
        self._gripper = float(state["gripper"])
        position, rotation = state["home"]
        self._home = (np.asarray(position, dtype=np.float64), np.asarray(rotation, dtype=np.float64))
        self._hold_pose = None
        self._primitive_version, self._primitive_steps = None, 0
        self._check_goals()

    # --- internals -----------------------------------------------------------------------------

    def _build(self) -> None:
        if sys.platform.startswith("linux"):
            os.environ.setdefault("MUJOCO_GL", "egl")
        _ensure_libero_config()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_utils import robosuite_parse_problem

        suites = benchmark.get_benchmark_dict()
        if self._suite_name not in suites:
            raise ValueError(
                f"LIBERO has no suite {self._suite_name!r}; choose one of {', '.join(sorted(suites))}"
            )
        suite = suites[self._suite_name]()
        tasks = {suite.get_task(index).name: index for index in range(suite.n_tasks)}
        unknown = [name for name in [*self._goal_names, self._init_from] if name not in tasks]
        if unknown:
            raise ValueError(
                f"{self._suite_name} has no task(s) {', '.join(sorted(set(unknown)))}; its tasks are "
                f"{', '.join(tasks)}"
            )
        bddl_root = Path(get_libero_path("bddl_files"))

        def bddl_path(name: str) -> Path:
            task = suite.get_task(tasks[name])
            return bddl_root / task.problem_folder / task.bddl_file

        start = bddl_path(self._init_from)
        scene = scene_of(start.read_text(encoding="utf-8"))
        others = [name for name in self._goal_names if scene_of(bddl_path(name).read_text("utf-8")) != scene]
        if others:
            raise ValueError(
                f"goal(s) {', '.join(others)} have a different scene from {self._init_from}: "
                "a long task needs tasks of one scene (e.g. LIBERO-Goal's)"
            )
        self._goal_states = [
            [list(condition) for condition in robosuite_parse_problem(str(bddl_path(name)))["goal_state"]]
            for name in self._goal_names
        ]
        self._languages = [suite.get_task(tasks[name]).language for name in self._goal_names]
        self._init_states = suite.get_task_init_states(tasks[self._init_from])
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(start), camera_heights=self._image_size, camera_widths=self._image_size
        )

    def _step(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        self._gripper = float(action[6])
        self._raw, _reward, _done, _info = self._env.step(action.tolist())
        self._steps += 1
        self._check_goals()
        if self._success or self._steps >= self._max_steps:
            self._terminated = True
        return action

    def _check_goals(self) -> None:
        evaluate = self._env.env._eval_predicate
        results = [[bool(evaluate(condition)) for condition in goal] for goal in self._goal_states]
        self._done = [all(result) for result in results]
        self._conditions_done = sum(sum(result) for result in results)
        for index, done in enumerate(self._done):
            if done and self._first_at[index] is None:
                self._first_at[index] = self._steps
        self._success = all(self._done)

    def _score(self) -> dict[str, int]:
        chain, last = 0, -1
        for first in self._first_at:  # goals reached in order, from the first (CALVIN's chain length)
            if first is None or first < last:
                break
            chain, last = chain + 1, first
        return {
            "goals": len(self._goal_names),
            "satisfied": sum(self._done),
            "reached": sum(first is not None for first in self._first_at),
            "chain": chain,
            "conditions": sum(len(goal) for goal in self._goal_states),
            "conditions_satisfied": self._conditions_done,
        }

    def _state(self) -> np.ndarray:
        raw = self._raw
        return np.concatenate(
            [
                np.asarray(raw["robot0_eef_pos"], dtype=np.float64),
                quat_to_axisangle(raw["robot0_eef_quat"]),
                np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64),
            ]
        ).astype(np.float32)

    def _pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(self._raw["robot0_eef_pos"], dtype=np.float64).copy()
        return position, _quat_to_mat(self._raw["robot0_eef_quat"])

    def _toward(
        self, position: np.ndarray, rotation: np.ndarray, step_m: float, step_rad: float
    ) -> np.ndarray:
        """The 6 pose values that move the end effector towards a pose, at most step_m and step_rad."""
        current_position, current_rotation = self._pose()
        move = position - current_position
        distance = float(np.linalg.norm(move))
        if distance > step_m:
            move *= step_m / distance
        # robosuite turns the end effector by the axis-angle given in the world frame: R_goal = R_delta R.
        turn = _mat_to_axisangle(rotation @ current_rotation.T)
        angle = float(np.linalg.norm(turn))
        if angle > step_rad:
            turn *= step_rad / angle
        return np.concatenate([move / _POS_SCALE, turn / _ROT_SCALE])

    def _primitive(self, name: str, version: Any) -> np.ndarray:
        if name != "go_home":
            raise ValueError(f"LIBERO has no primitive {name!r}; it offers {', '.join(PRIMITIVES)}")
        if version != self._primitive_version:
            self._primitive_version, self._primitive_steps = version, 0
        self._primitive_steps += 1
        assert self._home is not None
        if self._primitive_steps <= _RELEASE_STEPS:  # let go of whatever is in the hand where it is
            return np.array([*np.zeros(6), _GRIPPER_OPEN])
        move = self._toward(*self._home, _HOME_STEP_M, _HOME_STEP_RAD)
        return np.concatenate([move, [_GRIPPER_OPEN]])

    def _gripper_model(self) -> Any:
        return self._env.env.robots[0].gripper
