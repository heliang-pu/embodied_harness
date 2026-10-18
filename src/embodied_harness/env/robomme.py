"""RoboMME (ManiSkill/SAPIEN) episodes as a harness env. Linux with an NVIDIA GPU; needs `robomme`.

Episodes come from RoboMME's fixed splits: seed k runs episode `first_episode + k` of `split`, so
`embodied-harness run --seed 0 --episodes 50` covers the 50 test episodes, as RoboMME's own
evaluation does (joint-angle actions, 1300 steps at most). The instruction is `task_goal[0]`.
`difficulty: [medium, hard]` keeps only the episodes of those difficulties; seed k then runs the
k-th of them (after `first_episode`).

`composite` (PickXtimes only) replaces the episode's task with a composed one, e.g.
[[red, 2], [green, 1]]: put the red cube on the target twice, then the green cube once, then
press the button. The scene is RoboMME's; the task list is rebuilt from RoboMME's own checks, so
picking up a cube of another color or pressing the button early still fails the episode. Only
medium and hard episodes have cubes of all three colors.

Only cameras and proprio reach the planner and verifiers. RoboMME's verdict (status
success / fail / error) goes into `Observation.info` and is read once, when the episode ends,
for scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..core.types import Observation

COLORS = ("red", "green", "blue")
ORDINALS = ("first", "second", "third", "fourth", "fifth")  # RoboMME's subgoal numbering
_TIMES = {2: "two", 3: "three", 4: "four", 5: "five"}
PRIMITIVES = ("go_home",)
_HOME_STEP_RAD = 0.1  # joint change per control step while going home
_GRIPPER_OPEN = 1.0


def parse_composite(spec: Sequence[Any]) -> list[tuple[str, int]]:
    """[[red, 2], [green, 1]] (or {color, times} mappings) -> [("red", 2), ("green", 1)]."""
    segments: list[tuple[str, int]] = []
    totals: dict[str, int] = {}
    for item in spec:
        color, times = (item["color"], item["times"]) if isinstance(item, Mapping) else item
        color, times = str(color), int(times)
        if color not in COLORS:
            raise ValueError(f"composite: unknown color {color!r}; RoboMME has {', '.join(COLORS)}")
        if times < 1:
            raise ValueError(f"composite: {color} must be placed at least once")
        totals[color] = totals.get(color, 0) + times
        if totals[color] > len(ORDINALS):
            raise ValueError(f"composite: {color} is placed {totals[color]} times; RoboMME counts up to five")
        segments.append((color, times))
    if not segments:
        raise ValueError("composite needs at least one [color, times] segment")
    return segments


def composite_instruction(segments: Sequence[tuple[str, int]]) -> str:
    """The instruction in RoboMME's PickXtimes wording, one clause per segment."""
    clauses = []
    for color, times in segments:
        clause = f"pick up the {color} cube and place it on the target"
        if times > 1:
            clause += f", repeating this action {_TIMES[times]} times"
        clauses.append(clause)
    return ", then ".join(clauses) + ", then press the button to stop"


def composite_subgoals(segments: Sequence[tuple[str, int]]) -> list[str]:
    """The subgoals in order. Picks are numbered per color across the whole task, as RoboMME does."""
    picked: dict[str, int] = {}
    subgoals = []
    for color, times in segments:
        for _ in range(times):
            count = picked.get(color, 0)
            picked[color] = count + 1
            subgoals.append(f"pick up the {color} cube for the {ORDINALS[count]} time")
            subgoals.append(f"place the {color} cube onto the target")
    subgoals.append("press the button to stop")
    return subgoals


def _composite_tasks(env: Any, segments: Sequence[tuple[str, int]]) -> list[dict[str, Any]]:
    """A PickXtimes task list for the composed task, built from RoboMME's own checks."""
    from robomme.robomme_env.utils.subgoal_evaluate_func import (
        is_any_obj_pickup,
        is_button_pressed,
        is_obj_dropped_onto,
        is_obj_pickup,
    )

    cubes = {color: getattr(env, f"{color}_cubes", None) for color in COLORS}
    missing = sorted({color for color, _ in segments if not cubes[color]})
    if missing:
        raise RuntimeError(
            f"this episode has no {' or '.join(missing)} cube; composed tasks need difficulty medium or hard"
        )
    subgoals = iter(composite_subgoals(segments))
    tasks: list[dict[str, Any]] = []
    for color, times in segments:
        cube = cubes[color][0]
        others = [other for other in env.all_cubes if other is not cube]

        def failed(others: list[Any] = others) -> list[Any]:
            # Picking up another cube, or pressing the button early, fails the episode.
            return [is_any_obj_pickup(env, others), is_button_pressed(env, obj=env.button)]

        for _ in range(times):
            pick, place = next(subgoals), next(subgoals)
            tasks.append(
                {
                    "func": lambda cube=cube: is_obj_pickup(env, obj=cube),
                    "name": pick,
                    "subgoal_segment": pick.replace(" cube for", " cube at <> for"),
                    "choice_label": "pick up the cube",
                    "demonstration": False,
                    "failure_func": failed,
                    "segment": cube,
                }
            )
            tasks.append(
                {
                    "func": lambda cube=cube: is_obj_dropped_onto(env, obj=cube, target=env.target),
                    "name": place,
                    "subgoal_segment": f"{place} at <>",
                    "choice_label": "place the cube onto the target",
                    "demonstration": False,
                    "failure_func": failed,
                    "segment": env.target,
                }
            )
    tasks.append(
        {
            "func": lambda: is_button_pressed(env, obj=env.button),
            "name": next(subgoals),
            "subgoal_segment": "press the button at <> to stop",
            "choice_label": "press the button to stop",
            "demonstration": False,
            "failure_func": lambda: is_any_obj_pickup(env, env.all_cubes),
            "segment": env.cap_link,
        }
    )
    return tasks


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _last(value: Any) -> Any:
    return value[-1] if isinstance(value, (list, tuple)) else value


def _flag(value: Any) -> bool:
    return bool(_numpy(value).reshape(-1)[0]) if value is not None else False


class RoboMMEEnv:
    CAMERAS = ("front", "wrist")
    # A simulator without snapshot()/restore(): `run --resume` refuses it rather than continuing the
    # recorded ledger and history in a freshly reset scene.
    simulated = True

    def __init__(self, params: dict[str, Any]) -> None:
        self._task = str(params.get("task", "PickXtimes"))
        self._split = str(params.get("split", "test"))
        self._first = int(params.get("first_episode", 0))
        self._max_steps = int(params.get("max_steps", 1300))
        difficulty = params.get("difficulty")
        self._difficulty = {difficulty} if isinstance(difficulty, str) else set(difficulty or ())
        composite = params.get("composite")
        self._composite = parse_composite(composite) if composite else None
        if self._composite and self._task != "PickXtimes":
            raise ValueError("composite tasks are built in the PickXtimes scene; set task: PickXtimes")
        self._episodes: list[int] | None = None
        self._builder: Any = None
        self._env: Any = None
        self._images: dict[str, np.ndarray] = {}
        self._state = np.zeros(8, dtype=np.float32)
        self._home = np.zeros(7, dtype=np.float32)  # joints at the start of the episode
        self._gripper_command = _GRIPPER_OPEN
        self._status = "ongoing"
        self._terminated = False
        self._steps = 0
        self._instruction: str | None = None
        self.episode_index: int | None = None

    @classmethod
    def proprio_keys(cls, params: dict[str, Any]) -> list[str]:
        return ["gripper_width", "state"]

    @classmethod
    def camera_keys(cls, params: dict[str, Any]) -> list[str]:
        return list(cls.CAMERAS)

    def reset(self, seed: int | None = None) -> Observation:
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        if self._builder is None:
            self._builder = BenchmarkEnvBuilder(
                env_id=self._task,
                dataset=self._split,
                action_space="joint_angle",
                gui_render=False,
                max_steps=self._max_steps,
            )
        self.close()
        self.episode_index = self._episode_for(seed or 0)
        self._env = self._builder.make_env_for_episode(self.episode_index)
        obs, info = self._env.reset()
        if self._composite:
            # The scene is built at reset; its task list is checked on every step from now on.
            self._env.unwrapped.task_list = _composite_tasks(self._env.unwrapped, self._composite)
            self._instruction = composite_instruction(self._composite)
        else:
            goals = (info or {}).get("task_goal")
            self._instruction = str(goals[0] if isinstance(goals, (list, tuple)) else goals)
        self._status, self._terminated, self._steps = "ongoing", False, 0
        self._gripper_command = _GRIPPER_OPEN
        self._store(obs)
        self._home = self._state[:7].copy()
        return self.observe()

    def _episode_for(self, seed: int) -> int:
        """The split's episode index for a seed: the seed-th episode from first_episode, of the kept ones."""
        if self._episodes is None:
            count = self._builder.get_episode_num()
            self._episodes = [
                index
                for index in range(count)
                if not self._difficulty or self._builder.resolve_episode(index)[1] in self._difficulty
            ]
        position = self._first + seed
        if not 0 <= position < len(self._episodes):
            kept = f" of difficulty {', '.join(sorted(self._difficulty))}" if self._difficulty else ""
            raise ValueError(
                f"seed {seed} with first_episode {self._first} is past the {len(self._episodes)} "
                f"{self._split} episode(s){kept}"
            )
        return self._episodes[position]

    def instruction(self) -> str | None:
        return self._instruction

    def observe(self) -> Observation:
        return Observation(
            images=dict(self._images),
            proprio={"gripper_width": float(self._state[7]), "state": self._state.copy()},
            info={"status": self._status, "success": self._status == "success", "steps": self._steps},
            terminated=self._terminated,
        )

    def act(self, action: Any) -> np.ndarray | None:
        """Step the sim; returns the joint-angle action sent (primitives turned into joints)."""
        if self._terminated:
            return None
        if isinstance(action, Mapping):  # a harness primitive from the scripted executor
            action = self._primitive(str(action.get("primitive")))
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._gripper_command = float(action[7])
        obs, _reward, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        self._status = str((info or {}).get("status", "ongoing"))
        if self._status == "error" or _flag(terminated) or _flag(truncated):
            self._terminated = True
        self._store(obs)
        return action

    def _primitive(self, name: str) -> np.ndarray:
        """One step of a primitive as a joint-angle action.

        go_home moves back to the episode's start pose with the gripper open, at most _HOME_STEP_RAD
        per joint and step; a skill that runs it for enough steps (runtime.primitive_steps) arrives.
        The policy has only ever started a new cube's first pick from there.
        """
        if name != "go_home":
            raise ValueError(f"RoboMME has no primitive {name!r}; it offers {', '.join(PRIMITIVES)}")
        joints = self._state[:7]
        target = joints + np.clip(self._home - joints, -_HOME_STEP_RAD, _HOME_STEP_RAD)
        return np.concatenate([target, [_GRIPPER_OPEN]]).astype(np.float32)

    def hold(self) -> np.ndarray | None:
        """Command the current joint positions and the last gripper command; returns that action."""
        if self._env is None or self._terminated:
            return None
        return self.act(np.concatenate([self._state[:7], [self._gripper_command]]).astype(np.float32))

    def stop(self) -> None:
        pass

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def _store(self, obs: dict[str, Any]) -> None:
        self._images = {
            "front": _numpy(_last(obs["front_rgb_list"])).astype(np.uint8),
            "wrist": _numpy(_last(obs["wrist_rgb_list"])).astype(np.uint8),
        }
        joints = _numpy(_last(obs["joint_state_list"])).astype(np.float32).reshape(-1)[:7]
        gripper = _numpy(_last(obs["gripper_state_list"])).astype(np.float32).reshape(-1)[:1]
        self._state = np.concatenate([joints, gripper])
