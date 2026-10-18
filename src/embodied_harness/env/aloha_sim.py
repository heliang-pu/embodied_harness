"""gym-aloha's bimanual ALOHA simulator (MuJoCo) as a harness env. Needs `gym-aloha`; renders headless
with MUJOCO_GL=egl unless that variable is already set.

Seed k resets the sim with seed k, which places the objects, so `embodied-harness run --seed 0 --episodes 50`
runs 50 fixed scenes. Actions are the 14 joint targets of LeRobot's ALOHA datasets: per arm 6 joints and a
normalized gripper (1 open, 0 closed), left arm first. The observation is the top camera and the same 14
values, named for verifiers and the planner (`left_gripper`, ...) and whole as `state` for the policy.

The env ends the episode when the task succeeds (gym-aloha's full reward, e.g. the peg inside the socket)
or after `max_steps` steps (400 = 8 s at 50 Hz, the length of the scripted demonstrations). The verdict
goes into `Observation.info` and is read once, when the episode ends, for scoring.

One harness primitive: `go_home` opens both grippers and moves the joints back to the episode's start
pose, at most _HOME_STEP_RAD per joint and step. A policy trained on demonstrations only ever started its
pick from there, so a retry after a missed grasp goes home first.

`hold()` sends the last joint targets again (the start pose right after a reset), grippers included: the
position controllers keep a grasp closed and the arms where they were told to be. Targets rebuilt from the
measured joints would let the arms sag, since gravity pulls every reading a little below its target.

`snapshot()` / `restore()` save and restore MuJoCo's integration state (joints, objects, velocities, solver
warm start) plus this adapter's own state, so `run --resume` continues in the same world and `probe` can
rerun a call from where it started.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..core.types import Observation

JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")
STATE_KEYS = tuple(f"{side}_{joint}" for side in ("left", "right") for joint in JOINTS)
TASKS = {"insertion": "gym_aloha/AlohaInsertion-v0", "transfer_cube": "gym_aloha/AlohaTransferCube-v0"}
_GRIPPERS = [STATE_KEYS.index("left_gripper"), STATE_KEYS.index("right_gripper")]
_ARM_JOINTS = [index for index in range(len(STATE_KEYS)) if index not in _GRIPPERS]
PRIMITIVES = ("go_home",)
_HOME_STEP_RAD = 0.05  # joint change per control step while going home (2.5 rad/s at 50 Hz)
_GRIPPER_OPEN = 1.0


class AlohaSimEnv:
    CAMERAS = ("top",)
    simulated = True  # a simulator: snapshot()/restore() bring the whole world back on a resume

    def __init__(self, params: dict[str, Any]) -> None:
        task = str(params.get("task", "insertion"))
        if task not in TASKS:
            raise ValueError(f"gym-aloha has no task {task!r}; choose one of {', '.join(TASKS)}")
        self._env_id = TASKS[task]
        self._max_steps = int(params.get("max_steps", 400))
        if self._max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._env: Any = None
        self._images: dict[str, np.ndarray] = {}
        self._state = np.zeros(len(STATE_KEYS), dtype=np.float32)
        self._target = self._state.copy()  # the last 14 targets sent, gripper commands included
        self._home = self._state.copy()  # the joints at the start of the episode
        self._success = False
        self._terminated = False
        self._steps = 0

    @classmethod
    def proprio_keys(cls, params: dict[str, Any]) -> list[str]:
        return [*STATE_KEYS, "state"]

    @classmethod
    def camera_keys(cls, params: dict[str, Any]) -> list[str]:
        return list(cls.CAMERAS)

    def reset(self, seed: int | None = None) -> Observation:
        if self._env is None:
            os.environ.setdefault("MUJOCO_GL", "egl")
            import gym_aloha  # noqa: F401  (registers the envs)
            import gymnasium as gym

            # Our own step limit ends the episode; gymnasium's TimeLimit would cut it at the registered 300.
            self._env = gym.make(
                self._env_id,
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                max_episode_steps=self._max_steps,
            )
        obs, _info = self._env.reset(seed=seed)
        self._success, self._terminated, self._steps = False, False, 0
        self._store(obs)
        self._target = self._state.copy()  # holding before any action keeps the start pose
        self._home = self._state.copy()
        return self.observe()

    def observe(self) -> Observation:
        values = zip(STATE_KEYS, self._state, strict=True)
        proprio: dict[str, Any] = {key: float(value) for key, value in values}
        proprio["state"] = self._state.copy()
        return Observation(
            images=dict(self._images),
            proprio=proprio,
            info={"success": self._success, "steps": self._steps},
            terminated=self._terminated,
        )

    def act(self, action: Any) -> np.ndarray | None:
        """Step the sim with 14 joint targets, or one step of a primitive; returns the joint action sent."""
        if self._env is None or self._terminated:
            return None
        if isinstance(action, Mapping):  # a harness primitive from the scripted executor
            action = self._primitive(str(action.get("primitive")))
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (len(STATE_KEYS),):
            raise ValueError(f"gym-aloha takes {len(STATE_KEYS)} joint targets, got {action.shape[0]}")
        self._target = action.copy()
        obs, _reward, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        self._success = bool((info or {}).get("is_success", False))
        if self._success or terminated or truncated or self._steps >= self._max_steps:
            self._terminated = True
        self._store(obs)
        return action

    def _primitive(self, name: str) -> np.ndarray:
        if name != "go_home":
            raise ValueError(f"gym-aloha has no primitive {name!r}; it offers {', '.join(PRIMITIVES)}")
        target = self._state.copy()
        step = np.clip(self._home[_ARM_JOINTS] - self._state[_ARM_JOINTS], -_HOME_STEP_RAD, _HOME_STEP_RAD)
        target[_ARM_JOINTS] = self._state[_ARM_JOINTS] + step
        target[_GRIPPERS] = _GRIPPER_OPEN
        return target

    def hold(self) -> np.ndarray | None:
        """Send the last targets again (the start pose after a reset), so a grasp stays closed and the arms
        stay where they were told to be instead of sinking a little further every step."""
        if self._env is None or self._terminated:
            return None
        return self.act(self._target.copy())

    def snapshot(self) -> dict[str, Any] | None:
        """The world for env_snapshot.json (resume) and trace.snapshots (probe), as plain JSON values:
        MuJoCo's integration state and this adapter's step count, verdict, last targets and start pose."""
        if self._env is None:
            return None
        import mujoco

        physics = self._physics()
        sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        values = np.empty(mujoco.mj_stateSize(physics.model.ptr, sig), dtype=np.float64)
        mujoco.mj_getState(physics.model.ptr, physics.data.ptr, values, sig)
        return {
            "env_id": self._env_id,
            "mujoco": {"sig": sig, "state": values.tolist()},
            "steps": self._steps,
            "success": self._success,
            "terminated": self._terminated,
            "target": self._target.tolist(),
            "home": self._home.tolist(),
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Put the sim back where snapshot() was taken; reset() must have built the sim first."""
        if self._env is None:
            raise RuntimeError("reset() the env before restoring a snapshot")
        if state.get("env_id") != self._env_id:
            raise ValueError(f"the snapshot is of {state.get('env_id')!r}, this env runs {self._env_id!r}")
        import mujoco

        physics = self._physics()
        model, data = physics.model.ptr, physics.data.ptr
        sig = int(state["mujoco"]["sig"])
        values = np.asarray(state["mujoco"]["state"], dtype=np.float64)
        size = mujoco.mj_stateSize(model, sig)
        if values.shape != (size,):
            raise ValueError(f"the snapshot holds {values.size} MuJoCo state values, this sim expects {size}")
        mujoco.mj_setState(model, data, values, sig)
        physics.forward()  # positions, contacts and sensors of the restored state, for rendering and stepping
        self._steps = int(state["steps"])
        self._success = bool(state["success"])
        self._terminated = bool(state["terminated"])
        self._target = np.asarray(state["target"], dtype=np.float32).reshape(-1)
        self._home = np.asarray(state["home"], dtype=np.float32).reshape(-1)
        sim = self._env.unwrapped  # the observation step() would have returned in this state
        self._store(sim._format_raw_obs(sim._env.task.get_observation(physics)))

    def _physics(self) -> Any:
        """The dm_control physics inside gym-aloha's gymnasium env."""
        return self._env.unwrapped._env.physics

    def stop(self) -> None:
        pass

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None

    def _store(self, obs: dict[str, Any]) -> None:
        self._images = {"top": np.asarray(obs["pixels"]["top"], dtype=np.uint8)}
        self._state = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
