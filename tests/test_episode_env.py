"""Episodes shaped by the env: a per-episode instruction as the goal, and the env ending the episode."""

import json

import pytest

from conftest import make_bundle, of_type, read_events, run_episode, small_task
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.planner.oracle import MockSortingOracle

INSTRUCTION = "sort the three packages, small ones first"


class InstructedWorld(MockSortingWorld):
    """The mock world plus an instruction, and optionally an env-side end after some actions."""

    def __init__(self, params, *, instruction=INSTRUCTION, end_after=None, success=True):
        super().__init__(params)
        self._instruction = instruction
        self._end_after = end_after
        self._success = success
        self.actions = 0

    def instruction(self):
        return self._instruction

    def act(self, action):
        super().act(action)
        self.actions += 1

    def observe(self):
        obs = super().observe()
        if self._end_after is not None and self.actions >= self._end_after:
            obs.terminated = True
            obs.info = {
                **obs.info,
                "success": self._success,
                "status": "success" if self._success else "fail",
            }
        return obs


class RecordingOracle(MockSortingOracle):
    def __init__(self):
        super().__init__()
        self.goals = []

    def plan(self, context):
        self.goals.append(context.task.goal)
        return super().plan(context)


def world(**kwargs):
    return InstructedWorld({"num_packages": 3, "barcode_down_prob": 0.0}, **kwargs)


def test_episode_instruction_becomes_the_planner_goal(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3, goal_source="episode"))
    planner = RecordingOracle()
    run = run_episode(bundle, tmp_path, env=world(), planner=planner)
    assert run.result.status == "done"
    assert planner.goals and set(planner.goals) == {INSTRUCTION}
    assert bundle.task.goal != INSTRUCTION  # the config itself is untouched
    events = read_events(run)
    assert of_type(events, "instruction")[0]["text"] == INSTRUCTION


def test_config_goal_ignores_the_env_instruction(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3))
    planner = RecordingOracle()
    run_episode(bundle, tmp_path, env=world(), planner=planner)
    assert set(planner.goals) == {bundle.task.goal}


def test_episode_goal_needs_an_instruction(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3, goal_source="episode"))
    with pytest.raises(RuntimeError, match="gave no instruction"):
        run_episode(bundle, tmp_path, env=world(instruction=None))


@pytest.mark.parametrize("success", [True, False])
def test_env_can_end_the_episode(tmp_path, success):
    bundle = make_bundle(tmp_path, task=small_task(3))
    env = world(end_after=8, success=success)
    run = run_episode(bundle, tmp_path, env=env)
    result = run.result
    assert result.status == "ended"
    assert result.env_success is success
    assert env.actions == 8  # nothing is sent after the env ends the episode
    events = read_events(run)
    end = of_type(events, "env_end")[0]
    assert end["success"] is success and end["active"] is not None
    assert of_type(events, "episode_end")[0]["env_success"] is success
    outcome = json.loads((run.episode_dir / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["status"] == "ended" and outcome["env_success"] is success
