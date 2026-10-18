"""RoboMME env adapter pieces that run without the simulator: composed tasks and episode selection."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from embodied_harness.env.robomme import (
    RoboMMEEnv,
    composite_instruction,
    composite_subgoals,
    parse_composite,
)

LIBRARY = Path(__file__).resolve().parents[1] / "configs" / "robomme" / "mme" / "library_pickx_simplesg.yaml"


def test_parse_composite_accepts_pairs_and_mappings():
    assert parse_composite([["red", 2], {"color": "green", "times": 1}]) == [("red", 2), ("green", 1)]


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ([["purple", 1]], "unknown color"),
        ([["red", 0]], "at least once"),
        ([["red", 3], ["green", 1], ["red", 3]], "red is placed 6 times"),
        ([], "at least one"),
    ],
)
def test_parse_composite_rejects(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_composite(spec)


def test_instruction_follows_robomme_wording():
    segments = parse_composite([["red", 2], ["green", 1]])
    assert composite_instruction(segments) == (
        "pick up the red cube and place it on the target, repeating this action two times, "
        "then pick up the green cube and place it on the target, then press the button to stop"
    )


def test_picks_are_numbered_per_color_across_the_task():
    segments = parse_composite([["red", 1], ["green", 1], ["red", 1]])
    assert composite_subgoals(segments) == [
        "pick up the red cube for the first time",
        "place the red cube onto the target",
        "pick up the green cube for the first time",
        "place the green cube onto the target",
        "pick up the red cube for the second time",
        "place the red cube onto the target",
        "press the button to stop",
    ]


def test_every_subgoal_is_a_text_the_policy_was_trained_on():
    texts = {entry["text"] for entry in yaml.safe_load(LIBRARY.read_text(encoding="utf-8"))["prompts"]}
    segments = parse_composite([["red", 5], ["green", 5], ["blue", 5]])
    assert set(composite_subgoals(segments)) <= texts


class FakeBuilder:
    difficulties = ("easy", "medium", "hard", "easy", "medium", "hard")

    def get_episode_num(self):
        return len(self.difficulties)

    def resolve_episode(self, index):
        return 1000 + index, self.difficulties[index]


def test_seeds_run_the_kept_difficulties_in_order():
    env = RoboMMEEnv({"difficulty": ["medium", "hard"], "first_episode": 1})
    env._builder = FakeBuilder()
    assert [env._episode_for(seed) for seed in range(3)] == [2, 4, 5]
    with pytest.raises(ValueError, match="past the 4 test episode"):
        env._episode_for(3)


def test_without_a_difficulty_every_episode_is_kept():
    env = RoboMMEEnv({})
    env._builder = FakeBuilder()
    assert [env._episode_for(seed) for seed in range(6)] == list(range(6))


def test_go_home_steps_toward_the_start_pose_with_the_gripper_open():
    env = RoboMMEEnv({})
    env._home = np.zeros(7, dtype=np.float32)
    env._state = np.array([0.35, -0.05, 0.0, 0.2, 0.0, 0.0, 0.0, 0.018], dtype=np.float32)
    action = env._primitive("go_home")
    np.testing.assert_allclose(action[:7], [0.25, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], atol=1e-6)
    assert action[7] == 1.0
    with pytest.raises(ValueError, match="no primitive 'retract'"):
        env._primitive("retract")


def test_composite_needs_the_pickxtimes_scene():
    with pytest.raises(ValueError, match="PickXtimes scene"):
        RoboMMEEnv({"task": "BinFill", "composite": [["red", 1]]})
