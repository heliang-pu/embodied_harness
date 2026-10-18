"""Terminal observations are separate from policy action targets."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from conftest import make_bundle, run_episode, small_task
from test_from_runs import CAMERAS, NumericWorld, read_frames


class TerminalWorld(NumericWorld):
    def __init__(self, success):
        self.actions = 0
        self.success = success
        super().__init__(
            {"num_packages": 3, "barcode_down_prob": 0.0, "cameras": CAMERAS, "image_size": [64, 48]}
        )

    def act(self, action):
        sent = super().act(action)
        self.actions += 1
        return sent

    def observe(self):
        obs = super().observe()
        obs.proprio["state"].append(float(self.actions))
        if self.actions >= 8:
            obs.terminated = True
            obs.info["success"] = self.success
        return obs


@pytest.mark.parametrize("success", [True, False])
def test_environment_terminal_is_not_an_extra_action_frame(tmp_path, success):
    bundle = make_bundle(tmp_path, task=small_task(3), run={"trace": {"frames": True}})
    env = TerminalWorld(success)
    run = run_episode(bundle, tmp_path, env=env)
    rows = read_frames(run.episode_dir)
    terminal = json.loads((run.episode_dir / "terminal.json").read_text())

    assert run.result.status == "ended"
    assert terminal["success"] is success
    assert len(rows) == env.actions == 8
    assert rows[-1]["proprio"]["state"][-1] == 7.0
    assert terminal["proprio"]["state"][-1] == 8.0
    assert "action" not in terminal
    assert set(terminal["images"]) == set(CAMERAS)
    for relative in terminal["images"].values():
        with Image.open(run.episode_dir / relative) as frame:
            assert frame.size == (64, 48)


def test_supervisor_completion_records_the_verified_terminal_state(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(1), run={"trace": {"frames": True}})
    env = NumericWorld({"num_packages": 1, "barcode_down_prob": 0.0, "cameras": CAMERAS})
    run = run_episode(bundle, tmp_path, env=env)
    terminal = json.loads((run.episode_dir / "terminal.json").read_text())
    assert terminal["status"] == "done"
    assert terminal["success"] is True
    assert sum(terminal["ledger"]["sorted"].values()) == 1
    assert terminal["ledger"] == run.result.ledger


def test_terminal_capture_is_disabled_without_trace_frames(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3))
    run = run_episode(bundle, tmp_path, env=TerminalWorld(True))
    assert not (run.episode_dir / "terminal.json").exists()
