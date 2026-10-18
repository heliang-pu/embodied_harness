import json
import logging
import sys
import threading
import types

import numpy as np
import pytest
import yaml

from conftest import CONFIGS, RUN_CONFIG, make_bundle, run_episode, small_task, write_configs
from embodied_harness.cli import main
from embodied_harness.core.types import ActionChunk
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.runtime.logs.setup import ROOT_LOGGER


@pytest.fixture
def isolated_logging(monkeypatch):
    """`run` installs crash handlers and a console handler; undo both after the test."""
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    logger = logging.getLogger(ROOT_LOGGER)
    before = list(logger.handlers)
    yield
    for handler in list(logger.handlers):
        if handler not in before:
            logger.removeHandler(handler)
            handler.close()
    logger.propagate = True


def test_validate_ok(capsys):
    assert main(["validate", str(RUN_CONFIG)]) == 0
    assert capsys.readouterr().out.startswith("OK:")


def test_validate_reports_problems(tmp_path, capsys):
    path = write_configs(tmp_path, skills={"skills": {"pick": {"prompt": "nope"}}})
    assert main(["validate", str(path)]) == 2
    err = capsys.readouterr().err
    assert "E_CONFIG" in err
    assert "'nope' is not in prompt library" in err


def test_run_writes_traces(tmp_path, monkeypatch, capsys, isolated_logging):
    monkeypatch.chdir(tmp_path)
    args = ["run", str(RUN_CONFIG), "--episodes", "2", "--seed", "3", "--run-id", "t"]
    code = main([*args, "--hold-policy", "abort", "--log-level", "ERROR"])
    out = capsys.readouterr().out
    assert code == 0, out
    run_dir = tmp_path / "runs" / "t"
    assert (run_dir / "run.log").is_file()
    for episode in ("ep_0001", "ep_0002"):
        assert (run_dir / episode / "outcome.json").is_file()
        assert f"{episode}: done" in out
    assert "steps=" not in out  # the oracle's config has no fixed plan


@pytest.mark.parametrize("success, expected_code", [(True, 0), (False, 1)])
def test_run_exit_code_uses_environment_verdict(
    tmp_path, monkeypatch, capsys, isolated_logging, success, expected_code
):
    from test_episode_env import world

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "embodied_harness.runtime.runner.build_env",
        lambda bundle: world(end_after=8, success=success),
    )
    code = main(["run", str(RUN_CONFIG), "--run-id", "env-end", "--hold-policy", "abort"])
    assert code == expected_code
    assert f"ended env_success={success}" in capsys.readouterr().out


class JudgingWorld(MockSortingWorld):
    """A simulator that reports its own verdict, and disagrees with the harness's done."""

    def observe(self):
        obs = super().observe()
        obs.info = {**obs.info, "success": False}
        return obs


def test_run_scores_a_done_episode_by_the_envs_own_verdict(tmp_path, monkeypatch, capsys, isolated_logging):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "embodied_harness.runtime.runner.build_env",
        lambda bundle: JudgingWorld({"num_packages": 10, "barcode_down_prob": 0.0}),
    )
    code = main(["run", str(RUN_CONFIG), "--seed", "0", "--run-id", "judged", "--hold-policy", "abort"])
    assert code == 1
    assert "ep_0001: done env_success=False" in capsys.readouterr().out


def test_run_resume_refuses_a_simulator_that_cannot_restore(tmp_path, monkeypatch, capsys, isolated_logging):
    from test_resume import WORLD, SimulatorWithoutRestore

    bundle = make_bundle(tmp_path, task=small_task(3))
    first = run_episode(bundle, tmp_path, env=SimulatorWithoutRestore(WORLD), max_ticks=100)
    monkeypatch.setattr(
        "embodied_harness.runtime.runner.build_env", lambda bundle: SimulatorWithoutRestore(WORLD)
    )
    code = main(["run", str(bundle.path), "--resume", str(first.episode_dir), "--hold-policy", "abort"])
    assert code == 2
    assert "simulator without restore" in capsys.readouterr().err


def test_validate_prints_the_plan_and_its_rollbacks(capsys):
    assert main(["validate", str(CONFIGS / "runs" / "mock_sort_plan.yaml")]) == 0
    out = capsys.readouterr().out
    assert "plan: 12 steps (task.plan), played by the sequence planner" in out
    assert "rollback: [4] flip on dropped -> [3] pick" in out


def test_run_prints_how_far_along_the_plan_each_episode_got(tmp_path, monkeypatch, capsys, isolated_logging):
    monkeypatch.chdir(tmp_path)
    config = str(CONFIGS / "runs" / "mock_sort_plan.yaml")
    code = main(
        ["run", config, "--seed", "0", "--run-id", "p", "--hold-policy", "abort", "--log-level", "ERROR"]
    )
    out = capsys.readouterr().out
    assert code == 0 and "ep_0001: done" in out and "steps=12/12" in out


def recorded_seeds(monkeypatch):
    """Builds every episode's env, one per episode as the runner does, and records the seeds they get."""
    from test_episode_env import InstructedWorld

    seeds = []

    class SeedRecordingWorld(InstructedWorld):
        def reset(self, seed=None):
            seeds.append(seed)
            return super().reset(seed)

    monkeypatch.setattr(
        "embodied_harness.runtime.runner.build_env",
        lambda bundle: SeedRecordingWorld({"num_packages": 3, "barcode_down_prob": 0.0}, end_after=8),
    )
    return seeds


def test_run_without_a_seed_gives_every_episode_its_own_scene(tmp_path, monkeypatch, isolated_logging):
    # With a None seed, RoboMME, LIBERO and RoboTwin reset to the same scene in every episode.
    monkeypatch.chdir(tmp_path)
    seeds = recorded_seeds(monkeypatch)
    assert main(["run", str(RUN_CONFIG), "--episodes", "3", "--run-id", "s", "--hold-policy", "abort"]) == 0
    assert seeds == [0, 1, 2]
    events = [tmp_path / "runs" / "s" / f"ep_000{index}" / "events.jsonl" for index in (1, 2, 3)]
    assert [json.loads(path.read_text().splitlines()[0])["seed"] for path in events] == [0, 1, 2]


def test_run_starts_at_the_configs_runtime_seed_unless_given_one(tmp_path, monkeypatch, isolated_logging):
    monkeypatch.chdir(tmp_path)
    run = ["run", str(write_configs(tmp_path, run={"runtime": {"seed": 1000}})), "--hold-policy", "abort"]
    seeds = recorded_seeds(monkeypatch)
    assert main([*run, "--episodes", "2", "--run-id", "a"]) == 0
    assert main([*run, "--episodes", "2", "--seed", "5", "--run-id", "b"]) == 0
    assert seeds == [1000, 1001, 5, 6]


def test_suite_entries_start_at_the_configs_runtime_seed_unless_given_one(
    tmp_path, monkeypatch, isolated_logging
):
    # A suite over the RoboTwin eval config (runtime.seed 1000) must not replay training scenes 0, 1, ...
    monkeypatch.chdir(tmp_path)
    entry = {"config": str(write_configs(tmp_path, run={"runtime": {"seed": 1000}})), "category": "A"}
    entries = [
        {**entry, "name": "config", "episodes": 2},
        {**entry, "name": "override", "episodes": 2, "overrides": {"runtime": {"seed": 2000}}},
        {**entry, "name": "given", "episodes": 2, "seed": 5},
    ]
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"suite": "seeds", "entries": entries}), encoding="utf-8")
    seeds = recorded_seeds(monkeypatch)
    assert main(["suite", "run", str(suite), "--run-id", "s"]) == 0
    assert seeds == [1000, 1001, 2000, 2001, 5, 6]


def test_run_resume_without_a_seed_resets_with_the_recorded_one(tmp_path, monkeypatch, isolated_logging):
    from test_resume import WORLD, SeedRecordingWorld

    bundle = make_bundle(tmp_path, task=small_task(3), run={"runtime": {"seed": 1000}})
    first = run_episode(bundle, tmp_path, env=SeedRecordingWorld(WORLD), seed=7, max_ticks=100)
    world = SeedRecordingWorld(WORLD)
    monkeypatch.setattr("embodied_harness.runtime.runner.build_env", lambda bundle: world)
    code = main(["run", str(bundle.path), "--resume", str(first.episode_dir), "--hold-policy", "abort"])
    assert code == 0 and world.seeds == [7]
    assert "resuming with the recorded seed" not in (first.episode_dir / "episode.log").read_text()


class HoldStill:
    """A policy that sends back the joint targets it observes."""

    def __init__(self, library):
        self.library, self.version = library, 0

    def prompt_library(self):
        return self.library

    def capabilities(self):
        return types.SimpleNamespace(chunk_size=4)

    def reset(self):
        self.version = 0

    def set_instruction(self, call, prompt, version):
        self.version = version

    def infer_chunk(self, obs):
        return ActionChunk(self.version, [np.asarray(obs.proprio["state"]).copy() for _ in range(4)])

    def is_done(self):
        return False


@pytest.mark.parametrize(("runtime", "seeds"), [({}, [0, 1, 2]), ({"seed": 1000}, [1000, 1001, 1002])])
def test_every_robotwin_episode_gets_its_own_seed(tmp_path, monkeypatch, isolated_logging, runtime, seeds):
    # Every episode builds a fresh RobotwinEnv, whose reset without a seed falls back to params.episode_index:
    # 0, the scene of training demonstration 0, in every episode.
    from test_robotwin_native import ROBOTWIN_CONFIGS, install_fake_robotwin

    sims = install_fake_robotwin(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "embodied_harness.runtime.runner.build_policy", lambda bundle: HoldStill(bundle.library)
    )
    task = {"id": "bottles", "goal": "put the bottles into the dustbin", "skills": ["pick_bottle"]}
    (tmp_path / "task.yaml").write_text(yaml.safe_dump({"task": task}), encoding="utf-8")
    # A minimal skills file of its own, so edits to the eval skills (verifiers, ledger) cannot break it.
    pick = {
        "args": {"bottle": {"type": "enum", "values": ["yellow", "clear_blue", "black_red"]}},
        "prompt": {"yellow": "pick_yellow", "clear_blue": "pick_clear_blue", "black_red": "pick_black_red"},
        "timeout_s": 45,
    }
    skills = {"version": 1, "library": "robotwin_bottles_objects_v2", "skills": {"pick_bottle": pick}}
    (tmp_path / "skills.yaml").write_text(yaml.safe_dump(skills), encoding="utf-8")
    call = {"skill": "pick_bottle", "args": {"bottle": "yellow"}}
    run = {
        "name": "robotwin_seeds",
        "task": "task.yaml",
        "skills": "skills.yaml",
        "env": {
            "type": "robotwin",
            "params": {"backend": "native", "task": "fake_task", "root": str(tmp_path)},
        },
        "executor": {"type": "websocket", "library": str(ROBOTWIN_CONFIGS / "library_bottles_v2.yaml")},
        "planner": {"type": "sequence", "params": {"calls": [call]}},
        "runtime": runtime,
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    code = main(["run", "run.yaml", "--episodes", "3", "--run-id", "r", "--hold-policy", "abort"])
    assert code == 0  # the fake simulator ends every episode with its success verdict
    assert [sim.setup_kwargs["seed"] for sim in sims] == seeds
