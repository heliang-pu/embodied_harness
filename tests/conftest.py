"""Shared helpers: config variants in a temp dir, and mock episodes run to completion."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from embodied_harness.config.loader import Bundle, load_bundle
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.runtime.commands import AutoCommands, CommandSource
from embodied_harness.runtime.factory import check_bundle
from embodied_harness.runtime.runner import EpisodeRun, EpisodeRunner
from fake_openai import FakeChatServer

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
RUN_CONFIG = CONFIGS / "runs" / "mock_sort.yaml"
CONFIG_FILES = {
    "run": "runs/mock_sort.yaml",
    "task": "tasks/sort_by_size.yaml",
    "skills": "skills/sorting.yaml",
    "library": "libraries/mock_sort_v1.yaml",
}


REPLACE = "__replace__"  # put in a patch dict to replace that section instead of merging into it


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and value.get(REPLACE):
            merged[key] = {k: copy.deepcopy(v) for k, v in value.items() if k != REPLACE}
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def write_configs(tmp_path: Path, **patches: dict[str, Any] | None) -> Path:
    """Copy the mock configs into tmp_path with patches merged in; returns the run config path."""
    root = tmp_path / "configs"
    for name, relative in CONFIG_FILES.items():
        data = yaml.safe_load((CONFIGS / relative).read_text(encoding="utf-8"))
        patch = patches.get(name)
        if patch:
            data = deep_merge(data, patch)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return root / CONFIG_FILES["run"]


def make_bundle(tmp_path: Path, **patches: dict[str, Any] | None) -> Bundle:
    bundle = load_bundle(write_configs(tmp_path, **patches))
    check_bundle(bundle)
    return bundle


def small_task(total: int = 3, **extra: Any) -> dict[str, Any]:
    return {"task": {"expected_total": total, **extra}}


def run_episode(
    bundle: Bundle,
    tmp_path: Path,
    *,
    world: dict[str, Any] | None = None,
    env: Any = None,
    planner: Any = None,
    policy: Any = None,
    commands: CommandSource | None = None,
    seed: int = 0,
    episode_id: str = "ep_0001",
    resume: bool = False,
    max_ticks: int | None = None,
) -> EpisodeRun:
    if env is None:
        env = MockSortingWorld({"num_packages": 3, "barcode_down_prob": 0.0, **(world or {})})
    runner = EpisodeRunner(
        bundle,
        run_dir=tmp_path / "runs",
        episode_id=episode_id,
        seed=seed,
        commands=commands or AutoCommands("abort"),
        env=env,
        planner=planner,
        policy=policy,
        resume=resume,
        max_ticks=max_ticks,
    )
    return runner.run()


def read_events(run: EpisodeRun) -> list[dict[str, Any]]:
    lines = (run.episode_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def of_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event for event in events if event["type"] == kind]


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return make_bundle(tmp_path, task=small_task(3))


@pytest.fixture
def servers():
    """Starts fake OpenAI-compatible servers; each takes a reply function. All are closed afterwards."""
    started = []

    def start(reply):
        server = FakeChatServer(reply)
        started.append(server)
        return server

    yield start
    for server in started:
        server.close()
