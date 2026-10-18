"""Builds components from a loaded bundle. New backends are registered in the tables below."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.errors import ConfigError
from ..config.loader import Bundle
from ..env.aloha_sim import AlohaSimEnv
from ..env.libero import LiberoEnv
from ..env.mock_sorting import MockSortingWorld
from ..env.robomme import RoboMMEEnv
from ..env.robotwin import RobotwinEnv
from ..planner.base import settings_differences, unrecorded_warning
from ..planner.libero_oracle import LiberoOracle
from ..planner.openai_compat import OpenAICompatPlanner
from ..planner.oracle import MockSortingOracle
from ..planner.robomme_subgoal import RoboMMESubgoalPlanner
from ..planner.sequence import SequencePlanner
from ..policy.executor.mock import MockPolicyExecutor
from ..policy.executor.scripted import ScriptedExecutor
from ..policy.executor.websocket_policy import WebsocketPolicyExecutor

# Adapters import their heavy dependencies (robomme, gym-aloha, libero, websockets) only when an instance is
# built.
ENVS: dict[str, Any] = {
    "mock_sorting": MockSortingWorld,
    "robomme": RoboMMEEnv,
    "aloha_sim": AlohaSimEnv,
    "libero": LiberoEnv,
    "robotwin": RobotwinEnv,
}
EXECUTORS: dict[str, Any] = {"mock": MockPolicyExecutor, "websocket": WebsocketPolicyExecutor}
PLANNERS: dict[str, Any] = {
    "mock_oracle": MockSortingOracle,
    "libero_oracle": LiberoOracle,
    "sequence": SequencePlanner,
    "openai_compat": OpenAICompatPlanner,
    "robomme_subgoal": RoboMMESubgoalPlanner,
}


def _lookup(table: dict[str, Any], kind: str, name: str) -> Any:
    try:
        return table[name]
    except KeyError:
        raise ConfigError([f"unknown {kind} type {name!r}; available: {', '.join(sorted(table))}"]) from None


def _construct(kind: str, factory: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return factory(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError([f"{kind}: {exc}"]) from exc


def env_class(bundle: Bundle) -> Any:
    return _lookup(ENVS, "env", bundle.run.env.type)


def check_bundle(bundle: Bundle) -> None:
    """Checks that need the env type: names used by expressions and cameras used by the planner; and the
    planner type's own checks (an optional classmethod `check_config(bundle) -> list[str]`)."""
    _lookup(EXECUTORS, "executor", bundle.run.executor.type)
    planner_cls = _lookup(PLANNERS, "planner", bundle.run.planner.type)
    env_cls = env_class(bundle)
    problems = bundle.registry.check_names(env_cls.proprio_keys(bundle.run.env.params))
    cameras = bundle.run.planner.params.get("cameras")
    camera_keys = getattr(env_cls, "camera_keys", None)
    if cameras and callable(camera_keys):
        available = set(camera_keys(bundle.run.env.params))
        missing = [camera for camera in cameras if camera not in available]
        if missing:
            problems.append(
                f"planner.params.cameras: the env has no camera(s) {', '.join(missing)}; "
                f"it provides {', '.join(sorted(available)) or 'none'}"
            )
    check_config = getattr(planner_cls, "check_config", None)
    if callable(check_config):
        problems += check_config(bundle)
    if problems:
        raise ConfigError(problems)


def build_env(bundle: Bundle) -> Any:
    return _construct("env", env_class(bundle), bundle.run.env.params)


def build_policy(bundle: Bundle) -> Any:
    factory = _lookup(EXECUTORS, "executor", bundle.run.executor.type)
    return _construct("executor", factory, bundle.library, **bundle.run.executor.params)


def build_scripted(bundle: Bundle) -> ScriptedExecutor:
    runtime = bundle.run.runtime
    return ScriptedExecutor(runtime.primitive_steps, runtime.default_primitive_steps)


def build_planner(bundle: Bundle) -> Any:
    factory = _lookup(PLANNERS, "planner", bundle.run.planner.type)
    return _construct("planner", factory, **bundle.run.planner.params)


def bind_planner(planner: Any, bundle: Bundle) -> None:
    """Planners that need the skill vocabulary (e.g. for the output schema) get it here."""
    bind = getattr(planner, "bind", None)
    if callable(bind):
        bind(registry=bundle.registry, task=bundle.task, planner_config=bundle.run.planner)


def trained_settings(path: Path) -> dict[str, Any]:
    """The planner settings a `data planner` export recorded; `path` is its manifest.json or directory."""
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{path}: file not found")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: cannot read it as JSON: {exc}") from exc
    settings = manifest.get("planner") if isinstance(manifest, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: no planner settings; not a `data planner` manifest")
    return settings


def trained_settings_warnings(planner: Any, bundle: Bundle) -> list[str]:
    """planner.trained_manifest: one message per setting of the bound planner that differs from what its
    training export recorded, compared like `data eval` does. Empty when the field is not set. A manifest
    that cannot be read is a warning too: the check is advice, and the export may live elsewhere."""
    path = getattr(bundle, "trained_manifest", None)
    if path is None:
        return []
    source = f"planner.trained_manifest ({bundle.run.planner.trained_manifest})"
    try:
        trained = trained_settings(path)
    except ValueError as exc:
        return [f"planner.trained_manifest: {exc}; the planner settings are not compared"]
    settings = getattr(planner, "settings", None)
    if not callable(settings):
        return [f"planner type {bundle.run.planner.type!r} records no settings; {source} is not checked"]
    current = settings()
    hints = {
        "goal": " (the task goal)",
        "system": " (the system prompt: rules, lessons, skills or the done sentence)",
    }
    messages = [
        f"planner setting {key}{hints.get(key, '')} differs from {source}: trained {trained.get(key)!r}, "
        f"now {current.get(key)!r}; the model is not seeing what it was trained on"
        for key in settings_differences(current, trained)
    ]
    unrecorded = unrecorded_warning(current, trained)
    if unrecorded is not None:
        messages.append(f"{source}: {unrecorded}")
    return messages
