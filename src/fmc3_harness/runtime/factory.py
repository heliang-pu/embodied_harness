"""Builds components from a loaded bundle. New backends are registered in the tables below."""

from __future__ import annotations

from typing import Any

from ..config.errors import ConfigError
from ..config.loader import Bundle
from ..env.mock_sorting import MockSortingWorld
from ..executor.mock import MockVLAExecutor
from ..executor.scripted import ScriptedExecutor
from ..planner.openai_compat import OpenAICompatPlanner
from ..planner.oracle import MockSortingOracle
from ..planner.sequence import SequencePlanner

ENVS: dict[str, Any] = {"mock_sorting": MockSortingWorld}
EXECUTORS: dict[str, Any] = {"mock": MockVLAExecutor}
PLANNERS: dict[str, Any] = {
    "mock_oracle": MockSortingOracle,
    "sequence": SequencePlanner,
    "openai_compat": OpenAICompatPlanner,
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
    """Checks that need the env type: names used by expressions and cameras used by the planner."""
    _lookup(EXECUTORS, "executor", bundle.run.executor.type)
    _lookup(PLANNERS, "planner", bundle.run.planner.type)
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
    if problems:
        raise ConfigError(problems)


def build_env(bundle: Bundle) -> Any:
    return _construct("env", env_class(bundle), bundle.run.env.params)


def build_vla(bundle: Bundle) -> Any:
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
