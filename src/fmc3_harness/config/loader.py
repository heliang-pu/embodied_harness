"""Loads a run config and everything it references, then checks that the pieces fit together."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from ..logs.codes import ErrorCode
from ..skills.registry import SkillRegistry
from .errors import ConfigError
from .schema import PromptLibrary, RunConfig, SkillsFile, TaskFile, TaskSpec

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class Bundle:
    """A validated run config together with the task, skills and prompt library it points to."""

    path: Path
    run: RunConfig
    task: TaskSpec
    skills: SkillsFile
    library: PromptLibrary
    registry: SkillRegistry

    def snapshot(self) -> dict[str, Any]:
        """The fully resolved config, written to config.snapshot.yaml."""
        return {
            "config": str(self.path),
            "run": self.run.model_dump(mode="json"),
            "task": self.task.model_dump(mode="json"),
            "skills": self.skills.model_dump(mode="json"),
            "library": self.library.model_dump(mode="json"),
        }


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load(model: type[ModelT], path: Path, problems: list[str]) -> ModelT | None:
    if not path.is_file():
        problems.append(f"{path}: file not found")
        return None
    try:
        data = read_yaml(path)
    except yaml.YAMLError as exc:
        problems.append(f"{path}: invalid YAML: {exc}")
        return None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            problems.append(f"{path.name}: {location}: {error['msg']}")
        return None


def load_bundle(path: str | Path) -> Bundle:
    config_path = Path(path).resolve()
    problems: list[str] = []
    run = _load(RunConfig, config_path, problems)
    if run is None:
        raise ConfigError(problems)

    base = config_path.parent
    task_file = _load(TaskFile, (base / run.task).resolve(), problems)
    skills = _load(SkillsFile, (base / run.skills).resolve(), problems)
    library = _load(PromptLibrary, (base / run.executor.library).resolve(), problems)
    if task_file is None or skills is None or library is None:
        raise ConfigError(problems)

    task = task_file.task
    if skills.library is not None and skills.library != library.library:
        problems.append(
            f"the skills file is written for prompt library {skills.library!r}, "
            f"but the executor ships {library.library!r}"
        )
    for name in task.skills:
        if name not in skills.skills:
            problems.append(f"task.skills: {name!r} is not defined in the skills file")

    registry: SkillRegistry | None = None
    try:
        registry = SkillRegistry(skills, library, task)
    except ConfigError as exc:
        problems.extend(exc.problems)
    if problems or registry is None:
        raise ConfigError(problems)

    trained = run.planner.trained_library
    if trained is not None and trained != library.library:
        message = f"the planner was trained on prompt library {trained!r}"
        raise ConfigError(
            [f"{message}, but the executor ships {library.library!r}"], code=ErrorCode.E_LIBRARY_MISMATCH
        )
    return Bundle(config_path, run, task, skills, library, registry)
