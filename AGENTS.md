# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ package managed by `pyproject.toml`. Production code lives in `src/embodied_harness/`, with seven top-level packages: `config`, `core`, `data`, `env`, `planner`, `policy`, and `runtime`. Model families and shared policy execution, serving, transport, and integrations live under `policy/`. Control, verification, logging, and tracing live under `runtime/`; skill compilation lives in `config/` and the ledger lives in `core/`. YAML task, skill, library, and run bundles are under `configs/`; generated traces and exports belong in the gitignored `runs/` and `exports/` directories. Tests are in `tests/` and use shared fixtures from `conftest.py`; helper modules are imported directly because `tests/` is not a package. Training utilities are in `scripts/`.

## Build, Test, and Development Commands

Install editable development dependencies with `python -m pip install -e ".[dev]"`. Key commands:

```bash
python -m pytest -q                         # run the full suite
python -m pytest -q -k resume                # run a focused subset
ruff check src tests scripts                 # lint
ruff format --check src tests scripts        # verify formatting
embodied-harness validate configs/runs/mock_sort.yaml
embodied-harness run configs/runs/mock_sort.yaml --episodes 3 --seed 0 --hold-policy abort
```

Use `embodied-harness replay ...` to inspect a completed episode. Data commands requiring video decoding use the `[data]` extra and a LeRobot-capable environment.

## Coding Style & Naming Conventions

Use four-space indentation, English code/comments/docstrings, and double-quoted strings. Keep lines at 110 characters or fewer and run Ruff before submitting. New non-`__init__.py` modules should begin with `from __future__ import annotations`. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and descriptive YAML filenames such as `configs/runs/mock_sort.yaml`.

## Testing Guidelines

Tests use pytest and should be named `tests/test_<area>.py`, with test functions named `test_<behavior>`. Prefer deterministic mock episodes and temporary configuration bundles over network or model dependencies. Run the full suite plus targeted tests for changed behavior; no coverage threshold is currently configured.

## Commit & Pull Request Guidelines

Existing history uses concise Conventional Commit-style subjects, for example `feat: add ...`. Use an imperative, focused subject such as `fix: validate planner camera keys`. Pull requests should explain the behavior change, identify relevant config or CLI changes, list validation commands, and link an issue when applicable. Include sample output or screenshots when changing user-facing CLI behavior or generated artifacts.

## Configuration & Safety

Validate changed bundles before running them. Keep credentials, model endpoints, and generated episode data out of Git. Preserve path-resolution conventions: task, skill, and library paths are relative to the run config, while trace output defaults to `runs/`.
