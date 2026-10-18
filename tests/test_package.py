"""Public package and CLI entry points stay usable without model dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import RUN_CONFIG

SOURCE = Path(__file__).resolve().parents[1] / "src"


def run_python(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(SOURCE), env.get("PYTHONPATH"))))
    result = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    return result


def test_package_import_is_lightweight():
    run_python(
        "-c",
        """
import sys
import embodied_harness

assert embodied_harness.__version__
assert not {"numpy", "torch", "jax", "openpi", "lerobot"}.intersection(sys.modules)
""",
    )


def test_module_cli_reports_package_name():
    result = run_python("-m", "embodied_harness.cli", "--version")
    assert result.stdout.startswith("embodied-harness ")
    assert result.stderr == ""


def test_cli_starts_without_the_optional_extras():
    # The CLI imports every adapter, so the extras' packages must load only where they are used: the
    # documented `data planner` env (lerobot plus `.[data]`) has no msgpack, and a base install has only
    # pydantic, PyYAML and numpy.
    result = run_python(
        "-c",
        f"""
import sys

for name in ("msgpack", "websockets", "PIL", "pyarrow"):
    sys.modules[name] = None  # importing it now raises ImportError

from embodied_harness.cli import main

try:
    main(["--help"])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("--help must exit")
assert main(["validate", {str(RUN_CONFIG)!r}]) == 0
""",
    )
    assert "OK:" in result.stdout


@pytest.mark.parametrize(
    "module",
    [
        "policy.vla.openpi",
        "policy.vla.lerobot",
        "policy.serving.openpi_policy",
        "policy.serving.lerobot_policy",
    ],
)
def test_policy_entry_points_accept_help_without_model_dependencies(module):
    result = run_python(
        "-c",
        f"""
import runpy
import sys

sys.argv = [{module!r}, "--help"]
try:
    runpy.run_module("embodied_harness." + {module!r}, run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("--help must exit successfully")
assert not {{"torch", "jax", "openpi", "lerobot"}}.intersection(sys.modules)
""",
    )
    assert "--checkpoint" in result.stdout
    assert "--max-request-mb" in result.stdout
    assert result.stderr == ""
