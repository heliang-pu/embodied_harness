"""Canonical policy package imports and legacy serving aliases."""

from __future__ import annotations

import importlib

from embodied_harness.policy.serving.lerobot_policy import LeRobotPolicy as LegacyLeRobotPolicy
from embodied_harness.policy.serving.openpi_policy import OpenPIPolicy as LegacyOpenPIPolicy
from embodied_harness.policy.vla.lerobot import LeRobotPolicy
from embodied_harness.policy.vla.openpi import OpenPIPolicy


def test_vla_adapters_have_canonical_and_legacy_imports() -> None:
    assert LegacyOpenPIPolicy is OpenPIPolicy
    assert LegacyLeRobotPolicy is LeRobotPolicy


def test_policy_family_packages_are_available() -> None:
    for name in ("embodied_harness.policy.rl", "embodied_harness.policy.va", "embodied_harness.policy.wam"):
        assert importlib.import_module(name)
