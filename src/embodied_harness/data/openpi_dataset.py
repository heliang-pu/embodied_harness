"""Read LeRobot v3.0 samples for OpenPI without depending on OpenPI's legacy dataset loader."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..policy.integrations.openpi import OpenPIDatasetSpec, validate_lerobot_v3_dataset


class OpenPILeRobotDataset:
    """Random-access dataset of repacked, unnormalized OpenPI inputs.

    LeRobot owns video decoding, task-index resolution and episode-aware action padding. This
    adapter owns the storage-version check and input mapping. OpenPI's robot/model transforms and
    normalization must still run after this dataset, using the same settings as the served policy.
    The training process needs a v3-capable ``lerobot.datasets`` installation.
    """

    def __init__(
        self,
        root: str | Path,
        spec: OpenPIDatasetSpec,
        *,
        action_horizon: int,
        repo_id: str | None = None,
        episodes: list[int] | None = None,
        video_backend: str | None = None,
    ) -> None:
        if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        self.metadata = validate_lerobot_v3_dataset(Path(root), spec)
        # LeRobot's decoded RGB tensors use CHW even though storage metadata declares HWC.
        self.spec = replace(spec, image_input_layout="chw")
        self.action_horizon = action_horizon
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "OpenPILeRobotDataset needs a LeRobot v3-capable installation "
                "(lerobot.datasets.lerobot_dataset); OpenPI's older pinned LeRobot is not sufficient"
            ) from exc
        kwargs: dict[str, Any] = {
            "root": self.metadata.root,
            "delta_timestamps": {
                spec.action_key: [step / self.metadata.fps for step in range(action_horizon)]
            },
        }
        if episodes is not None:
            kwargs["episodes"] = episodes
        if video_backend is not None:
            kwargs["video_backend"] = video_backend
        self._dataset = LeRobotDataset(repo_id or f"local/{self.metadata.root.name}", **kwargs)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        sample = self._dataset[index]
        state = np.asarray(sample[self.spec.state_key])
        actions = np.asarray(sample[self.spec.action_key])
        if state.shape != (self.metadata.state_dim,):
            raise ValueError(f"state shape {state.shape} differs from metadata ({self.metadata.state_dim},)")
        expected = (self.action_horizon, self.metadata.action_dim)
        if actions.shape != expected:
            raise ValueError(f"action chunk shape {actions.shape} differs from expected {expected}")
        for name, value in (("state", state), ("actions", actions)):
            if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite numeric values")
        return self.spec.repack(sample)
