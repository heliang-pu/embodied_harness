"""Camera frames and feature metadata of a LeRobot dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol


class FrameSource(Protocol):
    def images(self, episode_index: int, frame: int, keys: Sequence[str]) -> dict[str, Any]:
        """The images of one frame, by dataset feature key (e.g. observation.images.head)."""
        ...


def dataset_features(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found; is {root} a LeRobot v3 dataset?") from None
    return dict(info.get("features") or {})


class LeRobotFrames:
    """Decodes frames with LeRobot itself, video or image features alike.

    Needs the `lerobot` package, so run the export in the environment you train in.
    """

    def __init__(self, root: Path, repo_id: str | None = None) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise RuntimeError(
                "decoding camera frames needs the lerobot package; "
                "install fmc3-harness into the environment that has lerobot"
            ) from exc
        self._dataset = LeRobotDataset(repo_id or f"local/{root.name}", root=root)
        episodes = self._dataset.meta.episodes
        self._starts = {
            int(episode): int(start)
            for episode, start in zip(episodes["episode_index"], episodes["dataset_from_index"], strict=True)
        }

    def images(self, episode_index: int, frame: int, keys: Sequence[str]) -> dict[str, Any]:
        item = self._dataset[self._starts[episode_index] + frame]
        missing = [key for key in keys if key not in item]
        if missing:
            raise KeyError(f"the dataset has no {', '.join(missing)}")
        return {key: item[key] for key in keys}
