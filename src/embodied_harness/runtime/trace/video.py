"""Episode video: every control tick's camera images in a grid, streamed to episode.mp4.

For looking at what happened, not for replay (planner replay uses the saved planner inputs).
Needs imageio with its ffmpeg plugin; without them the video is skipped with one warning.
A `--resume` writes its own file (`episode_resume_<k>.mp4` for the k-th), so earlier footage is kept.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("embodied_harness.trace")


def episode_video_path(episode_dir: Path, segment: int = 0) -> Path:
    """episode.mp4 for an episode's first run, episode_resume_<k>.mp4 for its k-th --resume."""
    return episode_dir / ("episode.mp4" if segment == 0 else f"episode_resume_{segment}.mp4")


class EpisodeVideo:
    def __init__(self, path: Path, *, fps: float = 10.0, cameras: Sequence[str] | None = None) -> None:
        self.path = path
        self._fps = fps
        self._cameras = list(cameras) if cameras else None
        self._writer: Any = None
        self._disabled = False
        self.frames = 0

    def add(self, images: Mapping[str, Any]) -> None:
        if self._disabled:
            return
        names = self._cameras or sorted(images)
        frames = [np.asarray(images[name]) for name in names if name in images]
        frames = [frame for frame in frames if frame.ndim == 3 and frame.shape[-1] >= 3]
        frame = self._compose(frames)
        if frame is None:
            return
        try:
            if self._writer is None:
                import imageio.v2 as imageio

                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = imageio.get_writer(self.path, fps=self._fps, macro_block_size=1)
            self._writer.append_data(frame)
            self.frames += 1
        except Exception as exc:  # no imageio / ffmpeg on this machine: skip the video, keep the run
            logger.warning("episode video disabled: %s", exc)
            self._disabled = True
            self.close()

    @staticmethod
    def _compose(frames: Sequence[np.ndarray]) -> np.ndarray | None:
        """Arrange equal-sized RGB frames in a near-square grid without squashing them."""
        if not frames:
            return None
        first = np.asarray(frames[0])
        if first.ndim != 3 or first.shape[-1] < 3:
            return None
        height, width = first.shape[:2]
        if any(np.asarray(frame).shape[:2] != (height, width) for frame in frames):
            return None
        count = len(frames)
        columns = max(1, int(np.ceil(np.sqrt(count))))
        rows = int(np.ceil(count / columns))
        canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
        for index, frame in enumerate(frames):
            row, column = divmod(index, columns)
            canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = np.asarray(
                frame[..., :3], dtype=np.uint8
            )
        return canvas

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
