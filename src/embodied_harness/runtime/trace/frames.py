"""Per-tick capture, for turning runs into a LeRobot dataset (`embodied-harness data from-runs`).

With `trace.frames: true` every control tick records what the policy saw and what was done:
- each camera's image, one video per camera under `frames/` (lossless enough for training: x264 CRF 18);
- `frames.jsonl`: the proprio, the action actually sent to the env, where it came from (`policy`,
  `primitive` or `hold`) and the skill call and prompt running at that tick.

The action sent is what `env.act` / `env.hold` return; for an env that returns nothing, a numeric
policy action is recorded as given and the rest stays null. Videos need imageio (with ffmpeg) or PyAV;
without either they are skipped with one warning and frames.jsonl is still written.

A `--resume` never rewrites what was recorded: it is a new segment whose videos go to
`frames/resume_<k>/` (the k-th resume; segment 0 is `frames/` itself), and each row says where its
images are (`segment`, and `frame`, the index in that segment's videos; null without video). Tick
numbers continue across segments. Rows written before segments existed have neither field: their images
are frame <row index> of `frames/`, which only holds for a trace that was never resumed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .writer import last_record, repair_tail, to_jsonable

logger = logging.getLogger("embodied_harness.trace")

FRAMES_DIR = "frames"
FRAMES_FILE = "frames.jsonl"
TERMINAL_FILE = "terminal.json"
SOURCES = ("policy", "primitive", "hold")
RESUME_DIR = "resume_{segment}"  # under frames/: the videos of the episode's segment-th --resume


def segment_dir(episode_dir: Path, segment: int = 0) -> Path:
    """Where one recording segment's camera videos are: frames/ for the episode's first run,
    frames/resume_<k>/ for its k-th --resume."""
    base = episode_dir / FRAMES_DIR
    return base if segment == 0 else base / RESUME_DIR.format(segment=segment)


def row_video(row: Mapping[str, Any], index: int) -> tuple[int, int | None]:
    """(segment, frame index in that segment's videos) of the `index`-th row of frames.jsonl. Rows from
    before segments existed are frame `index` of segment 0."""
    if "segment" not in row:
        return 0, index
    frame = row.get("frame")
    return int(row["segment"]), None if frame is None else int(frame)


def plain(value: Any) -> Any:
    """Numbers, bools, strings and lists of them: what JSON keeps exactly (the trace's one normalizer,
    with any value it does not know as its text)."""
    return to_jsonable(value, default=str)


def numeric_action(action: Any) -> list[float] | None:
    """The action as a list of floats, or None when it is not a numeric vector (e.g. a primitive dict)."""
    if action is None or isinstance(action, (str, Mapping)):
        return None
    try:
        array = np.asarray(action, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array.tolist()


class _Video:
    """One camera's frames, written with imageio when it is installed, else with PyAV."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self._fps = fps
        self._append: Any = None
        self._close: Any = None

    def add(self, frame: np.ndarray) -> None:
        if self._append is None:
            self._open(frame)
        self._append(frame)

    def _open(self, frame: np.ndarray) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio
        except ImportError:
            imageio = None
        if imageio is not None:
            writer = imageio.get_writer(
                self.path,
                fps=self._fps,
                macro_block_size=1,
                codec="libx264",
                quality=None,
                output_params=["-crf", "18"],
            )
            self._append, self._close = writer.append_data, writer.close
            return
        import av

        container = av.open(str(self.path), mode="w")
        stream = container.add_stream("libx264", rate=max(1, round(self._fps)))
        stream.width, stream.height = frame.shape[1], frame.shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}

        def append(image: np.ndarray) -> None:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)

        def close() -> None:
            for packet in stream.encode():
                container.mux(packet)
            container.close()

        self._append, self._close = append, close

    def close(self) -> None:
        if self._close is not None:
            try:
                self._close()
            finally:
                self._append = self._close = None


class EpisodeFrames:
    def __init__(self, episode_dir: Path, *, fps: float, segment: int = 0) -> None:
        """`segment` is 0 for an episode's first run and k for its k-th --resume, whose videos go to
        their own directory so the footage recorded before is kept."""
        self._dir = episode_dir
        self._fps = fps
        self.segment = segment
        self._videos: dict[str, _Video] = {}
        self._videos_ok = True
        self._frames = 0  # frames written to this segment's videos
        path = episode_dir / FRAMES_FILE
        repair_tail(path)
        last = last_record(path)
        self._tick_offset = int(last.get("tick") or 0) if last else 0  # ticks continue across resumes
        self._file = path.open("a", encoding="utf-8")
        self.count = 0

    def add(
        self,
        *,
        tick: int,
        t: float,
        images: Mapping[str, Any],
        proprio: Mapping[str, Any],
        action: Any,
        source: str,
        call_id: str | None,
        call: str | None,
        prompt: str | None,
    ) -> None:
        """`tick` counts this run's ticks from 1; the row continues the numbering of earlier segments."""
        frame = None
        if self._videos_ok and self._add_images(images):
            frame, self._frames = self._frames, self._frames + 1
        record = {
            "tick": self._tick_offset + tick,
            "t": round(float(t), 6),
            "proprio": plain(proprio),
            "action": numeric_action(action),
            "source": source,
            "call_id": call_id,
            "call": call,
            "prompt": prompt,
            "segment": self.segment,
            "frame": frame,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1

    def _add_images(self, images: Mapping[str, Any]) -> bool:
        """Add this tick's camera images to the segment's videos; True when any was written."""
        written = False
        try:
            for camera in sorted(images):
                frame = np.asarray(images[camera])
                if frame.ndim != 3:
                    continue
                video = self._videos.get(camera)
                if video is None:
                    path = segment_dir(self._dir, self.segment) / f"{camera}.mp4"
                    video = self._videos[camera] = _Video(path, self._fps)
                video.add(np.ascontiguousarray(frame.astype(np.uint8)))
                written = True
        except Exception as exc:  # no imageio / PyAV here: keep the run and frames.jsonl
            logger.warning("frame videos disabled: %s", exc)
            self._videos_ok = False
            for video in self._videos.values():
                video.close()
            return False
        return written

    def finish(
        self,
        *,
        t: float,
        images: Mapping[str, Any],
        proprio: Mapping[str, Any],
        status: str,
        success: bool | None,
        ledger: Mapping[str, Any],
        memory: str,
    ) -> None:
        """Keep the terminal observation separately, without inventing an action target.

        The per-tick videos remain aligned one-to-one with actual actions. A planner may learn
        `done` from these terminal images; a policy must not learn a fabricated terminal action.
        """
        paths = {}
        for index, (camera, frame) in enumerate(sorted(images.items())):
            relative = Path("terminal") / f"camera_{index:03d}.png"
            path = self._dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v2 as imageio

                imageio.imwrite(path, np.asarray(frame, dtype=np.uint8))
            except ImportError:
                from PIL import Image

                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path, format="PNG")
            paths[camera] = relative.as_posix()
        record = {
            "t": round(float(t), 6),
            "images": paths,
            "proprio": plain(proprio),
            "status": status,
            "success": success,
            "ledger": plain(ledger),
            "memory": memory,
        }
        path = self._dir / TERMINAL_FILE
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def close(self) -> None:
        for video in self._videos.values():
            video.close()
        self._file.close()
