"""A small LeRobot v3 dataset as lerobot-annotate leaves it, without needing lerobot itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 10
CAMERA_KEYS = {"head": "observation.images.head", "wrist": "observation.images.wrist"}
HF_METADATA = {b"huggingface": b'{"info": {"features": {}}}'}

# episode -> frames, subtask rows (t, text) and memory rows (t, text)
EPISODES: dict[int, dict[str, Any]] = {
    0: {
        "frames": 40,
        "subtasks": [
            (0.0, "Pick up the package."),
            (1.0, "flip the package"),
            (2.0, "put it in bin a"),
            (2.5, "place the package into bin A"),
        ],
        "memories": [(1.0, "picked package 1"), (2.0, "package 1 flipped"), (2.5, "package 1 above bin A")],
    },
    1: {
        "frames": 30,
        "subtasks": [(0.0, "grab the parcel"), (1.0, "wave at the camera"), (1.5, "put it in bin c")],
        "memories": [(1.0, "holding package 2"), (1.5, "waved at the camera")],
    },
    2: {"frames": 10, "subtasks": [], "memories": []},  # never annotated
}
FILES = {"data/chunk-000/file-000.parquet": [0, 1], "data/chunk-000/file-001.parquet": [2]}


def state(frame: int) -> list[float]:
    return [0.1 * frame, 0.0, 0.04 + 0.001 * frame]


def persistent_rows(episode: int) -> list[dict[str, Any]]:
    spec = EPISODES[episode]
    rows = [
        {
            "role": "assistant",
            "content": text,
            "style": style,
            "timestamp": t,
            "camera": None,
            "tool_calls": None,
        }
        for style, items in (("subtask", spec["subtasks"]), ("memory", spec["memories"]))
        for t, text in items
    ]
    return sorted(rows, key=lambda row: (row["timestamp"], row["style"]))


def write_dataset(root: Path) -> Path:
    index = 0
    for relative, episodes in FILES.items():
        columns: dict[str, list[Any]] = {
            key: []
            for key in (
                "index",
                "episode_index",
                "frame_index",
                "timestamp",
                "observation.state",
                "task_index",
            )
        }
        persistent: list[list[dict[str, Any]]] = []
        for episode in episodes:
            rows = persistent_rows(episode)
            for frame in range(EPISODES[episode]["frames"]):
                columns["index"].append(index)
                columns["episode_index"].append(episode)
                columns["frame_index"].append(frame)
                columns["timestamp"].append(frame / FPS)
                columns["observation.state"].append(state(frame))
                columns["task_index"].append(0)
                persistent.append(rows)
                index += 1
        table = pa.table(
            {
                "index": pa.array(columns["index"], pa.int64()),
                "episode_index": pa.array(columns["episode_index"], pa.int64()),
                "frame_index": pa.array(columns["frame_index"], pa.int64()),
                "timestamp": pa.array(columns["timestamp"], pa.float32()),
                "observation.state": pa.array(columns["observation.state"], pa.list_(pa.float32())),
                "task_index": pa.array(columns["task_index"], pa.int64()),
                # lerobot-annotate lets pyarrow infer these types; do the same.
                "language_persistent": pa.array(persistent),
                "language_events": pa.array([[] for _ in persistent]),
            }
        ).replace_schema_metadata(HF_METADATA)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(path, table.schema) as writer:
            episode_ids = columns["episode_index"]
            for episode in episodes:
                start = episode_ids.index(episode)
                writer.write_table(table.slice(start, EPISODES[episode]["frames"]))

    features = {
        "observation.state": {"dtype": "float32", "shape": [3], "names": ["x", "y", "gripper"]},
        **{key: {"dtype": "video", "shape": [48, 64, 3]} for key in CAMERA_KEYS.values()},
        "language_persistent": {"dtype": "language", "shape": [1], "names": None},
    }
    info = {"codebase_version": "v3.0", "fps": FPS, "features": features}
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    video = root / "videos" / CAMERA_KEYS["head"] / "chunk-000" / "file-000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"not really a video")
    return root


class FakeFrames:
    """Synthetic frames whose pixel value encodes the episode, frame and camera."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, tuple[str, ...]]] = []

    @staticmethod
    def image(episode: int, frame: int, key: str) -> np.ndarray:
        value = (episode * 50 + frame * 3 + (7 if key.endswith("wrist") else 0)) % 256
        return np.full((48, 64, 3), value, dtype=np.uint8)

    def images(self, episode_index: int, frame: int, keys: Any) -> dict[str, Any]:
        self.calls.append((episode_index, frame, tuple(keys)))
        return {key: self.image(episode_index, frame, key) for key in keys}


MAPPING = {
    "state_key": "observation.state",
    "cameras": dict(CAMERA_KEYS),
    "proprio": {"gripper_width": "state[2]", "gripper_closed": "state[2] < 0.01"},
}
