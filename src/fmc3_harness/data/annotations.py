"""Reads lerobot-annotate output from a LeRobot v3 dataset, with pyarrow only.

lerobot-annotate writes a `language_persistent` column that holds the same list of rows on
every frame of an episode. Two row styles are used here:
- `subtask` rows, stamped at the start of each subtask span;
- `memory` rows, stamped at every subtask start except the first. Each holds the compressed
  memory of everything completed before that start.

That matches the planner contract. At the start of subtask i the planner writes the memory
row stamped at that start and chooses subtask i. After the last subtask it declares done.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LANGUAGE_PERSISTENT = "language_persistent"
SUBTASK_STYLE = "subtask"
MEMORY_STYLE = "memory"
_EPSILON_S = 1e-3  # rows and frames carry float32 timestamps; allow for rounding
_BATCH_ROWS = 8192


@dataclass(frozen=True)
class Decision:
    """One planner decision recovered from the annotations."""

    index: int
    frame: int  # frame_index where the decision is made: the frame nearest to `t`
    t: float
    label: str | None  # the free-form subtask label; None for the final "done" decision
    duration_s: float  # until the next subtask starts, or the episode ends
    memory: str  # the memory written at this decision: the latest memory row at or before `t`

    @property
    def done(self) -> bool:
        return self.label is None


@dataclass(frozen=True)
class AnnotatedEpisode:
    episode_index: int
    num_frames: int
    decisions: tuple[Decision, ...]  # empty when the episode has no subtask rows
    states: dict[int, list[float]]  # state vector at each decision frame


def data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}; is this a LeRobot v3 dataset?")
    return files


def read_annotated_episodes(
    root: Path, *, state_key: str = "observation.state", episodes: list[int] | None = None
) -> list[AnnotatedEpisode]:
    import pyarrow.parquet as pq

    wanted = set(episodes) if episodes is not None else None
    files = data_files(root)
    frames: dict[int, list[tuple[int, float]]] = defaultdict(list)
    language: dict[int, list[dict[str, Any]]] = {}
    for path in files:
        parquet = pq.ParquetFile(path)
        if LANGUAGE_PERSISTENT not in parquet.schema_arrow.names:
            raise ValueError(f"{path} has no {LANGUAGE_PERSISTENT} column; run lerobot-annotate first")
        columns = ["episode_index", "frame_index", "timestamp", LANGUAGE_PERSISTENT]
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=columns):
            episode_ids = batch.column("episode_index").to_pylist()
            frame_ids = batch.column("frame_index").to_pylist()
            stamps = batch.column("timestamp").to_pylist()
            persistent = batch.column(LANGUAGE_PERSISTENT)
            for row, episode in enumerate(episode_ids):
                if wanted is not None and episode not in wanted:
                    continue
                if episode not in language:
                    # Every frame of an episode carries the same rows; read them once.
                    language[episode] = persistent[row].as_py() or []
                frames[episode].append((frame_ids[row], float(stamps[row])))

    decisions = {episode: _decisions(language[episode], sorted(frames[episode])) for episode in frames}
    needed = {(episode, d.frame) for episode, items in decisions.items() for d in items}
    states = _read_states(files, state_key, needed)
    return [
        AnnotatedEpisode(
            episode_index=episode,
            num_frames=len(frames[episode]),
            decisions=tuple(decisions[episode]),
            states={
                d.frame: states[(episode, d.frame)]
                for d in decisions[episode]
                if (episode, d.frame) in states
            },
        )
        for episode in sorted(frames)
    ]


def _read_states(
    files: list[Path], state_key: str, needed: set[tuple[int, int]]
) -> dict[tuple[int, int], list[float]]:
    import pyarrow.parquet as pq

    states: dict[tuple[int, int], list[float]] = {}
    if not needed:
        return states
    for path in files:
        parquet = pq.ParquetFile(path)
        if state_key not in parquet.schema_arrow.names:
            continue
        for batch in parquet.iter_batches(
            batch_size=_BATCH_ROWS, columns=["episode_index", "frame_index", state_key]
        ):
            episode_ids = batch.column("episode_index").to_pylist()
            frame_ids = batch.column("frame_index").to_pylist()
            column = batch.column(state_key)
            for row, key in enumerate(zip(episode_ids, frame_ids, strict=True)):
                if key in needed:
                    states[key] = [float(value) for value in column[row].as_py()]
    return states


def _stamp(row: dict[str, Any]) -> float:
    return float(row["timestamp"])


def _decisions(persistent: list[dict[str, Any]], frames: list[tuple[int, float]]) -> list[Decision]:
    subtasks = sorted((r for r in persistent if r.get("style") == SUBTASK_STYLE), key=_stamp)
    if not subtasks or not frames:
        return []
    memories = sorted((r for r in persistent if r.get("style") == MEMORY_STYLE), key=_stamp)
    memory_times = [_stamp(r) for r in memories]
    frame_times = [t for _, t in frames]
    end_t = frame_times[-1]

    def memory_at(t: float) -> str:
        position = bisect.bisect_right(memory_times, t + _EPSILON_S)
        return (memories[position - 1].get("content") or "").strip() if position else ""

    def frame_at(t: float) -> int:
        position = bisect.bisect_left(frame_times, t)
        if position == 0:
            return frames[0][0]
        if position == len(frames):
            return frames[-1][0]
        before, after = frame_times[position - 1], frame_times[position]
        return frames[position][0] if after - t <= t - before else frames[position - 1][0]

    decisions = []
    for index, row in enumerate(subtasks):
        start = _stamp(row)
        end = _stamp(subtasks[index + 1]) if index + 1 < len(subtasks) else end_t
        label = (row.get("content") or "").strip()
        decisions.append(
            Decision(index, frame_at(start), start, label, max(0.0, end - start), memory_at(start))
        )
    decisions.append(Decision(len(subtasks), frames[-1][0], end_t, None, 0.0, memory_at(end_t)))
    return decisions
