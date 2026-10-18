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
import json
from collections import defaultdict
from collections.abc import Sequence
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
    recorded: dict[str, Any] | None = None  # accepted run decision with its original planner context

    @property
    def done(self) -> bool:
        return self.label is None


@dataclass(frozen=True)
class AnnotatedEpisode:
    episode_index: int
    num_frames: int
    decisions: tuple[Decision, ...]  # empty when the episode has no subtask rows
    states: dict[int, list[float]]  # state vector at each decision frame (and offset frames)
    instruction: str | None = None  # from the dataset's instruction column, when one is named

    def frame_at(self, frame: int, offset: int) -> int:
        """`frame + offset`, kept inside the episode."""
        return max(0, min(self.num_frames - 1, frame + offset))


def data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}; is this a LeRobot v3 dataset?")
    return files


def read_annotated_episodes(
    root: Path,
    *,
    state_key: str = "observation.state",
    episodes: list[int] | None = None,
    instruction_key: str | None = None,
    frame_offsets: Sequence[int] = (0,),
    planner: bool = False,
) -> list[AnnotatedEpisode]:
    """`frame_offsets` also loads the states of frames around each decision, for sample augmentation."""
    import pyarrow.parquet as pq

    wanted = set(episodes) if episodes is not None else None
    files = data_files(root)
    frames: dict[int, list[tuple[int, float]]] = defaultdict(list)
    language: dict[int, list[dict[str, Any]]] = {}
    instructions: dict[int, str] = {}
    sidecar = root / ".embodied_harness" / "from_runs.json"
    recorded: dict[int, dict[str, Any]] | None = None
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 2:
            raise ValueError(
                f"{sidecar}: legacy from-runs metadata has no planner context; export the original runs again"
            )
        recorded = {int(item["episode_index"]): item for item in metadata["episodes"]}
    annotated = False
    for path in files:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        # A trial run (lerobot-annotate --only_episodes) rewrites only the files holding its episodes;
        # the episodes of the other files read as not annotated.
        has_language = LANGUAGE_PERSISTENT in names
        annotated = annotated or has_language
        has_instruction = instruction_key is not None and instruction_key in names
        if instruction_key is not None and not has_instruction and recorded is None:
            raise ValueError(f"{path} has no instruction column {instruction_key!r}")
        columns = ["episode_index", "frame_index", "timestamp"]
        if has_language:
            columns.append(LANGUAGE_PERSISTENT)
        if has_instruction:
            columns.append(instruction_key)
        for batch in parquet.iter_batches(batch_size=_BATCH_ROWS, columns=columns):
            episode_ids = batch.column("episode_index").to_pylist()
            frame_ids = batch.column("frame_index").to_pylist()
            stamps = batch.column("timestamp").to_pylist()
            persistent = batch.column(LANGUAGE_PERSISTENT) if has_language else None
            instruction_column = batch.column(instruction_key) if has_instruction else None
            for row, episode in enumerate(episode_ids):
                if wanted is not None and episode not in wanted:
                    continue
                if episode not in language:
                    # Every frame of an episode carries the same rows; read them once.
                    language[episode] = (persistent[row].as_py() or []) if persistent is not None else []
                    if instruction_column is not None:
                        value = instruction_column[row].as_py()
                        if value is not None and str(value).strip():
                            instructions[episode] = str(value).strip()
                elif instruction_column is not None:
                    value = instruction_column[row].as_py()
                    if value is not None and str(value).strip():
                        text = str(value).strip()
                        previous = instructions.get(episode)
                        if previous is not None and previous != text:
                            raise ValueError(
                                f"episode {episode} has inconsistent instructions {previous!r} and {text!r}"
                            )
                        instructions[episode] = text
                frames[episode].append((frame_ids[row], float(stamps[row])))
    if not annotated:
        raise ValueError(f"no file of {root} has a {LANGUAGE_PERSISTENT} column; run lerobot-annotate first")

    decisions = {
        episode: _decisions(language[episode], sorted(frames[episode]), append_done=recorded is None)
        for episode in frames
    }
    if recorded is not None:
        for episode in frames:
            if episode not in recorded or "planner_decisions" not in recorded[episode]:
                raise ValueError(f"{sidecar}: no planner decision metadata for episode {episode}")
            item = recorded[episode]
            goal = str(item.get("goal") or "").strip()
            if instructions.get(episode) and goal != instructions[episode]:
                raise ValueError(f"episode {episode}: instruction differs from the recorded run goal")
            instructions[episode] = goal
            if not planner:
                continue
            items = item["planner_decisions"]
            decisions[episode] = [
                Decision(
                    index=index,
                    frame=int(decision["frame"]),
                    t=float(decision["t"]),
                    label=decision.get("label"),
                    duration_s=max(0.0, float(items[index + 1]["t"]) - float(decision["t"]))
                    if index + 1 < len(items)
                    else 0.0,
                    memory=str(decision["target"].get("memory") or ""),
                    recorded=decision,
                )
                for index, decision in enumerate(items)
            ]
    needed = set()
    for episode, items in decisions.items():
        last = len(frames[episode]) - 1
        for decision in items:
            needed.add((episode, max(0, min(last, decision.frame))))
            needed.update((episode, max(0, min(last, decision.frame + offset))) for offset in frame_offsets)
    states = _read_states(files, state_key, needed)
    by_episode: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for (episode, frame), state in states.items():
        by_episode[episode][frame] = state
    return [
        AnnotatedEpisode(
            episode_index=episode,
            num_frames=len(frames[episode]),
            decisions=tuple(decisions[episode]),
            states=by_episode[episode],
            instruction=instructions.get(episode),
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


def _decisions(
    persistent: list[dict[str, Any]], frames: list[tuple[int, float]], *, append_done: bool = True
) -> list[Decision]:
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
    if append_done:
        # A candidate only: the reader knows no task. episode_steps drops it when no span maps to a call,
        # and the export keeps it only where task.done_when holds on the rebuilt ledger.
        decisions.append(Decision(len(subtasks), frames[-1][0], end_t, None, 0.0, memory_at(end_t)))
    return decisions
