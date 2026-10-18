"""`embodied-harness data from-runs`: runs recorded with `trace.frames` as a LeRobot v3 dataset.

Every control tick becomes a frame: each camera's image, `observation.state` (the proprio vector named
by `state_key`), `action` (what was sent to the env), `harness.source` (0 policy, 1 primitive, 2 hold)
and `task`, the episode's instruction. The language columns are then written the way lerobot-annotate
writes them: a `subtask` row where each skill started (its prompt-library text; a scripted primitive's
description) and a `memory` row wherever the planner rewrote its memory. The result is the same kind of
dataset `data annotate` makes, so `data planner` and policy training read it as they are.

Needs lerobot (and PyAV, which it brings) in the environment that runs it.
"""

from __future__ import annotations

import bisect
import json
import math
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..config.schema import PromptLibrary, SkillsFile, TaskSpec
from ..config.skills import SkillRegistry
from ..core.ledger import Ledger
from ..core.types import SkillCall
from ..runtime.trace.frames import FRAMES_DIR, FRAMES_FILE, SOURCES
from ..runtime.trace.progress import episode_succeeded
from ..runtime.trace.reader import read_events, restore_state
from .export import commit_effects
from .steps import Step

SOURCE_CODES = {source: code for code, source in enumerate(SOURCES)}
WORK_DIR = ".embodied_harness"


@dataclass
class RecordedEpisode:
    directory: Path
    frames: list[dict[str, Any]]
    events: list[dict[str, Any]]
    outcome: dict[str, Any]
    snapshot: dict[str, Any]

    @property
    def goal(self) -> str:
        instruction = next((e.get("text") for e in self.events if e.get("type") == "instruction"), None)
        return str(instruction or self.snapshot.get("task", {}).get("goal", ""))

    @property
    def fps(self) -> float:
        return float(self.snapshot.get("run", {}).get("runtime", {}).get("control_hz", 10.0))

    @property
    def succeeded(self) -> bool:
        """The rule `run`, `report` and `suite` score by: the env's verdict when it gave one, else done."""
        return episode_succeeded(self.outcome.get("status"), self.outcome.get("env_success"))

    @property
    def cameras(self) -> list[str]:
        return sorted(path.stem for path in (self.directory / FRAMES_DIR).glob("*.mp4"))


def find_episodes(paths: Sequence[Path]) -> list[Path]:
    """Episode directories with frames, from run directories or episode directories."""
    found: list[Path] = []
    for path in paths:
        if (path / FRAMES_FILE).is_file():
            found.append(path)
        else:
            found += [episode for episode in sorted(path.glob("ep_*")) if (episode / FRAMES_FILE).is_file()]
    return found


def load_episode(directory: Path) -> RecordedEpisode:
    import yaml

    frames = [json.loads(line) for line in (directory / FRAMES_FILE).read_text().splitlines() if line.strip()]
    outcome_path = directory / "outcome.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    snapshot_path = directory / "config.snapshot.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text()) if snapshot_path.exists() else {}
    return RecordedEpisode(directory, frames, list(read_events(directory)), outcome, snapshot or {})


def check_episode(episode: RecordedEpisode, state_key: str) -> None:
    where = episode.directory
    fps = episode.fps
    if not math.isfinite(fps) or fps <= 0 or not math.isclose(fps, round(fps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{where}: control_hz={fps:g} is not an integer; from-runs requires an integer FPS so "
            "LeRobot timestamps and action chunks stay aligned"
        )
    if any(event.get("type") == "episode_resume" for event in episode.events):
        raise ValueError(
            f"{where}: resumed traces cannot be imported by from-runs because frames.jsonl and the single "
            "camera video do not identify the resume segment; export the original segments separately"
        )
    if not episode.frames:
        raise ValueError(f"{where}: no frames recorded")
    missing_action = sum(frame.get("action") is None for frame in episode.frames)
    if missing_action:
        raise ValueError(
            f"{where}: {missing_action} of {len(episode.frames)} tick(s) have no numeric action; the env "
            "must return the action it sends from act() and hold() for a run to become training data"
        )
    missing_state = sum(not isinstance(frame["proprio"].get(state_key), list) for frame in episode.frames)
    if missing_state:
        raise ValueError(f"{where}: {missing_state} tick(s) have no proprio vector {state_key!r}")
    if not episode.cameras:
        raise ValueError(f"{where}: no camera videos under {FRAMES_DIR}/ (were imageio or PyAV installed?)")


def _decode(path: Path) -> Iterator[np.ndarray]:
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def _call_texts(snapshot: dict[str, Any]) -> dict[str, str]:
    """Scripted skill name -> the text a label map compares it with (description, else the name)."""
    skills = snapshot.get("skills", {}).get("skills", {}) or {}
    return {
        name: (spec or {}).get("description") or name.replace("_", " ")
        for name, spec in skills.items()
        if (spec or {}).get("executor") == "scripted"
    }


def language_rows(episode: RecordedEpisode, fps: float) -> list[dict[str, Any]]:
    """Label the actions actually executed, including failures and recovery primitives.

    ``finish_chunk`` may send old actions after the next skill starts. Frame ownership,
    not a dispatch timestamp, determines the policy's language condition. Planner
    supervision is separately preserved in the sidecar and never inferred from these spans.
    """
    times = [float(frame["t"]) for frame in episode.frames]
    scripted = _call_texts(episode.snapshot)
    starts = {event.get("call_id"): event for event in episode.events if event.get("type") == "skill_start"}
    rows: dict[tuple[str, int], dict[str, Any]] = {}

    def add(style: str, index: int, text: str) -> None:
        rows[(style, index)] = {
            "role": "assistant",
            "content": text,
            "style": style,
            "timestamp": index / fps,
            "camera": None,
            "tool_calls": None,
        }

    if all("call_id" in frame for frame in episode.frames):
        previous: str | None = None
        for index, frame in enumerate(episode.frames):
            call_id = frame.get("call_id")
            if call_id is None or call_id == previous:
                continue
            event = starts.get(call_id, {})
            skill = (event.get("call") or {}).get("skill", "")
            text = frame.get("prompt") or scripted.get(skill) or skill
            if text:
                add("subtask", index, text)
                previous = call_id
    else:
        # Compatibility for hand-authored traces. Current runtime traces always
        # tag frames; these rows cannot safely recover finish_chunk boundaries.
        for event in episode.events:
            if event.get("type") != "skill_start":
                continue
            prompt = event.get("prompt") or {}
            call = event.get("call") or {}
            text = prompt.get("text") or scripted.get(call.get("skill", ""), call.get("skill", ""))
            if text:
                index = min(bisect.bisect_left(times, float(event["t"]) - 1e-9), len(times) - 1)
                add("subtask", index, text)
    for event in episode.events:
        if event.get("type") == "memory_update":
            index = min(bisect.bisect_left(times, float(event["t"]) - 1e-9), len(times) - 1)
            add("memory", index, event.get("memory") or "")
    return sorted(rows.values(), key=lambda row: (row["timestamp"], row["style"]))


def call_metadata(episode: RecordedEpisode) -> list[dict[str, Any]]:
    """Describe every executor call, its outcome and its actual action frames."""
    ends = {
        str(event.get("call_id")): event
        for event in episode.events
        if event.get("type") == "skill_end" and event.get("call_id") is not None
    }
    calls: list[dict[str, Any]] = []
    for event in episode.events:
        if event.get("type") != "skill_start":
            continue
        call_id = str(event.get("call_id", ""))
        end = ends.get(call_id)
        end_t = float(end["t"]) if end is not None and end.get("t") is not None else None
        start_t = float(event.get("t", 0.0))
        frame_indices = [
            index for index, frame in enumerate(episode.frames) if frame.get("call_id") == call_id
        ]
        outcome = (end or {}).get("outcome")
        calls.append(
            {
                "call_id": call_id,
                "purpose": event.get("purpose", "main"),
                "call": event.get("call"),
                "prompt": (event.get("prompt") or {}).get("text"),
                "start_t": start_t,
                "end_t": end_t,
                "outcome": outcome,
                "frame_indices": frame_indices,
                "success": bool((outcome or {}).get("status") == "success"),
            }
        )
    return calls


def planner_metadata(episode: RecordedEpisode, root: Path, episode_index: int) -> list[dict[str, Any]]:
    """Preserve actual accepted planner inputs, separately from executor action spans."""
    times = [float(frame["t"]) for frame in episode.frames]
    requests = {e["call_id"]: e for e in episode.events if e.get("type") == "planner_request"}
    refused = {
        e["call_id"] for e in episode.events if e.get("type") in {"decision_rejected", "switch_blocked"}
    }
    calls = call_metadata(episode)
    labels = {
        json.dumps(call["call"], sort_keys=True): call["prompt"] or (call["call"] or {}).get("skill", "")
        for call in calls
    }

    registry: SkillRegistry | None = None

    def snapshot_registry(problem: str) -> SkillRegistry:
        """The run's own skills, library and task: the immutable snapshot, not today's config files."""
        nonlocal registry
        if registry is None:
            if not all(name in episode.snapshot for name in ("skills", "library", "task")):
                raise ValueError(f"{episode.directory}: {problem}")
            registry = SkillRegistry(
                SkillsFile.model_validate(episode.snapshot["skills"]),
                PromptLibrary.model_validate(episode.snapshot["library"]),
                TaskSpec.model_validate(episode.snapshot["task"]),
            )
        return registry

    def prompt_for(call: dict[str, Any]) -> str:
        key = json.dumps(call, sort_keys=True)
        if labels.get(key):
            return labels[key]
        # A valid response can be interrupted before dispatch: resolve its text from the snapshot.
        skills = snapshot_registry("accepted planner call has no dispatch or complete config snapshot")
        skill_call = SkillCall(call["skill"], call.get("args") or {})
        prompt = skills.resolve_prompt(skill_call)
        return (
            prompt.text
            if prompt is not None
            else (skills.skill(skill_call.skill).spec.description or skill_call.skill.replace("_", " "))
        )

    def copy_image(relative: str, directory: str) -> str:
        source = (episode.directory / relative).resolve()
        source.relative_to(episode.directory.resolve())
        target = Path(WORK_DIR) / f"episode_{episode_index:06d}" / directory / source.name
        (root / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, root / target)
        return target.as_posix()

    decisions = []
    for event in episode.events:
        if event.get("type") != "planner_response" or event.get("errors"):
            continue
        call_id = event["call_id"]
        target = event.get("decision")
        # Done requires an actual successful terminal observation. It is added below.
        if call_id in refused or not target or target.get("done"):
            continue
        path = episode.directory / "planner" / f"{call_id}.json"
        if not path.is_file() or call_id not in requests:
            raise ValueError(
                f"{episode.directory}: {call_id} has no recorded planner input; re-record the run"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        stamp = float(requests[call_id]["t"])
        frame = min(bisect.bisect_left(times, stamp - 1e-9), len(times) - 1)
        images = {
            Path(relative).stem: copy_image(relative, call_id) for relative in record.get("attachments", [])
        }
        call = {"skill": target["skill"], "args": target.get("args") or {}}
        decisions.append(
            {
                "call_id": call_id,
                "frame": frame,
                "t": stamp,
                "label": prompt_for(call),
                "input": record["input"],
                "target": target,
                "images": images,
                "terminal": False,
            }
        )
    terminal_path = episode.directory / "terminal.json"
    if not (episode.succeeded and terminal_path.is_file()):
        return decisions
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    env_end = next((event for event in episode.events if event.get("type") == "env_end"), None)
    if terminal.get("success") is False or (env_end is not None and env_end.get("success") is not True):
        return decisions  # the env judged a failure, or ended the episode without a verdict
    restored = restore_state(episode.events)
    task = TaskSpec.model_validate(episode.snapshot["task"])
    ledger = Ledger.replay(task.ledger, restored.ledger_ops)
    last = restored.last
    running = _running_call(episode.events, env_end) if env_end is not None else None
    if running is not None:
        # The env's success ended the call before any verifier judged it, so nothing committed its
        # effects; that success is its outcome. The done decision sees the call committed, the way
        # Supervisor._commit leaves the ledger and the last result after a verified success.
        registry = snapshot_registry("the call running at the env's end needs a complete config snapshot")
        try:
            commit_effects(registry, running, dict(terminal["proprio"]), ledger, task)
        except ValueError as exc:
            raise ValueError(
                f"{episode.directory}: cannot apply the effects of {running.key()}, which ran when the env "
                f"ended the episode: {exc}"
            ) from exc
        last = {"call": running.key(), "status": "success", "reason": None}
    inputs: dict[str, Any] = {
        "goal": episode.goal,
        "proprio": terminal["proprio"],
        "memory": restored.memory,
        "reason": "skill_done",
        "history": [call.to_dict() for call in restored.history],
        "ledger": ledger.view(),
        "last": last,
    }
    memory = restored.memory
    declared = _planner_done(episode.events) if env_end is None else None
    if declared is not None:
        # The planner's own done answer ended the episode. Its input is what the planner saw then: the
        # state restored from the whole trace already holds the answer's memory and, during a skill,
        # the preemption the answer caused.
        path = episode.directory / "planner" / f"{declared['call_id']}.json"
        if not path.is_file():
            raise ValueError(
                f"{episode.directory}: {declared['call_id']} has no recorded planner input; re-record the run"
            )
        seen = json.loads(path.read_text(encoding="utf-8"))["input"]
        keys = ("memory", "reason", "history", "ledger", "last", "resample", "rejected")
        inputs.update({key: seen[key] for key in keys if key in seen})
        memory = str((declared.get("decision") or {}).get("memory") or "")
    decisions.append(
        {
            "call_id": "terminal",
            "frame": len(times) - 1,
            "t": terminal["t"],
            "label": None,
            "input": inputs,
            "target": {"done": True, "memory": memory},
            "images": {
                camera: copy_image(relative, "terminal")
                for camera, relative in terminal.get("images", {}).items()
            },
            "terminal": True,
        }
    )
    return decisions


def _running_call(events: Sequence[dict[str, Any]], env_end: dict[str, Any]) -> SkillCall | None:
    """The main call the env ended the episode during: dispatched, and not ended by a verdict or a switch.

    None while the supervisor planned or held after a verdict, and during a scripted prelude, which
    commits nothing.
    """
    ended = {event.get("call_id") for event in events if event.get("type") == "skill_end"}
    start = next((event for event in reversed(events) if event.get("type") == "skill_start"), None)
    if start is None or start.get("call_id") in ended or start.get("purpose", "main") != "main":
        return None
    call = SkillCall(str(start["call"]["skill"]), dict(start["call"].get("args") or {}))
    return call if env_end.get("active", call.key()) == call.key() else None


def _planner_done(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The planner's done answer (its planner_response) when that answer ended the episode.

    None when task.done_when ended it after a commit: the last check was not the planner's claim.
    """
    checks = [index for index, event in enumerate(events) if event.get("type") == "check"]
    if not checks:
        return None
    check = events[checks[-1]]
    if not (check.get("passed") and check.get("claimed_by_planner")):
        return None
    # The claim is checked right after the answer that made it.
    response = next((e for e in reversed(events[: checks[-1]]) if e.get("type") == "planner_response"), None)
    if response is None or response.get("errors") or not (response.get("decision") or {}).get("done"):
        return None
    return response


def successful_call_steps(root: Path) -> list[Step]:
    """Use verified main-call durations for library statistics, not planner polling intervals."""
    metadata = json.loads((root / WORK_DIR / "from_runs.json").read_text(encoding="utf-8"))
    steps = []
    for episode in metadata["episodes"]:
        for call in episode["calls"]:
            if call["purpose"] != "main" or not call["success"] or call["end_t"] is None:
                continue
            target = call["call"]
            steps.append(
                Step(
                    episode["episode_index"],
                    len(steps),
                    call["frame_indices"][0] if call["frame_indices"] else 0,
                    call["start_t"],
                    SkillCall(target["skill"], target.get("args") or {}),
                    (call["prompt"],) if call["prompt"] else (),
                    max(0.0, call["end_t"] - call["start_t"]),
                    "",
                    "",
                )
            )
    return steps


def export_runs(
    paths: Sequence[Path],
    out: Path,
    *,
    repo_id: str | None = None,
    state_key: str = "state",
    success_only: bool = False,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new directory")
    directories = find_episodes(paths)
    if not directories:
        raise FileNotFoundError("no episodes with frames.jsonl; record runs with `trace.frames: true`")
    episodes = [load_episode(directory) for directory in directories]
    skipped = [e.directory for e in episodes if success_only and not e.succeeded]
    episodes = [e for e in episodes if not (success_only and not e.succeeded)]
    if not episodes:
        raise ValueError("no successful episode to export")
    for episode in episodes:
        check_episode(episode, state_key)

    first = episodes[0]
    fps, cameras = first.fps, first.cameras
    shapes = {
        camera: next(_decode(first.directory / FRAMES_DIR / f"{camera}.mp4")).shape for camera in cameras
    }
    state_dim = len(first.frames[0]["proprio"][state_key])
    action_dim = len(first.frames[0]["action"])
    for episode in episodes[1:]:
        if episode.fps != fps or episode.cameras != cameras:
            raise ValueError(f"{episode.directory}: fps or cameras differ from {first.directory}")
    features: dict[str, Any] = {
        f"observation.images.{camera}": {
            "dtype": "video",
            "shape": tuple(shapes[camera]),
            "names": ["height", "width", "channels"],
        }
        for camera in cameras
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": [f"state_{i}" for i in range(state_dim)],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": [f"action_{i}" for i in range(action_dim)],
    }
    features["harness.source"] = {"dtype": "int64", "shape": (1,), "names": None}

    dataset = LeRobotDataset.create(
        repo_id=repo_id or f"local/{out.name}", fps=round(fps), features=features, root=out, use_videos=True
    )
    provenance = []
    for index, episode in enumerate(episodes):
        videos = [_decode(episode.directory / FRAMES_DIR / f"{camera}.mp4") for camera in cameras]
        count = 0
        for record, images in zip(episode.frames, zip(*videos, strict=False), strict=False):
            frame: dict[str, Any] = {
                f"observation.images.{c}": image for c, image in zip(cameras, images, strict=True)
            }
            frame["observation.state"] = np.asarray(record["proprio"][state_key], dtype=np.float32)
            frame["action"] = np.asarray(record["action"], dtype=np.float32)
            frame["harness.source"] = np.asarray([SOURCE_CODES[record["source"]]], dtype=np.int64)
            frame["task"] = episode.goal
            dataset.add_frame(frame)
            count += 1
        if count != len(episode.frames):
            raise ValueError(
                f"{episode.directory}: {len(episode.frames)} ticks but only {count} video frames"
            )
        dataset.save_episode(parallel_encoding=False)
        provenance.append(
            {
                "episode_index": index,
                "source": str(episode.directory),
                "frames": count,
                "goal": episode.goal,
                "status": episode.outcome.get("status"),
                "env_success": episode.outcome.get("env_success"),
                "calls": call_metadata(episode),
                "planner_decisions": planner_metadata(episode, out, index),
            }
        )
    dataset.finalize()

    rows = [language_rows(episode, float(round(fps))) for episode in episodes]
    _write_language(out, rows)
    work = out / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 2,
        "created": datetime.now().isoformat(timespec="seconds"),
        "embodied_harness": __version__,
        "state_key": state_key,
        "episodes": provenance,
        "skipped_unsuccessful": [str(path) for path in skipped],
    }
    (work / "from_runs.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "episodes": len(episodes),
        "frames": sum(item["frames"] for item in provenance),
        "skipped": len(skipped),
        "cameras": cameras,
        "subtask_rows": sum(row["style"] == "subtask" for episode_rows in rows for row in episode_rows),
        "memory_rows": sum(row["style"] == "memory" for episode_rows in rows for row in episode_rows),
    }


def _write_language(root: Path, rows: Sequence[list[dict[str, Any]]]) -> None:
    """Write `language_persistent` with lerobot-annotate's own writer, and declare it in meta/info.json."""
    from lerobot.annotations.steerable_pipeline.executor import Executor
    from lerobot.annotations.steerable_pipeline.reader import iter_episodes
    from lerobot.annotations.steerable_pipeline.staging import EpisodeStaging
    from lerobot.annotations.steerable_pipeline.writer import LanguageColumnsWriter

    staging = root / ".annotate_staging"
    for episode_index, episode_rows in enumerate(rows):
        EpisodeStaging(staging, episode_index).write("plan", episode_rows)
    LanguageColumnsWriter().write_all(list(iter_episodes(root)), staging, root)
    Executor._ensure_annotation_metadata_in_info(root)
    shutil.rmtree(staging)
