"""`embodied-harness policy steer-check`: does the policy still follow its instruction?

The same observations go to the policy with the prompt of each chosen call, a few times each. A policy
that listens answers different prompts with clearly different action chunks; one that has lost its
steerability (narrow fine-tuning can do that) answers them alike. The check compares how far apart two
prompts' mean chunks are (separation) with how far apart repeated answers to one prompt are (noise). A
deterministic policy has no noise; the spread of all its actions is the yardstick then.

It needs neither a simulator nor ground truth: observations come from runs recorded with `trace.frames`
(robot or sim) or from env resets. Run it before and after fine-tuning; a pair that stops being
distinguished is a regression.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config.loader import Bundle
from ..core.types import Observation, SkillCall
from .trace.frames import FRAMES_FILE, row_video, segment_dir
from .trace.reader import read_events


@dataclass
class SteerObservation(Observation):
    """A sampled observation with its episode instruction, for goal-conditioned policies."""

    goal: str | None = None


@dataclass
class PairResult:
    first: str
    second: str
    separation: float  # RMS distance of the two prompts' mean chunks, averaged over observations
    ratio: float | None  # separation / noise; None for a deterministic policy
    relative: float  # separation / action scale
    distinguished: bool


@dataclass
class SteerReport:
    observations: int
    repeats: int
    calls: list[str]
    noise: float  # RMS distance between repeated answers to one prompt
    scale: float  # RMS spread of all actions around their mean
    pairs: list[PairResult]

    @property
    def passed(self) -> bool:
        return all(pair.distinguished for pair in self.pairs)


def policy_calls(bundle: Bundle) -> list[SkillCall]:
    """Every call of the task that the policy runs: one per prompt it can be given."""
    calls = []
    for name in bundle.task.skills:
        skill = bundle.registry.skill(name)
        if skill.scripted:
            continue
        if isinstance(skill.spec.prompt, dict):
            ((arg, spec),) = skill.spec.args.items()
            calls += [SkillCall(name, {arg: value}) for value in spec.values]
        else:
            calls.append(SkillCall(name))
    return calls


def query(
    executor: Any,
    bundle: Bundle,
    observations: Sequence[Observation],
    calls: Sequence[SkillCall],
    repeats: int,
) -> np.ndarray:
    """Action chunks as an array [observation, call, repeat, step, dim], cut to the shortest chunk."""
    chunks: list[list[list[np.ndarray]]] = []
    version = 0
    for obs in observations:
        task = bundle.task
        if task.goal_source == "episode":
            goal = getattr(obs, "goal", None)
            if not goal:
                raise ValueError("goal_source=episode needs an instruction for each steer-check observation")
            task = task.model_copy(update={"goal": goal})
        begin = getattr(executor, "begin_episode", None)
        if callable(begin):
            begin(task)
        per_call = []
        for call in calls:
            prompt = bundle.registry.resolve_prompt(call)
            answers = []
            for _ in range(repeats):
                version += 1
                executor.set_instruction(call, prompt, version)
                chunk = executor.infer_chunk(obs)
                try:
                    array = np.asarray(chunk.actions, dtype=np.float64)
                except (TypeError, ValueError):
                    raise ValueError(
                        "steer-check needs numeric actions; this executor sends something else"
                    ) from None
                if array.ndim != 2:
                    raise ValueError(f"expected a chunk of action vectors, got shape {array.shape}")
                answers.append(array)
            per_call.append(answers)
        chunks.append(per_call)
    steps = min(answer.shape[0] for per_call in chunks for answers in per_call for answer in answers)
    return np.asarray(
        [[[answer[:steps] for answer in answers] for answers in per_call] for per_call in chunks]
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def analyze(
    chunks: np.ndarray, calls: Sequence[str], *, min_ratio: float = 2.0, min_relative: float = 0.05
) -> SteerReport:
    """`chunks` is [observation, call, repeat, step, dim]."""
    observations, count, repeats = chunks.shape[:3]
    noise_terms = [
        _rms(chunks[o, c, a] - chunks[o, c, b])
        for o in range(observations)
        for c in range(count)
        for a, b in itertools.combinations(range(repeats), 2)
    ]
    noise = float(np.mean(noise_terms)) if noise_terms else 0.0
    flat = chunks.reshape(-1, chunks.shape[-1])
    scale = _rms(flat - flat.mean(axis=0))
    means = chunks.mean(axis=2)  # [observation, call, step, dim]
    pairs = []
    for first, second in itertools.combinations(range(count), 2):
        separation = float(np.mean([_rms(means[o, first] - means[o, second]) for o in range(observations)]))
        ratio = separation / noise if noise > 0 else None
        relative = separation / scale if scale > 0 else 0.0
        distinguished = ratio >= min_ratio if ratio is not None else relative >= min_relative
        pairs.append(PairResult(calls[first], calls[second], separation, ratio, relative, distinguished))
    pairs.sort(key=lambda pair: (pair.distinguished, pair.ratio if pair.ratio is not None else pair.relative))
    return SteerReport(observations, repeats, list(calls), noise, scale, pairs)


def format_steer(report: SteerReport) -> str:
    kept = sum(pair.distinguished for pair in report.pairs)
    lines = [
        f"{report.observations} observation(s) x {len(report.calls)} prompt(s) x {report.repeats} repeat(s); "
        f"noise {report.noise:.4g}, action scale {report.scale:.4g}; "
        f"{kept}/{len(report.pairs)} prompt pair(s) distinguished",
        "",
        "| prompt pair | separation | / noise | / scale | distinguished |",
        "|---|---|---|---|---|",
    ]
    for pair in report.pairs:
        ratio = "-" if pair.ratio is None else f"{pair.ratio:.2f}"
        lines.append(
            f"| {pair.first} vs {pair.second} | {pair.separation:.4g} | {ratio} | {pair.relative:.3f} | "
            f"{'yes' if pair.distinguished else 'NO'} |"
        )
    return "\n".join(lines)


# --- where the observations come from ---------------------------------------------------------------


def observations_from_env(bundle: Bundle, count: int, seed: int = 0, env: Any = None) -> list[Observation]:
    """The first observation after `count` resets of the configured env (seeds seed, seed+1, ...)."""
    from .factory import build_env

    env = env if env is not None else build_env(bundle)
    try:
        observations = []
        for index in range(count):
            obs = env.reset(seed + index)
            instruction = getattr(env, "instruction", None)
            goal = instruction() if callable(instruction) else None
            observations.append(
                SteerObservation(
                    t=obs.t,
                    images=obs.images,
                    proprio=obs.proprio,
                    info=obs.info,
                    terminated=obs.terminated,
                    goal=goal,
                )
            )
        return observations
    finally:
        env.close()


def read_video_frames(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    """The listed frames of one camera video (imageio with ffmpeg, else PyAV)."""
    try:
        import imageio.v3 as iio

        frames: Iterator[Any] = iio.imiter(path)
    except ImportError:
        import av

        container = av.open(str(path))
        frames = (frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))
    found: dict[int, np.ndarray] = {}
    for index, frame in enumerate(frames):
        if index in indices:
            found[index] = np.asarray(frame)
        if len(found) == len(indices):
            break
    return found


def _frame_rows(episode: Path, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The episode's frames.jsonl rows (a torn last line from a crash is left out), each with the
    `segment` and `frame` of its images. Rows from before segments existed only map to the videos by
    their line number, which a resume of that time broke; such a trace is refused."""
    rows = []
    resumed = any(event.get("type") == "episode_resume" for event in events)
    # A crash can cut the last line inside a multi-byte character; only that line is lost.
    lines = (episode / FRAMES_FILE).read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            if index == len(lines) - 1:
                break  # the torn tail a crash can leave
            raise
        if "segment" not in record and resumed:
            raise ValueError(
                f"{episode}: frames.jsonl was recorded before resumed runs kept their own videos, so "
                "its rows no longer match the camera videos; use episodes that were not resumed"
            )
        record["segment"], record["frame"] = row_video(record, len(rows))
        rows.append(record)
    return rows


def observations_from_frames(
    paths: Sequence[Path], count: int, during: str | None = None
) -> list[Observation]:
    """`count` ticks spread evenly over runs recorded with trace.frames; with `during`, only ticks while
    that skill ran (e.g. `pick`, to compare pick targets where picking starts). Each row's images come
    from the video of its own recording segment (a resumed episode has one per resume)."""
    episodes = []
    for path in paths:
        if (path / FRAMES_FILE).is_file():
            episodes.append(path)
        else:
            episodes += [p for p in sorted(path.glob("ep_*")) if (p / FRAMES_FILE).is_file()]
    if not episodes:
        raise FileNotFoundError(
            "no frames.jsonl under the given paths; record runs with `trace.frames: true`"
        )
    goals: dict[Path, str | None] = {}
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for episode in episodes:
        events = read_events(episode) if (episode / "events.jsonl").is_file() else []
        goals[episode] = next(
            (event.get("text") for event in events if event.get("type") == "instruction"), None
        )
        for record in _frame_rows(episode, events):
            call = record.get("call") or ""
            if during is None or call == during or call.startswith(f"{during}("):
                candidates.append((episode, record))
    if not candidates:
        raise ValueError(f"no recorded tick while {during!r} ran")
    picks = sorted({int(i) for i in np.linspace(0, len(candidates) - 1, min(count, len(candidates))).round()})
    chosen = [candidates[i] for i in picks]
    images: dict[tuple[Path, int], dict[str, dict[int, np.ndarray]]] = {}
    for episode, segment in dict.fromkeys((path, record["segment"]) for path, record in chosen):
        wanted = {
            record["frame"]
            for path, record in chosen
            if path == episode and record["segment"] == segment and record["frame"] is not None
        }
        directory = segment_dir(episode, segment)
        cameras = sorted(p.stem for p in directory.glob("*.mp4"))
        images[(episode, segment)] = {
            camera: read_video_frames(directory / f"{camera}.mp4", wanted) if wanted else {}
            for camera in cameras
        }
    observations = []
    for episode, record in chosen:
        videos, frame = images[(episode, record["segment"])], record["frame"]
        observations.append(
            SteerObservation(
                t=float(record["t"]),
                images={camera: found[frame] for camera, found in videos.items() if frame in found},
                proprio=dict(record["proprio"]),
                goal=goals[episode],
            )
        )
    return observations
