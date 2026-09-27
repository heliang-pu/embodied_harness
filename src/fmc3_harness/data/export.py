"""Exports from annotated demonstrations: prompt-library statistics and planner fine-tuning samples."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..config.errors import ConfigError
from ..config.loader import Bundle
from ..config.schema import DurationStats, PromptLibrary
from ..core.types import DecisionReason
from ..planner.base import PlannerContext
from ..planner.render import render_target, to_sft_record
from ..skills.registry import SkillRegistry
from .annotations import AnnotatedEpisode
from .frames import FrameSource, dataset_features
from .labels import LabelMap
from .mapping import DataMapping, ProprioMapper
from .steps import Step, StepStats, episode_steps

SAMPLES_FILE = "samples.jsonl"
MANIFEST_FILE = "manifest.json"


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between the closest ranks, like numpy's default."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of no values")
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def library_with_stats(
    library: PromptLibrary, registry: SkillRegistry, steps: Iterable[Step]
) -> tuple[PromptLibrary, list[str]]:
    """Fill `episodes` and `duration_s` of each prompt from the annotated spans.

    Returns the new library and the ids of prompts that no span maps to.
    """
    durations: dict[str, list[float]] = defaultdict(list)
    episodes: dict[str, set[int]] = defaultdict(set)
    for step in steps:
        prompt = registry.resolve_prompt(step.call) if step.call is not None else None
        if prompt is not None:
            durations[prompt.id].append(step.duration_s)
            episodes[prompt.id].add(step.episode_index)
    prompts = []
    unused = []
    for entry in library.prompts:
        values = durations.get(entry.id)
        if not values:
            unused.append(entry.id)
            prompts.append(entry)
            continue
        stats = DurationStats(p50=round(percentile(values, 50), 2), p95=round(percentile(values, 95), 2))
        prompts.append(entry.model_copy(update={"episodes": len(episodes[entry.id]), "duration_s": stats}))
    return library.model_copy(update={"prompts": prompts}), unused


def split_episodes(episodes: Sequence[int], val_fraction: float, seed: int) -> dict[int, str]:
    """Hold out whole episodes, so validation never sees a frame of a training episode."""
    ordered = sorted(episodes)
    count = round(len(ordered) * val_fraction)
    if val_fraction > 0 and count == 0 and len(ordered) > 1:
        count = 1
    held_out = set(random.Random(seed).sample(ordered, count)) if count else set()
    return {episode: "val" if episode in held_out else "train" for episode in ordered}


def check_mapping(mapping: DataMapping, planner: Any, features: dict[str, Any]) -> ProprioMapper:
    problems = []
    for camera in planner.cameras:
        key = mapping.cameras.get(camera)
        if key is None:
            problems.append(f"mapping.cameras: no dataset key for planner camera {camera!r}")
        elif key not in features:
            problems.append(f"mapping.cameras.{camera}: the dataset has no feature {key!r}")
    mapper = ProprioMapper(mapping.proprio)
    if planner.proprio_keys is not None:
        missing = [key for key in planner.proprio_keys if key not in mapper.keys]
        if missing:
            problems.append(f"mapping.proprio: no expression for planner proprio key(s) {', '.join(missing)}")
    if mapper.keys and mapping.state_key not in features:
        problems.append(f"mapping.state_key: the dataset has no feature {mapping.state_key!r}")
    if problems:
        raise ConfigError(problems)
    return mapper


def export_planner_samples(
    *,
    bundle: Bundle,
    planner: Any,
    dataset: Path,
    episodes: Sequence[AnnotatedEpisode],
    labels: LabelMap,
    mapping: DataMapping,
    frames: FrameSource,
    out_dir: Path,
    val_fraction: float = 0.1,
    seed: int = 0,
    absolute_paths: bool = False,
) -> dict[str, Any]:
    """Write planner fine-tuning samples rendered exactly as the harness renders a live call.

    `planner` is a bound OpenAICompatPlanner (anything with render_request()). Output:
    train.jsonl and val.jsonl (messages + images), samples.jsonl (inputs and targets of every
    sample, read by `data eval`), images/ and manifest.json.
    """
    mapper = check_mapping(mapping, planner, dataset_features(dataset))
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; choose a new directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = bundle.run.planner.memory_max_chars
    # Only annotated episodes produce samples, so only they are split.
    annotated = [episode.episode_index for episode in episodes if episode.decisions]
    splits = split_episodes(annotated, val_fraction, seed)
    stats = StepStats()
    counts: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    split_episode_ids: dict[str, list[int]] = defaultdict(list)
    memory_over_limit = 0
    longest_memory = 0

    handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "val")}
    samples = (out_dir / SAMPLES_FILE).open("w", encoding="utf-8")
    try:
        for episode in episodes:
            split = splits.get(episode.episode_index, "train")
            steps = episode_steps(episode, labels, stats)
            if steps:
                split_episode_ids[split].append(episode.episode_index)
            for step in steps:
                state = episode.states.get(step.frame)
                if state is None and mapper.keys:
                    raise ValueError(
                        f"episode {episode.episode_index} frame {step.frame} has no {mapping.state_key!r}"
                    )
                proprio = mapper(state or [])
                keys = [mapping.cameras[camera] for camera in planner.cameras]
                images_by_key = frames.images(episode.episode_index, step.frame, keys)
                images = {camera: images_by_key[mapping.cameras[camera]] for camera in planner.cameras}
                longest_memory = max(longest_memory, len(step.memory_out))
                memory_over_limit += len(step.memory_out) > limit
                # Truncated the way Supervisor._update_memory truncates what the planner writes.
                memory_in, memory_out = step.memory_in[:limit], step.memory_out[:limit]
                reason = DecisionReason.START if step.index == 0 else DecisionReason.SKILL_DONE
                context = PlannerContext(
                    task=bundle.task,
                    images=images,
                    proprio=proprio,
                    memory=memory_in,
                    reason=reason,
                    skills=list(bundle.task.skills),
                )
                request = planner.render_request(context)
                paths = {}
                for camera, data in request.images.items():
                    relative = (
                        Path("images")
                        / f"episode_{episode.episode_index:06d}"
                        / f"frame_{step.frame:06d}_{camera}.{planner.image_extension}"
                    )
                    path = out_dir / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    paths[camera] = str(path.resolve()) if absolute_paths else relative.as_posix()
                target = {**step.target(), "memory": memory_out}
                record = to_sft_record(request.prompt, render_target(target), list(paths.values()))
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                sample = {
                    "split": split,
                    "line": counts[split],
                    "episode_index": episode.episode_index,
                    "step": step.index,
                    "frame": step.frame,
                    "t": round(step.t, 4),
                    "labels": list(step.labels),
                    "reason": reason.value,
                    "proprio": proprio,
                    "memory_in": memory_in,
                    "images": paths,
                    "target": target,
                }
                samples.write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts[split] += 1
                calls["done" if step.call is None else step.call.key()] += 1
    finally:
        for handle in (*handles.values(), samples):
            handle.close()
    if not counts["val"]:
        (out_dir / "val.jsonl").unlink()

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "fmc3_harness": __version__,
        "config": str(bundle.path),
        "dataset": str(dataset),
        "task": bundle.task.id,
        "library": bundle.library.library,
        "planner": planner.settings(),
        "mapping": mapping.model_dump(mode="json"),
        "val_fraction": val_fraction,
        "seed": seed,
        "image_paths": "absolute" if absolute_paths else "relative to this directory",
        "splits": {
            split: {"episodes": sorted(split_episode_ids[split]), "samples": counts[split]}
            for split in ("train", "val")
        },
        "calls": dict(sorted(calls.items())),
        "spans": {
            "episodes": stats.episodes,
            "unannotated_episodes": stats.unannotated,
            "spans": stats.spans,
            "skipped": stats.skipped,
            "merged": stats.merged,
        },
        "memory": {"max_chars": limit, "over_limit": memory_over_limit, "longest": longest_memory},
    }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
