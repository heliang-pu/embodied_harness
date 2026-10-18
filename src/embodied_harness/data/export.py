"""Exports from annotated demonstrations: prompt-library statistics and planner fine-tuning samples."""

from __future__ import annotations

import contextlib
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..config.errors import ConfigError
from ..config.expressions import ExpressionError
from ..config.loader import Bundle
from ..config.schema import DurationStats, PromptLibrary, TaskSpec
from ..config.skills import SkillRegistry, task_namespace
from ..core.ledger import Ledger
from ..core.types import DecisionReason, SkillCall
from ..planner.base import PlannerContext
from ..planner.render import render_target, to_sft_record
from .annotations import AnnotatedEpisode, read_annotated_episodes
from .frames import FrameSource, dataset_features
from .labels import LabelMap
from .mapping import DataMapping, ProprioMapper
from .steps import Step, StepStats, episode_steps

SAMPLES_FILE = "samples.jsonl"
MANIFEST_FILE = "manifest.json"

logger = logging.getLogger("embodied_harness.data")


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


def episode_task(task: TaskSpec, episode: AnnotatedEpisode) -> TaskSpec:
    """The task as the planner saw it in one episode (goal_source episode: its instruction is the goal)."""
    if task.goal_source != "episode":
        return task
    if not episode.instruction:
        raise ValueError(
            f"episode {episode.episode_index} has no instruction; the task takes its goal from the episode, "
            "so name the dataset column in mapping.instruction_key"
        )
    return task.model_copy(update={"goal": episode.instruction})


def commit_effects(
    registry: SkillRegistry, call: SkillCall, proprio: dict[str, Any], ledger: Ledger, task: TaskSpec
) -> None:
    """Apply a demonstrated call's `effects.on_success` to the ledger, as Supervisor._commit does after
    a verified success: a demonstration's steps all succeeded."""
    namespace: dict[str, Any] = dict(proprio)
    # The mapping may give only some proprio keys; an effect that reads a missing predicate says so.
    with contextlib.suppress(ExpressionError):
        namespace.update(registry.predicates(proprio))
    namespace.update(
        ledger=ledger.view(), task=task_namespace(task), args=dict(call.args), executor_done=True
    )
    for effect in registry.skill(call.skill).effects:
        ledger.apply(effect.evaluate(namespace))


def dispatched_calls(steps: Sequence[Step], end: int) -> list[dict[str, Any]]:
    """The calls dispatched before steps[end], as Supervisor._history keeps them for the planner: in
    order, with consecutive repeats once. This remains true even when an excluded annotation sits between
    two equal calls."""
    history: list[dict[str, Any]] = []
    last_key: str | None = None
    for prior in steps[:end]:
        if prior.call is None or prior.call.key() == last_key:
            continue
        history.append(prior.call.to_dict())
        last_key = prior.call.key()
    return history


def ended_by_planner(registry: SkillRegistry, call: SkillCall) -> bool:
    """Whether only a periodic planner check ends this call: a policy skill without success conditions
    (a policy executor never reports done), so no verifier ends it with a success and nothing commits."""
    skill = registry.skill(call.skill)
    return not skill.scripted and not skill.success


def dataset_fps(root: Path) -> float:
    path = root / "meta" / "info.json"
    fps = json.loads(path.read_text(encoding="utf-8")).get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        raise ValueError(f"{path} has no fps; the periodic planner samples are placed by it")
    return float(fps)


def spread(frames: Sequence[int], limit: int | None) -> list[int]:
    """At most `limit` of the frames, evenly spaced, the first and the last included."""
    if limit is None or len(frames) <= limit:
        return list(frames)
    if limit <= 1:
        return list(frames[:limit])
    return [frames[round(index * (len(frames) - 1) / (limit - 1))] for index in range(limit)]


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
    frame_offsets: Sequence[int] = (0,),
    periodic_limit: int | None = None,
) -> dict[str, Any]:
    """Write planner fine-tuning samples rendered exactly as the harness renders a live call.

    `planner` is a bound OpenAICompatPlanner (anything with render_request()). Output:
    train.jsonl and val.jsonl (messages + images), samples.jsonl (inputs and targets of every
    sample, read by `data eval`), images/ and manifest.json.

    With planner.call periodic, demonstrations also give the decisions a periodic check asks while a
    call runs (Supervisor._tick_periodic), one every `every_steps` control steps of each step's span,
    at most `periodic_limit` per step. A call that no verifier can end (ended_by_planner) ends at the
    check that switches to the next step, so that decision is a periodic one too.
    """
    mapper = check_mapping(mapping, planner, dataset_features(dataset))
    if (dataset / ".embodied_harness" / "from_runs.json").is_file():
        episodes = read_annotated_episodes(
            dataset,
            state_key=mapping.state_key,
            episodes=[episode.episode_index for episode in episodes],
            instruction_key=mapping.instruction_key,
            frame_offsets=frame_offsets,
            planner=True,
        )
    recorded_data = any(d.recorded is not None for episode in episodes for d in episode.decisions)
    if recorded_data and tuple(frame_offsets) != (0,):
        raise ValueError(
            "frame_offsets must be [0] for runtime exports: use the exact recorded planner observations"
        )
    if not frame_offsets:
        # No decision would give a sample, and a periodic export would only hold its in-step checks.
        raise ValueError("frame_offsets is empty: every decision needs at least one frame, e.g. 0")
    if periodic_limit is not None and periodic_limit < 0:
        raise ValueError(f"periodic_limit must be 0 or more, not {periodic_limit}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; choose a new directory")
    run = bundle.run
    # Frames between two periodic checks; 0 when the planner is not periodic. Recorded runs already hold
    # the periodic decisions they made, with their exact inputs.
    check_every, fps = 0, None
    if periodic_limit is not None and (run.planner.call != "periodic" or recorded_data):
        raise ValueError(
            "periodic_limit only applies to demonstrations exported for planner.call: periodic; "
            "recorded runs keep the periodic checks they made"
        )
    if run.planner.call == "periodic" and not recorded_data:
        fps = dataset_fps(dataset)
        check_every = max(1, round(run.planner.every_steps * fps / run.runtime.control_hz))
        if mapper.keys:  # the proprio at the checks, between the decisions
            longest = max((episode.num_frames for episode in episodes), default=0)
            episodes = read_annotated_episodes(
                dataset,
                state_key=mapping.state_key,
                episodes=[episode.episode_index for episode in episodes],
                instruction_key=mapping.instruction_key,
                frame_offsets=sorted({*frame_offsets, *range(check_every, longest, check_every)}),
            )
    stats = StepStats()
    episode_step_lists = [episode_steps(episode, labels, stats) for episode in episodes]
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = run.planner.memory_max_chars
    # Only episodes with steps produce samples, so only they are split.
    with_steps = [
        episode.episode_index for episode, steps in zip(episodes, episode_step_lists, strict=True) if steps
    ]
    splits = split_episodes(with_steps, val_fraction, seed)
    counts: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    split_episode_ids: dict[str, list[int]] = defaultdict(list)
    memory_over_limit = 0
    longest_memory = 0
    settings = planner.settings() if callable(getattr(planner, "settings", None)) else {}
    # Progress is the ledger the supervisor would have had: the effects of every earlier step.
    track_ledger = bool(settings.get("show_progress"))
    # The supervisor refuses a planner's done while task.done_when is false, so a done target is only
    # kept where it holds; the ledger is rebuilt for that check even when progress is not shown.
    check_done = bool(bundle.task.done_when)
    unfinished: list[int] = []  # episodes that end before task.done_when holds: no done sample
    done_dropped = 0  # done samples (every frame offset) left out because planner.allow_done is false
    periodic_samples = 0
    switched_steps = 0

    def proprio_at(episode: AnnotatedEpisode, frame: int) -> dict[str, Any]:
        state = episode.states.get(frame)
        if state is None and mapper.keys:
            raise ValueError(f"episode {episode.episode_index} frame {frame} has no {mapping.state_key!r}")
        return mapper(state or [])

    def ends_unfinished(episode: AnnotatedEpisode, at_end: dict[str, Any] | None, task: TaskSpec) -> bool:
        """Whether task.done_when is false on the ledger the episode ends with."""
        if not check_done or at_end is None:
            return False
        try:
            return bundle.registry.is_done(at_end, task_namespace(task)) is False
        except ExpressionError as exc:
            raise ValueError(
                f"episode {episode.episode_index}: cannot check task.done_when on its ledger {at_end}: {exc}"
            ) from exc

    def write_samples(
        episode: AnnotatedEpisode,
        step: Step,
        split: str,
        task: TaskSpec,
        sample_frames: Sequence[int],
        *,
        reason: DecisionReason,
        history: list[dict[str, Any]],
        memory_in: str,
        last: dict[str, Any] | None,
        progress: dict[str, Any] | None,
        target: dict[str, Any],
    ) -> None:
        """One sample per frame: the decision of `step`'s target in this context."""
        recorded = step.recorded
        original = recorded["input"] if recorded is not None else None
        for frame in sample_frames:
            proprio = dict(original["proprio"]) if original is not None else proprio_at(episode, frame)
            keys = [mapping.cameras[camera] for camera in planner.cameras]
            saved_images = recorded.get("images", {}) if recorded is not None else {}
            missing = [camera for camera in planner.cameras if camera not in saved_images]
            if recorded is not None and recorded.get("terminal") and missing:
                raise ValueError(f"terminal observation is missing planner cameras: {', '.join(missing)}")
            images_by_key = (
                frames.images(episode.episode_index, frame, keys) if missing or not recorded else {}
            )
            images = (
                {camera: images_by_key[mapping.cameras[camera]] for camera in missing}
                if recorded
                else {camera: images_by_key[mapping.cameras[camera]] for camera in planner.cameras}
            )
            if saved_images:
                from PIL import Image

                for camera in planner.cameras:
                    if camera in saved_images:
                        with Image.open(dataset / saved_images[camera]) as saved:
                            images[camera] = saved.convert("RGB")
            context = PlannerContext(
                task=task,
                images=images,
                proprio=proprio,
                memory=memory_in,
                reason=reason,
                skills=list(task.skills),
                history=history,
                ledger=progress,
                last=last,
                resample=int(original.get("resample", 0)) if original is not None else 0,
                rejected=list(original.get("rejected") or []) if original is not None else [],
            )
            request = planner.render_request(context)
            paths = {}
            for camera, data in request.images.items():
                relative = (
                    Path("images")
                    / f"episode_{episode.episode_index:06d}"
                    / (f"step_{step.index:06d}_" if recorded is not None else "")
                    / f"frame_{frame:06d}_{camera}.{planner.image_extension}"
                )
                path = out_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                paths[camera] = str(path.resolve()) if absolute_paths else relative.as_posix()
            record = to_sft_record(request.prompt, render_target(target), list(paths.values()))
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            sample = {
                "split": split,
                "line": counts[split],
                "episode_index": episode.episode_index,
                "step": step.index,
                "frame": frame,
                "offset": frame - step.frame,
                "t": round(step.t, 4),
                "labels": list(step.labels),
                "reason": reason.value,
                "goal": task.goal,
                "proprio": proprio,
                "memory_in": memory_in,
                # Keep the same dispatched-call history that shaped the rendered prompt.
                # `data eval` must be able to reconstruct show_history=True requests.
                "history": history,
                "ledger": progress,
                "last": last,
                "resample": context.resample,
                "rejected": context.rejected,
                "terminal": bool(recorded and recorded.get("terminal")),
                "images": paths,
                "target": target,
            }
            samples.write(json.dumps(sample, ensure_ascii=False) + "\n")
            counts[split] += 1
            calls["done" if step.call is None else step.call.key()] += 1

    handles = {split: (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "val")}
    samples = (out_dir / SAMPLES_FILE).open("w", encoding="utf-8")
    try:
        for episode, steps in zip(episodes, episode_step_lists, strict=True):
            split = splits.get(episode.episode_index, "train")
            if steps:
                split_episode_ids[split].append(episode.episode_index)
            task = episode_task(bundle.task, episode)
            ledger = Ledger(bundle.task.ledger) if track_ledger or check_done else None
            last: dict[str, Any] | None = None  # how the previous step ended: in a demonstration, well
            # Periodic: the result the previous call gets once this decision, asked by a periodic check
            # while it still ran, switches away from it.
            switched: dict[str, Any] | None = None
            for position, step in enumerate(steps):
                recorded = step.recorded
                original = recorded["input"] if recorded is not None else None
                # Offsets: the same decision seen a few frames around the annotated boundary, since
                # at run time the planner is called when a verifier fires, not at that exact frame.
                sample_frames = (
                    [step.frame]
                    if recorded is not None
                    else sorted({episode.frame_at(step.frame, offset) for offset in frame_offsets})
                )
                if step.call is None:
                    if not run.planner.allow_done:
                        # The planner's schema and prompt rule done out; only the env ends the episode.
                        done_dropped += len(sample_frames)
                        continue
                    # The ledger at the end: rebuilt from the demonstrated steps' effects, or the one
                    # recorded with a run's terminal observation.
                    if original is not None:
                        at_end = original.get("ledger")
                    else:
                        at_end = ledger.view() if ledger is not None else None
                    if ends_unfinished(episode, at_end, task):
                        unfinished.append(episode.episode_index)
                        continue
                if recorded is not None and step.call is not None:
                    problems = bundle.registry.check_call(step.call)
                    if step.call.skill not in task.skills or problems:
                        raise ValueError(
                            f"recorded call {step.call.key()} is not valid in this task: {problems}"
                        )
                # The calls dispatched before this step, as the supervisor keeps them for the planner.
                history = dispatched_calls(steps, position)
                longest_memory = max(longest_memory, len(step.memory_out))
                memory_over_limit += len(step.memory_out) > limit
                # Truncated the way Supervisor._update_memory truncates what the planner writes.
                memory_in, memory_out = step.memory_in[:limit], step.memory_out[:limit]
                reason = DecisionReason.START if step.index == 0 else DecisionReason.SKILL_DONE
                if switched is not None:
                    reason = DecisionReason.PERIODIC
                if original is not None:
                    history = list(original.get("history") or [])
                    last = original.get("last")
                    reason = DecisionReason(original["reason"])
                target = {**step.target(), "memory": memory_out}
                progress = ledger.view() if track_ledger and ledger is not None else None
                if original is not None:
                    progress = original.get("ledger")
                    if track_ledger and progress is None:
                        raise ValueError(
                            "recorded planner input has no ledger; re-record with show_progress support"
                        )
                write_samples(
                    episode,
                    step,
                    split,
                    task,
                    sample_frames,
                    reason=reason,
                    history=history,
                    memory_in=memory_in,
                    last=last,
                    progress=progress,
                    target=target,
                )
                if switched is not None:
                    if step.call is None or step.call.key() != switched["call"]:
                        last = switched  # the previous call ended preempted; a confirm leaves it running
                    switched = None
                if step.call is not None and recorded is None:
                    if check_every:
                        # The checks while this call runs: inside its own spans (not in a span mapped to
                        # null after them), and before the next step's first sample. The call is in the
                        # history now, the planner's memory is what it wrote when it chose the call, and
                        # the answer keeps both.
                        span_end = step.frame + round(step.duration_s * fps)
                        if position + 1 < len(steps):
                            following = steps[position + 1].frame
                            shifted = [episode.frame_at(following, offset) for offset in frame_offsets]
                            span_end = min([span_end, following, *shifted])
                        grid = range(step.frame + check_every, span_end, check_every)
                        checks = spread(grid, periodic_limit)
                        write_samples(
                            episode,
                            step,
                            split,
                            task,
                            checks,
                            reason=DecisionReason.PERIODIC,
                            history=dispatched_calls(steps, position + 1),
                            memory_in=memory_out,
                            last=last,
                            progress=progress,
                            target=target,
                        )
                        periodic_samples += len(checks)
                    if check_every and ended_by_planner(bundle.registry, step.call):
                        # A planner switch preempts the call: no effects, and the next decision is a check.
                        switched = {"call": step.call.key(), "status": "preempted", "reason": "planner"}
                        switched_steps += 1
                    else:
                        if ledger is not None:
                            # The call ends where the next decision is made.
                            end = steps[position + 1].frame if position + 1 < len(steps) else step.frame
                            commit_effects(bundle.registry, step.call, proprio_at(episode, end), ledger, task)
                        last = {"call": step.call.key(), "status": "success", "reason": None}
    finally:
        for handle in (*handles.values(), samples):
            handle.close()
    if not counts["val"]:
        (out_dir / "val.jsonl").unlink()
    if unfinished:
        logger.warning(
            "%d episode(s) end before task.done_when (%s) holds, so they give no done sample: %s",
            len(unfinished),
            bundle.task.done_when,
            ", ".join(str(index) for index in unfinished),
        )

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "embodied_harness": __version__,
        "config": str(bundle.path),
        "dataset": str(dataset),
        "task": bundle.task.id,
        "library": bundle.library.library,
        "planner": planner.settings(),
        "mapping": mapping.model_dump(mode="json"),
        "val_fraction": val_fraction,
        "seed": seed,
        "frame_offsets": list(frame_offsets),
        "image_paths": "absolute" if absolute_paths else "relative to this directory",
        "splits": {
            split: {"episodes": sorted(split_episode_ids[split]), "samples": counts[split]}
            for split in ("train", "val")
        },
        "calls": dict(sorted(calls.items())),
        "spans": {
            "episodes": stats.episodes,
            "unannotated_episodes": stats.unannotated,
            "skipped_episodes": stats.skipped_episodes,
            "spans": stats.spans,
            "skipped": stats.skipped,
            "merged": stats.merged,
        },
        # None: every demonstration's end is taken as done. Otherwise done samples need it to hold.
        "done_when": bundle.task.done_when or None,
        "unfinished_episodes": sorted(unfinished),
        # planner.allow_done false: the planner never says done, so no demonstration ends in a done sample.
        "done_dropped": done_dropped,
        "memory": {"max_chars": limit, "over_limit": memory_over_limit, "longest": longest_memory},
    }
    if check_every:
        manifest["periodic"] = {
            "every_steps": run.planner.every_steps,
            "control_hz": run.runtime.control_hz,
            "fps": fps,
            "frames_per_check": check_every,
            "limit": periodic_limit,
            "samples": periodic_samples,  # decisions inside a step's span
            "switched_steps": switched_steps,  # calls only a planner switch ends (ended_by_planner)
        }
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
