"""`embodied-harness data annotate`: lerobot-annotate on a copy of a dataset, steered by the skill vocabulary.

lerobot-annotate (part of the lerobot package) watches each episode with a VLM, cuts it into subtask
spans and writes a `subtask` row at every span start and a `memory` row at every later start into
`language_persistent`, the column the other `data` commands read. The harness runs it with:

- a copy of the dataset, because lerobot-annotate rewrites the data shards in place (videos are linked);
- only what the harness uses, subtasks and memory: no plan rows, task paraphrases, interjections or VQA;
- the task's prompt library as the label vocabulary: the segmentation prompt asks for the library's
  exact texts, so most labels map onto a skill call without a human in the loop;
- the planner's memory limit and the task rules in the memory prompt.

The prompts go in through lerobot's own override hook (`LEROBOT_PROMPT_OVERRIDE_<name>`): the packaged
templates are extended with extra paragraphs, never rewritten, and keep all their fields.
"""

from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import __version__
from ..config.loader import Bundle
from ..config.schema import TaskSpec
from ..config.skills import SkillRegistry
from .annotations import read_annotated_episodes
from .labels import candidate_calls, normalize_label

STEERED_PROMPTS = ("plan_subtasks", "plan_subtask_relabel", "plan_memory")
PROMPT_ENV = "LEROBOT_PROMPT_OVERRIDE_{name}"
WORK_DIR = ".embodied_harness"  # inside the annotated copy: the prompts used and a record of the run


def packaged_prompt(name: str) -> str:
    """A prompt template as the installed lerobot ships it (ignoring any override already set)."""
    try:
        from lerobot.annotations.steerable_pipeline import prompts
    except ImportError as exc:
        raise RuntimeError(
            "lerobot-annotate is part of the lerobot package; run `data annotate` where lerobot is installed"
        ) from exc
    return (Path(prompts.__file__).parent / f"{name}.txt").read_text(encoding="utf-8")


def annotate_executable() -> list[str]:
    found = shutil.which("lerobot-annotate")
    return [found] if found else [sys.executable, "-m", "lerobot.scripts.lerobot_annotate"]


def label_vocabulary(registry: SkillRegistry, task: TaskSpec) -> list[str]:
    """The library text of every call the policy can run in this task, in vocabulary order."""
    texts: list[str] = []
    for candidate in candidate_calls(registry, task.skills):
        prompt = registry.resolve_prompt(candidate.call)
        if prompt is not None and prompt.text not in texts:
            texts.append(prompt.text)
    return texts


def label_descriptions(registry: SkillRegistry, task: TaskSpec) -> dict[str, str]:
    """What each vocabulary text means: the description of the skill whose call has that library text."""
    found: dict[str, str] = {}
    for candidate in candidate_calls(registry, task.skills):
        prompt = registry.resolve_prompt(candidate.call)
        description = registry.skill(candidate.call.skill).spec.description
        if prompt is not None and description and prompt.text not in found:
            found[prompt.text] = description
    return found


def _escape(text: str) -> str:
    """Literal text inside a str.format template."""
    return text.replace("{", "{{").replace("}", "}}")


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def _insert(template: str, block: str, anchors: Sequence[str]) -> str:
    """Put `block` before the first anchor found, or at the end."""
    for anchor in anchors:
        position = template.find(anchor)
        if position >= 0:
            return f"{template[:position]}{block}\n\n{template[position:]}"
    return f"{template.rstrip()}\n\n{block}\n"


def steer_prompts(
    templates: Mapping[str, str],
    *,
    vocabulary: Sequence[str],
    rules: Sequence[str],
    memory_max_chars: int,
    descriptions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The lerobot-annotate templates with the harness's paragraphs added.

    Without a vocabulary the segmentation prompts are left alone; the memory prompt always gets the
    planner's memory limit, plus the task rules when there are any. `descriptions` (vocabulary text ->
    what the skill does) follow their phrases, to help the annotator recognize each skill in the frames.
    """
    steered: dict[str, str] = {}
    if vocabulary:
        described = descriptions or {}
        phrases = "\n".join(
            f'  - "{_escape(text)}"' + (f" ({_escape(described[text])})" if described.get(text) else "")
            for text in vocabulary
        )
        explained = " (what each skill looks like follows it in brackets; copy only the quoted phrase)"
        labels = (
            "Label vocabulary. These are the robot's skills, and they override the label style above:\n"
            "label a segment with EXACTLY one of these phrases whenever one of them describes it, copied\n"
            f"verbatim, articles included{explained if described else ''}:\n"
            f"{phrases}\n"
            "Only when none of them fits the segment, write a short imperative phrase of your own.\n"
            "The vocabulary lists what the robot can do, not steps that must appear: label only\n"
            "what visibly happens in the frames. Never add a segment for a step the frames do not\n"
            "show, even when the vocabulary or the task description mentions it (recordings often\n"
            "stop before the task is done)."
        )
        steered["plan_subtasks"] = _insert(
            templates["plan_subtasks"], labels, ("Timing:", "Output strictly valid JSON")
        )
        steered["plan_subtask_relabel"] = _insert(templates["plan_subtask_relabel"], labels, ("Rules:",))
    memory = [
        f"Keep the memory under {memory_max_chars} characters: a robot task planner reads it at every step."
    ]
    if rules:
        memory.append(
            "The task has these rules; keep what later steps need to follow them (e.g. how many are done):"
        )
        memory += [f"  - {_escape(rule)}" for rule in rules]
    steered["plan_memory"] = _insert(
        templates["plan_memory"], "\n".join(memory), ("Output strictly valid JSON",)
    )
    for name, text in steered.items():
        added = _fields(text) - _fields(templates[name])
        if added:
            raise ValueError(f"the steered {name} prompt gained template fields {sorted(added)}")
    return steered


def copy_dataset(source: Path, target: Path, *, copy_videos: bool) -> dict[str, int]:
    """Copy everything but the videos, which are linked unless copy_videos. Returns file counts."""
    counts = {"copied": 0, "linked": 0}
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.name in (WORK_DIR, ".annotate_staging"):
            continue
        destination = target / entry.name
        if entry.name == "videos" and entry.is_dir() and not copy_videos:
            for path in sorted(p for p in entry.rglob("*") if p.is_file()):
                link = destination / path.relative_to(entry)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(path.resolve())
                counts["linked"] += 1
        elif entry.is_dir():
            shutil.copytree(entry, destination)
            counts["copied"] += sum(1 for p in destination.rglob("*") if p.is_file())
        else:
            shutil.copy2(entry, destination)
            counts["copied"] += 1
    return counts


@dataclass(frozen=True)
class AnnotateOptions:
    vlm_model: str | None = None  # --vlm.model_id; lerobot's default when None
    vlm_url: str | None = None  # an OpenAI-compatible server already running; lerobot starts one when None
    camera: str | None = None  # the dataset image key the VLM watches; lerobot's default when None
    episodes: tuple[int, ...] | None = None  # annotate only these episodes (a trial run); all when None
    concurrency: int | None = None  # requests in flight at once; lerobot's default (16) when None
    extra: tuple[str, ...] = field(default_factory=tuple)  # more lerobot-annotate arguments, passed last


def annotate_args(root: Path, options: AnnotateOptions) -> list[str]:
    args = [
        f"--root={root}",
        "--plan.emit_memory=true",
        "--plan.emit_plan=false",
        "--plan.n_task_rephrasings=0",
        "--interjections.enabled=false",
        "--vqa.enabled=false",
    ]
    if options.vlm_model:
        args.append(f"--vlm.model_id={options.vlm_model}")
    if options.vlm_url:
        args += [f"--vlm.api_base={options.vlm_url}", "--vlm.auto_serve=false"]
    if options.camera:
        args.append(f"--vlm.camera_key={options.camera}")
    if options.episodes is not None:
        args.append(f"--only_episodes=[{','.join(str(index) for index in options.episodes)}]")
    if options.concurrency is not None:
        args.append(f"--vlm.client_concurrency={options.concurrency}")
    return args + list(options.extra)


def annotate_dataset(
    *,
    bundle: Bundle,
    dataset: Path,
    out: Path,
    options: AnnotateOptions,
    steer: bool = True,
    copy_videos: bool = False,
    dry_run: bool = False,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Copy the dataset to `out` and run lerobot-annotate on the copy. `dataset` is never written to."""
    dataset, out = dataset.resolve(), out.resolve()
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"{dataset / 'meta' / 'info.json'} not found; is {dataset} a LeRobot v3 dataset?"
        )
    if out == dataset or dataset in out.parents:
        raise ValueError("write the annotated copy outside the source dataset")
    if out.exists() and any(entry.name != WORK_DIR for entry in out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new directory")

    vocabulary = label_vocabulary(bundle.registry, bundle.task) if steer else []
    templates = {name: packaged_prompt(name) for name in STEERED_PROMPTS}
    prompts = steer_prompts(
        templates,
        vocabulary=vocabulary,
        rules=bundle.task.rules,
        memory_max_chars=bundle.run.planner.memory_max_chars,
        descriptions=label_descriptions(bundle.registry, bundle.task) if steer else None,
    )
    work = out / WORK_DIR
    (work / "prompts").mkdir(parents=True, exist_ok=True)
    for name, text in prompts.items():
        (work / "prompts" / f"{name}.txt").write_text(text, encoding="utf-8")
    command = [*annotate_executable(), *annotate_args(out, options)]
    record = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "embodied_harness": __version__,
        "config": str(bundle.path),
        "source": str(dataset),
        "command": command,
        "vocabulary": vocabulary,
        "steered_prompts": sorted(prompts),
    }
    (work / "annotate.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result: dict[str, Any] = {"command": command, "prompts": work / "prompts", "vocabulary": vocabulary}
    if dry_run:
        return result

    result["files"] = copy_dataset(dataset, out, copy_videos=copy_videos)
    env = dict(os.environ)
    env.update({PROMPT_ENV.format(name=name): text for name, text in prompts.items()})
    completed = run(command, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"lerobot-annotate exited with code {completed.returncode}; the copy is in {out}")
    result["report"] = annotation_report(out, vocabulary)
    return result


def annotation_report(root: Path, vocabulary: Sequence[str]) -> dict[str, Any]:
    """How much of the dataset got annotated, and how many span labels are library texts verbatim."""
    episodes = read_annotated_episodes(root)
    labels = [d.label for e in episodes for d in e.decisions if d.label is not None]
    known = {normalize_label(text) for text in vocabulary}
    return {
        "episodes": len(episodes),
        "annotated": sum(bool(episode.decisions) for episode in episodes),
        "spans": len(labels),
        "distinct_labels": len({normalize_label(label) for label in labels}),
        "in_vocabulary": sum(normalize_label(label) in known for label in labels),
    }
