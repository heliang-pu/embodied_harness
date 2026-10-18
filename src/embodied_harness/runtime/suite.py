"""`embodied-harness suite`: a composition benchmark over run configs.

A suite file lists entries, each a run config (optionally with overrides) run for some episodes, tagged
with a category and a split:
- category A (atomic), AP (atomic with perturbations), C (composition), CP (composition with
  perturbations), after RoboHiMan's task types;
- split seen or unseen: whether that composition was in the training data.
Running it with the sequence planner (a known-correct plan) measures the executor and the transitions;
with the VLM planner, the whole hierarchy. The summary gives success per entry and per category x split,
and lists the transitions (previous call -> next call) that only unseen entries exercise: where a
composition asks for a step order no seen task had.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config.loader import Bundle, load_bundle, read_yaml
from ..config.schema import SuiteEntry, SuiteFile
from .commands import AutoCommands
from .factory import check_bundle
from .runner import EpisodeRunner
from .trace.report import build_report, composition_steps, episode_dirs, read_episode

RUN_CONFIG_FILE = "run_config.yaml"
TASK_FILE = "task.yaml"


def load_suite(path: Path) -> SuiteFile:
    return SuiteFile.model_validate(read_yaml(path))


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def prepare_entry(suite_path: Path, entry: SuiteEntry, entry_dir: Path) -> Bundle:
    """Write the entry's run config (and task, when overridden) into its directory, with absolute paths
    to everything else, and load it. The directory then records exactly what was run."""
    config_path = (suite_path.parent / entry.config).resolve()
    base = config_path.parent
    run = deep_merge(read_yaml(config_path), entry.overrides)
    entry_dir.mkdir(parents=True, exist_ok=True)
    task_path = (base / run["task"]).resolve()
    if entry.task_overrides:
        task = read_yaml(task_path)
        task["task"] = deep_merge(task["task"], entry.task_overrides)
        task_path = entry_dir / TASK_FILE
        task_path.write_text(yaml.safe_dump(task, allow_unicode=True, sort_keys=False), encoding="utf-8")
    run["task"] = str(task_path)
    run["skills"] = str((base / run["skills"]).resolve())
    run["executor"]["library"] = str((base / run["executor"]["library"]).resolve())
    if run.get("planner", {}).get("lessons"):
        run["planner"]["lessons"] = str((base / run["planner"]["lessons"]).resolve())
    if run.get("planner", {}).get("trained_manifest"):
        run["planner"]["trained_manifest"] = str((base / run["planner"]["trained_manifest"]).resolve())
    path = entry_dir / RUN_CONFIG_FILE
    path.write_text(yaml.safe_dump(run, allow_unicode=True, sort_keys=False), encoding="utf-8")
    bundle = load_bundle(path)
    check_bundle(bundle)
    return bundle


def run_suite(
    suite_path: Path, suite: SuiteFile, run_dir: Path, *, only: Sequence[str] = (), print_line: Any = print
) -> None:
    unknown = [name for name in only if name not in {entry.name for entry in suite.entries}]
    if unknown:
        raise ValueError(f"no suite entry named {', '.join(unknown)}")
    selected = [entry for entry in suite.entries if not only or entry.name in only]
    # Check every selected cell before writing configs or executing any new episode.
    for entry in selected:
        entry_dir = run_dir / entry.name
        if any(entry_dir.glob("ep_*/events.jsonl")):
            raise FileExistsError(
                f"{entry_dir} already has episode traces; "
                "choose a new suite --run-id before rewriting its config"
            )
    for entry in selected:
        entry_dir = run_dir / entry.name
        bundle = prepare_entry(suite_path, entry, entry_dir)
        for index in range(entry.episodes):
            episode_id = f"ep_{index + 1:04d}"
            runner = EpisodeRunner(
                bundle,
                run_dir=entry_dir,
                episode_id=episode_id,
                seed=(bundle.run.runtime.seed if entry.seed is None else entry.seed) + index,
                commands=AutoCommands("abort"),
            )
            result = runner.run().result
            status = "stopped" if result is None else result.status
            if result is not None and result.env_success is not None:
                status += f" env_success={result.env_success}"
            print_line(f"{entry.name}/{episode_id}: {status}")


@dataclass
class EntrySummary:
    name: str
    category: str
    split: str
    episodes: int
    success: int
    causes: dict[str, int]
    transitions: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # [starts, successes]

    @property
    def rate(self) -> float | None:
        return self.success / self.episodes if self.episodes else None


def summarize_suite(suite: SuiteFile, run_dir: Path) -> list[EntrySummary]:
    summaries = []
    for entry in suite.entries:
        directory = run_dir / entry.name
        episodes = [read_episode(path) for path in episode_dirs([directory])] if directory.is_dir() else []
        report = build_report(episodes) if episodes else {"causes": {}}
        transitions: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        for episode in episodes:
            for previous, call, succeeded in composition_steps(episode):
                counts = transitions[(previous, call)]
                counts[0] += 1
                counts[1] += succeeded
        causes = {cause: count for cause, count in report["causes"].items() if count}
        summaries.append(
            EntrySummary(
                entry.name,
                entry.category,
                entry.split,
                len(episodes),
                causes.get("success", 0),
                causes,
                dict(transitions),
            )
        )
    return summaries


def _rate(success: int, total: int) -> str:
    return f"{success}/{total} ({success / total:.0%})" if total else "-"


def format_suite(suite: SuiteFile, summaries: Sequence[EntrySummary]) -> str:
    lines = [
        f"suite {suite.suite}",
        "",
        "| entry | category | split | success | causes |",
        "|---|---|---|---|---|",
    ]
    for item in summaries:
        causes = (
            ", ".join(f"{cause} {count}" for cause, count in item.causes.items() if cause != "success") or "-"
        )
        rate = _rate(item.success, item.episodes)
        lines.append(f"| {item.name} | {item.category} | {item.split} | {rate} | {causes} |")
    lines += ["", "| category | seen | unseen |", "|---|---|---|"]
    totals: Counter[tuple[str, str, str]] = Counter()
    for item in summaries:
        totals[(item.category, item.split, "episodes")] += item.episodes
        totals[(item.category, item.split, "success")] += item.success
    for category in ("A", "AP", "C", "CP"):
        cells = [
            _rate(totals[(category, split, "success")], totals[(category, split, "episodes")])
            for split in ("seen", "unseen")
        ]
        if any(cell != "-" for cell in cells):
            lines.append(f"| {category} | {cells[0]} | {cells[1]} |")
    seen = {pair for item in summaries if item.split == "seen" for pair in item.transitions}
    novel: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for item in summaries:
        if item.split == "unseen":
            for pair, (starts, successes) in item.transitions.items():
                if pair not in seen:
                    novel[pair][0] += starts
                    novel[pair][1] += successes
    if novel:
        lines += [
            "",
            "transitions only unseen entries exercise:",
            "",
            "| transition | starts | success |",
            "|---|---|---|",
        ]
        for (before, after), (starts, successes) in sorted(
            novel.items(), key=lambda item: item[1][1] / item[1][0]
        ):
            lines.append(f"| {before} -> {after} | {starts} | {_rate(successes, starts)} |")
    return "\n".join(lines)


def suite_json(summaries: Sequence[EntrySummary]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.name,
            "category": item.category,
            "split": item.split,
            "episodes": item.episodes,
            "success": item.success,
            "rate": item.rate,
            "causes": item.causes,
            "transitions": [
                {"previous": before, "next": after, "starts": starts, "success": successes}
                for (before, after), (starts, successes) in item.transitions.items()
            ],
        }
        for item in summaries
    ]
