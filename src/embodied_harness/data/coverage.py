"""`embodied-harness data coverage`: which step orders the training data covers.

Composition fails less for lack of skills than for lack of the dependencies between them: a skill that
was never demonstrated right after a given other skill may not start from the state that one leaves
behind. This lists every transition (previous call -> call, with START for the first) in the annotated
dataset and, given runs, how often each transition the runs needed appears in training. Collect data for
the uncovered ones rather than for every combination.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..runtime.trace.report import START, composition_steps, episode_dirs, read_episode
from .annotations import AnnotatedEpisode
from .labels import LabelMap
from .steps import episode_steps


def training_transitions(episodes: Iterable[AnnotatedEpisode], labels: LabelMap) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for episode in episodes:
        previous = START
        for step in episode_steps(episode, labels):
            if step.call is None:
                break
            counts[(previous, step.call.key())] += 1
            previous = step.call.key()
    return counts


def run_transitions(paths: Sequence[Path]) -> dict[tuple[str, str], list[int]]:
    """Transition -> [starts, successes] over the runs' episodes."""
    found: dict[tuple[str, str], list[int]] = {}
    for directory in episode_dirs(paths):
        for previous, call, succeeded in composition_steps(read_episode(directory)):
            counts = found.setdefault((previous, call), [0, 0])
            counts[0] += 1
            counts[1] += succeeded
    return found


def format_coverage(training: Counter[tuple[str, str]], runs: dict[tuple[str, str], list[int]] | None) -> str:
    if runs is None:
        lines = ["| transition | in training |", "|---|---|"]
        for (before, after), count in sorted(training.items(), key=lambda item: -item[1]):
            lines.append(f"| {before} -> {after} | {count} |")
        return "\n".join(lines)
    lines = ["| transition | in training | in runs | run success |", "|---|---|---|---|"]
    ordered = sorted(runs.items(), key=lambda item: (training.get(item[0], 0), item[1][1] / item[1][0]))
    for (before, after), (starts, successes) in ordered:
        rate = f"{successes}/{starts} ({successes / starts:.0%})"
        lines.append(f"| {before} -> {after} | {training.get((before, after), 0)} | {starts} | {rate} |")
    missing = [pair for pair in runs if not training.get(pair)]
    lines += [
        "",
        f"{len(missing)} of {len(runs)} transition(s) the runs needed never appear in the training data",
    ]
    return "\n".join(lines)
