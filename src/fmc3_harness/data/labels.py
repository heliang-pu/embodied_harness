"""The label map: which skill call each free-form subtask label from lerobot-annotate stands for.

lerobot-annotate asks a VLM for a short imperative phrase per span, so one skill shows up
under many wordings ("pick up the box", "grab the parcel", ...). The planner may only
output calls from the skill vocabulary, and the executor must be trained on exactly the
library text it will receive, so every label is mapped to a call once, by a human, in a
YAML file. `fmc3-harness data labels` writes that file with suggestions to review.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config.errors import ConfigError
from ..config.loader import read_yaml
from ..config.schema import TaskSpec
from ..core.types import SkillCall
from ..skills.registry import SkillRegistry
from .annotations import AnnotatedEpisode

_CALL = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*$")
_TOKEN = re.compile(r"[a-z0-9]+|[㐀-鿿]")
_TRAILING = " .。!！;；,，"
_ENTRY_KEYS = {"label", "count", "call", "reviewed"}


def normalize_label(text: str) -> str:
    """Labels that differ only in case, spacing or final punctuation are the same label."""
    return " ".join(text.lower().split()).rstrip(_TRAILING)


def parse_call(text: str) -> SkillCall:
    """`pick` or `place(bin=A)`, the same form as SkillCall.key()."""
    match = _CALL.match(text)
    if match is None:
        raise ValueError(f"{text!r} is not a skill call like `pick` or `place(bin=A)`")
    name, inner = match.group(1), match.group(2)
    args: dict[str, str] = {}
    if inner and inner.strip():
        for part in inner.split(","):
            key, sep, value = part.partition("=")
            if not sep or not key.strip() or not value.strip():
                raise ValueError(f"{text!r}: arguments are written as name=value")
            args[key.strip()] = value.strip()
    return SkillCall(name, args)


@dataclass(frozen=True)
class Candidate:
    call: SkillCall
    text: str  # what labels are compared with: the prompt text, else the description, else the name


def candidate_calls(registry: SkillRegistry, skills: Iterable[str]) -> list[Candidate]:
    """Every concrete call the planner can make, e.g. pick, place(bin=A), place(bin=B), ..."""
    candidates = []
    for name in skills:
        spec = registry.skill(name).spec
        choices = [[(arg, value) for value in arg_spec.values] for arg, arg_spec in spec.args.items()]
        for combo in itertools.product(*choices):
            call = SkillCall(name, dict(combo))
            prompt = registry.resolve_prompt(call)
            text = prompt.text if prompt is not None else (spec.description or name.replace("_", " "))
            candidates.append(Candidate(call, text))
    return candidates


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def rank_candidates(label: str, candidates: list[Candidate]) -> list[tuple[float, Candidate]]:
    """Word overlap (Jaccard) between the label and each candidate; a hint for the reviewer only."""
    words = _tokens(label)
    scored = []
    for candidate in candidates:
        other = _tokens(candidate.text)
        union = words | other
        scored.append((len(words & other) / len(union) if union else 0.0, candidate))
    scored.sort(key=lambda item: -item[0])  # stable, so ties keep vocabulary order
    return scored


def count_labels(episodes: Iterable[AnnotatedEpisode]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        for decision in episode.decisions:
            if decision.label is not None:
                counts[normalize_label(decision.label)] += 1
    return counts


@dataclass(frozen=True)
class LabelEntry:
    label: str
    count: int
    call: SkillCall | None  # None: leave these spans out of the planner data, keep their text
    reviewed: bool


@dataclass(frozen=True)
class LabelMap:
    entries: dict[str, LabelEntry]

    def lookup(self, label: str) -> LabelEntry:
        entry = self.entries.get(normalize_label(label))
        if entry is None:
            raise KeyError(f"subtask label {label!r} is not in the label map; run `fmc3-harness data labels`")
        return entry

    def missing(self, labels: Iterable[str]) -> list[str]:
        return sorted({normalize_label(label) for label in labels} - set(self.entries))


def _quoted(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)  # a JSON string is a valid YAML double-quoted scalar


def write_label_file(
    path: Path,
    counts: Counter[str],
    candidates: list[Candidate],
    *,
    dataset: Path,
    library: str,
    previous: LabelMap | None = None,
) -> dict[str, int]:
    """Write the label map for review. Entries already in `previous` keep their call and review state."""
    lines = [
        "# Subtask label map, generated by `fmc3-harness data labels`.",
        "# For each label, check `call`: the skill call it means, or null to leave those spans out of",
        "# the planner data. Then set `reviewed: true`; the export refuses unreviewed entries.",
        "# The comment after each call lists the closest calls by word overlap. It is only a hint.",
        f"dataset: {_quoted(str(dataset))}",
        f"library: {_quoted(library)}",
        "labels:",
    ]
    known = set(previous.entries) if previous is not None else set()
    stats = {"labels": 0, "new": 0, "kept": 0, "absent": 0}
    for label in sorted(set(counts) | known, key=lambda item: (-counts.get(item, 0), item)):
        ranked = rank_candidates(label, candidates)
        old = previous.entries.get(label) if previous is not None else None
        if old is not None:
            call, reviewed = old.call, old.reviewed
            stats["kept"] += 1
        else:
            best_score, best = ranked[0] if ranked else (0.0, None)
            call = best.call if best is not None and best_score > 0 else None
            reviewed = False
            stats["new"] += 1
        if label not in counts:
            hint = "not in this dataset"
            stats["absent"] += 1
        else:
            hint = "closest: " + ", ".join(f"{c.call.key()} {score:.2f}" for score, c in ranked[:3])
        call_text = _quoted(call.key()) if call is not None else "null"
        lines += [
            f"  - label: {_quoted(label)}",
            f"    count: {counts.get(label, 0)}",
            f"    call: {call_text}  # {hint}",
            f"    reviewed: {'true' if reviewed else 'false'}",
        ]
        stats["labels"] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def load_label_map(
    path: Path, registry: SkillRegistry, task: TaskSpec, *, allow_unreviewed: bool = False
) -> LabelMap:
    try:
        data = read_yaml(path)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError([f"{path}: {exc}"]) from exc
    if not isinstance(data, dict) or not isinstance(data.get("labels"), list):
        raise ConfigError([f"{path}: expected a mapping with a `labels` list"])
    problems: list[str] = []
    entries: dict[str, LabelEntry] = {}
    seen: set[str] = set()
    unreviewed: list[str] = []
    for position, item in enumerate(data["labels"]):
        where = f"{path.name}: labels[{position}]"
        if not isinstance(item, dict) or "label" not in item:
            problems.append(f"{where}: each entry needs a `label`")
            continue
        unknown = set(item) - _ENTRY_KEYS
        if unknown:
            problems.append(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
        label = normalize_label(str(item["label"]))
        if label in seen:
            problems.append(f"{where}: duplicate label {label!r}")
            continue
        seen.add(label)
        call = None
        if item.get("call") is not None:
            try:
                call = parse_call(str(item["call"]))
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
                continue
            if call.skill not in task.skills:
                problems.append(
                    f"{where}: {call.skill!r} is not one of the task's skills ({', '.join(task.skills)})"
                )
                continue
            call_problems = registry.check_call(call)
            if call_problems:
                problems += [f"{where}: {problem}" for problem in call_problems]
                continue
        reviewed = item.get("reviewed", False)
        if not isinstance(reviewed, bool):
            problems.append(f"{where}: reviewed must be true or false")
            continue
        if not reviewed:
            unreviewed.append(label)
        entries[label] = LabelEntry(label, int(item.get("count") or 0), call, reviewed)
    if unreviewed and not allow_unreviewed:
        problems.append(
            f"{len(unreviewed)} label(s) are not reviewed yet, e.g. {unreviewed[0]!r}: "
            "check them and set `reviewed: true`, or pass --allow-unreviewed for a quick experiment"
        )
    if problems:
        raise ConfigError(problems)
    return LabelMap(entries)
