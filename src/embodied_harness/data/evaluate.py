"""Offline planner evaluation on exported samples, usually the held-out validation episodes.

Each sample is sent through the normal planner code path with the exported JPEG bytes, which
pass through the encoder untouched, so the model sees exactly the training-time input.
Like replay, this is open loop: every step starts from the demonstration's own state and
memory, so it scores single decisions, not whole episodes.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config.loader import Bundle
from ..core.types import DecisionReason, SkillCall
from ..planner.base import PlannerContext
from ..planner.decision import DecisionError, parse_decision
from ..runtime.trace.replay import compare_decisions
from .export import SAMPLES_FILE, percentile


@dataclass
class EvalRow:
    split: str
    episode_index: int
    step: int
    target: dict[str, Any]
    predicted: dict[str, Any] | None
    error: str | None
    same_skill: bool
    same_call: bool
    memory_similarity: float | None
    latency_s: float | None
    # Why the planner is asked: the sample's reason, with a periodic check split into "periodic keep" (the
    # target is the running call) and "periodic switch".
    kind: str = ""


def _kind(sample: dict[str, Any]) -> str:
    if sample["reason"] != DecisionReason.PERIODIC.value:
        return str(sample["reason"])
    target, history = sample["target"], sample.get("history") or []
    running = SkillCall(history[-1]["skill"], history[-1].get("args") or {}).key() if history else None
    keeps = not target.get("done") and SkillCall(target["skill"], target.get("args") or {}).key() == running
    return "periodic keep" if keeps else "periodic switch"


def _call_key(decision: dict[str, Any] | None) -> str:
    if decision is None:
        return "error"
    if decision.get("done"):
        return "done"
    args = decision.get("args") or {}
    if not args:
        return str(decision.get("skill"))
    inner = ", ".join(f"{name}={value}" for name, value in sorted(args.items()))
    return f"{decision.get('skill')}({inner})"


def evaluate_samples(
    planner: Any, bundle: Bundle, export_dir: Path, *, split: str = "val", limit: int | None = None
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    settings = getattr(planner, "settings", None)
    current = settings() if callable(settings) else {}
    # What each option needs from a sample; an export made without it cannot reproduce the prompt.
    needs = {"show_history": "history", "show_state": "last", "show_progress": "ledger"}
    with (export_dir / SAMPLES_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            if split != "all" and sample["split"] != split:
                continue
            if limit is not None and len(rows) >= limit:
                break
            for option, key in needs.items():
                missing = key not in sample or (key == "ledger" and sample[key] is None)
                if current.get(option) and missing:
                    raise ValueError(
                        f"this export has no {key} for {option}=True; "
                        "regenerate it with `embodied-harness data planner`"
                    )
            images = {camera: (export_dir / path).read_bytes() for camera, path in sample["images"].items()}
            task = bundle.task
            if sample.get("goal") and sample["goal"] != task.goal:
                task = task.model_copy(update={"goal": sample["goal"]})
            context = PlannerContext(
                task=task,
                images=images,
                proprio=dict(sample["proprio"]),
                memory=sample["memory_in"],
                reason=DecisionReason(sample["reason"]),
                skills=list(task.skills),
                history=list(sample.get("history") or []),
                ledger=sample.get("ledger"),
                last=sample.get("last"),
                resample=int(sample.get("resample", 0)),
                rejected=list(sample.get("rejected") or []),
            )
            target = sample["target"]
            predicted: dict[str, Any] | None = None
            error: str | None = None
            latency: float | None = None
            started = time.perf_counter()
            try:
                output = planner.plan(context)
                latency = time.perf_counter() - started
                predicted = parse_decision(output.raw).to_dict()
            except DecisionError as exc:
                error = f"unparseable output: {exc}"
            except Exception as exc:
                error = repr(exc)
            same_skill, same_call = compare_decisions(target, predicted)
            similarity = (
                SequenceMatcher(None, target["memory"], predicted["memory"]).ratio()
                if predicted is not None
                else None
            )
            rows.append(
                EvalRow(
                    sample["split"],
                    sample["episode_index"],
                    sample["step"],
                    target,
                    predicted,
                    error,
                    same_skill,
                    same_call,
                    similarity,
                    latency,
                    _kind(sample),
                )
            )
    return rows


def summarize_eval(rows: list[EvalRow]) -> dict[str, Any]:
    count = len(rows)
    done_targets = [row for row in rows if row.target.get("done")]
    other = [row for row in rows if not row.target.get("done")]
    similarities = [row.memory_similarity for row in rows if row.memory_similarity is not None]
    latencies = [row.latency_s for row in rows if row.latency_s is not None]
    mistakes = Counter(
        f"{_call_key(row.target)} -> {_call_key(row.predicted)}" for row in rows if not row.same_call
    )
    kinds: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        kinds[row.kind].append(row)

    def rate(part: int, whole: int) -> float | None:
        return part / whole if whole else None

    return {
        "samples": count,
        "errors": sum(row.error is not None for row in rows),
        "same_skill": sum(row.same_skill for row in rows),
        "same_call": sum(row.same_call for row in rows),
        "same_skill_rate": rate(sum(row.same_skill for row in rows), count),
        "same_call_rate": rate(sum(row.same_call for row in rows), count),
        "done_recall": rate(sum(row.same_call for row in done_targets), len(done_targets)),
        "premature_done": sum(bool(row.predicted and row.predicted.get("done")) for row in other),
        "memory_similarity": sum(similarities) / len(similarities) if similarities else None,
        "latency_p50_s": percentile(latencies, 50) if latencies else None,
        "latency_p95_s": percentile(latencies, 95) if latencies else None,
        "mistakes": dict(mistakes.most_common(10)),
        "by_kind": {
            kind: {"samples": len(group), "same_call": sum(row.same_call for row in group)}
            for kind, group in sorted(kinds.items())
        },
    }
