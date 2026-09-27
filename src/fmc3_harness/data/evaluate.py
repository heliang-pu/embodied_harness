"""Offline planner evaluation on exported samples, usually the held-out validation episodes.

Each sample is sent through the normal planner code path with the exported JPEG bytes, which
pass through the encoder untouched, so the model sees exactly the training-time input.
Like replay, this is open loop: every step starts from the demonstration's own state and
memory, so it scores single decisions, not whole episodes.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config.loader import Bundle
from ..core.types import DecisionReason
from ..planner.base import PlannerContext
from ..planner.decision import DecisionError, parse_decision
from ..trace.replay import compare_decisions
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
    with (export_dir / SAMPLES_FILE).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            if split != "all" and sample["split"] != split:
                continue
            if limit is not None and len(rows) >= limit:
                break
            images = {camera: (export_dir / path).read_bytes() for camera, path in sample["images"].items()}
            context = PlannerContext(
                task=bundle.task,
                images=images,
                proprio=dict(sample["proprio"]),
                memory=sample["memory_in"],
                reason=DecisionReason(sample["reason"]),
                skills=list(bundle.task.skills),
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
    }
