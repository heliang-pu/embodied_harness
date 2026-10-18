"""`embodied-harness lessons`: draft lessons for the planner from failures that keep coming back.

A failure pattern is a call that failed right after another call the planner chose (transition and recovery
primitives are invisible to the planner, so they do not count), for the same reason. When one shows up
at least `min_count` times (three by default), it becomes a draft: a factual sentence plus where it came
from, marked `reviewed: false`. Rewrite the sentence into advice the planner can act on (an order to prefer,
a step to take first), then set `reviewed: true`; only reviewed lessons reach the planner (`planner.lessons`
in the run config). Planner claims of "done" that the task check refused are counted the same way.

The file is merged, not overwritten: existing entries keep their text and review, and a pattern already in
the file (same source key) is not drafted again.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .report import START, EpisodeReport, build_report

KIND_TEXTS = {
    "readiness": "多半是起点不对，下一步没能开始",
    "grounding": "目标没找对",
    "control": "开始正常，执行中失败",
    "unknown": "原因不明",
}


def draft_lessons(episodes: Sequence[EpisodeReport], *, min_count: int = 3) -> list[dict[str, Any]]:
    build_report(episodes)  # fills each failed call's kind
    starts: Counter[tuple[str, str]] = Counter()
    failures: Counter[tuple[str, str, str]] = Counter()
    kinds: dict[tuple[str, str, str], Counter[str]] = {}
    for episode in episodes:
        previous = START  # the planner's own previous call: it never sees transition or recovery primitives
        for row in episode.calls:
            if row.purpose != "main":
                continue
            starts[(previous, row.key)] += 1
            if row.status == "failure":
                pattern = (previous, row.key, row.reason or "failure")
                failures[pattern] += 1
                kinds.setdefault(pattern, Counter())[row.kind or "unknown"] += 1
            previous = row.key
    drafts = []
    for (previous, call, reason), count in failures.most_common():
        if count < min_count:
            continue
        total = starts[(previous, call)]
        kind = kinds[(previous, call, reason)].most_common(1)[0][0]
        where = "一开始就执行" if previous == "START" else f"在 {previous} 之后直接执行"
        drafts.append(
            {
                "text": f"{where} {call} 常常失败：{count}/{total} 次（{reason}），{KIND_TEXTS[kind]}。",
                "reviewed": False,
                "source": f"report: {previous} -> {call}, {reason}, {count}/{total} failed, {kind}",
            }
        )
    false_done = sum(episode.false_done for episode in episodes)
    if false_done >= min_count:
        drafts.append(
            {
                "text": f"任务还没完成时，你声称完成了 {false_done} 次；声称完成前先核对进度。",
                "reviewed": False,
                "source": f"report: planner claimed done {false_done} times before the task check passed",
            }
        )
    return drafts


def merge_lessons(path: Path, drafts: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Add the drafts not in the file yet. Returns (added, already there)."""
    data: dict[str, Any] = {"version": 1, "lessons": []}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or data
        data.setdefault("lessons", [])
    known = {str(entry.get("source")) for entry in data["lessons"] if entry.get("source")}
    known |= {str(entry.get("text")) for entry in data["lessons"]}
    new = [draft for draft in drafts if draft["source"] not in known and draft["text"] not in known]
    data["lessons"] += new
    header = (
        "# Lessons for the planner (planner.lessons in the run config). Only entries with reviewed: true\n"
        "# are shown to it: rewrite each draft into advice first. Drafted by `embodied-harness lessons`.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return len(new), len(drafts) - len(new)
