"""`embodied-harness report`: where episodes fail, per skill, per transition, and one cause per failed
episode.

It reads traces only (events.jsonl and outcome.json). Every skill call becomes one row: the call, the
call that started right before it (START for the first), when it started and how it ended. From those
rows come three tables:
- per skill (or per call with `by="call"`): how often it started and how it ended;
- per transition "previous -> next": the same for the next call, so a skill that works after one skill
  and fails after another stands out;
- one cause per failed episode (see `attribute`).

Each failed call also gets a kind (see `failure_kind`): readiness (it could not start from the state it
was given), grounding (it went for the wrong target), control (it started right and failed while
executing) or unknown. A verifier failure condition can name its kind; otherwise it is read from how
soon the call failed. `embodied-harness probe` checks a readiness suspicion directly.

An episode counts as a success by the env's own verdict when it gave one, else by finishing as done
(`episode_succeeded`, in trace/progress.py), the same rule `run`, `suite`, the plan summaries and
`data from-runs --success-only` use.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...core.types import SkillCall
from .progress import PlanProgress, episode_succeeded, plan_progress, read_snapshot, reference_plan
from .reader import read_events

START = "START"
CAUSES = ("success", "planner", "transition", "skill", "harness", "other")
KINDS = ("readiness", "grounding", "control", "unknown")
DONE_WELL = ("success", "ended_success")  # ended_*: the env ended the episode while the call ran
DONE_BADLY = ("failure", "ended_failure", "aborted")
EARLY_FRACTION = 1 / 3  # a failure within this share of the skill's typical duration: it never got going


@dataclass
class CallRow:
    call_id: str
    skill: str
    key: str
    purpose: str
    previous_skill: str
    previous_key: str
    start: float
    end: float | None = None
    # success, failure, preempted, ended_success, ended_failure, aborted, open. preempted with reason
    # `hold`: a HOLD stopped it; when that hold ended the episode it counts like an open call.
    status: str = "open"
    reason: str | None = None
    tagged_kind: str | None = None  # the `kind` of the verifier failure condition that fired
    kind: str | None = None  # failed calls: readiness, grounding, control or unknown (build_report)

    @property
    def elapsed(self) -> float:
        return (self.end if self.end is not None else self.start) - self.start


@dataclass
class EpisodeReport:
    directory: str
    status: str | None
    error_code: str | None
    message: str | None
    env_success: bool | None
    calls: list[CallRow]
    rejected: int  # planner answers the supervisor refused
    false_done: int  # "done" claims the task check did not accept
    planner_done: bool = False  # it finished on the planner's accepted "done"
    cause: str = "other"
    detail: str = ""
    kind: str | None = None  # the failed call's kind, for transition and skill causes
    progress: PlanProgress | None = None  # how far along the config's fixed plan it got (trace/progress.py)


def episode_dirs(paths: Iterable[Path]) -> list[Path]:
    """Episode directories, from run directories or episode directories."""
    found: list[Path] = []
    for path in paths:
        if (path / "events.jsonl").is_file():
            found.append(path)
        else:
            found += [p for p in sorted(path.glob("ep_*")) if (p / "events.jsonl").is_file()]
    return found


def read_episode(directory: Path) -> EpisodeReport:
    events = read_events(directory)
    outcome_path = directory / "outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else {}
    rows: list[CallRow] = []
    by_id: dict[str, CallRow] = {}
    previous = (START, START)
    env_end: dict[str, Any] | None = None
    held: CallRow | None = None  # the call the last HOLD stopped, while nothing has started since
    last_check: dict[str, Any] = {}
    rejected = false_done = 0
    for event in events:
        kind = event.get("type")
        if kind == "skill_start":
            call = SkillCall(event["call"]["skill"], dict(event["call"].get("args") or {}))
            row = CallRow(
                call_id=event["call_id"],
                skill=call.skill,
                key=call.key(),
                purpose=event.get("purpose", "main"),
                previous_skill=previous[0],
                previous_key=previous[1],
                start=float(event["t"]),
            )
            rows.append(row)
            by_id[row.call_id] = row
            held = None
            if row.purpose != "before_retry":
                # A recovery primitive belongs to the retry around it: the retried call still follows
                # the call that failed, so its failure reads as the skill's own, not a transition.
                previous = (call.skill, call.key())
        elif kind == "skill_end" and event.get("call_id") in by_id:
            row = by_id[event["call_id"]]
            ended = event.get("outcome") or {}
            row.end, row.status, row.reason = float(event["t"]), str(ended.get("status")), ended.get("reason")
            row.tagged_kind = (ended.get("evidence") or {}).get("kind")
            if row.status == "preempted" and row.reason == "hold":
                held = row
        elif kind == "env_end":
            env_end = event
        elif kind == "decision_rejected":
            rejected += 1
        elif kind == "check":
            last_check = event
            if event.get("claimed_by_planner") and not event.get("passed"):
                false_done += 1
    for row in rows:
        # Open calls (the env ended the episode during them, or a trace from before holds closed
        # them), and the call the final HOLD stopped, end the way the episode did.
        if row.status != "open" and row is not held:
            continue
        if env_end is not None:
            row.status = "ended_success" if env_end.get("success") else "ended_failure"
            if row.end is None:
                row.end = float(env_end["t"])
        elif outcome.get("status") == "aborted":
            row.status = "aborted"
    snapshot = read_snapshot(directory)
    plan = reference_plan(snapshot) if snapshot else None
    report = EpisodeReport(
        directory=str(directory),
        status=outcome.get("status"),
        error_code=outcome.get("error_code"),
        message=outcome.get("message"),
        env_success=outcome.get("env_success"),
        calls=rows,
        rejected=rejected,
        false_done=false_done,
        planner_done=bool(last_check.get("claimed_by_planner") and last_check.get("passed")),
        progress=None if plan is None else plan_progress(plan, events, outcome),
    )
    return report


def typical_durations(episodes: Sequence[EpisodeReport]) -> dict[str, float]:
    """Per skill, the median duration of its successful calls: how long it normally takes."""
    durations: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        for row in episode.calls:
            if row.status == "success":
                durations[row.skill].append(row.elapsed)
    return {skill: statistics.median(values) for skill, values in durations.items()}


def failure_kind(
    row: CallRow, typical: dict[str, float], *, window_s: float = 3.0, early: float = EARLY_FRACTION
) -> str:
    """What kind of failure a failed call was, from the trace alone:

    - the kind its verifier failure condition names, when it names one;
    - readiness: it stalled (the robot stopped moving), or it failed within `early` of the skill's
      typical successful duration (`window_s` when it never succeeded): it never got going;
    - control: a failure condition fired later, while it was executing;
    - unknown: anything else, e.g. a timeout, which a confused start and a slow execution both give.
    Grounding can only come from a condition that names it (e.g. a visual check of the target).
    """
    if row.tagged_kind in KINDS:
        return row.tagged_kind
    if row.status != "failure":
        return "unknown"  # the env ended the episode or the run was aborted during the call
    if row.reason == "stall" or row.elapsed < early * (typical.get(row.skill) or window_s):
        return "readiness"
    if row.reason not in (None, "timeout"):
        return "control"
    return "unknown"


def composition_steps(episode: EpisodeReport) -> list[tuple[str, str, bool]]:
    """The episode as steps of its composition: (previous call, call, succeeded) for each main call,
    with retries of the same call folded into one step that succeeds if any attempt does."""
    steps: list[tuple[str, str, bool]] = []
    previous = START
    for row in episode.calls:
        if row.purpose != "main":
            continue  # primitives are not steps of the composition
        succeeded = row.status in DONE_WELL
        if steps and row.key == previous:
            if succeeded:
                steps[-1] = (steps[-1][0], steps[-1][1], True)
            continue
        steps.append((previous, row.key, succeeded))
        previous = row.key
    return steps


def _culprit(episode: EpisodeReport) -> CallRow | None:
    """The call a failed episode's failure belongs to: the one running when it ended, else the last
    failed call."""
    return next((row for row in reversed(episode.calls) if row.status in DONE_BADLY), None)


def attribute(
    episode: EpisodeReport, *, window_s: float = 3.0, typical: dict[str, float] | None = None
) -> tuple[str, str]:
    """One cause per episode. A heuristic, from the trace alone:

    - success: the env called it a success; without an env verdict (a real robot), the episode
      finished as done;
    - planner: the planner's answers or calls ended the episode (invalid output, timeouts, its call
      budget used up, or a "done" the env did not confirm);
    - harness: an internal or config error;
    - transition / skill: the call the failure belongs to (the one running when the env ended the
      episode, else the last failed call) failed right after a different call and sooner than that
      skill normally takes to succeed (its median successful duration, at least `window_s`): it went
      wrong from the start it was given (transition). Later failures, and failures after a retry of the
      same call, count as the skill's own (skill);
    - other: anything else.
    Without ground truth, a valid but wrong planner decision shows up as the skill that ran; runs with
    a known-correct plan (the sequence planner) separate the two.
    """
    message = episode.message or ""
    if episode_succeeded(episode.status, episode.env_success):
        return "success", ""
    if episode.status == "done":  # the harness finished it, but the env's own verdict says otherwise
        if episode.planner_done:
            return "planner", "the planner declared done; env_success=False"
        return "other", "done_when passed; env_success=False"
    code = episode.error_code
    if code in ("E_PLANNER_INVALID", "E_PLANNER_TIMEOUT") or (
        code == "E_BUDGET" and "planner call" in message
    ):
        return "planner", f"{code}: {message}"
    if code in ("E_INTERNAL", "E_CONFIG"):
        return "harness", f"{code}: {message}"
    culprit = _culprit(episode)
    if culprit is None:
        return "other", message or str(episode.status)
    switched = culprit.previous_key not in (START, culprit.key)
    window = max(window_s, (typical or {}).get(culprit.skill, 0.0))
    cause = "transition" if switched and culprit.elapsed < window else "skill"
    how = culprit.reason or culprit.status
    return cause, f"{culprit.key} after {culprit.previous_key}: {how} {culprit.elapsed:.1f}s in"


@dataclass
class Tally:
    starts: int = 0
    success: int = 0
    failure: int = 0
    preempted: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    kinds: Counter[str] = field(default_factory=Counter)

    def add(self, row: CallRow) -> None:
        self.starts += 1
        if row.status in DONE_WELL:
            self.success += 1
        elif row.status in DONE_BADLY:
            self.failure += 1
            self.reasons[row.reason or row.status] += 1
            self.kinds[row.kind or "unknown"] += 1
        elif row.status == "preempted":
            self.preempted += 1

    @property
    def rate(self) -> float | None:
        decided = self.success + self.failure
        return self.success / decided if decided else None


def build_report(
    episodes: Sequence[EpisodeReport],
    *,
    by: str = "skill",
    window_s: float = 3.0,
    early: float = EARLY_FRACTION,
) -> dict[str, Any]:
    typical = typical_durations(episodes)
    for episode in episodes:
        for row in episode.calls:
            if row.status in DONE_BADLY:
                row.kind = failure_kind(row, typical, window_s=window_s, early=early)
        episode.cause, episode.detail = attribute(episode, window_s=window_s, typical=typical)
        culprit = _culprit(episode)
        if episode.cause in ("transition", "skill") and culprit is not None:
            episode.kind = culprit.kind
    key = (lambda row: row.skill) if by == "skill" else (lambda row: row.key)
    previous = (lambda row: row.previous_skill) if by == "skill" else (lambda row: row.previous_key)
    skills: dict[str, Tally] = defaultdict(Tally)
    transitions: dict[tuple[str, str], Tally] = defaultdict(Tally)
    for episode in episodes:
        for row in episode.calls:
            skills[key(row)].add(row)
            transitions[(previous(row), key(row))].add(row)
    causes = Counter(episode.cause for episode in episodes)
    kinds = {
        cause: dict(Counter(e.kind or "unknown" for e in episodes if e.cause == cause))
        for cause in ("transition", "skill")
    }
    return {
        "episodes": len(episodes),
        "causes": {cause: causes.get(cause, 0) for cause in CAUSES},
        "kinds": kinds,  # per cause, the kinds of the failed calls behind it
        "typical_s": typical,
        "skills": skills,
        "transitions": transitions,
        "failed": [episode for episode in episodes if episode.cause != "success"],
        "plan": plan_summaries(episodes),  # one entry per distinct fixed plan; empty without one
    }


def plan_summaries(episodes: Sequence[EpisodeReport]) -> list[dict[str, Any]]:
    """Long-horizon numbers per distinct plan: average completed steps (CALVIN's average length), the chain
    before the first failure, success@k, and per step where episodes stall."""
    groups: dict[tuple[str, ...], list[EpisodeReport]] = defaultdict(list)
    for episode in episodes:
        if episode.progress is not None:
            groups[tuple(episode.progress.plan)].append(episode)
    summaries = []
    for plan, members in groups.items():
        progresses = [episode.progress for episode in members if episode.progress is not None]
        count, n = len(progresses), len(plan)

        def mean(values: Iterable[float], count: int = count) -> float:
            return sum(values) / count

        completed = mean(p.completed for p in progresses)
        first_failure = mean(p.first_failure for p in progresses)
        steps = []
        for index, call in enumerate(plan):
            reasons: Counter[str] = Counter()
            for p in progresses:
                reasons.update(p.failures[index])
            steps.append(
                {
                    "step": index + 1,
                    "call": call,
                    "reached": sum(p.attempts[index] > 0 or p.skipped[index] > 0 for p in progresses),
                    "completed": sum(p.completed > index for p in progresses),
                    "first_try": sum(p.first_try[index] is True for p in progresses),
                    "attempts": sum(p.attempts[index] for p in progresses),
                    "rollbacks_from": sum(p.rollbacks_from[index] for p in progresses),
                    "stopped": sum(p.stopped_at == index for p in progresses),
                    "failures": dict(reasons),
                }
            )
        summaries.append(
            {
                "plan": list(plan),
                "steps": n,
                "episodes": count,
                "done": sum(episode_succeeded(e.status, e.env_success) for e in members),
                "exact": all(p.exact for p in progresses),
                "avg_completed": completed,
                "avg_first_failure": first_failure,
                "avg_furthest": mean(p.furthest for p in progresses),
                "recovered": completed - first_failure,
                "success_at": [sum(p.completed >= k for p in progresses) / count for k in range(1, n + 1)],
                "rollbacks": sum(p.rollbacks for p in progresses),
                "off_plan": sum(p.off_plan for p in progresses),
                "per_step": steps,
            }
        )
    return summaries


def _rate(tally: Tally) -> str:
    return "-" if tally.rate is None else f"{tally.rate:.0%}"


def _reasons(tally: Tally) -> str:
    return ", ".join(f"{reason} x{count}" for reason, count in tally.reasons.most_common(3)) or "-"


def _kinds(counts: Counter[str] | dict[str, int]) -> str:
    return ", ".join(f"{kind} {counts[kind]}" for kind in KINDS if counts.get(kind)) or "-"


def format_report(report: dict[str, Any]) -> str:
    causes = []
    for cause, count in report["causes"].items():
        kinds = report.get("kinds", {}).get(cause)
        causes.append(f"{cause} {count}" + (f" ({_kinds(kinds)})" if count and kinds else ""))
    lines = [f"{report['episodes']} episode(s): " + ", ".join(causes)]
    lines += [
        "",
        "| skill | starts | success | failed | preempted | rate | typical s | failure reasons |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, tally in sorted(report["skills"].items(), key=lambda item: -item[1].starts):
        typical = report["typical_s"].get(name.split("(", 1)[0])
        lines.append(
            f"| {name} | {tally.starts} | {tally.success} | {tally.failure} | {tally.preempted} | "
            f"{_rate(tally)} | {'-' if typical is None else f'{typical:.1f}'} | {_reasons(tally)} |"
        )
    lines += [
        "",
        "| transition | starts | success | failed | rate | failure reasons | failure kinds |",
        "|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(
        report["transitions"].items(),
        key=lambda item: (item[1].rate if item[1].rate is not None else 2, -item[1].starts),
    )
    for (before, after), tally in ordered:
        lines.append(
            f"| {before} -> {after} | {tally.starts} | {tally.success} | {tally.failure} | {_rate(tally)} | "
            f"{_reasons(tally)} | {_kinds(tally.kinds)} |"
        )
    if report["failed"]:
        lines += ["", "| failed episode | cause | kind | detail |", "|---|---|---|---|"]
        for episode in report["failed"]:
            lines.append(
                f"| {episode.directory} | {episode.cause} | {episode.kind or '-'} | {episode.detail} |"
            )
    for plan in report.get("plan") or ():
        lines += ["", *_format_plan(plan)]
    return "\n".join(lines)


def _format_plan(plan: dict[str, Any]) -> list[str]:
    n, count = plan["steps"], plan["episodes"]
    how = "exact positions" if plan["exact"] else "positions matched by call"
    lines = [
        f"plan: {n} steps, {count} episode(s), {plan['done']} done ({how}): "
        f"completed {plan['avg_completed']:.1f}/{n} on average, {plan['avg_first_failure']:.1f}/{n} before "
        f"the first failure (recovery +{plan['recovered']:.1f}); furthest {plan['avg_furthest']:.1f}; "
        f"rollbacks {plan['rollbacks']}, off-plan calls {plan['off_plan']}",
        "success@k: " + " ".join(f"{k}:{rate:.0%}" for k, rate in enumerate(plan["success_at"], 1)),
        "",
        "| step | call | reached | completed | first try | attempts | rollbacks from | stopped here "
        "| failure reasons |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in plan["per_step"]:
        reasons = Counter(row["failures"])
        text = ", ".join(f"{reason} x{number}" for reason, number in reasons.most_common(3)) or "-"
        lines.append(
            f"| {row['step']} | {row['call']} | {row['reached']} | {row['completed']} | "
            f"{row['first_try']}/{row['reached']} | {row['attempts']} | {row['rollbacks_from']} | "
            f"{row['stopped']} | {text} |"
        )
    return lines


def report_json(report: dict[str, Any], episodes: Sequence[EpisodeReport]) -> dict[str, Any]:
    def tally(value: Tally) -> dict[str, Any]:
        data = asdict(value)
        data["reasons"] = dict(value.reasons)
        data["kinds"] = dict(value.kinds)
        data["rate"] = value.rate
        return data

    def episode(value: EpisodeReport) -> dict[str, Any]:
        data = asdict(value)
        if data["progress"] is None:
            del data["progress"]  # runs without a plan keep today's JSON
        return data

    data = {
        "episodes": [episode(value) for value in episodes],
        "causes": report["causes"],
        "kinds": report["kinds"],
        "typical_s": report["typical_s"],
        "skills": {name: tally(value) for name, value in report["skills"].items()},
        "transitions": [
            {"previous": before, "next": after, **tally(value)}
            for (before, after), value in report["transitions"].items()
        ],
    }
    if report.get("plan"):
        data["plan"] = report["plan"]
    return data
