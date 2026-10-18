"""Score real RoboTwin traces using only recorded terminal simulator verdicts.

Unlike the general harness report, this evaluator has no fallback from a
controller's ``done`` status to success. Missing outcomes and episodes aborted
by the planner or harness count as failures in the attempted-episode denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from embodied_harness.runtime.trace.progress import read_snapshot


def planner_cameras(directory: Path) -> list[str] | None:
    """The cameras the episode's planner saw, from its config.snapshot.yaml; None when unknown."""
    planner = ((read_snapshot(directory) or {}).get("run") or {}).get("planner") or {}
    cameras = (planner.get("params") or {}).get("cameras")
    return list(cameras) if isinstance(cameras, list) else None


def score_run(run_dir: Path, *, expected_episodes: int | None = None) -> dict[str, Any]:
    """Return strict terminal success and failure details for one run directory."""
    directories = sorted(path for path in run_dir.glob("ep_*") if path.is_dir())
    if not directories:
        raise ValueError(f"No episode directories under {run_dir}")
    if expected_episodes is not None and expected_episodes < len(directories):
        raise ValueError("expected episode count is smaller than the recorded episode count")
    rows: list[dict[str, Any]] = []
    for directory in directories:
        outcome_path = directory / "outcome.json"
        outcome = json.loads(outcome_path.read_text()) if outcome_path.is_file() else {}
        events_path = directory / "events.jsonl"
        events = (
            [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
            if events_path.is_file()
            else []
        )
        start = next((event for event in events if event.get("type") == "episode_start"), {})
        terminal = next((event for event in reversed(events) if event.get("type") == "env_end"), {})
        success = (
            outcome.get("status") == "ended"
            and outcome.get("env_success") is True
            and terminal.get("success") is True
        )
        false_done = sum(
            event.get("type") == "check"
            and event.get("claimed_by_planner") is True
            and event.get("passed") is False
            for event in events
        )
        rows.append(
            {
                "episode": directory.name,
                "seed": start.get("seed"),
                "success": success,
                "status": outcome.get("status", "incomplete"),
                "env_success": outcome.get("env_success"),
                "error_code": outcome.get("error_code"),
                "message": outcome.get("message"),
                "skill_calls": outcome.get("skill_calls"),
                "planner_calls": outcome.get("planner_calls"),
                "false_done_claims": false_done,
                "duration_s": outcome.get("duration_s"),
                "planner_cameras": planner_cameras(directory),
            }
        )
    attempted = expected_episodes or len(rows)
    succeeded = sum(row["success"] for row in rows)
    return {
        "evaluation": "robotwin_closed_loop_terminal_verdict",
        "run_dir": str(run_dir.resolve()),
        "attempted_episodes": attempted,
        "recorded_episodes": len(rows),
        "unrecorded_episodes": attempted - len(rows),
        "successful_episodes": succeeded,
        "failed_episodes": attempted - succeeded,
        "success_rate": succeeded / attempted,
        "scoring_rule": "status=ended AND outcome.env_success=true AND env_end.success=true",
        "controller_inputs": "RGB camera images, 14D joint state, task text, and controller history",
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.expected_episodes is not None and args.expected_episodes < 1:
        parser.error("--expected-episodes must be positive")
    result = score_run(args.run_dir, expected_episodes=args.expected_episodes)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
