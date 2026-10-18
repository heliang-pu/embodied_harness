"""Command line entry point: `embodied-harness validate`, `run`, `replay`, `report`, `probe`, `lessons`,
`suite`, `data` and `policy`."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .config.errors import ConfigError
from .config.loader import Bundle, load_bundle
from .core.types import SkillCall
from .data.commands import register as register_data_commands
from .policy.integrations.commands import register as register_policy_commands
from .runtime.commands import AutoCommands, CommandSource, TerminalCommands
from .runtime.factory import bind_planner, build_planner, check_bundle, trained_settings_warnings
from .runtime.logs.setup import close_file_log, configure_logging, install_crash_handlers, open_file_log
from .runtime.runner import EpisodeRunner, ResumeError
from .runtime.trace.progress import episode_progress
from .runtime.trace.replay import replay, summarize
from .runtime.trace.report import episode_succeeded
from .runtime.trace.writer import to_jsonable

logger = logging.getLogger("embodied_harness.cli")


def _load(path: Path) -> Bundle | None:
    try:
        bundle = load_bundle(path)
        check_bundle(bundle)
    except ConfigError as exc:
        print(f"{exc.code.value}: {path} has {len(exc.problems)} problem(s):", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return None
    return bundle


def _validate(args: argparse.Namespace) -> int:
    bundle = _load(args.config)
    if bundle is None:
        return 2
    run = bundle.run
    print(
        f"OK: task {bundle.task.id!r}, {len(bundle.task.skills)} skills, "
        f"prompt library {bundle.library.library!r} ({len(bundle.library.prompts)} prompts), "
        f"env {run.env.type}, executor {run.executor.type}, planner {run.planner.type}"
    )
    if bundle.task.lessons or bundle.unreviewed_lessons:
        print(
            f"lessons: {len(bundle.task.lessons)} shown to the planner, "
            f"{bundle.unreviewed_lessons} not reviewed yet (left out)"
        )
    for line in _plan_lines(bundle):
        print(line)
    if run.planner.trained_manifest is not None:
        # The run-start check, ahead of time; building a planner opens no connection.
        try:
            planner = build_planner(bundle)
            bind_planner(planner, bundle)
        except (ConfigError, ValueError) as exc:  # what `run` would stop at
            print(f"E_CONFIG: {args.config}: the planner cannot be built:\n{exc}", file=sys.stderr)
            return 2
        warnings = trained_settings_warnings(planner, bundle)
        for warning in warnings:
            print(f"warning: {warning}")
        if not warnings:
            print(f"planner settings match planner.trained_manifest ({run.planner.trained_manifest})")
    return 0


def _plan_lines(bundle: Bundle) -> list[str]:
    """The fixed plan and where each rollback rule of its steps goes back to."""
    planner = bundle.run.planner
    plan, source = bundle.registry.plan, "task.plan"
    calls = planner.params.get("calls") if planner.type == "sequence" else None
    if plan is None and isinstance(calls, list):
        plan, source = (
            [SkillCall(c["skill"], dict(c.get("args") or {})) for c in calls],
            "planner.params.calls",
        )
    if plan is None:
        return []
    played = planner.type == "sequence"
    how = (
        "played by the sequence planner"
        if played
        else f"the reference for report; {planner.type} plans itself"
    )
    lines = [f"plan: {len(plan)} steps ({source}), {how}"]
    if not played:
        return lines
    table = bundle.registry.plan_rollbacks(plan)
    for index, call in enumerate(plan):
        for reason, action in bundle.skills.skills[call.skill].on_failure.items():
            if action.rollback is None:
                continue
            target = table[index].get(reason)
            where = (
                f"[{target}] {plan[target].key()}"
                if target is not None
                else f"nowhere (no earlier {action.rollback}), the step is repeated"
            )
            lines.append(f"  rollback: [{index}] {call.key()} on {reason} -> {where}")
    return lines


def _commands(policy: str | None) -> CommandSource:
    if policy is None:
        policy = "ask" if sys.stdin.isatty() else "abort"
    return TerminalCommands() if policy == "ask" else AutoCommands(policy)


def _run(args: argparse.Namespace) -> int:
    bundle = _load(args.config)
    if bundle is None:
        return 2
    configure_logging(args.log_level or bundle.run.logging.console_level)
    install_crash_handlers()

    if args.resume is not None:
        episode_dir = args.resume.resolve()
        run_dir, episode_ids = episode_dir.parent, [episode_dir.name]
    else:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(bundle.run.trace.root).expanduser().resolve() / run_id
        episode_ids = [f"ep_{index + 1:04d}" for index in range(args.episodes)]

    run_log = open_file_log(run_dir / "run.log", "INFO")
    try:
        logger.info(
            "embodied-harness %s, python %s on %s; config %s",
            __version__,
            platform.python_version(),
            platform.platform(),
            bundle.path,
        )
        # Only the values that differ from the defaults: the quickest way to see what a run changed.
        non_default = bundle.run.model_dump(mode="json", exclude_defaults=True)
        logger.info("run config (non-default values): %s", json.dumps(non_default, ensure_ascii=False))

        failures = 0
        # Without --seed a new run starts at the config's runtime.seed, so its episodes get distinct scenes
        # (scene-indexed simulators replay one scene for a None seed); a resume keeps the recorded seed.
        first_seed = args.seed
        if first_seed is None and args.resume is None:
            first_seed = bundle.run.runtime.seed
        for index, episode_id in enumerate(episode_ids):
            seed = None if first_seed is None else first_seed + index
            runner = EpisodeRunner(
                bundle,
                run_dir=run_dir,
                episode_id=episode_id,
                seed=seed,
                commands=_commands(args.hold_policy),
                resume=args.resume is not None,
            )
            try:
                episode = runner.run()
            except ResumeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            result = episode.result
            if result is None:
                failures += 1
                print(f"{episode_id}: stopped early  dir={episode.episode_dir}")
                continue
            # The env's own verdict when it gave one (a simulator), else finishing as done.
            if not episode_succeeded(result.status, result.env_success):
                failures += 1
            code = f" {result.error_code}" if result.error_code else ""
            if result.env_success is not None:
                code += f" env_success={result.env_success}"
            if result.env_score is not None:
                code += f" env_score={json.dumps(result.env_score, separators=(',', ':'))}"
            ledger = json.dumps(result.ledger, ensure_ascii=False)
            progress = episode_progress(episode.episode_dir)
            steps = "" if progress is None else f"  steps={progress.completed}/{progress.steps}"
            print(
                f"{episode_id}: {result.status}{code}  ledger={ledger}{steps}  "
                f"skill_calls={result.skill_calls}  planner_calls={result.planner_calls}  "
                f"sim_time={result.duration_s:.1f}s  dir={episode.episode_dir}"
            )
        return 1 if failures else 0
    finally:
        close_file_log(run_log)


def _replay(args: argparse.Namespace) -> int:
    bundle = _load(args.config)
    if bundle is None:
        return 2
    try:
        planner = build_planner(bundle)
    except ConfigError as exc:
        print(f"{exc.code.value}: {exc}", file=sys.stderr)
        return 2
    bind_planner(planner, bundle)
    rows = replay(planner, [path.resolve() for path in args.episodes], bundle.task)
    summary = summarize(rows)

    def rate(value: float | None) -> str:
        return "-" if value is None else f"{value:.1%}"

    compared = summary["compared"]
    print(
        f"replayed {summary['calls']} planner call(s): "
        f"same skill {summary['same_skill']}/{compared} ({rate(summary['same_skill_rate'])}), "
        f"same call {summary['same_call']}/{compared} ({rate(summary['same_call_rate'])}), "
        f"errors {summary['errors']}"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {"config": str(bundle.path), "summary": summary, "rows": to_jsonable(rows)}
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _report(args: argparse.Namespace) -> int:
    from .runtime.trace.report import build_report, episode_dirs, format_report, read_episode, report_json

    directories = episode_dirs(args.runs)
    if not directories:
        print("no episodes with events.jsonl under the given paths", file=sys.stderr)
        return 2
    episodes = [read_episode(directory) for directory in directories]
    report = build_report(episodes, by=args.by, window_s=args.window, early=args.early)
    print(format_report(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report_json(report, episodes), ensure_ascii=False, indent=2) + "\n"
        args.out.write_text(text, encoding="utf-8")
    return 0


def _suite(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from .runtime.suite import format_suite, load_suite, run_suite, suite_json, summarize_suite

    try:
        suite = load_suite(args.suite)
    except (OSError, ValidationError) as exc:
        print(f"error: {args.suite}: {exc}", file=sys.stderr)
        return 2
    if args.suite_command == "run":
        configure_logging(args.log_level or "WARNING")
        run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(suite.root).expanduser().resolve() / suite.suite / run_id
        try:
            run_suite(args.suite.resolve(), suite, run_dir, only=args.only)
        except ConfigError as exc:
            print(f"{exc.code.value}: {exc}", file=sys.stderr)
            return 2
        except (FileExistsError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        run_dir = args.run_dir.resolve()
    summaries = summarize_suite(suite, run_dir)
    print(format_suite(suite, summaries))
    (run_dir / "suite.json").write_text(
        json.dumps({"suite": suite.suite, "entries": suite_json(summaries)}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nresults in {run_dir}")
    return 0


def _lessons(args: argparse.Namespace) -> int:
    from .runtime.trace.lessons import draft_lessons, merge_lessons
    from .runtime.trace.report import episode_dirs, read_episode

    directories = episode_dirs(args.runs)
    if not directories:
        print("no episodes with events.jsonl under the given paths", file=sys.stderr)
        return 2
    drafts = draft_lessons([read_episode(directory) for directory in directories], min_count=args.min_count)
    for draft in drafts:
        print(f"- {draft['text']}\n  ({draft['source']})")
    if not drafts:
        print(f"no failure pattern came back {args.min_count} times or more in {len(directories)} episode(s)")
    added, known = merge_lessons(args.out, drafts)
    print(f"{args.out}: {added} draft(s) added, {known} already there; review them, then set reviewed: true")
    return 0


def _probe(args: argparse.Namespace) -> int:
    from .runtime.probe import format_probe, load_start_states, parse_calls, probe
    from .runtime.probe import summarize as summarize_probe

    bundle = _load(args.config)
    if bundle is None:
        return 2
    configure_logging(args.log_level or "WARNING")
    try:
        starts = [start for spec in args.at for start in load_start_states(spec, bundle)]
        calls = parse_calls(args.call)
        trials = probe(bundle, starts, calls, trials=args.trials, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ConfigError as exc:
        print(f"{exc.code.value}: {exc}", file=sys.stderr)
        return 2
    rows = summarize_probe(trials)
    print(format_probe(rows))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {"config": str(bundle.path), "rows": to_jsonable(rows), "trials": to_jsonable(trials)}
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embodied-harness", description=__doc__)
    parser.add_argument("--version", action="version", version=f"embodied-harness {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="load a run config and check that every piece fits")
    validate.add_argument("config", type=Path)
    validate.set_defaults(handler=_validate)

    run = commands.add_parser("run", help="run episodes and write their traces")
    run.add_argument("config", type=Path)
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the first episode (default: the config's runtime.seed, 0 unless set); later ones add 1 "
        "(with --resume the recorded seed is used)",
    )
    run.add_argument("--run-id", default=None, help="directory name under trace.root (default: a timestamp)")
    run.add_argument(
        "--hold-policy",
        choices=["ask", "abort", "resume"],
        default=None,
        help="what to do on HOLD (default: ask on a terminal, abort otherwise)",
    )
    run.add_argument(
        "--resume", type=Path, default=None, metavar="EPISODE_DIR", help="continue an interrupted episode"
    )
    run.add_argument("--log-level", default=None, help="console log level (default from the config)")
    run.set_defaults(handler=_run)

    replay_cmd = commands.add_parser(
        "replay", help="run the config's planner on recorded planner calls and compare its decisions"
    )
    replay_cmd.add_argument("config", type=Path, help="run config whose planner is replayed")
    replay_cmd.add_argument("episodes", type=Path, nargs="+", help="episode directories with recorded calls")
    replay_cmd.add_argument("--out", type=Path, default=None, help="write the per-call comparison as JSON")
    replay_cmd.set_defaults(handler=_replay)

    report = commands.add_parser(
        "report", help="where episodes fail: per skill, per transition, and one cause per failed episode"
    )
    report.add_argument("runs", type=Path, nargs="+", help="run directories or episode directories")
    report.add_argument(
        "--by",
        choices=["skill", "call"],
        default="skill",
        help="group by skill, or by call with its arguments",
    )
    report.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="minimum seconds after a switch within which a failure counts as a transition failure "
        "(the skill's typical successful duration when longer)",
    )
    report.add_argument(
        "--early",
        type=float,
        default=1 / 3,
        help="a failure within this share of the skill's typical duration is a readiness failure "
        "(it never got going); default 1/3",
    )
    report.add_argument("--out", type=Path, default=None, help="also write everything as JSON")
    report.set_defaults(handler=_report)

    suite = commands.add_parser(
        "suite", help="a composition benchmark: run a suite file, or summarize its runs"
    )
    suite_sub = suite.add_subparsers(dest="suite_command", required=True)
    suite_run = suite_sub.add_parser("run", help="run every entry of a suite file and summarize")
    suite_run.add_argument("suite", type=Path, help="suite file")
    suite_run.add_argument(
        "--run-id", default=None, help="directory name under <root>/<suite> (default: a timestamp)"
    )
    suite_run.add_argument("--only", nargs="+", default=[], metavar="ENTRY", help="run only these entries")
    suite_run.add_argument("--log-level", default=None, help="console log level (default WARNING)")
    suite_run.set_defaults(handler=_suite)
    suite_report = suite_sub.add_parser("report", help="summarize the runs of a suite")
    suite_report.add_argument("suite", type=Path, help="suite file")
    suite_report.add_argument("run_dir", type=Path, help="<root>/<suite>/<run id> written by `suite run`")
    suite_report.set_defaults(handler=_suite)

    lessons = commands.add_parser(
        "lessons", help="draft planner lessons from failure patterns that keep coming back in runs"
    )
    lessons.add_argument("runs", type=Path, nargs="+", help="run directories or episode directories")
    lessons.add_argument("--out", type=Path, required=True, help="lessons file to add the drafts to")
    lessons.add_argument(
        "--min-count", type=int, default=3, help="how often a pattern must fail to be drafted (default 3)"
    )
    lessons.set_defaults(handler=_lessons)

    probe = commands.add_parser(
        "probe", help="rerun skill calls from the start states a run recorded (trace.snapshots, sim only)"
    )
    probe.add_argument("config", type=Path, help="run config: env, executor and skills to probe with")
    probe.add_argument(
        "--at",
        action="append",
        required=True,
        metavar="EPISODE_DIR[:CALL_ID]",
        help="start where this call started (e.g. runs/x/ep_0001:sc_0003); a bare episode directory "
        "means every main call of it; repeatable",
    )
    probe.add_argument(
        "--call",
        action="append",
        default=[],
        metavar="CALL",
        help="run this call instead of the recorded one, e.g. pick or 'place(bin=A)'; repeatable",
    )
    probe.add_argument("--trials", type=int, default=5, help="runs per start state and call (default 5)")
    probe.add_argument(
        "--seed", type=int, default=0, help="env reset seed of the first trial; later ones add 1"
    )
    probe.add_argument("--out", type=Path, default=None, help="also write every trial as JSON")
    probe.add_argument("--log-level", default=None, help="console log level (default WARNING)")
    probe.set_defaults(handler=_probe)

    register_data_commands(commands, _load)
    register_policy_commands(commands, _load)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
