"""Command line entry point: `fmc3-harness validate`, `run`, `replay` and `data`."""

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
from .data.commands import register as register_data_commands
from .logs.setup import close_file_log, configure_logging, install_crash_handlers, open_file_log
from .runtime.commands import AutoCommands, CommandSource, TerminalCommands
from .runtime.factory import bind_planner, build_planner, check_bundle
from .runtime.runner import EpisodeRunner
from .trace.replay import replay, summarize
from .trace.writer import to_jsonable

logger = logging.getLogger("fmc3_harness.cli")


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
    return 0


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
            "fmc3-harness %s, python %s on %s; config %s",
            __version__,
            platform.python_version(),
            platform.platform(),
            bundle.path,
        )
        # Only the values that differ from the defaults: the quickest way to see what a run changed.
        non_default = bundle.run.model_dump(mode="json", exclude_defaults=True)
        logger.info("run config (non-default values): %s", json.dumps(non_default, ensure_ascii=False))

        failures = 0
        for index, episode_id in enumerate(episode_ids):
            seed = None if args.seed is None else args.seed + index
            runner = EpisodeRunner(
                bundle,
                run_dir=run_dir,
                episode_id=episode_id,
                seed=seed,
                commands=_commands(args.hold_policy),
                resume=args.resume is not None,
            )
            episode = runner.run()
            result = episode.result
            if result is None:
                failures += 1
                print(f"{episode_id}: stopped early  dir={episode.episode_dir}")
                continue
            if result.status != "done":
                failures += 1
            code = f" {result.error_code}" if result.error_code else ""
            ledger = json.dumps(result.ledger, ensure_ascii=False)
            print(
                f"{episode_id}: {result.status}{code}  ledger={ledger}  "
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fmc3-harness", description=__doc__)
    parser.add_argument("--version", action="version", version=f"fmc3-harness {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="load a run config and check that every piece fits")
    validate.add_argument("config", type=Path)
    validate.set_defaults(handler=_validate)

    run = commands.add_parser("run", help="run episodes and write their traces")
    run.add_argument("config", type=Path)
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument("--seed", type=int, default=None, help="seed of the first episode; later ones add 1")
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

    register_data_commands(commands, _load)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
