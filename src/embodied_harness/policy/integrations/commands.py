"""`embodied-harness policy ...`: preparing policy checkpoints for training with the harness's data, and
checking that a policy still follows its prompts."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .checkpoint import prepare_checkpoint


def _openpi_check(args: argparse.Namespace) -> int:
    from .openpi import load_openpi_spec, validate_lerobot_v3_dataset

    try:
        spec = load_openpi_spec(args.mapping)
        metadata = validate_lerobot_v3_dataset(args.dataset, spec)
        report = metadata.to_dict()
        if args.sample is not None:
            from ...data.openpi_dataset import OpenPILeRobotDataset

            dataset = OpenPILeRobotDataset(
                args.dataset,
                spec,
                action_horizon=args.action_horizon,
                repo_id=args.repo_id,
                video_backend=args.video_backend,
            )
            if not 0 <= args.sample < len(dataset):
                raise ValueError(f"--sample must be in [0, {len(dataset)})")
            item = dataset[args.sample]
            report["decoded_sample"] = args.sample
            report["action_horizon"] = args.action_horizon
            report["input_keys"] = sorted(item)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"OK: LeRobot v3.0 metadata, {metadata.fps:g} fps, state_dim={metadata.state_dim}, "
        f"action_dim={metadata.action_dim}, {len(metadata.image_features)} camera(s)"
    )
    if args.sample is None:
        print("metadata only; use --sample 0 to check decoding and action chunks in a LeRobot v3 environment")
    else:
        print(f"decoded sample {args.sample}, action_horizon={args.action_horizon}")
    return 0


def _subtask_checkpoint(args: argparse.Namespace) -> int:
    try:
        report = prepare_checkpoint(
            args.checkpoint, args.out, style=args.style, copy_weights=args.copy_weights
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"prepared {args.out}: {report['copied']} file(s) copied, {report['linked']} weight file(s) linked")
    print("preprocessor steps: " + " -> ".join(report["steps"]))
    print(
        f"fine-tune with: lerobot-train --policy.path={args.out} --dataset.root=<annotated dataset> ...\n"
        "(install embodied_harness where lerobot-train runs: LeRobot imports the step by its class path)"
    )
    return 0


def _steer_check(load: Callable[[Path], Any], args: argparse.Namespace) -> int:
    from ...data.labels import parse_call
    from ...runtime.factory import build_policy
    from ...runtime.steer import (
        analyze,
        format_steer,
        observations_from_env,
        observations_from_frames,
        policy_calls,
        query,
    )
    from ...runtime.trace.writer import to_jsonable

    bundle = load(args.config)
    if bundle is None:
        return 2
    try:
        calls = [parse_call(text) for text in args.call] or policy_calls(bundle)
        for call in calls:
            problems = bundle.registry.check_call(call)
            if not problems and bundle.registry.skill(call.skill).scripted:
                problems = [f"{call.key()} is a scripted primitive, not a policy prompt"]
            if problems:
                raise ValueError("; ".join(problems))
        if len(calls) < 2:
            raise ValueError("compare at least two calls (prompts)")
        if args.frames:
            observations = observations_from_frames(args.frames, args.samples, args.during)
        else:
            observations = observations_from_env(bundle, args.samples, args.seed)
        executor = build_policy(bundle)
        try:
            executor.reset()
            chunks = query(executor, bundle, observations, calls, args.repeats)
        finally:
            close = getattr(executor, "close", None)
            if callable(close):
                close()
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    keys = [call.key() for call in calls]
    report = analyze(chunks, keys, min_ratio=args.min_ratio, min_relative=args.min_relative)
    print(format_steer(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        data = {"config": str(bundle.path), **to_jsonable(report), "passed": report.passed}
        args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report.passed else 1


def register(commands: Any, load: Callable[[Path], Any] | None = None) -> None:
    policy = commands.add_parser(
        "policy", help="prepare policy checkpoints for training, and check a policy follows its prompts"
    )
    sub = policy.add_subparsers(dest="policy_command", required=True)
    check = sub.add_parser(
        "openpi-check", help="check a LeRobot v3.0 dataset against an OpenPI input mapping"
    )
    check.add_argument("dataset", type=Path, help="local LeRobot v3.0 dataset root")
    check.add_argument("--mapping", type=Path, required=True, help="OpenPI input mapping YAML")
    check.add_argument("--sample", type=int, default=None, help="also decode this sample (needs LeRobot v3)")
    check.add_argument("--action-horizon", type=int, default=50, help="chunk length for --sample")
    check.add_argument("--repo-id", default=None, help="dataset repository id (default local/<dir>)")
    check.add_argument("--video-backend", default=None, help="optional LeRobot decoder, e.g. pyav")
    check.add_argument("--out", type=Path, default=None, help="write validated metadata as JSON")
    check.set_defaults(handler=_openpi_check)
    if load is not None:
        steer = sub.add_parser(
            "steer-check",
            help="does the policy answer different prompts differently? (e.g. before and after fine-tuning)",
        )
        steer.add_argument(
            "config", type=Path, help="run config: the executor (policy server) and its prompts"
        )
        steer.add_argument(
            "--frames",
            type=Path,
            nargs="+",
            default=None,
            help="runs or episodes recorded with trace.frames to take observations from "
            "(default: env resets)",
        )
        steer.add_argument("--samples", type=int, default=8, help="observations to use (default 8)")
        steer.add_argument("--during", default=None, help="with --frames: only ticks while this skill ran")
        steer.add_argument(
            "--call",
            action="append",
            default=[],
            metavar="CALL",
            help="a call whose prompt to compare, e.g. 'pick(color=red)'; repeatable "
            "(default: every policy call)",
        )
        steer.add_argument(
            "--repeats", type=int, default=3, help="answers per prompt and observation (default 3)"
        )
        steer.add_argument("--seed", type=int, default=0, help="first env reset seed, without --frames")
        steer.add_argument(
            "--min-ratio", type=float, default=2.0, help="separation must exceed this many times the noise"
        )
        steer.add_argument(
            "--min-relative",
            type=float,
            default=0.05,
            help="for a deterministic policy: separation as a share of the action scale",
        )
        steer.add_argument("--out", type=Path, default=None, help="also write the result as JSON")
        steer.set_defaults(handler=lambda args: _steer_check(load, args))
    prepare = sub.add_parser(
        "subtask-checkpoint",
        help="copy a LeRobot checkpoint so fine-tuning prompts with each frame's subtask, not the task",
    )
    prepare.add_argument("checkpoint", type=Path, help="LeRobot policy checkpoint, e.g. pi05_base")
    prepare.add_argument("--out", type=Path, required=True, help="new directory for the prepared copy")
    prepare.add_argument("--style", default="subtask", help="language row style to read (default: subtask)")
    prepare.add_argument(
        "--copy-weights", action="store_true", help="copy weight files instead of linking them"
    )
    prepare.set_defaults(handler=_subtask_checkpoint)
