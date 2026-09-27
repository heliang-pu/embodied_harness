"""`fmc3-harness data ...`: training data and offline evaluation from lerobot-annotate datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from ..config.errors import ConfigError
from ..config.loader import Bundle
from ..runtime.factory import bind_planner, build_planner
from .annotations import AnnotatedEpisode, read_annotated_episodes
from .evaluate import evaluate_samples, summarize_eval
from .export import MANIFEST_FILE, export_planner_samples, library_with_stats
from .frames import FrameSource, LeRobotFrames
from .labels import LabelMap, candidate_calls, count_labels, load_label_map, write_label_file
from .mapping import load_data_mapping
from .relabel import relabel_dataset
from .steps import episode_steps

Loader = Callable[[Path], "Bundle | None"]


def frame_source(dataset: Path, repo_id: str | None) -> FrameSource:
    """Where `data planner` gets camera frames. Tests replace this with synthetic frames."""
    return LeRobotFrames(dataset, repo_id)


def _rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _labels_for(args: argparse.Namespace, bundle: Bundle, episodes: list[AnnotatedEpisode]) -> LabelMap:
    labels = load_label_map(args.labels, bundle.registry, bundle.task, allow_unreviewed=args.allow_unreviewed)
    missing = labels.missing(d.label for e in episodes for d in e.decisions if d.label is not None)
    if missing:
        raise ConfigError(
            [
                f"{len(missing)} subtask label(s) of the dataset are not in {args.labels}, "
                f"e.g. {missing[0]!r}; run `fmc3-harness data labels` again to add them"
            ]
        )
    return labels


def _labels(args: argparse.Namespace, bundle: Bundle) -> int:
    episodes = read_annotated_episodes(args.dataset)
    counts = count_labels(episodes)
    if not counts:
        print(f"no subtask rows in {args.dataset}; run lerobot-annotate first", file=sys.stderr)
        return 1
    previous = None
    if args.out.exists():
        previous = load_label_map(args.out, bundle.registry, bundle.task, allow_unreviewed=True)
    stats = write_label_file(
        args.out,
        counts,
        candidate_calls(bundle.registry, bundle.task.skills),
        dataset=args.dataset.resolve(),
        library=bundle.library.library,
        previous=previous,
    )
    unannotated = sum(not episode.decisions for episode in episodes)
    print(
        f"{stats['labels']} label(s) from {len(episodes)} episode(s) ({unannotated} without subtasks) "
        f"written to {args.out}: {stats['new']} new, {stats['kept']} kept from the existing file."
    )
    print("Check each `call`, then set `reviewed: true`.")
    return 0


def _library(args: argparse.Namespace, bundle: Bundle) -> int:
    episodes = read_annotated_episodes(args.dataset)
    labels = _labels_for(args, bundle, episodes)
    steps = [step for episode in episodes for step in episode_steps(episode, labels)]
    library, unused = library_with_stats(bundle.library, bundle.registry, steps)
    annotated = sum(bool(episode.decisions) for episode in episodes)
    header = (
        f"# Prompt library {library.library}. episodes and duration_s come from {annotated} annotated "
        f"episode(s) of {args.dataset.resolve()} (fmc3-harness data library).\n"
    )
    body = yaml.safe_dump(
        library.model_dump(mode="json", exclude_none=True), allow_unicode=True, sort_keys=False
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + body, encoding="utf-8")
    for entry in library.prompts:
        duration = entry.duration_s
        if entry.id in unused or duration is None:
            continue
        print(f"  {entry.id}: {entry.episodes} episode(s), p50 {duration.p50:.2f}s, p95 {duration.p95:.2f}s")
    if unused:
        print(f"no annotated spans for: {', '.join(unused)} (statistics left as they were)")
    print(f"wrote {args.out}")
    return 0


def _planner(args: argparse.Namespace, bundle: Bundle) -> int:
    planner = build_planner(bundle)
    if not hasattr(planner, "render_request"):
        raise ConfigError(
            [
                f"planner type {bundle.run.planner.type!r} renders no prompts; "
                "use the run config of the deployed VLM planner (openai_compat)"
            ]
        )
    bind_planner(planner, bundle)
    mapping = load_data_mapping(args.mapping)
    episodes = read_annotated_episodes(args.dataset, state_key=mapping.state_key)
    labels = _labels_for(args, bundle, episodes)
    manifest = export_planner_samples(
        bundle=bundle,
        planner=planner,
        dataset=args.dataset.resolve(),
        episodes=episodes,
        labels=labels,
        mapping=mapping,
        frames=frame_source(args.dataset, args.repo_id),
        out_dir=args.out,
        val_fraction=args.val_fraction,
        seed=args.seed,
        absolute_paths=args.absolute_paths,
    )
    splits, spans, memory = manifest["splits"], manifest["spans"], manifest["memory"]
    print(
        f"train: {splits['train']['samples']} sample(s) from {len(splits['train']['episodes'])} episode(s); "
        f"val: {splits['val']['samples']} sample(s) from {len(splits['val']['episodes'])} episode(s)"
    )
    print(
        f"spans: {spans['spans']}, merged into the previous step: {spans['merged']}, "
        f"left out (label mapped to null): {spans['skipped']}; "
        f"episodes without subtasks: {spans['unannotated_episodes']}"
    )
    print("calls: " + ", ".join(f"{call} {count}" for call, count in manifest["calls"].items()))
    if memory["over_limit"]:
        print(
            f"warning: {memory['over_limit']} memory text(s) exceed {memory['max_chars']} characters "
            f"(longest {memory['longest']}) and were truncated; consider raising planner.memory_max_chars"
        )
    print(f"wrote {args.out}")
    return 0


def _relabel(args: argparse.Namespace, bundle: Bundle) -> int:
    episodes = read_annotated_episodes(args.dataset)
    labels = _labels_for(args, bundle, episodes)
    report = relabel_dataset(args.dataset, args.out, labels, bundle.registry, copy_videos=args.copy_videos)
    print(
        f"rewrote {report['files']} data file(s): "
        f"{report['labels_replaced']} label(s) now use library texts; videos {report['videos']}"
    )
    if report["labels_kept"]:
        kept = len(report["labels_kept"])
        print(f"kept as annotated (mapped to null or to a scripted skill): {kept} label(s)")
    print(f"wrote {args.out}")
    return 0


def _eval(args: argparse.Namespace, bundle: Bundle) -> int:
    manifest = json.loads((args.export / MANIFEST_FILE).read_text(encoding="utf-8"))
    planner = build_planner(bundle)
    bind_planner(planner, bundle)
    settings = getattr(planner, "settings", None)
    if callable(settings):
        current, exported = settings(), manifest.get("planner", {})
        changed = sorted(key for key in set(current) | set(exported) if current.get(key) != exported.get(key))
        if changed:
            print(
                f"warning: planner settings differ from the export ({', '.join(changed)}); "
                "the model is not seeing what it was trained on",
                file=sys.stderr,
            )
    rows = evaluate_samples(planner, bundle, args.export, split=args.split, limit=args.limit)
    summary = summarize_eval(rows)
    count = summary["samples"]
    print(
        f"{count} {args.split} sample(s): same call {summary['same_call']}/{count} "
        f"({_rate(summary['same_call_rate'])}), same skill {summary['same_skill']}/{count} "
        f"({_rate(summary['same_skill_rate'])}), done recall {_rate(summary['done_recall'])}, "
        f"premature done {summary['premature_done']}, errors {summary['errors']}"
    )
    if summary["memory_similarity"] is not None:
        print(f"memory similarity to the annotation: {summary['memory_similarity']:.2f} (1.0 = identical)")
    if summary["latency_p50_s"] is not None:
        print(f"latency p50 {summary['latency_p50_s']:.2f}s, p95 {summary['latency_p95_s']:.2f}s")
    for mistake, times in summary["mistakes"].items():
        print(f"  {times} x {mistake}")
    if args.out is not None:
        from ..trace.writer import to_jsonable

        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": str(bundle.path),
            "export": str(args.export),
            "summary": summary,
            "rows": to_jsonable(rows),
        }
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _run(handler: Callable[[argparse.Namespace, Bundle], int], load: Loader, args: argparse.Namespace) -> int:
    bundle = load(args.config)
    if bundle is None:
        return 2
    try:
        return handler(args, bundle)
    except ConfigError as exc:
        print(f"{exc.code.value}: {len(exc.problems)} problem(s):", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 2


def register(commands: Any, load: Loader) -> None:
    data = commands.add_parser(
        "data", help="training data and offline planner evaluation from lerobot-annotate datasets"
    )
    sub = data.add_subparsers(dest="data_command", required=True)

    def command(name: str, handler: Callable[[argparse.Namespace, Bundle], int], help_text: str) -> Any:
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument("config", type=Path, help="run config: task, skills, prompt library and planner")
        parser.set_defaults(handler=partial(_run, handler, load))
        return parser

    def labelled(parser: Any) -> None:
        parser.add_argument("dataset", type=Path, help="LeRobot dataset annotated by lerobot-annotate")
        parser.add_argument("--labels", type=Path, required=True, help="reviewed label map (data labels)")
        parser.add_argument(
            "--allow-unreviewed", action="store_true", help="accept label entries not marked reviewed"
        )

    labels = command("labels", _labels, "collect subtask labels into a label map to review")
    labels.add_argument("dataset", type=Path, help="LeRobot dataset annotated by lerobot-annotate")
    labels.add_argument(
        "--out", type=Path, required=True, help="label map YAML; entries already in it keep their review"
    )

    library = command("library", _library, "fill prompt-library statistics from the annotated spans")
    labelled(library)
    library.add_argument("--out", type=Path, required=True, help="prompt library YAML to write")

    planner = command("planner", _planner, "export planner fine-tuning samples (messages + images JSONL)")
    labelled(planner)
    planner.add_argument("--mapping", type=Path, required=True, help="dataset cameras and proprio mapping")
    planner.add_argument("--out", type=Path, required=True, help="new directory for the export")
    planner.add_argument("--val-fraction", type=float, default=0.1, help="episodes held out for data eval")
    planner.add_argument("--seed", type=int, default=0, help="seed of the train/val episode split")
    planner.add_argument(
        "--repo-id", default=None, help="repo id passed to LeRobotDataset (default local/<dir>)"
    )
    planner.add_argument("--absolute-paths", action="store_true", help="write absolute image paths")

    relabel = command(
        "relabel", _relabel, "copy the dataset with library texts as subtasks, for the executor"
    )
    labelled(relabel)
    relabel.add_argument("--out", type=Path, required=True, help="new directory for the relabeled copy")
    relabel.add_argument("--copy-videos", action="store_true", help="copy videos instead of linking them")

    evaluate = command("eval", _eval, "run the config's planner on exported samples and score it")
    evaluate.add_argument("export", type=Path, help="directory written by `data planner`")
    evaluate.add_argument("--split", choices=["val", "train", "all"], default="val")
    evaluate.add_argument("--limit", type=int, default=None, help="evaluate at most this many samples")
    evaluate.add_argument("--out", type=Path, default=None, help="write per-sample results as JSON")
