"""`embodied-harness data ...`: training data and offline evaluation from lerobot-annotate datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from ..config.errors import ConfigError
from ..config.loader import Bundle
from ..planner.base import PER_SAMPLE, settings_differences, unrecorded_warning
from ..runtime.factory import bind_planner, build_planner
from .annotate import WORK_DIR, AnnotateOptions, annotate_dataset
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
                f"e.g. {missing[0]!r}; run `embodied-harness data labels` again to add them"
            ]
        )
    return labels


def _write_labels(dataset: Path, out: Path, bundle: Bundle, *, accept_exact: bool) -> int:
    episodes = read_annotated_episodes(dataset)
    counts = count_labels(episodes)
    if (dataset / ".embodied_harness" / "from_runs.json").is_file():
        # A periodic response can be accepted without sending an action before the
        # environment ends. Such a planner label still needs a review entry.
        planner_counts = count_labels(read_annotated_episodes(dataset, planner=True))
        counts.update({label: count for label, count in planner_counts.items() if label not in counts})
    if not counts:
        print(f"no subtask rows in {dataset}; run `embodied-harness data annotate` first", file=sys.stderr)
        return 1
    previous = None
    if out.exists():
        previous = load_label_map(out, bundle.registry, bundle.task, allow_unreviewed=True)
    stats = write_label_file(
        out,
        counts,
        candidate_calls(bundle.registry, bundle.task.skills),
        dataset=dataset.resolve(),
        library=bundle.library.library,
        previous=previous,
        accept_exact=accept_exact,
    )
    unannotated = sum(not episode.decisions for episode in episodes)
    accepted = f", {stats['accepted']} accepted as exact library texts" if accept_exact else ""
    print(
        f"{stats['labels']} label(s) from {len(episodes)} episode(s) ({unannotated} without subtasks) "
        f"written to {out}: {stats['new']} new{accepted}, {stats['kept']} kept from the existing file."
    )
    print("Check each `call` that is not reviewed yet, then set `reviewed: true`.")
    return 0


def _labels(args: argparse.Namespace, bundle: Bundle) -> int:
    return _write_labels(args.dataset, args.out, bundle, accept_exact=args.accept_exact)


def _episode_list(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    try:
        return tuple(int(value) for value in text.split(",") if value.strip())
    except ValueError:
        raise ValueError(f"--episodes takes episode indices like 0,1,2, not {text!r}") from None


def _annotate(args: argparse.Namespace, bundle: Bundle) -> int:
    from .vlm_proxy import OPENAI_API, OpenAIProxy, codex_provider

    options = AnnotateOptions(
        vlm_model=args.vlm_model,
        vlm_url=args.vlm_url,
        camera=args.camera,
        episodes=_episode_list(args.episodes),
        concurrency=args.concurrency,
        extra=tuple(args.lerobot_arg),
    )
    proxy = None
    if args.vlm_api is not None:
        # openai: OpenAI's API (or --vlm-upstream) with the key from an environment variable;
        # codex: `codex exec`, with whatever provider and login the Codex CLI is set up with.
        if args.vlm_api == "codex":
            model = args.vlm_model or codex_provider().get("model")
            upstream, endpoint, key_env = OPENAI_API, "codex", "OPENAI_API_KEY"
            where = "codex exec (the Codex CLI's own provider and login)"
        else:
            model = args.vlm_model
            upstream = args.vlm_upstream or OPENAI_API
            endpoint = args.vlm_endpoint or "responses"
            key_env = args.api_key_env or "OPENAI_API_KEY"
            where = f"{upstream} ({endpoint} endpoint, key from ${key_env})"
        if not model:
            raise ValueError(f"--vlm-api {args.vlm_api} needs --vlm-model, e.g. gpt-6-sol")
        if args.vlm_url:
            raise ValueError(f"--vlm-api {args.vlm_api} starts its own local endpoint; drop --vlm-url")
        options = replace(options, vlm_model=model)
        print(f"VLM: {model} via {where}")
        if args.dry_run:  # nothing is sent, so no key is needed
            options = replace(options, vlm_url="http://127.0.0.1:<port>/v1")
        else:
            proxy = OpenAIProxy(
                model=model,
                upstream=upstream,
                endpoint=endpoint,
                api_key_env=key_env,
                reasoning_effort=args.reasoning_effort,
            )
            options = replace(options, vlm_url=proxy.base_url)
    try:
        result = annotate_dataset(
            bundle=bundle,
            dataset=args.dataset,
            out=args.out,
            options=options,
            steer=not args.free_labels,
            copy_videos=args.copy_videos,
            dry_run=args.dry_run,
        )
    finally:
        if proxy is not None:
            proxy.close()
            print(proxy.usage.summary(proxy.model))
            usage = {**proxy.usage.to_dict(), "model": proxy.model, "cost_usd": proxy.usage.cost(proxy.model)}
            work = args.out / WORK_DIR
            if work.is_dir():
                (work / "usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    vocabulary = result["vocabulary"]
    print(f"label vocabulary: {len(vocabulary)} library text(s)" if vocabulary else "label vocabulary: none")
    print(f"prompts: {result['prompts']}")
    if args.dry_run:
        print("command (run with the prompts as LEROBOT_PROMPT_OVERRIDE_<name>):")
        print("  " + " ".join(result["command"]))
        return 0
    report = result["report"]
    in_vocabulary = f"{report['in_vocabulary']}/{report['spans']}" if vocabulary else "-"
    print(
        f"annotated {report['annotated']}/{report['episodes']} episode(s): {report['spans']} span(s), "
        f"{report['distinct_labels']} distinct label(s), library texts verbatim: {in_vocabulary}"
    )
    print(f"wrote {args.out}")
    if args.labels is not None:
        return _write_labels(args.out, args.labels, bundle, accept_exact=True)
    print(f"next: embodied-harness data labels {args.config} {args.out} --accept-exact --out <labels.yaml>")
    return 0


def _library(args: argparse.Namespace, bundle: Bundle) -> int:
    episodes = read_annotated_episodes(args.dataset)
    labels = _labels_for(args, bundle, episodes)
    if (args.dataset / ".embodied_harness" / "from_runs.json").is_file():
        from .from_runs import successful_call_steps

        steps = successful_call_steps(args.dataset)
    else:
        steps = [step for episode in episodes for step in episode_steps(episode, labels)]
    library, unused = library_with_stats(bundle.library, bundle.registry, steps)
    annotated = sum(bool(episode.decisions) for episode in episodes)
    header = (
        f"# Prompt library {library.library}. episodes and duration_s come from {annotated} annotated "
        f"episode(s) of {args.dataset.resolve()} (embodied-harness data library).\n"
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
    offsets = [int(value) for value in args.frame_offsets.split(",") if value.strip()]
    episodes = read_annotated_episodes(
        args.dataset,
        state_key=mapping.state_key,
        instruction_key=mapping.instruction_key,
        frame_offsets=offsets,
        planner=True,
    )
    excluded = set(_episode_list(args.exclude_episodes) or ())
    if excluded:
        unknown = sorted(excluded - {episode.episode_index for episode in episodes})
        if unknown:
            raise ValueError(f"--exclude-episodes: the dataset has no episode(s) {unknown}")
        episodes = [episode for episode in episodes if episode.episode_index not in excluded]
        print(f"left out {len(excluded)} episode(s): {', '.join(str(index) for index in sorted(excluded))}")
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
        frame_offsets=offsets,
        periodic_limit=args.periodic_limit,
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
    periodic = manifest.get("periodic")
    if periodic is not None:
        cap = f", at most {periodic['limit']} per step" if periodic["limit"] is not None else ""
        print(
            f"periodic checks: {periodic['samples']} sample(s) inside the steps, one every "
            f"{periodic['frames_per_check']} frame(s){cap}; {periodic['switched_steps']} step(s) end "
            "only by a planner switch (no success condition)"
        )
    if manifest["done_dropped"]:
        print(f"left out {manifest['done_dropped']} done sample(s): planner.allow_done is false")
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
    target = args.dataset if args.in_place else args.out
    report = relabel_dataset(args.dataset, target, labels, bundle.registry, copy_videos=args.copy_videos)
    print(
        f"rewrote {report['files']} data file(s): "
        f"{report['labels_replaced']} label(s) now use library texts; videos {report['videos']}"
    )
    if report["labels_kept"]:
        kept = len(report["labels_kept"])
        print(f"kept as annotated (mapped to null or to a scripted skill): {kept} label(s)")
    print(f"rewrote {args.dataset} in place" if args.in_place else f"wrote {args.out}")
    return 0


def _eval(args: argparse.Namespace, bundle: Bundle) -> int:
    manifest = json.loads((args.export / MANIFEST_FILE).read_text(encoding="utf-8"))
    planner = build_planner(bundle)
    bind_planner(planner, bundle)
    settings = getattr(planner, "settings", None)
    if callable(settings):
        # Each sample is rendered with the goal it recorded, so the configured goal does not matter here.
        current, exported = settings(), manifest.get("planner", {})
        changed = settings_differences(current, exported, skip=PER_SAMPLE)
        if changed:
            print(
                f"warning: planner settings differ from the export ({', '.join(changed)}); "
                "the model is not seeing what it was trained on",
                file=sys.stderr,
            )
        unrecorded = unrecorded_warning(current, exported, skip=PER_SAMPLE)
        if unrecorded is not None:
            print(f"warning: {unrecorded}", file=sys.stderr)
    rows = evaluate_samples(planner, bundle, args.export, split=args.split, limit=args.limit)
    summary = summarize_eval(rows)
    count = summary["samples"]
    print(
        f"{count} {args.split} sample(s): same call {summary['same_call']}/{count} "
        f"({_rate(summary['same_call_rate'])}), same skill {summary['same_skill']}/{count} "
        f"({_rate(summary['same_skill_rate'])}), done recall {_rate(summary['done_recall'])}, "
        f"premature done {summary['premature_done']}, errors {summary['errors']}"
    )
    if len(summary["by_kind"]) > 1:
        kinds = [f"{kind} {row['same_call']}/{row['samples']}" for kind, row in summary["by_kind"].items()]
        print("same call by kind: " + ", ".join(kinds))
    if summary["memory_similarity"] is not None:
        print(f"memory similarity to the annotation: {summary['memory_similarity']:.2f} (1.0 = identical)")
    if summary["latency_p50_s"] is not None:
        print(f"latency p50 {summary['latency_p50_s']:.2f}s, p95 {summary['latency_p95_s']:.2f}s")
    for mistake, times in summary["mistakes"].items():
        print(f"  {times} x {mistake}")
    if args.out is not None:
        from ..runtime.trace.writer import to_jsonable

        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "config": str(bundle.path),
            "export": str(args.export),
            "summary": summary,
            "rows": to_jsonable(rows),
        }
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _coverage(args: argparse.Namespace, bundle: Bundle) -> int:
    from .coverage import format_coverage, run_transitions, training_transitions

    episodes = read_annotated_episodes(args.dataset)
    labels = _labels_for(args, bundle, episodes)
    training = training_transitions(episodes, labels)
    runs = run_transitions(args.runs) if args.runs else None
    annotated = sum(bool(episode.decisions) for episode in episodes)
    print(f"training data: {annotated} annotated episode(s), {len(training)} distinct transition(s)")
    print(format_coverage(training, runs))
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


def _from_runs(args: argparse.Namespace) -> int:
    from .from_runs import export_runs

    try:
        report = export_runs(
            args.runs,
            args.out,
            repo_id=args.repo_id,
            state_key=args.state_key,
            success_only=args.success_only,
        )
    except ImportError as exc:
        print(f"error: {exc}; `data from-runs` needs lerobot (and PyAV) in this environment", file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.out}: {report['episodes']} episode(s), {report['frames']} frame(s), "
        f"cameras {', '.join(report['cameras'])}; {report['subtask_rows']} subtask and "
        f"{report['memory_rows']} memory row(s)"
    )
    if report["skipped"]:
        print(f"left out {report['skipped']} unsuccessful episode(s)")
    return 0


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

    annotate = command(
        "annotate", _annotate, "run lerobot-annotate on a copy of a dataset, asking for the library's texts"
    )
    annotate.add_argument("dataset", type=Path, help="LeRobot v3 dataset to annotate; it is not modified")
    annotate.add_argument("--out", type=Path, required=True, help="new directory for the annotated copy")
    annotate.add_argument(
        "--labels", type=Path, default=None, help="then write this label map, exact library texts accepted"
    )
    annotate.add_argument(
        "--vlm-model", default=None, help="VLM that annotates (default: lerobot-annotate's)"
    )
    annotate.add_argument(
        "--vlm-url", default=None, help="OpenAI-compatible server to use instead of starting one"
    )
    annotate.add_argument("--camera", default=None, help="dataset image key the VLM watches (default: first)")
    annotate.add_argument(
        "--free-labels", action="store_true", help="let the VLM word labels freely, without the library"
    )
    annotate.add_argument("--copy-videos", action="store_true", help="copy videos instead of linking them")
    annotate.add_argument("--dry-run", action="store_true", help="write the prompts, print the command, stop")
    annotate.add_argument(
        "--episodes",
        default=None,
        metavar="I,J,...",
        help="annotate only these episodes, e.g. 0,1 for a trial",
    )
    annotate.add_argument(
        "--concurrency", type=int, default=None, help="VLM requests in flight at once (lerobot default 16)"
    )
    annotate.add_argument(
        "--vlm-api",
        choices=["openai", "codex"],
        default=None,
        help="annotate with a hosted model through a local endpoint: openai (OpenAI's API or --vlm-upstream, "
        "key from an environment variable) or codex (`codex exec` with the Codex CLI's own login)",
    )
    annotate.add_argument(
        "--vlm-endpoint",
        choices=["responses", "chat"],
        default=None,
        help="with --vlm-api: the Responses API (current OpenAI models) or Chat Completions",
    )
    annotate.add_argument("--vlm-upstream", default=None, help="with --vlm-api: the API base URL")
    annotate.add_argument(
        "--api-key-env", default=None, help="with --vlm-api: environment variable with the key"
    )
    annotate.add_argument(
        "--reasoning-effort", default="low", help="with --vlm-api: the model's reasoning effort (default low)"
    )
    annotate.add_argument(
        "--lerobot-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="one more lerobot-annotate argument, repeatable: --lerobot-arg=--plan.min_subtask_seconds=2",
    )

    from_runs = sub.add_parser(
        "from-runs", help="turn runs recorded with trace.frames into a LeRobot dataset (needs lerobot)"
    )
    from_runs.add_argument("runs", type=Path, nargs="+", help="run directories or episode directories")
    from_runs.add_argument("--out", type=Path, required=True, help="new directory for the dataset")
    from_runs.add_argument("--repo-id", default=None, help="dataset repo id (default local/<out name>)")
    from_runs.add_argument(
        "--state-key", default="state", help="proprio key holding the state vector (default: state)"
    )
    from_runs.add_argument(
        "--success-only",
        action="store_true",
        help="only successful episodes, by the rule run and report score with: the env's own verdict when "
        "it gave one, else finishing as done",
    )
    from_runs.set_defaults(handler=_from_runs)

    labels = command("labels", _labels, "collect subtask labels into a label map to review")
    labels.add_argument("dataset", type=Path, help="LeRobot dataset annotated by lerobot-annotate")
    labels.add_argument(
        "--out", type=Path, required=True, help="label map YAML; entries already in it keep their review"
    )
    labels.add_argument(
        "--accept-exact",
        action="store_true",
        help="mark labels that are word for word the text of one call as reviewed",
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
    planner.add_argument(
        "--frame-offsets",
        default="0",
        help="comma-separated frame offsets around each decision, e.g. 0,2,4; each gives one sample",
    )
    planner.add_argument(
        "--exclude-episodes",
        default=None,
        metavar="I,J,...",
        help="leave these episodes out, e.g. ones whose labels failed a check",
    )
    planner.add_argument(
        "--periodic-limit",
        type=int,
        default=None,
        metavar="N",
        help="planner.call periodic: at most N samples inside each step, spread over its checks "
        "(default: one per check)",
    )

    relabel = command("relabel", _relabel, "replace subtask labels by the library texts, for the executor")
    labelled(relabel)
    target = relabel.add_mutually_exclusive_group(required=True)
    target.add_argument("--out", type=Path, help="new directory for a relabeled copy")
    target.add_argument(
        "--in-place", action="store_true", help="rewrite the dataset itself (the copy `data annotate` made)"
    )
    relabel.add_argument("--copy-videos", action="store_true", help="copy videos instead of linking them")

    coverage = command(
        "coverage", _coverage, "which step orders (previous call -> call) the training data covers"
    )
    labelled(coverage)
    coverage.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=None,
        help="runs whose transitions to look up in the training data",
    )

    evaluate = command("eval", _eval, "run the config's planner on exported samples and score it")
    evaluate.add_argument("export", type=Path, help="directory written by `data planner`")
    evaluate.add_argument("--split", choices=["val", "train", "all"], default="val")
    evaluate.add_argument("--limit", type=int, default=None, help="evaluate at most this many samples")
    evaluate.add_argument("--out", type=Path, default=None, help="write per-sample results as JSON")
