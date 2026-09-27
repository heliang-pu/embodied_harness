"""Data tools for lerobot-annotate datasets: labels, library statistics, planner samples, relabel, eval."""

import base64
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from conftest import REPLACE, make_bundle, small_task, write_configs
from fake_openai import chat_reply
from fmc3_harness.cli import main
from fmc3_harness.config.errors import ConfigError
from fmc3_harness.core.types import DecisionReason, SkillCall
from fmc3_harness.data import commands
from fmc3_harness.data.annotations import read_annotated_episodes
from fmc3_harness.data.evaluate import evaluate_samples, summarize_eval
from fmc3_harness.data.export import export_planner_samples, library_with_stats, percentile, split_episodes
from fmc3_harness.data.labels import (
    candidate_calls,
    count_labels,
    load_label_map,
    normalize_label,
    parse_call,
    rank_candidates,
    write_label_file,
)
from fmc3_harness.data.mapping import DataMapping, ProprioMapper
from fmc3_harness.data.relabel import relabel_dataset
from fmc3_harness.data.steps import StepStats, episode_steps
from fmc3_harness.planner.base import PlannerContext
from fmc3_harness.planner.decision import parse_decision
from fmc3_harness.planner.render import RENDER_VERSION
from fmc3_harness.runtime.factory import bind_planner, build_planner
from lerobot_fixture import MAPPING, FakeFrames, persistent_rows, state, write_dataset

CAMERAS = ["head", "wrist"]
REVIEWED = {
    "pick up the package": "pick",
    "flip the package": "flip",
    "put it in bin a": "place(bin=A)",
    "place the package into bin a": "place(bin=A)",
    "grab the parcel": "pick",
    "wave at the camera": None,
    "put it in bin c": "place(bin=C)",
}
DATA_FILE = "data/chunk-000/file-000.parquet"


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "dataset")


def vlm_run(base_url="http://127.0.0.1:9/v1", **planner_fields):
    params = {
        "base_url": base_url,
        "model": "fake-planner",
        "cameras": CAMERAS,
        "image_size": [32, 24],
        "proprio_keys": ["gripper_width", "gripper_closed"],
    }
    return {
        "env": {"params": {"cameras": CAMERAS, "image_size": [32, 24]}},
        "planner": {REPLACE: True, "type": "openai_compat", "params": params, **planner_fields},
    }


def vlm_bundle(tmp_path, base_url="http://127.0.0.1:9/v1", **planner_fields):
    return make_bundle(tmp_path, task=small_task(3), run=vlm_run(base_url, **planner_fields))


def review(path, calls=REVIEWED):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in data["labels"]:
        entry["call"] = calls[entry["label"]]
        entry["reviewed"] = True
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def reviewed_labels(tmp_path, bundle, dataset):
    path = tmp_path / "labels.yaml"
    episodes = read_annotated_episodes(dataset)
    candidates = candidate_calls(bundle.registry, bundle.task.skills)
    write_label_file(
        path, count_labels(episodes), candidates, dataset=dataset, library=bundle.library.library
    )
    review(path)
    return load_label_map(path, bundle.registry, bundle.task)


def bound_planner(bundle):
    planner = build_planner(bundle)
    bind_planner(planner, bundle)
    return planner


def export(tmp_path, bundle, dataset, **kwargs):
    kwargs.setdefault("val_fraction", 0.0)
    return export_planner_samples(
        bundle=bundle,
        planner=bound_planner(bundle),
        dataset=dataset,
        episodes=read_annotated_episodes(dataset),
        labels=reviewed_labels(tmp_path, bundle, dataset),
        mapping=DataMapping.model_validate(MAPPING),
        frames=kwargs.pop("frames", FakeFrames()),
        out_dir=tmp_path / "export",
        **kwargs,
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def placeholder_text(body):
    """The user message of a live request, with each image replaced by the training placeholder."""
    content = body["messages"][1]["content"]
    return "".join(part["text"] if part["type"] == "text" else "<image>" for part in content)


def request_images(body):
    content = body["messages"][1]["content"]
    urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    return [base64.b64decode(url.split(",", 1)[1]) for url in urls]


# --- reading the annotations ---------------------------------------------------------------------


def test_read_annotated_episodes(dataset):
    episodes = read_annotated_episodes(dataset)
    assert [episode.episode_index for episode in episodes] == [0, 1, 2]
    first = episodes[0]
    assert first.num_frames == 40
    assert [(d.frame, d.label) for d in first.decisions] == [
        (0, "Pick up the package."),
        (10, "flip the package"),
        (20, "put it in bin a"),
        (25, "place the package into bin A"),
        (39, None),
    ]
    # The memory written at a decision is the row stamped at its start; there is none at the first.
    assert [d.memory for d in first.decisions] == [
        "",
        "picked package 1",
        "package 1 flipped",
        "package 1 above bin A",
        "package 1 above bin A",
    ]
    assert first.decisions[0].duration_s == pytest.approx(1.0)
    assert first.decisions[3].duration_s == pytest.approx(1.4, abs=1e-5)
    assert first.states[20] == pytest.approx(state(20))
    assert set(first.states) == {0, 10, 20, 25, 39}
    assert episodes[2].decisions == ()
    assert [e.episode_index for e in read_annotated_episodes(dataset, episodes=[1])] == [1]


def test_read_rejects_a_dataset_without_annotations(tmp_path):
    path = tmp_path / "plain" / DATA_FILE
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": [0], "frame_index": [0], "timestamp": [0.0]}), path)
    with pytest.raises(ValueError, match="run lerobot-annotate first"):
        read_annotated_episodes(tmp_path / "plain")
    with pytest.raises(FileNotFoundError):
        read_annotated_episodes(tmp_path / "missing")


# --- label map -----------------------------------------------------------------------------------


def test_labels_are_suggested_and_must_be_reviewed(tmp_path, bundle, dataset):
    counts = count_labels(read_annotated_episodes(dataset))
    assert counts == {label: 1 for label in REVIEWED}
    candidates = candidate_calls(bundle.registry, bundle.task.skills)
    assert [c.call.key() for c in candidates] == [
        "pick",
        "flip",
        "place(bin=A)",
        "place(bin=B)",
        "place(bin=C)",
        "retract",
    ]
    assert rank_candidates("put it in bin c", candidates)[0][1].call == SkillCall("place", {"bin": "C"})

    path = tmp_path / "labels.yaml"
    stats = write_label_file(path, counts, candidates, dataset=dataset, library=bundle.library.library)
    assert stats == {"labels": 7, "new": 7, "kept": 0, "absent": 0}
    text = path.read_text(encoding="utf-8")
    assert "closest: pick 0.80" in text
    entries = {entry["label"]: entry for entry in yaml.safe_load(text)["labels"]}
    assert entries["pick up the package"]["call"] == "pick"
    assert entries["flip the package"]["call"] == "flip"
    assert entries["put it in bin a"]["call"] == "place(bin=A)"
    assert entries["place the package into bin a"]["call"] == "place(bin=A)"
    assert not any(entry["reviewed"] for entry in entries.values())

    with pytest.raises(ConfigError, match="not reviewed"):
        load_label_map(path, bundle.registry, bundle.task)
    labels = load_label_map(path, bundle.registry, bundle.task, allow_unreviewed=True)
    assert labels.lookup("  Pick up the   PACKAGE. ").call == SkillCall("pick")
    with pytest.raises(KeyError, match="not in the label map"):
        labels.lookup("juggle")


def test_label_map_rejects_bad_entries(tmp_path, bundle):
    path = tmp_path / "labels.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "labels": [
                    {"label": "a", "call": "place(bin=D)", "reviewed": True},
                    {"label": "b", "call": "dance", "reviewed": True},
                    {"label": "c", "call": "place(bin A)", "reviewed": True},
                    {"label": "A.", "call": "pick", "reviewed": True},
                    {"label": "e", "call": None, "reviewed": "yes"},
                    {"label": "f", "call": "pick", "reviewed": True, "note": "?"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as info:
        load_label_map(path, bundle.registry, bundle.task)
    problems = "\n".join(info.value.problems)
    assert len(info.value.problems) == 6
    assert "'D' is not one of" in problems
    assert "'dance' is not one of the task's skills" in problems
    assert "name=value" in problems
    assert "duplicate label 'a'" in problems
    assert "reviewed must be true or false" in problems
    assert "unknown key(s) note" in problems


def test_parse_call_and_normalize_label():
    assert parse_call("place(bin=A)") == SkillCall("place", {"bin": "A"})
    assert parse_call(" place ( bin = A ) ") == SkillCall("place", {"bin": "A"})
    assert parse_call("pick") == SkillCall("pick")
    with pytest.raises(ValueError):
        parse_call("place bin A")
    assert normalize_label("  Pick up\tthe Box。 ") == "pick up the box"


def test_label_file_keeps_reviewed_entries(tmp_path, bundle, dataset):
    labels = reviewed_labels(tmp_path, bundle, dataset)
    counts = count_labels(read_annotated_episodes(dataset))
    del counts["grab the parcel"]
    counts["lift the package"] = 3
    candidates = candidate_calls(bundle.registry, bundle.task.skills)
    path = tmp_path / "labels.yaml"
    stats = write_label_file(path, counts, candidates, dataset=dataset, library="x", previous=labels)
    assert stats == {"labels": 8, "new": 1, "kept": 7, "absent": 1}
    entries = {entry["label"]: entry for entry in yaml.safe_load(path.read_text(encoding="utf-8"))["labels"]}
    assert entries["grab the parcel"] == {
        "label": "grab the parcel",
        "count": 0,
        "call": "pick",
        "reviewed": True,
    }
    assert entries["wave at the camera"]["call"] is None and entries["wave at the camera"]["reviewed"] is True
    assert entries["lift the package"]["reviewed"] is False
    assert next(iter(entries)) == "lift the package"  # most frequent first
    assert "not in this dataset" in path.read_text(encoding="utf-8")


# --- planner steps and library statistics ------------------------------------------------------


def test_steps_merge_skip_and_chain_memory(tmp_path, bundle, dataset):
    labels = reviewed_labels(tmp_path, bundle, dataset)
    episodes = read_annotated_episodes(dataset)
    stats = StepStats()
    first = episode_steps(episodes[0], labels, stats)
    assert [(s.frame, s.call.key() if s.call else "done") for s in first] == [
        (0, "pick"),
        (10, "flip"),
        (20, "place(bin=A)"),
        (39, "done"),
    ]
    assert [(s.memory_in, s.memory_out) for s in first] == [
        ("", ""),
        ("", "picked package 1"),
        ("picked package 1", "package 1 flipped"),
        ("package 1 flipped", "package 1 above bin A"),
    ]
    assert first[2].labels == ("put it in bin a", "place the package into bin A")
    assert first[2].duration_s == pytest.approx(1.9, abs=1e-5)
    assert first[3].target() == {"done": True, "memory": "package 1 above bin A"}

    second = episode_steps(episodes[1], labels, stats)
    assert [(s.frame, s.call.key() if s.call else "done") for s in second] == [
        (0, "pick"),
        (15, "place(bin=C)"),
        (29, "done"),
    ]
    # The skipped span leaves no step, so the next one still receives the memory of the pick.
    assert [(s.memory_in, s.memory_out) for s in second] == [
        ("", ""),
        ("", "waved at the camera"),
        ("waved at the camera", "waved at the camera"),
    ]
    assert episode_steps(episodes[2], labels, stats) == []
    assert (stats.episodes, stats.unannotated, stats.spans, stats.skipped, stats.merged) == (3, 1, 7, 1, 1)


def test_library_statistics(tmp_path, bundle, dataset):
    labels = reviewed_labels(tmp_path, bundle, dataset)
    steps = [step for episode in read_annotated_episodes(dataset) for step in episode_steps(episode, labels)]
    library, unused = library_with_stats(bundle.library, bundle.registry, steps)
    by_id = {prompt.id: prompt for prompt in library.prompts}
    assert by_id["pick_next"].episodes == 2
    assert (by_id["pick_next"].duration_s.p50, by_id["pick_next"].duration_s.p95) == (1.0, 1.0)
    assert by_id["place_bin_a"].duration_s.p95 == 1.9
    assert by_id["place_bin_c"].duration_s.p50 == 1.4
    assert unused == ["place_bin_b"]
    assert by_id["place_bin_b"] == bundle.library.get("place_bin_b")
    assert [p.text for p in library.prompts] == [p.text for p in bundle.library.prompts]


def test_percentile_and_split():
    assert percentile([4, 1, 3, 2], 50) == 2.5
    assert percentile([5.0], 95) == 5.0
    assert percentile([0, 10], 95) == pytest.approx(9.5)
    episodes = list(range(10))
    split = split_episodes(episodes, 0.3, seed=1)
    assert sum(value == "val" for value in split.values()) == 3
    assert split == split_episodes(episodes, 0.3, seed=1)
    assert set(split_episodes(episodes, 0.0, seed=1).values()) == {"train"}
    assert list(split_episodes([3, 4], 0.1, seed=0).values()).count("val") == 1


def test_proprio_mapper():
    mapper = ProprioMapper(
        {"gripper_width": "state[2]", "gripper_closed": "state[2] < 0.01", "n": "len(state)"}
    )
    assert mapper([0.0, 1.0, 0.005]) == {"gripper_width": 0.005, "gripper_closed": True, "n": 3}
    values = mapper(np.array([0.0, 1.0, 0.5], dtype=np.float32))
    assert values == {"gripper_width": 0.5, "gripper_closed": False, "n": 3}
    assert type(values["gripper_width"]) is float and type(values["gripper_closed"]) is bool
    with pytest.raises(ConfigError) as info:
        ProprioMapper({"a": "ledger.x", "b": "state[", "c": "__import__('os')"})
    assert len(info.value.problems) == 3


# --- planner samples -----------------------------------------------------------------------------


def test_planner_export(tmp_path, dataset):
    bundle = vlm_bundle(tmp_path)
    frames = FakeFrames()
    manifest = export(tmp_path, bundle, dataset, frames=frames)
    out = tmp_path / "export"
    assert manifest["splits"]["train"] == {"episodes": [0, 1], "samples": 7}
    assert manifest["splits"]["val"] == {"episodes": [], "samples": 0}
    assert not (out / "val.jsonl").exists()
    assert manifest["calls"] == {"done": 2, "flip": 1, "pick": 2, "place(bin=A)": 1, "place(bin=C)": 1}
    assert manifest["planner"]["render"] == RENDER_VERSION
    assert manifest["planner"]["image_size"] == [32, 24]
    assert manifest["spans"] == {
        "episodes": 3,
        "unannotated_episodes": 1,
        "spans": 7,
        "skipped": 1,
        "merged": 1,
    }

    records = read_jsonl(out / "train.jsonl")
    samples = read_jsonl(out / "samples.jsonl")
    assert len(records) == len(samples) == 7
    first = records[0]
    assert [message["role"] for message in first["messages"]] == ["system", "user", "assistant"]
    assert bundle.task.goal in first["messages"][0]["content"]
    user = first["messages"][1]["content"]
    assert user.startswith("[head] <image>\n[wrist] <image>\n")
    assert "本体状态：gripper_width=0.040, gripper_closed=false\n" in user
    assert user.endswith("记忆：（空）")
    assert parse_decision(first["messages"][2]["content"]).to_dict() == {
        "skill": "pick",
        "args": {},
        "memory": "",
    }
    assert first["images"] == [
        "images/episode_000000/frame_000000_head.jpg",
        "images/episode_000000/frame_000000_wrist.jpg",
    ]
    with Image.open(out / first["images"][0]) as image:
        assert image.size == (32, 24)

    place = records[2]
    assert place["messages"][1]["content"].endswith("记忆：picked package 1")
    assert json.loads(place["messages"][2]["content"]) == {
        "skill": "place",
        "args": {"bin": "A"},
        "memory": "package 1 flipped",
    }
    assert json.loads(records[3]["messages"][2]["content"]) == {
        "done": True,
        "memory": "package 1 above bin A",
    }
    assert samples[2]["labels"] == ["put it in bin a", "place the package into bin A"]
    assert samples[2]["proprio"] == {"gripper_width": pytest.approx(0.06), "gripper_closed": False}
    assert samples[2]["reason"] == "skill_done" and samples[0]["reason"] == "start"
    assert (0, 20, ("observation.images.head", "observation.images.wrist")) in frames.calls


def test_export_is_what_the_live_planner_sends(tmp_path, dataset, servers):
    server = servers(lambda body, headers: chat_reply(json.dumps({"skill": "pick", "memory": ""})))
    bundle = vlm_bundle(tmp_path, server.base_url)
    export(tmp_path, bundle, dataset)
    out = tmp_path / "export"
    record, sample = read_jsonl(out / "train.jsonl")[2], read_jsonl(out / "samples.jsonl")[2]

    # A live call from the same raw frame, proprio and memory.
    raw = FakeFrames().images(0, 20, list(MAPPING["cameras"].values()))
    context = PlannerContext(
        task=bundle.task,
        images={camera: raw[key] for camera, key in MAPPING["cameras"].items()},
        proprio=sample["proprio"],
        memory=sample["memory_in"],
        reason=DecisionReason.SKILL_DONE,
        skills=list(bundle.task.skills),
    )
    bound_planner(bundle).plan(context)
    body = server.requests[-1]
    assert body["messages"][0]["content"] == record["messages"][0]["content"]
    assert placeholder_text(body) == record["messages"][1]["content"]
    assert request_images(body) == [(out / path).read_bytes() for path in record["images"]]


def test_export_truncates_memory_like_the_harness(tmp_path, dataset):
    bundle = vlm_bundle(tmp_path, memory_max_chars=10)
    manifest = export(tmp_path, bundle, dataset)
    assert manifest["memory"] == {"max_chars": 10, "over_limit": 5, "longest": 21}
    records = read_jsonl(tmp_path / "export" / "train.jsonl")
    assert records[2]["messages"][1]["content"].endswith("记忆：picked pac")
    assert json.loads(records[2]["messages"][2]["content"])["memory"] == "package 1 "


def test_export_checks_the_mapping_and_output(tmp_path, dataset):
    bundle = vlm_bundle(tmp_path)
    bad = DataMapping(cameras={"head": "observation.images.front"}, proprio={"gripper_width": "state[2]"})
    with pytest.raises(ConfigError) as info:
        export_planner_samples(
            bundle=bundle,
            planner=bound_planner(bundle),
            dataset=dataset,
            episodes=[],
            labels=reviewed_labels(tmp_path, bundle, dataset),
            mapping=bad,
            frames=FakeFrames(),
            out_dir=tmp_path / "export",
        )
    problems = "\n".join(info.value.problems)
    assert "no dataset key for planner camera 'wrist'" in problems
    assert "no feature 'observation.images.front'" in problems
    assert "gripper_closed" in problems

    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "old.txt").write_text("x")
    with pytest.raises(FileExistsError):
        export(tmp_path, bundle, dataset)


# --- relabeled copy for the executor ------------------------------------------------------------


def test_relabel_rewrites_only_subtask_texts(tmp_path, bundle, dataset):
    labels = reviewed_labels(tmp_path, bundle, dataset)
    out = tmp_path / "relabeled"
    report = relabel_dataset(dataset, out, labels, bundle.registry)
    assert report == {
        "files": 2,
        "labels_replaced": 6,
        "labels_kept": ["wave at the camera"],
        "videos": "linked",
    }

    source = pq.read_table(dataset / DATA_FILE)
    result = pq.read_table(out / DATA_FILE)
    assert result.schema.equals(source.schema, check_metadata=True)
    for name in source.column_names:
        if name != "language_persistent":
            assert result.column(name).equals(source.column(name))
    persistent = result.column("language_persistent")
    rows = persistent[0].as_py()
    assert [r["content"] for r in rows if r["style"] == "subtask"] == [
        "pick up the next package",
        "flip the package over",
        "place the package into bin A",
        "place the package into bin A",
    ]
    assert [r["content"] for r in rows if r["style"] == "memory"] == [
        r["content"] for r in persistent_rows(0) if r["style"] == "memory"
    ]
    assert persistent[39].as_py() == rows  # every frame of the episode carries the same rows
    assert [r["content"] for r in persistent[40].as_py() if r["style"] == "subtask"] == [
        "pick up the next package",
        "wave at the camera",
        "place the package into bin C",
    ]
    assert pq.ParquetFile(out / DATA_FILE).num_row_groups == 2
    assert (out / "videos").is_symlink()
    assert (out / "meta" / "info.json").read_bytes() == (dataset / "meta" / "info.json").read_bytes()
    assert source.column("language_persistent")[0].as_py() == persistent_rows(0)  # source untouched

    relabeled = count_labels(read_annotated_episodes(out))
    assert relabeled == {
        "pick up the next package": 2,
        "flip the package over": 1,
        "place the package into bin a": 2,
        "wave at the camera": 1,
        "place the package into bin c": 1,
    }

    with pytest.raises(FileExistsError):
        relabel_dataset(dataset, out, labels, bundle.registry)
    with pytest.raises(ValueError, match="outside the source"):
        relabel_dataset(dataset, dataset / "copy", labels, bundle.registry)


def test_relabel_can_copy_videos(tmp_path, bundle, dataset):
    labels = reviewed_labels(tmp_path, bundle, dataset)
    out = tmp_path / "relabeled"
    assert relabel_dataset(dataset, out, labels, bundle.registry, copy_videos=True)["videos"] == "copied"
    assert not (out / "videos").is_symlink()
    assert (out / "videos" / "observation.images.head" / "chunk-000" / "file-000.mp4").is_file()


# --- offline evaluation --------------------------------------------------------------------------


def test_eval_scores_the_planner_on_exported_samples(tmp_path, dataset, servers):
    server = servers(lambda body, headers: chat_reply(json.dumps({"skill": "pick", "memory": ""})))
    bundle = vlm_bundle(tmp_path, server.base_url)
    export(tmp_path, bundle, dataset)
    out = tmp_path / "export"
    rows = evaluate_samples(bound_planner(bundle), bundle, out, split="all")
    summary = summarize_eval(rows)
    assert summary["samples"] == 7
    assert (summary["same_call"], summary["same_skill"], summary["errors"]) == (2, 2, 0)
    assert summary["done_recall"] == 0.0 and summary["premature_done"] == 0
    assert summary["mistakes"]["done -> pick"] == 2
    assert summary["mistakes"]["place(bin=A) -> pick"] == 1
    assert rows[0].memory_similarity == 1.0  # both empty

    # The model is shown exactly the training sample.
    record = read_jsonl(out / "train.jsonl")[0]
    body = server.requests[0]
    assert placeholder_text(body) == record["messages"][1]["content"]
    assert request_images(body) == [(out / path).read_bytes() for path in record["images"]]

    assert evaluate_samples(bound_planner(bundle), bundle, out, split="val") == []
    assert len(evaluate_samples(bound_planner(bundle), bundle, out, split="all", limit=3)) == 3


# --- command line --------------------------------------------------------------------------------


def test_cli_data_workflow(tmp_path, dataset, servers, monkeypatch, capsys):
    server = servers(lambda body, headers: chat_reply(json.dumps({"done": True, "memory": "all sorted"})))
    config = write_configs(tmp_path, task=small_task(3), run=vlm_run(server.base_url))
    labels, library = tmp_path / "labels.yaml", tmp_path / "library.yaml"
    assert main(["data", "labels", str(config), str(dataset), "--out", str(labels)]) == 0
    assert "7 label(s) from 3 episode(s) (1 without subtasks)" in capsys.readouterr().out

    assert (
        main(["data", "library", str(config), str(dataset), "--labels", str(labels), "--out", str(library)])
        == 2
    )
    assert "not reviewed" in capsys.readouterr().err

    review(labels)
    assert (
        main(["data", "library", str(config), str(dataset), "--labels", str(labels), "--out", str(library)])
        == 0
    )
    written = yaml.safe_load(library.read_text(encoding="utf-8"))
    assert {p["id"]: p.get("episodes") for p in written["prompts"]}["pick_next"] == 2
    assert library.read_text(encoding="utf-8").startswith("# Prompt library mock_sort_v1.")
    assert "no annotated spans for: place_bin_b" in capsys.readouterr().out

    relabeled = tmp_path / "relabeled"
    args = ["data", "relabel", str(config), str(dataset), "--labels", str(labels), "--out", str(relabeled)]
    assert main(args) == 0
    assert "6 label(s) now use library texts" in capsys.readouterr().out

    mapping, exported = tmp_path / "mapping.yaml", tmp_path / "export"
    mapping.write_text(yaml.safe_dump(MAPPING), encoding="utf-8")
    monkeypatch.setattr(commands, "frame_source", lambda dataset, repo_id: FakeFrames())
    args = ["data", "planner", str(config), str(dataset), "--labels", str(labels), "--mapping", str(mapping)]
    assert main([*args, "--out", str(exported), "--val-fraction", "0.5"]) == 0
    printed = capsys.readouterr().out
    assert "merged into the previous step: 1" in printed
    manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["samples"] + manifest["splits"]["val"]["samples"] == 7
    # Two annotated episodes, half held out: one each; the unannotated episode takes no slot.
    assert len(manifest["splits"]["train"]["episodes"]) == len(manifest["splits"]["val"]["episodes"]) == 1
    assert main([*args, "--out", str(exported)]) == 2  # refuses to write into a used directory
    assert "not empty" in capsys.readouterr().err

    report = tmp_path / "eval.json"
    assert main(["data", "eval", str(config), str(exported), "--split", "all", "--out", str(report)]) == 0
    assert "7 all sample(s): same call 2/7" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8"))["summary"]["done_recall"] == 1.0


def test_cli_planner_export_needs_the_vlm_config(tmp_path, dataset, capsys):
    config = write_configs(tmp_path, task=small_task(3))  # mock oracle planner
    labels = tmp_path / "labels.yaml"
    assert main(["data", "labels", str(config), str(dataset), "--out", str(labels)]) == 0
    review(labels)
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump(MAPPING), encoding="utf-8")
    args = ["data", "planner", str(config), str(dataset), "--labels", str(labels), "--mapping", str(mapping)]
    assert main([*args, "--out", str(tmp_path / "export")]) == 2
    assert "renders no prompts" in capsys.readouterr().err
