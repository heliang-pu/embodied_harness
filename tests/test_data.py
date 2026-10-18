"""Data tools for lerobot-annotate datasets: labels, library statistics, planner samples, relabel, eval."""

import base64
import hashlib
import json
import logging
import re
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from conftest import REPLACE, make_bundle, run_episode, small_task, write_configs
from embodied_harness.cli import main
from embodied_harness.config.errors import ConfigError
from embodied_harness.core.types import DecisionReason, SkillCall
from embodied_harness.data import commands
from embodied_harness.data.annotations import read_annotated_episodes
from embodied_harness.data.evaluate import evaluate_samples, summarize_eval
from embodied_harness.data.export import (
    export_planner_samples,
    library_with_stats,
    percentile,
    split_episodes,
)
from embodied_harness.data.labels import (
    candidate_calls,
    count_labels,
    load_label_map,
    normalize_label,
    parse_call,
    rank_candidates,
    write_label_file,
)
from embodied_harness.data.mapping import DataMapping, ProprioMapper
from embodied_harness.data.relabel import relabel_dataset
from embodied_harness.data.steps import StepStats, episode_steps
from embodied_harness.planner.base import PlannerContext
from embodied_harness.planner.decision import parse_decision
from embodied_harness.planner.render import RENDER_VERSION, render_target
from embodied_harness.runtime.factory import bind_planner, build_planner
from fake_openai import chat_reply, user_text
from lerobot_fixture import INSTRUCTIONS, MAPPING, FakeFrames, persistent_rows, state, write_dataset

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
    # Each annotated episode sorts one package: with one expected, task.done_when holds at its end.
    return make_bundle(tmp_path, task=small_task(1), run=vlm_run(base_url, **planner_fields))


def review(path, calls=REVIEWED):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in data["labels"]:
        entry["call"] = calls[entry["label"]]
        entry["reviewed"] = True
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def reviewed_labels(tmp_path, bundle, dataset, calls=REVIEWED):
    path = tmp_path / "labels.yaml"
    episodes = read_annotated_episodes(dataset)
    candidates = candidate_calls(bundle.registry, bundle.task.skills)
    write_label_file(
        path, count_labels(episodes), candidates, dataset=dataset, library=bundle.library.library
    )
    review(path, calls)
    return load_label_map(path, bundle.registry, bundle.task)


def bound_planner(bundle):
    planner = build_planner(bundle)
    bind_planner(planner, bundle)
    return planner


def export(tmp_path, bundle, dataset, calls=REVIEWED, **kwargs):
    kwargs.setdefault("val_fraction", 0.0)
    return export_planner_samples(
        bundle=bundle,
        planner=bound_planner(bundle),
        dataset=dataset,
        episodes=read_annotated_episodes(dataset, frame_offsets=kwargs.get("frame_offsets", (0,))),
        labels=reviewed_labels(tmp_path, bundle, dataset, calls),
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


def test_a_trial_annotation_leaves_the_other_files_unannotated(dataset):
    # lerobot-annotate --only_episodes rewrites only the files holding those episodes.
    path = dataset / "data/chunk-000/file-001.parquet"
    pq.write_table(pq.read_table(path).drop_columns(["language_persistent", "language_events"]), path)
    episodes = {episode.episode_index: episode for episode in read_annotated_episodes(dataset)}
    assert set(episodes) == {0, 1, 2}
    assert episodes[0].decisions and episodes[2].decisions == ()


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
    assert stats == {"labels": 7, "new": 7, "kept": 0, "absent": 0, "accepted": 0}
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
    assert stats == {"labels": 8, "new": 1, "kept": 7, "absent": 1, "accepted": 0}
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
    assert stats.skipped_episodes == 0


def test_an_episode_whose_spans_all_map_to_null_gives_no_step(tmp_path, bundle, dataset):
    # Every span of episode 1 left out: its done step alone used to become a START sample whose target
    # is done, with nothing demonstrated before it.
    calls = {**REVIEWED, "grab the parcel": None, "put it in bin c": None}
    labels = reviewed_labels(tmp_path, bundle, dataset, calls)
    episodes = read_annotated_episodes(dataset)
    stats = StepStats()
    first = episode_steps(episodes[0], labels, stats)
    assert [s.call.key() if s.call else "done" for s in first] == ["pick", "flip", "place(bin=A)", "done"]
    assert episode_steps(episodes[1], labels, stats) == []
    assert episode_steps(episodes[2], labels, stats) == []  # not annotated at all: not a skipped episode
    assert (stats.unannotated, stats.skipped_episodes, stats.spans, stats.skipped) == (1, 1, 7, 3)


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
        "skipped_episodes": 0,
        "spans": 7,
        "skipped": 1,
        "merged": 1,
    }
    assert manifest["done_when"] == bundle.task.done_when and manifest["unfinished_episodes"] == []

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


def test_steps_so_far_are_the_same_in_the_export_and_in_live_calls(tmp_path, dataset, servers):
    server = servers(lambda body, headers: chat_reply(json.dumps({"skill": "pick", "memory": ""})))
    run = vlm_run(server.base_url)
    run["planner"]["params"]["show_history"] = True
    bundle = make_bundle(tmp_path, task=small_task(3), run=run)
    manifest = export(tmp_path, bundle, dataset)
    assert manifest["planner"]["show_history"] is True
    out = tmp_path / "export"
    records, samples = read_jsonl(out / "train.jsonl"), read_jsonl(out / "samples.jsonl")
    record, sample = records[2], samples[2]
    earlier = [s["target"] for s in samples[:2] if s["episode_index"] == sample["episode_index"]]
    assert sample["step"] == 2 and all("skill" in target for target in earlier)
    keys = [SkillCall(target["skill"], target.get("args", {})).key() for target in earlier]
    user = record["messages"][1]["content"]
    assert f"已执行的步骤：1. {keys[0]}；2. {keys[1]}\n这次调用的原因：上一步已完成" in user

    # A live call at the same point of the episode renders the same user turn.
    raw = FakeFrames().images(0, 20, list(MAPPING["cameras"].values()))
    context = PlannerContext(
        task=bundle.task,
        images={camera: raw[key] for camera, key in MAPPING["cameras"].items()},
        proprio=sample["proprio"],
        memory=sample["memory_in"],
        reason=DecisionReason.SKILL_DONE,
        skills=list(bundle.task.skills),
        history=[{"skill": target["skill"], "args": target.get("args", {})} for target in earlier],
    )
    bound_planner(bundle).plan(context)
    assert placeholder_text(server.requests[-1]) == user

    # Offline evaluation must reconstruct the same history when show_history is enabled.
    server.requests.clear()
    evaluate_samples(bound_planner(bundle), bundle, out, split="all")
    assert f"已执行的步骤：1. {keys[0]}；2. {keys[1]}" in placeholder_text(server.requests[2])

    # Older exports must not silently evaluate a history-trained model without its history.
    for sample in samples:
        sample.pop("history", None)
    (out / "samples.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="regenerate"):
        evaluate_samples(bound_planner(bundle), bundle, out, split="all")


def test_state_and_progress_are_the_same_in_the_export_and_in_live_calls(tmp_path, dataset, servers):
    server = servers(lambda body, headers: chat_reply(json.dumps({"skill": "pick", "memory": ""})))
    run = vlm_run(server.base_url)
    run["planner"]["params"].update(show_state=True, show_progress=True)
    bundle = make_bundle(tmp_path, task=small_task(1), run=run)
    manifest = export(tmp_path, bundle, dataset)
    assert manifest["planner"]["show_state"] is True and manifest["planner"]["show_progress"] is True
    out = tmp_path / "export"
    records, samples = read_jsonl(out / "train.jsonl"), read_jsonl(out / "samples.jsonl")
    episode = [(r, s) for r, s in zip(records, samples, strict=True) if s["episode_index"] == 0]
    # Episode 0 demonstrates pick, flip, place(A), done: the ledger counts the place once it is done.
    assert [s["last"] and s["last"]["call"] for _, s in episode] == [None, "pick", "flip", "place(bin=A)"]
    assert [s["ledger"]["sorted"]["A"] for _, s in episode] == [0, 0, 0, 1]
    record, sample = episode[3]
    user = record["messages"][1]["content"]
    width = sample["proprio"]["gripper_width"]  # the predicates come from the mapped proprio
    state = (
        f"holding={str(0.01 < width < 0.08).lower()}, empty_grasp=false, released={str(width > 0.08).lower()}"
    )
    assert f"当前状态：{state}\n上一步：place(bin=A) 成功\n" in user
    assert "进度：sorted: A=1, B=0, C=0（共 1）；目标总数 1\n" in user

    # A live call at the same point renders the same user turn.
    raw = FakeFrames().images(0, sample["frame"], list(MAPPING["cameras"].values()))
    context = PlannerContext(
        task=bundle.task,
        images={camera: raw[key] for camera, key in MAPPING["cameras"].items()},
        proprio=sample["proprio"],
        memory=sample["memory_in"],
        reason=DecisionReason.SKILL_DONE,
        skills=list(bundle.task.skills),
        ledger=sample["ledger"],
        last=sample["last"],
    )
    bound_planner(bundle).plan(context)
    assert placeholder_text(server.requests[-1]) == user

    # Offline evaluation rebuilds both; an export made without them is refused.
    server.requests.clear()
    evaluate_samples(bound_planner(bundle), bundle, out, split="all")
    assert placeholder_text(server.requests[3]) == user
    for sample in samples:
        sample["ledger"] = None
    (out / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8")
    with pytest.raises(ValueError, match="no ledger for show_progress"):
        evaluate_samples(bound_planner(bundle), bundle, out, split="all")


def test_export_truncates_memory_like_the_harness(tmp_path, dataset):
    bundle = vlm_bundle(tmp_path, memory_max_chars=10)
    manifest = export(tmp_path, bundle, dataset)
    assert manifest["memory"] == {"max_chars": 10, "over_limit": 5, "longest": 21}
    records = read_jsonl(tmp_path / "export" / "train.jsonl")
    assert records[2]["messages"][1]["content"].endswith("记忆：picked pac")
    assert json.loads(records[2]["messages"][2]["content"])["memory"] == "package 1 "


def test_export_uses_episode_instructions_and_frame_offsets(tmp_path, dataset):
    bundle = make_bundle(tmp_path, task=small_task(1, goal_source="episode"), run=vlm_run())
    episodes = read_annotated_episodes(dataset, instruction_key="instruction", frame_offsets=(0, 2))
    assert [e.instruction for e in episodes] == [INSTRUCTIONS[0], INSTRUCTIONS[1], INSTRUCTIONS[2]]
    assert {0, 2, 10, 12, 20, 22, 39} <= set(episodes[0].states)
    assert episodes[0].states[12] == pytest.approx(state(12))
    assert episodes[0].frame_at(39, 2) == 39 and episodes[0].frame_at(0, -3) == 0
    manifest = export_planner_samples(
        bundle=bundle,
        planner=bound_planner(bundle),
        dataset=dataset,
        episodes=episodes,
        labels=reviewed_labels(tmp_path, bundle, dataset),
        mapping=DataMapping.model_validate({**MAPPING, "instruction_key": "instruction"}),
        frames=FakeFrames(),
        out_dir=tmp_path / "export",
        val_fraction=0.0,
        frame_offsets=(0, 2),
    )
    # Each step gives one sample per distinct frame: the last frame caps the done steps at one.
    assert manifest["splits"]["train"]["samples"] == (3 * 2 + 1) + (2 * 2 + 1)
    assert manifest["frame_offsets"] == [0, 2]
    records = read_jsonl(tmp_path / "export" / "train.jsonl")
    samples = read_jsonl(tmp_path / "export" / "samples.jsonl")
    assert [(s["frame"], s["offset"]) for s in samples[:3]] == [(0, 0), (2, 2), (10, 0)]
    assert records[0]["messages"][2]["content"] == records[1]["messages"][2]["content"]  # same target
    assert records[0]["images"] != records[1]["images"]
    for record, sample in zip(records, samples, strict=True):
        goal = INSTRUCTIONS[sample["episode_index"]]
        assert sample["goal"] == goal
        assert f"任务：{goal}" in record["messages"][0]["content"]

    with pytest.raises(ValueError, match="has no instruction"):
        export_planner_samples(
            bundle=bundle,
            planner=bound_planner(bundle),
            dataset=dataset,
            episodes=read_annotated_episodes(dataset),
            labels=reviewed_labels(tmp_path, bundle, dataset),
            mapping=DataMapping.model_validate(MAPPING),
            frames=FakeFrames(),
            out_dir=tmp_path / "export2",
        )


def test_episode_instruction_rejects_inconsistent_values(tmp_path, dataset):
    path = dataset / DATA_FILE
    table = pq.read_table(path)
    values = table.column("instruction").to_pylist()
    values[1] = "different instruction"
    table = table.set_column(table.schema.get_field_index("instruction"), "instruction", pa.array(values))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="inconsistent instructions"):
        read_annotated_episodes(dataset, instruction_key="instruction")


def test_null_instruction_does_not_become_a_literal_none_goal(dataset):
    path = dataset / DATA_FILE
    table = pq.read_table(path)
    values = table.column("instruction").to_pylist()
    values[0] = None
    table = table.set_column(table.schema.get_field_index("instruction"), "instruction", pa.array(values))
    pq.write_table(table, path)
    episodes = read_annotated_episodes(dataset, instruction_key="instruction")
    assert episodes[0].instruction == INSTRUCTIONS[0]


def test_progress_export_with_only_positive_offsets_keeps_boundary_states(tmp_path, dataset):
    run = vlm_run()
    run["planner"]["params"]["show_progress"] = True
    bundle = make_bundle(tmp_path, task=small_task(1), run=run)
    episodes = read_annotated_episodes(dataset, frame_offsets=[2])
    # Images use the requested offsets, but ledger effects still read the unshifted skill ends.
    assert {0, 2, 10, 12, 20, 22, 25, 27, 39} <= set(episodes[0].states)
    out = tmp_path / "export"
    manifest = export_planner_samples(
        bundle=bundle,
        planner=bound_planner(bundle),
        dataset=dataset,
        episodes=episodes,
        labels=reviewed_labels(tmp_path, bundle, dataset),
        mapping=DataMapping.model_validate(MAPPING),
        frames=FakeFrames(),
        out_dir=out,
        val_fraction=0.0,
        frame_offsets=[2],
    )
    assert manifest["frame_offsets"] == [2]
    assert manifest["splits"]["train"]["samples"] == 7
    samples = read_jsonl(out / "samples.jsonl")
    first = [sample for sample in samples if sample["episode_index"] == 0]
    second = [sample for sample in samples if sample["episode_index"] == 1]
    assert [(sample["frame"], sample["offset"]) for sample in first] == [
        (2, 2),
        (12, 2),
        (22, 2),
        (39, 0),
    ]
    assert [sample["ledger"]["sorted"]["A"] for sample in first] == [0, 0, 0, 1]
    assert [sample["ledger"]["sorted"]["C"] for sample in second] == [0, 0, 1]


def test_a_demonstration_that_ends_before_done_when_gives_no_done_sample(tmp_path, dataset, caplog):
    # Episode 1 without its place, as if the recording stopped after the pick: one package is expected
    # and none is sorted, so the supervisor would refuse a done there. It used to be a done target.
    bundle = vlm_bundle(tmp_path)
    with caplog.at_level(logging.WARNING, logger="embodied_harness.data"):
        manifest = export(tmp_path, bundle, dataset, {**REVIEWED, "put it in bin c": None})
    samples = read_jsonl(tmp_path / "export" / "samples.jsonl")
    targets = [(sample["episode_index"], sample["target"].get("skill", "done")) for sample in samples]
    assert targets == [(0, "pick"), (0, "flip"), (0, "place"), (0, "done"), (1, "pick")]
    # The ledger is rebuilt for the check without show_progress, and still not shown.
    assert all(sample["ledger"] is None for sample in samples)
    assert manifest["calls"]["done"] == 1
    assert manifest["done_when"] == "sum(ledger.sorted) == task.expected_total"
    assert manifest["unfinished_episodes"] == [1]
    assert "1 episode(s) end before task.done_when" in caplog.text


def test_done_samples_need_the_tasks_done_when(tmp_path, dataset):
    # Each demonstration sorts one package: a task expecting three is done at the end of none of them.
    three = make_bundle(tmp_path / "three", task=small_task(3), run=vlm_run())
    manifest = export(tmp_path / "three", three, dataset)
    assert "done" not in manifest["calls"]
    assert manifest["splits"]["train"] == {"episodes": [0, 1], "samples": 5}
    assert manifest["unfinished_episodes"] == [0, 1]
    # Without done_when, completion is the planner's call at run time: each demonstration ends done.
    free = make_bundle(tmp_path / "free", task=small_task(3, done_when=None), run=vlm_run())
    manifest = export(tmp_path / "free", free, dataset)
    assert manifest["done_when"] is None and manifest["unfinished_episodes"] == []
    assert manifest["calls"]["done"] == 2


def test_no_done_sample_when_the_planner_cannot_say_done(tmp_path, dataset):
    # planner.allow_done false: the schema has no done and the system prompt says never to output done, so
    # a done target would contradict its own sample. The demonstrations end where the env ends them.
    bundle = vlm_bundle(tmp_path, allow_done=False)
    manifest = export(tmp_path, bundle, dataset)
    records = read_jsonl(tmp_path / "export" / "train.jsonl")
    assert all("不要输出 done" in record["messages"][0]["content"] for record in records)
    assert "done" not in manifest["calls"] and manifest["done_dropped"] == 2
    assert manifest["splits"]["train"]["samples"] == 5 and manifest["unfinished_episodes"] == []
    assert manifest["planner"]["no_done"] is True
    # The count is of samples: with frame offsets, each frame a done step would have given.
    offsets = {"frame_offsets": (-3, 0)}
    allowed = export(tmp_path / "on", vlm_bundle(tmp_path / "on"), dataset, **offsets)
    refused = export(tmp_path / "off", vlm_bundle(tmp_path / "off", allow_done=False), dataset, **offsets)
    assert refused["done_dropped"] == allowed["calls"]["done"] == 4


def test_an_episode_without_a_call_step_gives_no_sample(tmp_path, dataset):
    bundle = vlm_bundle(tmp_path)
    calls = {**REVIEWED, "grab the parcel": None, "put it in bin c": None}
    # With val_fraction 0.5 and seed 1, the split holds out episode 0 of [0, 1]: the episode that gives
    # no sample must not take a slot.
    manifest = export(tmp_path, bundle, dataset, calls, val_fraction=0.5, seed=1)
    samples = read_jsonl(tmp_path / "export" / "samples.jsonl")
    assert {sample["episode_index"] for sample in samples} == {0}
    assert not any(sample["reason"] == "start" and sample["target"].get("done") for sample in samples)
    assert manifest["spans"]["skipped_episodes"] == 1
    assert manifest["splits"]["train"] == {"episodes": [0], "samples": 4}
    assert manifest["splits"]["val"] == {"episodes": [], "samples": 0}


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


# --- periodic planners ---------------------------------------------------------------------------

# Episode 0 of the fixture: each step's call and the memory written when choosing it.
DEMONSTRATION = [
    ({"skill": "pick"}, ""),
    ({"skill": "flip"}, "picked package 1"),
    ({"skill": "place", "args": {"bin": "A"}}, "package 1 flipped"),
    ({"done": True}, "package 1 above bin A"),
]
# Skills without success conditions: a periodic planner's switch is the only way a call ends.
NO_SUCCESS_CONDITIONS = {"skills": {name: {"verify": {REPLACE: True}} for name in ("pick", "flip", "place")}}


def periodic_run(base_url="http://127.0.0.1:9/v1", **params):
    run = vlm_run(base_url, call="periodic", every_steps=3)
    run["planner"]["params"].update(params)
    return run


def demonstrator(switch_at=None):
    """A planner that replays episode 0, reading the steps so far from its prompt (show_history). A
    periodic check keeps the running call, or from its `switch_at`-th check of a call on, moves on."""
    checks: Counter[int] = Counter()

    def reply(body, headers):
        text = user_text(body)
        listed = re.search("已执行的步骤：(.*)\n", text).group(1)
        position = 0 if listed == "（无）" else listed.count("；") + 1
        if "执行中的定期检查" in text:
            checks[position] += 1
            if switch_at is None or checks[position] < switch_at:
                position -= 1  # the running call
        decision, memory = DEMONSTRATION[position]
        return chat_reply(json.dumps({**decision, "memory": memory}))

    return reply


def state_text(text):
    """A user turn without the lines computed from the robot state (the proprio and the predicates),
    which a demonstration and a mock run cannot share."""
    return "\n".join(line for line in text.split("\n") if not line.startswith(("本体状态：", "当前状态：")))


def test_a_periodic_export_adds_a_sample_per_check_inside_each_step(tmp_path, dataset):
    def exported(name, run):
        bundle = make_bundle(tmp_path / name, task=small_task(1), run=run)
        manifest = export(tmp_path / name, bundle, dataset)
        return manifest, read_jsonl(tmp_path / name / "export" / "samples.jsonl")

    _, boundary = exported("boundary", vlm_run())
    manifest, samples = exported("periodic", periodic_run())
    checks = [sample for sample in samples if sample["reason"] == "periodic"]
    # The dataset runs at 10 fps like the control loop: every 3 frames while the step's call runs.
    assert [(sample["episode_index"], sample["frame"]) for sample in checks] == [
        *[(0, frame) for frame in (3, 6, 9, 13, 16, 19, 23, 26, 29, 32, 35, 38)],
        *[(1, frame) for frame in (3, 6, 9, 18, 21, 24, 27)],
    ]
    # The running call is in the history, the memory is what the planner wrote when it chose the call,
    # and the answer keeps both.
    for sample in checks:
        target = sample["target"]
        assert sample["history"][-1] == {"skill": target["skill"], "args": target["args"]}
        assert sample["memory_in"] == target["memory"]
    # The skills' verifiers end every call, so the decisions between steps stay as they were.
    others = [{**sample, "line": None} for sample in samples if sample["reason"] != "periodic"]
    assert others == [{**sample, "line": None} for sample in boundary]
    assert manifest["periodic"] == {
        "every_steps": 3,
        "control_hz": 10.0,
        "fps": 10.0,
        "frames_per_check": 3,
        "limit": None,
        "samples": 19,
        "switched_steps": 0,
    }
    assert manifest["planner"]["call"] == "periodic"


def test_the_checks_stay_inside_the_annotated_step(tmp_path, dataset):
    bundle = make_bundle(tmp_path, task=small_task(1), run=periodic_run())
    export(tmp_path, bundle, dataset, frame_offsets=(-2, 0))
    samples = read_jsonl(tmp_path / "export" / "samples.jsonl")

    def checks(episode, skill):
        return [
            s["frame"]
            for s in samples
            if s["episode_index"] == episode and s["reason"] == "periodic" and s["target"]["skill"] == skill
        ]

    # Episode 0's flip has its first sample at frame 8 (offset -2), which already teaches moving on from
    # pick: a check at 9 would teach keeping it.
    assert checks(0, "pick") == [3, 6]
    # Episode 1's pick ends at frame 10, where the demonstrator waves at the camera (a span mapped to null)
    # until place(bin=C) starts at 15: a check at 12 would keep a call the demonstration had finished.
    assert checks(1, "pick") == [3, 6, 9]


def test_a_call_no_verifier_ends_is_switched_by_a_periodic_check(tmp_path, dataset):
    run = periodic_run(show_state=True, show_progress=True)
    bundle = make_bundle(tmp_path, task=small_task(1, done_when=None), run=run, skills=NO_SUCCESS_CONDITIONS)
    manifest = export(tmp_path, bundle, dataset)
    samples = read_jsonl(tmp_path / "export" / "samples.jsonl")
    switches = [s for s in samples if s["episode_index"] == 0 and s["offset"] == 0]
    assert [s["reason"] for s in switches] == ["start", "periodic", "periodic", "periodic"]
    # Asked while the previous call still ran: it is preempted, so it commits nothing.
    assert [s["last"] and (s["last"]["call"], s["last"]["status"]) for s in switches] == [
        None,
        None,
        ("pick", "preempted"),
        ("flip", "preempted"),
    ]
    assert [s["ledger"]["sorted"]["A"] for s in switches] == [0, 0, 0, 0]
    assert manifest["periodic"]["switched_steps"] == 5


@pytest.mark.parametrize("verifiers", [True, False], ids=["verifiers_end_calls", "planner_switches_calls"])
def test_a_periodic_export_renders_what_a_periodic_planner_is_asked(tmp_path, dataset, servers, verifiers):
    server = servers(demonstrator(switch_at=None if verifiers else 2))
    bundle = make_bundle(
        tmp_path,
        task=small_task(1) if verifiers else small_task(1, done_when=None),
        run=periodic_run(server.base_url, show_history=True, show_state=True, show_progress=True),
        skills=None if verifiers else NO_SUCCESS_CONDITIONS,
    )
    world = {"num_packages": 1, "cameras": CAMERAS, "image_size": [32, 24]}
    live = run_episode(bundle, tmp_path, world=world)
    assert live.result.status == "done"
    paths = sorted((live.episode_dir / "planner").glob("pc_*.json"))
    calls = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(calls) == len(server.requests)
    # What the planner was asked at each check and what it answered (refused answers left out).
    asked = {
        (state_text(placeholder_text(body)), render_target(call["decision"]))
        for body, call in zip(server.requests, calls, strict=True)
        if not call["errors"] and call["input"]["reason"] == "periodic"
    }
    running = {re.search("已执行的步骤：(.*)\n", context).group(1).split("；")[-1] for context, _ in asked}
    assert {"1. pick", "3. place(bin=A)"} <= running
    export(tmp_path, bundle, dataset)
    out = tmp_path / "export"
    records, samples = read_jsonl(out / "train.jsonl"), read_jsonl(out / "samples.jsonl")
    demonstrated = {
        (state_text(record["messages"][1]["content"]), record["messages"][2]["content"])
        for record, sample in zip(records, samples, strict=True)
        if sample["episode_index"] == 0 and sample["reason"] == "periodic"
    }
    if verifiers:
        # A supervisor may skip the checks while a call's verified success is pending (flip's success only
        # waits for its min_duration_s), so the export can also cover states the planner is not asked in.
        assert asked <= demonstrated
    else:
        assert asked == demonstrated
    every_request = {state_text(placeholder_text(body)) for body in server.requests}
    assert every_request <= {state_text(record["messages"][1]["content"]) for record in records}
    assert {body["messages"][0]["content"] for body in server.requests} == {
        record["messages"][0]["content"] for record in records
    }


def test_eval_scores_keeping_and_switching_apart(tmp_path, dataset, servers, capsys):
    def keep_running(body, headers):
        """Never switches: answers the call it is running (the last of the steps so far), pick at first."""
        listed = re.search("已执行的步骤：(.*)\n", user_text(body)).group(1)
        running = listed.split("；")[-1].split(". ", 1)[-1]
        call = parse_call(running) if listed != "（无）" else SkillCall("pick")
        return chat_reply(json.dumps({**call.to_dict(), "memory": ""}))

    server = servers(keep_running)
    run = periodic_run(server.base_url, show_history=True)
    bundle = make_bundle(tmp_path, task=small_task(1, done_when=None), run=run, skills=NO_SUCCESS_CONDITIONS)
    export(tmp_path, bundle, dataset)
    capsys.readouterr()
    assert main(["data", "eval", str(bundle.path), str(tmp_path / "export"), "--split", "all"]) == 0
    printed = capsys.readouterr().out
    # 21 of 26 look right, but it never moves on.
    assert "26 all sample(s): same call 21/26" in printed
    assert "same call by kind: periodic keep 19/19, periodic switch 0/5, start 2/2" in printed


def test_cli_periodic_export_can_limit_the_checks(tmp_path, dataset, monkeypatch, capsys):
    config = write_configs(tmp_path, task=small_task(1), run=periodic_run())
    labels = tmp_path / "labels.yaml"
    assert main(["data", "labels", str(config), str(dataset), "--out", str(labels)]) == 0
    review(labels)
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump(MAPPING), encoding="utf-8")
    monkeypatch.setattr(commands, "frame_source", lambda dataset, repo_id: FakeFrames())
    args = ["data", "planner", str(config), str(dataset), "--labels", str(labels), "--mapping", str(mapping)]
    exported = tmp_path / "export"
    capsys.readouterr()
    assert main([*args, "--out", str(exported), "--val-fraction", "0", "--periodic-limit", "2"]) == 0
    printed = capsys.readouterr().out
    assert "periodic checks: 10 sample(s) inside the steps, one every 3 frame(s), at most 2 per step" in (
        printed
    )
    samples = read_jsonl(exported / "samples.jsonl")
    # The first and the last check of each step.
    assert [(s["episode_index"], s["frame"]) for s in samples if s["reason"] == "periodic"] == [
        *[(0, frame) for frame in (3, 9, 13, 19, 23, 38)],
        *[(1, frame) for frame in (3, 9, 18, 27)],
    ]
    assert main([*args, "--out", str(tmp_path / "other"), "--periodic-limit", "-1"]) == 2
    assert "periodic_limit must be 0 or more" in capsys.readouterr().err
    # Without an offset no decision has a sample: the export would hold nothing but the checks.
    assert main([*args, "--out", str(tmp_path / "other"), "--frame-offsets", ""]) == 2
    assert "frame_offsets is empty" in capsys.readouterr().err
    boundary = write_configs(tmp_path / "boundary", task=small_task(1), run=vlm_run())
    args[2] = str(boundary)
    assert main([*args, "--out", str(tmp_path / "other"), "--periodic-limit", "2"]) == 2
    assert "periodic_limit only applies to demonstrations exported for planner.call: periodic" in (
        capsys.readouterr().err
    )


def test_a_boundary_export_is_byte_for_byte_what_it_was(tmp_path, dataset):
    # Pinned: what these settings exported before periodic samples, the done drop and the planner settings
    # added since. A deliberate change to the boundary samples needs a new RENDER_VERSION or a note in the
    # docs, and new digests here.
    task = small_task(1, goal="把包裹分拣到对应的箱子", rules=["条码朝下的包裹先翻面再放"])
    run = vlm_run()
    run["planner"]["params"].update(show_history=True, show_state=True, show_progress=True)
    export(tmp_path, make_bundle(tmp_path, task=task, run=run), dataset)
    digests = {
        name: hashlib.sha256((tmp_path / "export" / name).read_bytes()).hexdigest()[:16]
        for name in ("train.jsonl", "samples.jsonl")
    }
    assert digests == {"train.jsonl": "2cdf4bfd9f6325b9", "samples.jsonl": "32caf5991d5238be"}


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


def test_relabel_in_place_rewrites_only_the_shards(tmp_path, bundle, dataset):
    labels = reviewed_labels(tmp_path, bundle, dataset)
    meta_before = (dataset / "meta" / "info.json").read_text()
    source = pq.read_table(dataset / DATA_FILE)
    report = relabel_dataset(dataset, dataset, labels, bundle.registry)
    assert report["labels_replaced"] == 6 and report["videos"] == "untouched"
    result = pq.read_table(dataset / DATA_FILE)
    assert result.schema.equals(source.schema, check_metadata=True)
    rows = result.column("language_persistent")[0].as_py()
    assert next(r["content"] for r in rows if r["style"] == "subtask") == "pick up the next package"
    assert (dataset / "meta" / "info.json").read_text() == meta_before
    assert not list(dataset.rglob("*.tmp"))


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


def test_eval_warns_when_the_done_mode_the_rules_or_the_call_mode_differ(tmp_path, dataset, servers, capsys):
    server = servers(lambda body, headers: chat_reply(json.dumps({"skill": "pick", "memory": ""})))
    export(tmp_path, vlm_bundle(tmp_path / "trained", server.base_url), dataset)
    exported = tmp_path / "export"

    def warned(name, task=None, **planner_fields):
        run = vlm_run(server.base_url, **planner_fields)
        config = write_configs(tmp_path / name, task=task or small_task(1), run=run)
        assert main(["data", "eval", str(config), str(exported), "--split", "all"]) == 0
        err = capsys.readouterr().err
        found = re.search(r"planner settings differ from the export \((.*)\)", err)
        unchecked = ["(no system fingerprint)"] if "export has no system fingerprint" in err else []
        return (found.group(1).split(", ") if found else []) + unchecked

    rules = small_task(1, rules=["先抓大的"])
    assert warned("same") == []
    assert warned("no_done", allow_done=False) == ["no_done", "system"]  # the completion sentence too
    assert warned("rules", task=rules) == ["system"]
    assert warned("periodic", call="periodic") == ["call"]
    # Every sample is sent with the goal it recorded, so the configured goal changes nothing here.
    assert warned("goal", task=small_task(1, goal="把所有包裹分拣好")) == []
    # An export made before these settings were recorded: what it did not record reads as the default,
    # and a changed system prompt cannot be told apart, which the warning says.
    manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
    for key in ("no_done", "system", "goal"):
        del manifest["planner"][key]
    (exported / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert warned("old") == warned("old_rules", task=rules) == ["(no system fingerprint)"]
    assert warned("old_no_done", allow_done=False) == ["no_done", "(no system fingerprint)"]


# --- command line --------------------------------------------------------------------------------


def test_cli_data_workflow(tmp_path, dataset, servers, monkeypatch, capsys):
    server = servers(lambda body, headers: chat_reply(json.dumps({"done": True, "memory": "all sorted"})))
    config = write_configs(tmp_path, task=small_task(1), run=vlm_run(server.base_url))
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


def test_a_deployment_config_can_name_the_export_it_is_about_to_make(tmp_path, dataset, monkeypatch, capsys):
    # A periodic planner's data is exported with the deployment's run config, which also names that export
    # as what the planner is trained on. Only `validate` and the run start read it.
    run = periodic_run()
    run["planner"]["trained_manifest"] = "../../export"  # relative to configs/runs/
    config = write_configs(tmp_path, task=small_task(1), run=run)
    capsys.readouterr()
    assert main(["validate", str(config)]) == 0
    missing = f"warning: planner.trained_manifest: {tmp_path.resolve() / 'export'}: file not found"
    assert missing in capsys.readouterr().out
    labels, mapping = tmp_path / "labels.yaml", tmp_path / "mapping.yaml"
    assert main(["data", "labels", str(config), str(dataset), "--out", str(labels)]) == 0
    review(labels)
    mapping.write_text(yaml.safe_dump(MAPPING), encoding="utf-8")
    monkeypatch.setattr(commands, "frame_source", lambda dataset, repo_id: FakeFrames())
    args = ["data", "planner", str(config), str(dataset), "--labels", str(labels), "--mapping", str(mapping)]
    assert main([*args, "--out", str(tmp_path / "export")]) == 0
    capsys.readouterr()
    assert main(["validate", str(config)]) == 0
    assert "planner settings match planner.trained_manifest (../../export)" in capsys.readouterr().out


def test_coverage_lists_the_step_orders_of_the_training_data(tmp_path, dataset, capsys):
    from conftest import run_episode

    config = write_configs(tmp_path, task=small_task(3))
    labels = tmp_path / "labels.yaml"
    assert main(["data", "labels", str(config), str(dataset), "--out", str(labels)]) == 0
    review(labels)
    capsys.readouterr()
    assert main(["data", "coverage", str(config), str(dataset), "--labels", str(labels)]) == 0
    printed = capsys.readouterr().out
    assert "training data: 2 annotated episode(s), 4 distinct transition(s)" in printed
    assert "| START -> pick | 2 |" in printed and "| flip -> place(bin=A) | 1 |" in printed

    # A mock run sorts three packages: place -> pick never happens in the demonstrations.
    run = run_episode(make_bundle(tmp_path / "run", task=small_task(3)), tmp_path)
    args = [
        "data",
        "coverage",
        str(config),
        str(dataset),
        "--labels",
        str(labels),
        "--runs",
        str(run.episode_dir),
    ]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "| START -> pick | 2 | 1 | 1/1 (100%) |" in printed
    assert re.search(r"\| place\(bin=[ABC]\) -> pick \| 0 \| \d+ \|", printed)  # placed, then picked again
    assert "transition(s) the runs needed never appear in the training data" in printed


def test_planner_export_can_leave_episodes_out(tmp_path, dataset, monkeypatch, capsys):
    config = write_configs(tmp_path, task=small_task(3), run=vlm_run())
    labels = tmp_path / "labels.yaml"
    assert main(["data", "labels", str(config), str(dataset), "--out", str(labels)]) == 0
    review(labels)
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump(MAPPING), encoding="utf-8")
    monkeypatch.setattr(commands, "frame_source", lambda dataset, repo_id: FakeFrames())
    args = ["data", "planner", str(config), str(dataset), "--labels", str(labels), "--mapping", str(mapping)]
    exported = tmp_path / "export"
    assert main([*args, "--out", str(exported), "--val-fraction", "0", "--exclude-episodes", "1"]) == 0
    assert "left out 1 episode(s): 1" in capsys.readouterr().out
    manifest = json.loads((exported / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["episodes"] == [0]
    assert main([*args, "--out", str(tmp_path / "other"), "--exclude-episodes", "7"]) == 2
    assert "has no episode(s) [7]" in capsys.readouterr().err
