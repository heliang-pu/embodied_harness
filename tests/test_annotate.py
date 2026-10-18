"""`embodied-harness data annotate`: lerobot-annotate on a copy of the dataset, steered by the prompt library.

lerobot itself is not installed here: the packaged prompt templates are replaced by small ones with
the same fields and anchors, and lerobot-annotate by a script that writes subtask and memory rows.
"""

import json
import sys

import pyarrow.parquet as pq
import pytest

from conftest import RUN_CONFIG
from embodied_harness.cli import main
from embodied_harness.config.loader import load_bundle
from embodied_harness.data import annotate
from embodied_harness.data.annotate import label_descriptions, label_vocabulary, steer_prompts
from embodied_harness.data.labels import load_label_map
from lerobot_fixture import write_dataset

TEMPLATES = {
    "plan_subtasks": (
        'The operator\'s goal was: "{episode_task}"\n'
        "{observation_block}Labels: short imperative phrases, drop articles.\n\n"
        "Timing:\n- stay within [0.0, {episode_duration}], at most {max_steps} segments, "
        "none shorter than {min_subtask_seconds} s.\n\n"
        'Output strictly valid JSON of shape:\n  {{"subtasks": []}}\n'
    ),
    "plan_subtask_relabel": (
        'Return only JSON:\n  {{"label": "<label>"}}\n'
        "Target segment: {segment_index} of {segment_count}, {start:.2f}s to {end:.2f}s, "
        'original label "{seed_label}".\n\nRules:\n- keep the label short.\n'
    ),
    "plan_memory": (
        'Episode task: "{episode_task}"\nPrevious memory: {prior_memory}\n'
        'Just-completed subtask: "{completed_subtask}"\nRemaining subtasks: {remaining_subtasks}\n\n'
        'Output strictly valid JSON:\n  {{ "memory": "<memory>" }}\n'
    ),
}

FAKE_ANNOTATE = r"""
import json, os, sys
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

root = Path(next(arg.split("=", 1)[1] for arg in sys.argv[1:] if arg.startswith("--root=")))
seen = {
    "argv": sys.argv[1:],
    "overrides": sorted(key for key in os.environ if key.startswith("LEROBOT_PROMPT_OVERRIDE_")),
    "plan_subtasks": os.environ.get("LEROBOT_PROMPT_OVERRIDE_plan_subtasks", ""),
}
(root / "seen_by_annotate.json").write_text(json.dumps(seen))
SUBTASKS = {
    0: [
        (0.0, "pick up the next package"),
        (1.0, "flip the package over"),
        (2.0, "place the package into bin A"),
    ],
    1: [(0.0, "grab the parcel"), (1.5, "place the package into bin C")],
}
MEMORY = {
    0: [(1.0, "I picked up package 1."), (2.0, "I flipped package 1.")],
    1: [(1.5, "I picked up package 2.")],
}

def rows(episode):
    out = [
        {"role": "assistant", "content": text, "style": style, "timestamp": t,
         "camera": None, "tool_calls": None}
        for style, items in (("subtask", SUBTASKS.get(episode, [])), ("memory", MEMORY.get(episode, [])))
        for t, text in items
    ]
    return sorted(out, key=lambda row: (row["timestamp"], row["style"]))

for path in sorted((root / "data").rglob("*.parquet")):
    table = pq.read_table(path)
    column = pa.array([rows(episode) for episode in table.column("episode_index").to_pylist()])
    index = table.schema.get_field_index("language_persistent")
    pq.write_table(table.set_column(index, "language_persistent", column), path)
sys.exit(int(os.environ.get("FAKE_ANNOTATE_EXIT", "0")))
"""


@pytest.fixture
def fake_lerobot(tmp_path, monkeypatch):
    script = tmp_path / "fake_lerobot_annotate.py"
    script.write_text(FAKE_ANNOTATE, encoding="utf-8")
    monkeypatch.setattr(annotate, "packaged_prompt", lambda name: TEMPLATES[name])
    monkeypatch.setattr(annotate, "annotate_executable", lambda: [sys.executable, str(script)])


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path / "dataset")


def first_labels(root):
    table = pq.read_table(sorted((root / "data").rglob("*.parquet"))[0])  # episode 0 comes first
    rows = table.column("language_persistent")[0].as_py()
    return [row["content"] for row in rows if row["style"] == "subtask"]


def test_the_vocabulary_is_the_library_text_of_every_policy_call():
    bundle = load_bundle(RUN_CONFIG)
    assert label_vocabulary(bundle.registry, bundle.task) == [
        "pick up the next package",
        "flip the package over",
        "place the package into bin A",
        "place the package into bin B",
        "place the package into bin C",
    ]  # retract is a scripted primitive: no library text


def test_steered_prompts_keep_the_template_fields():
    steered = steer_prompts(
        TEMPLATES,
        vocabulary=["pick up the cube", "put {it} away"],
        rules=["count {each} pick"],
        memory_max_chars=120,
    )
    subtasks = steered["plan_subtasks"].format(
        episode_task="t", observation_block="", episode_duration="3.0", max_steps=8, min_subtask_seconds=1.5
    )
    assert subtasks.index("Label vocabulary") < subtasks.index("Timing:")
    assert '  - "pick up the cube"\n  - "put {it} away"' in subtasks
    assert "copied\nverbatim, articles included" in subtasks
    relabel = steered["plan_subtask_relabel"].format(
        segment_index=1, segment_count=3, start=0.0, end=1.0, seed_label="x"
    )
    assert relabel.index('"put {it} away"') < relabel.index("Rules:")
    memory = steered["plan_memory"].format(
        episode_task="t", prior_memory="(none)", completed_subtask="x", remaining_subtasks="(none)"
    )
    assert "under 120 characters" in memory
    assert "  - count {each} pick" in memory
    assert memory.index("under 120 characters") < memory.index("Output strictly valid JSON")


def test_skill_descriptions_follow_their_phrases():
    bundle = load_bundle(RUN_CONFIG)
    descriptions = label_descriptions(bundle.registry, bundle.task)
    assert set(descriptions) <= set(label_vocabulary(bundle.registry, bundle.task))
    steered = steer_prompts(
        TEMPLATES,
        vocabulary=["pick up the cube", "put {it} away"],
        rules=[],
        memory_max_chars=120,
        descriptions={"pick up the cube": "the gripper closes on the {red} cube and lifts it"},
    )
    subtasks = steered["plan_subtasks"].format(
        episode_task="t", observation_block="", episode_duration="3.0", max_steps=8, min_subtask_seconds=1.5
    )
    assert (
        '  - "pick up the cube" (the gripper closes on the {red} cube and lifts it)\n  - "put {it} away"\n'
        in (subtasks)
    )
    assert "copy only the quoted phrase" in subtasks


def test_without_a_vocabulary_only_the_memory_prompt_changes():
    steered = steer_prompts(TEMPLATES, vocabulary=[], rules=[], memory_max_chars=200)
    assert sorted(steered) == ["plan_memory"]


def test_annotate_runs_on_a_copy_and_writes_the_label_map(tmp_path, dataset, fake_lerobot, capsys):
    out, labels = tmp_path / "annotated", tmp_path / "labels.yaml"
    before = first_labels(dataset)
    argv = ["data", "annotate", str(RUN_CONFIG), str(dataset), "--out", str(out), "--labels", str(labels)]
    assert main(argv) == 0

    assert first_labels(dataset) == before  # the source dataset is untouched
    assert first_labels(out)[0] == "pick up the next package"
    video = next((out / "videos").rglob("*.mp4"))
    assert video.is_symlink() and not (out / "meta" / "info.json").is_symlink()

    seen = json.loads((out / "seen_by_annotate.json").read_text())
    assert seen["overrides"] == [
        "LEROBOT_PROMPT_OVERRIDE_plan_memory",
        "LEROBOT_PROMPT_OVERRIDE_plan_subtask_relabel",
        "LEROBOT_PROMPT_OVERRIDE_plan_subtasks",
    ]
    assert '"place the package into bin A"' in seen["plan_subtasks"]
    assert {"--vqa.enabled=false", "--interjections.enabled=false", "--plan.emit_plan=false"} <= set(
        seen["argv"]
    )
    record = json.loads((out / ".embodied_harness" / "annotate.json").read_text())
    assert record["vocabulary"][0] == "pick up the next package"

    printed = capsys.readouterr().out
    assert "5 span(s), 5 distinct label(s), library texts verbatim: 4/5" in printed
    bundle = load_bundle(RUN_CONFIG)
    entries = load_label_map(labels, bundle.registry, bundle.task, allow_unreviewed=True).entries
    assert entries["place the package into bin a"].call.key() == "place(bin=A)"
    assert all(entry.reviewed for label, entry in entries.items() if label != "grab the parcel")
    assert not entries["grab the parcel"].reviewed  # not a library text: a human decides


def test_dry_run_writes_the_prompts_and_runs_nothing(tmp_path, dataset, fake_lerobot, capsys):
    out = tmp_path / "annotated"
    argv = ["data", "annotate", str(RUN_CONFIG), str(dataset), "--out", str(out), "--dry-run"]
    extra = ["--vlm-model", "Qwen/Qwen3-VL-8B-Instruct", "--lerobot-arg=--plan.min_subtask_seconds=2"]
    assert main([*argv, *extra]) == 0
    assert (out / ".embodied_harness" / "prompts" / "plan_subtasks.txt").exists()
    assert not (out / "data").exists()
    command = capsys.readouterr().out.splitlines()[-1]
    assert "--vlm.model_id=Qwen/Qwen3-VL-8B-Instruct" in command
    assert command.endswith("--plan.min_subtask_seconds=2")
    # A dry run leaves only the prompts behind, so the real run can use the same directory.
    assert main(["data", "annotate", str(RUN_CONFIG), str(dataset), "--out", str(out)]) == 0
    assert (out / "data").exists()


def test_annotate_refuses_to_overwrite_and_reports_failures(
    tmp_path, dataset, fake_lerobot, monkeypatch, capsys
):
    out = tmp_path / "annotated"
    (out / "something").mkdir(parents=True)
    assert main(["data", "annotate", str(RUN_CONFIG), str(dataset), "--out", str(out)]) == 2
    assert "is not empty" in capsys.readouterr().err

    monkeypatch.setenv("FAKE_ANNOTATE_EXIT", "3")
    assert main(["data", "annotate", str(RUN_CONFIG), str(dataset), "--out", str(tmp_path / "other")]) == 2
    assert "lerobot-annotate exited with code 3" in capsys.readouterr().err


def test_labels_can_accept_exact_library_texts(tmp_path, dataset, fake_lerobot):
    out = tmp_path / "annotated"
    assert main(["data", "annotate", str(RUN_CONFIG), str(dataset), "--out", str(out)]) == 0
    plain, exact = tmp_path / "plain.yaml", tmp_path / "exact.yaml"
    assert main(["data", "labels", str(RUN_CONFIG), str(out), "--out", str(plain)]) == 0
    assert main(["data", "labels", str(RUN_CONFIG), str(out), "--out", str(exact), "--accept-exact"]) == 0
    bundle = load_bundle(RUN_CONFIG)
    plain_map = load_label_map(plain, bundle.registry, bundle.task, allow_unreviewed=True)
    exact_map = load_label_map(exact, bundle.registry, bundle.task, allow_unreviewed=True)
    assert not any(entry.reviewed for entry in plain_map.entries.values())
    assert sum(entry.reviewed for entry in exact_map.entries.values()) == 4
