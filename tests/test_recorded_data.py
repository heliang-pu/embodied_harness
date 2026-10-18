"""Runtime datasets preserve planner observations separately from actual action labels."""

from __future__ import annotations

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from conftest import make_bundle, run_episode, small_task
from embodied_harness.core.types import SkillCall
from embodied_harness.data.annotations import read_annotated_episodes
from embodied_harness.data.export import export_planner_samples
from embodied_harness.data.from_runs import (
    call_metadata,
    check_episode,
    language_rows,
    load_episode,
    planner_metadata,
    successful_call_steps,
)
from embodied_harness.data.labels import (
    LabelEntry,
    LabelMap,
    candidate_calls,
    count_labels,
    load_label_map,
    normalize_label,
    write_label_file,
)
from embodied_harness.data.mapping import DataMapping
from embodied_harness.data.relabel import relabel_dataset
from embodied_harness.data.steps import episode_steps
from lerobot_fixture import CAMERA_KEYS, FakeFrames
from test_data import bound_planner, read_jsonl, vlm_run
from test_from_runs import NumericWorld, episode


def runtime_dataset(tmp_path, *, outcome=None):
    """Small Arrow dataset with real failed/retried runtime calls, requiring no LeRobot install."""
    bundle = make_bundle(
        tmp_path / "run",
        task=small_task(1),
        run={"trace": {"frames": True}},
        skills={"skills": {"place": {"timeout_s": 6, "on_failure": {"timeout": {"retry": 0}}}}},
    )
    env = NumericWorld(
        {
            "num_packages": 1,
            "barcode_down_prob": 0.0,
            "cameras": ["head", "wrist"],
            "image_size": [64, 48],
            "scripted_outcomes": {
                "pick_next": ["empty_grasp"],
                "place_bin_a": ["timeout"],
                "place_bin_b": ["timeout"],
                "place_bin_c": ["timeout"],
            },
        }
    )
    run = run_episode(bundle, tmp_path / "run", env=env)
    assert run.result.status == "done"
    recorded = load_episode(run.episode_dir)
    if outcome is not None:
        recorded.outcome = outcome
    root, training, labels = recording_dataset(tmp_path, recorded)
    return root, recorded, training, labels


def recording_dataset(tmp_path, recorded):
    """One recorded run as a small Arrow dataset with its sidecar, a training bundle and reviewed labels."""
    root = tmp_path / "dataset"
    data_path = root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    persistent = language_rows(recorded, 10)
    count = len(recorded.frames)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0] * count,
                "frame_index": list(range(count)),
                "timestamp": [i / 10 for i in range(count)],
                "observation.state": [frame["proprio"]["state"] for frame in recorded.frames],
                "language_persistent": [persistent] * count,
            }
        ),
        data_path,
    )
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 10,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    **{key: {"dtype": "video", "shape": [48, 64, 3]} for key in CAMERA_KEYS.values()},
                },
            }
        )
    )
    decisions = planner_metadata(recorded, root, 0)
    sidecar = root / ".embodied_harness/from_runs.json"
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "episodes": [
                    {
                        "episode_index": 0,
                        "goal": recorded.goal,
                        "planner_decisions": decisions,
                        "calls": call_metadata(recorded),
                    }
                ],
            }
        )
    )
    run_config = vlm_run()
    run_config["planner"]["params"].update(show_history=True, show_state=True, show_progress=True)
    training = make_bundle(tmp_path / "training", task=small_task(1, goal_source="episode"), run=run_config)
    labels_path = tmp_path / "labels.yaml"
    write_label_file(
        labels_path,
        count_labels(read_annotated_episodes(root)),
        candidate_calls(training.registry, training.task.skills),
        dataset=root,
        library=training.library.library,
        accept_exact=True,
    )
    labels = load_label_map(labels_path, training.registry, training.task)
    return root, training, labels


def export_runtime(root, bundle, labels, out, **kwargs):
    return export_planner_samples(
        bundle=bundle,
        planner=bound_planner(bundle),
        dataset=root,
        episodes=read_annotated_episodes(root),
        labels=labels,
        mapping=DataMapping.model_validate(
            {
                "instruction_key": "instruction",
                "cameras": CAMERA_KEYS,
                "proprio": {"gripper_width": "state[0]", "gripper_closed": "state[1] > 0.5"},
            }
        ),
        frames=FakeFrames(),
        out_dir=out,
        val_fraction=0,
        **kwargs,
    )


def test_action_language_uses_frame_ownership_even_after_failure_and_dispatch():
    frames = [
        {"t": i / 10, "call_id": call_id, "prompt": prompt}
        for i, (call_id, prompt) in enumerate(
            [
                ("a", "failed pick"),
                ("a", "failed pick"),
                ("a", "failed pick"),
                ("b", None),
                ("c", "retry pick"),
            ]
        )
    ]
    events = [
        {"type": "skill_start", "call_id": "a", "t": 0, "call": {"skill": "pick"}},
        {"type": "skill_end", "call_id": "a", "t": 0.1, "outcome": {"status": "failure"}},
        {
            "type": "skill_start",
            "call_id": "b",
            "t": 0.1,
            "purpose": "before_retry",
            "call": {"skill": "retract"},
        },
        {"type": "skill_start", "call_id": "c", "t": 0.2, "call": {"skill": "pick"}},
    ]
    rows = language_rows(episode(frames, events, {"retract": {"executor": "scripted"}}), 10)
    assert [(row["timestamp"], row["content"]) for row in rows] == [
        (0, "failed pick"),
        (0.3, "retract"),
        (0.4, "retry pick"),
    ]


def test_runtime_reader_preserves_policy_spans_and_dynamic_goal(tmp_path):
    root, recorded, _, labels = runtime_dataset(tmp_path)
    (language,) = read_annotated_episodes(root, instruction_key="instruction")
    (planner,) = read_annotated_episodes(root, instruction_key="instruction", planner=True)
    assert language.instruction == planner.instruction == recorded.goal
    assert all(decision.label is not None and decision.recorded is None for decision in language.decisions)
    assert "retract" in [decision.label for decision in language.decisions]
    assert all(step.call != SkillCall("retract") for step in episode_steps(planner, labels))
    accepted = [event for event in recorded.events if event["type"] == "planner_response"]
    assert len(planner.decisions) == len(accepted) + 1  # exactly one true terminal sample
    assert planner.decisions[-1].done
    assert sum(call["success"] for call in call_metadata(recorded) if call["call"]["skill"] == "place") == 1


def test_runtime_export_keeps_failure_context_and_one_real_terminal(tmp_path):
    root, recorded, bundle, labels = runtime_dataset(tmp_path)
    out = tmp_path / "planner"
    export_runtime(root, bundle, labels, out)
    samples = read_jsonl(out / "samples.jsonl")
    decisions = planner_metadata(recorded, tmp_path / "metadata_check", 0)
    assert len(samples) == len(decisions)
    for sample, decision in zip(samples, decisions, strict=True):
        original = decision["input"]
        assert sample["ledger"] == original["ledger"]
        assert sample["last"] == original["last"]
        assert sample["history"] == original["history"]
        assert sample["reason"] == original["reason"]
        assert sample["memory_in"] == original["memory"]
    failure = next(sample for sample in samples if sample["reason"] == "skill_failed")
    assert failure["last"]["status"] == "failure"
    assert sum(failure["ledger"]["sorted"].values()) == 0
    terminal = samples[-1]
    assert terminal["terminal"] and terminal["target"]["done"]
    assert sum(terminal["ledger"]["sorted"].values()) == 1
    source = json.loads((recorded.directory / "terminal.json").read_text())
    assert terminal["proprio"] == source["proprio"]
    for relative in terminal["images"].values():
        with Image.open(out / relative) as image:
            assert image.format == "JPEG" and image.size == (32, 24)
    assert terminal["images"] != samples[-2]["images"]
    with pytest.raises(ValueError, match="frame_offsets must be"):
        export_runtime(root, bundle, labels, tmp_path / "augmented", frame_offsets=(0, 2))


def test_failed_or_legacy_run_never_gets_a_fabricated_done(tmp_path):
    root, recorded, _, _ = runtime_dataset(tmp_path)
    assert planner_metadata(replace(recorded, outcome={"status": "aborted"}), root, 1)
    assert not any(
        item["target"].get("done")
        for item in planner_metadata(
            replace(recorded, outcome={"status": "aborted"}),
            root,
            1,
        )
    )
    (recorded.directory / "terminal.json").unlink()
    assert not any(item["target"].get("done") for item in planner_metadata(recorded, root, 2))


def test_runtime_library_uses_verified_calls_and_relabel_keeps_all_actions(tmp_path):
    root, recorded, bundle, labels = runtime_dataset(tmp_path)
    steps = successful_call_steps(root)
    assert sum(step.call.skill == "place" for step in steps) == 1
    assert all(step.call.skill != "retract" for step in steps)
    successful = [call for call in call_metadata(recorded) if call["success"] and call["purpose"] == "main"]
    assert [step.duration_s for step in steps] == [call["end_t"] - call["start_t"] for call in successful]
    relabeled = tmp_path / "relabeled"
    relabel_dataset(root, relabeled, labels, bundle.registry)
    assert count_labels(read_annotated_episodes(relabeled)) == count_labels(read_annotated_episodes(root))


def test_recorded_replanning_is_not_merged_and_refuses_label_remapping(tmp_path):
    root, _, _, labels = runtime_dataset(tmp_path)
    (annotated,) = read_annotated_episodes(root, planner=True)
    first = annotated.decisions[0]
    doubled = replace(annotated, decisions=(first, replace(first, index=1)))
    assert len(episode_steps(doubled, labels)) == 2
    bad = dict(labels.entries)
    key = normalize_label(first.label)
    bad[key] = LabelEntry(key, 1, SkillCall("flip"), True)
    with pytest.raises(ValueError, match="recorded planner call"):
        episode_steps(doubled, LabelMap(bad))
    bad[key] = LabelEntry(key, 1, None, True)
    assert episode_steps(doubled, LabelMap(bad)) == []


def test_from_runs_refuses_resumed_video_traces():
    trace = episode([], [{"type": "episode_resume"}])
    with pytest.raises(ValueError, match="resumed traces cannot be imported"):
        check_episode(trace, "state")


def test_undispatched_planner_call_uses_snapshot_prompt(tmp_path):
    root, recorded, _, _ = runtime_dataset(tmp_path)
    call_id = "pc_9999"
    decision = {"skill": "place", "args": {"bin": "C"}, "memory": "accepted before interruption"}
    first = next(event for event in recorded.events if event["type"] == "planner_response")
    original = json.loads((recorded.directory / "planner" / f"{first['call_id']}.json").read_text())
    original.update(call_id=call_id, decision=decision, attachments=[])
    (recorded.directory / "planner" / f"{call_id}.json").write_text(json.dumps(original))
    recorded.events.extend(
        [
            {"type": "planner_request", "call_id": call_id, "t": 0.5},
            {"type": "planner_response", "call_id": call_id, "t": 0.5, "decision": decision, "errors": []},
        ]
    )
    # Only bin B is actually dispatched in this deterministic run. Bin C must
    # resolve from the embedded snapshot even when its original config is gone.
    recorded.snapshot["config"] = "/does/not/exist/run.yaml"
    metadata = planner_metadata(recorded, root, 1)
    undelivered = next(row for row in metadata if row["call_id"] == call_id)
    assert undelivered["label"] == "place the package into bin C"


class EndsDuring(NumericWorld):
    """Ends the episode itself, with `success`, once `actions` actions whose prompt (the mock policy sends
    the library id) contains `text` went out: a simulator that judges the task done (gym-aloha: the peg
    is in) before any verifier fires."""

    def __init__(self, text, success=True, actions=2):
        self.text, self.success, self.limit, self.sent = text, success, actions, 0
        super().__init__(
            {
                "num_packages": 1,
                "barcode_down_prob": 0.0,
                "cameras": ["head", "wrist"],
                "image_size": [64, 48],
            }
        )

    def act(self, action):
        if self.text in ((action or {}).get("prompt") or ""):
            self.sent += 1
        return super().act(action)

    def observe(self):
        obs = super().observe()
        if self.sent >= self.limit:
            obs.terminated = True
            obs.info["success"] = self.success
        return obs


def env_ended_run(tmp_path, text, success=True):
    bundle = make_bundle(tmp_path / "run", task=small_task(1), run={"trace": {"frames": True}})
    run = run_episode(bundle, tmp_path / "run", env=EndsDuring(text, success))
    assert run.result.status == "ended" and run.result.env_success is success
    return bundle, load_episode(run.episode_dir)


def test_env_success_during_a_call_counts_that_call_as_succeeded(tmp_path):
    # The env ends the episode with success while the place runs, so no verifier commits the place. The
    # done decision used to see the ledger without it (nothing sorted, done_when false) and the pick as
    # the last result.
    bundle, recorded = env_ended_run(tmp_path, "place")
    (env_end,) = [event for event in recorded.events if event["type"] == "env_end"]
    placing = next(event for event in reversed(recorded.events) if event["type"] == "skill_start")
    assert env_end["active"] == SkillCall(placing["call"]["skill"], placing["call"]["args"]).key()
    terminal = planner_metadata(recorded, tmp_path / "meta", 0)[-1]
    assert terminal["terminal"] and terminal["target"]["done"]
    seen = terminal["input"]
    assert seen["last"] == {"call": env_end["active"], "status": "success", "reason": None}
    assert seen["history"][-1] == placing["call"]
    assert sum(seen["ledger"]["sorted"].values()) == 1
    assert bundle.registry.is_done(seen["ledger"], {"expected_total": bundle.task.expected_total}) is True
    root, training, labels = recording_dataset(tmp_path, recorded)
    manifest = export_runtime(root, training, labels, tmp_path / "planner")
    exported = read_jsonl(tmp_path / "planner" / "samples.jsonl")[-1]
    assert exported["terminal"] and exported["target"]["done"] and exported["last"] == seen["last"]
    assert manifest["unfinished_episodes"] == []

    # A call a verdict already ended is not the env's to count: here the place failed first.
    ended = {**placing, "type": "skill_end", "outcome": {"status": "failure", "reason": "timeout"}}
    recorded.events.insert(recorded.events.index(env_end), ended)
    seen = planner_metadata(recorded, tmp_path / "meta_failed", 0)[-1]["input"]
    assert seen["last"]["status"] == "failure" and sum(seen["ledger"]["sorted"].values()) == 0


@pytest.mark.parametrize("success", [False, None])
def test_env_end_without_a_success_verdict_gives_no_done(tmp_path, success):
    _, recorded = env_ended_run(tmp_path, "place", success)
    assert not any(item["target"].get("done") for item in planner_metadata(recorded, tmp_path / "meta", 0))
    # The env's own verdict in the trace decides, not an outcome that claims more.
    recorded.outcome = {**recorded.outcome, "env_success": True}
    decisions = planner_metadata(recorded, tmp_path / "meta_claimed", 0)
    assert not any(item["target"].get("done") for item in decisions)


def test_env_success_the_ledger_contradicts_gives_no_done_sample(tmp_path):
    # The env ends the episode with success during the pick: even counting the pick, nothing is sorted,
    # so task.done_when is false and the supervisor would refuse a done there.
    _, recorded = env_ended_run(tmp_path, "pick")
    terminal = planner_metadata(recorded, tmp_path / "meta", 0)[-1]
    assert terminal["target"]["done"] and terminal["input"]["last"]["call"] == "pick"
    root, training, labels = recording_dataset(tmp_path, recorded)
    manifest = export_runtime(root, training, labels, tmp_path / "planner")
    samples = read_jsonl(tmp_path / "planner" / "samples.jsonl")
    assert samples and not any(sample["target"].get("done") for sample in samples)
    assert manifest["unfinished_episodes"] == [0]


@pytest.mark.parametrize("call", ["boundary", "periodic"])
def test_a_planner_done_keeps_the_input_it_was_given(tmp_path, call):
    # Without done_when the planner's own done ends the episode. The terminal sample used to read the
    # state after that answer: the answer's memory as the input memory and, while a skill ran, the
    # preemption the answer itself caused as the last result.
    patch = {"trace": {"frames": True}}
    if call == "periodic":
        patch["planner"] = {"call": "periodic", "every_steps": 1}
    bundle = make_bundle(tmp_path / "run", task=small_task(2, done_when=None), run=patch)
    params = {
        "num_packages": 2,
        "barcode_down_prob": 0.0,
        "cameras": ["head", "wrist"],
        "image_size": [64, 48],
    }
    run = run_episode(bundle, tmp_path / "run", env=NumericWorld(params))
    assert run.result.status == "done"
    recorded = load_episode(run.episode_dir)
    answers = [event for event in recorded.events if event["type"] == "planner_response"]
    assert answers[-1]["decision"]["done"]
    live = json.loads((recorded.directory / "planner" / f"{answers[-1]['call_id']}.json").read_text())
    terminal = planner_metadata(recorded, tmp_path / "meta", 0)[-1]
    assert terminal["terminal"]
    assert terminal["target"] == {"done": True, "memory": live["decision"]["memory"]}
    assert terminal["input"]["memory"] != live["decision"]["memory"]
    for key in ("memory", "reason", "history", "ledger", "last"):
        assert terminal["input"][key] == live["input"][key]
