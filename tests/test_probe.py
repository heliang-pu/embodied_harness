"""`embodied-harness probe`: rerun skill calls from the start states a run recorded with trace.snapshots."""

import json

import pytest

from conftest import REPLACE, make_bundle, run_episode, small_task
from embodied_harness.cli import main
from embodied_harness.core.types import SkillCall
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.runtime.probe import format_probe, load_start_states, probe, summarize

# The second pick closes on nothing once; the recovery table retracts and retries it.
WORLD = {
    "num_packages": 3,
    "barcode_down_prob": 0.0,
    "scripted_outcomes": {"pick_next": ["success", "empty_grasp"]},
}


def recorded(tmp_path, snapshots=True):
    bundle = make_bundle(
        tmp_path,
        task=small_task(3),
        run={"env": {"params": {REPLACE: True, **WORLD}}, "trace": {"snapshots": snapshots}},
    )
    run = run_episode(bundle, tmp_path, world=WORLD)
    assert run.result.status == "done"
    return bundle, run


def test_every_call_start_is_kept(tmp_path):
    _, run = recorded(tmp_path)
    events = [json.loads(line) for line in (run.episode_dir / "events.jsonl").read_text().splitlines()]
    starts = [e["call_id"] for e in events if e["type"] == "skill_start"]
    assert sorted(p.stem for p in (run.episode_dir / "snapshots").glob("*.json")) == sorted(starts)
    first = json.loads((run.episode_dir / "snapshots" / "sc_0001.json").read_text())
    assert all(package["location"] == "table" for package in first["packages"])  # before anything moved


def test_the_same_call_from_a_clean_and_a_chained_start(tmp_path):
    bundle, run = recorded(tmp_path)
    clean = load_start_states(f"{run.episode_dir}:sc_0001", bundle)
    chained = load_start_states(f"{run.episode_dir}:sc_0003", bundle)
    assert [s.call.key() for s in clean + chained] == ["pick", "pick"]
    assert chained[0].previous.startswith("place(") and clean[0].previous == "START"
    assert sum(chained[0].ledger["sorted"].values()) == 1  # the first place had been counted
    rows = summarize(probe(bundle, clean + chained, trials=3, env=MockSortingWorld(WORLD)))
    assert [(row.start.split(":")[1], row.trials, row.success, row.rate) for row in rows] == [
        ("sc_0001", 3, 3, 1.0),
        ("sc_0003", 3, 0, 0.0),
    ]
    assert rows[1].reasons == {"empty_grasp": 3}
    table = format_probe(rows)
    assert (
        "| ep_0001:sc_0003 | place(bin=" in table
        and "| pick | 3 | 0 | 3 | 0% | - | empty_grasp x3 |" in table
    )


def test_other_calls_from_one_start(tmp_path):
    bundle, run = recorded(tmp_path)
    holding = load_start_states(f"{run.episode_dir}:sc_0002", bundle)  # where the first place started
    empty = load_start_states(f"{run.episode_dir}:sc_0001", bundle)
    calls = [SkillCall("place", {"bin": "B"}), SkillCall("flip")]
    rows = summarize(probe(bundle, holding + empty, calls, trials=2, env=MockSortingWorld(WORLD)))
    by_key = {(row.start.split(":")[1], row.call): row for row in rows}
    assert by_key[("sc_0002", "place(bin=B)")].rate == 1.0
    assert by_key[("sc_0002", "flip")].rate == 1.0
    not_ready = by_key[("sc_0001", "place(bin=B)")]
    assert not_ready.not_ready == 2 and not_ready.rate is None
    assert not_ready.reasons == {"not ready: holding": 2}


def test_a_whole_episode_probes_every_main_call(tmp_path):
    bundle, run = recorded(tmp_path)
    starts = load_start_states(str(run.episode_dir), bundle)
    assert [s.call_id for s in starts] == [
        "sc_0001",
        "sc_0002",
        "sc_0003",
        "sc_0005",
        "sc_0006",
        "sc_0007",
        "sc_0008",
    ]


def test_probing_needs_snapshots_and_valid_calls(tmp_path):
    bundle, run = recorded(tmp_path, snapshots=False)
    with pytest.raises(FileNotFoundError, match=r"trace\.snapshots: true"):
        load_start_states(f"{run.episode_dir}:sc_0001", bundle)
    with pytest.raises(ValueError, match="no skill call sc_0099"):
        load_start_states(f"{run.episode_dir}:sc_0099", bundle)
    with pytest.raises(ValueError, match="bin='D' is not one of"):
        probe(bundle, [], [SkillCall("place", {"bin": "D"})], env=MockSortingWorld(WORLD))


def test_the_command(tmp_path, capsys):
    bundle, run = recorded(tmp_path)
    out = tmp_path / "probe.json"
    code = main(
        [
            "probe",
            str(bundle.path),
            "--at",
            f"{run.episode_dir}:sc_0001",
            "--at",
            f"{run.episode_dir}:sc_0003",
            "--trials",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "| ep_0001:sc_0001 | START | pick | 2 | 2 | 0 | 100% |" in printed
    data = json.loads(out.read_text())
    assert len(data["trials"]) == 4 and data["rows"][1]["success"] == 0
    assert main(["probe", str(bundle.path), "--at", f"{run.episode_dir}:sc_0001", "--call", "jump"]) == 2


def test_probe_uses_the_episode_instruction_for_task_conditions(tmp_path):
    bundle = make_bundle(
        tmp_path,
        task=small_task(1, goal_source="episode"),
        skills={"skills": {"pick": {"preconditions": ["task.goal == 'sort this episode'"]}}},
        run={"trace": {"snapshots": True}},
    )

    class InstructedWorld(MockSortingWorld):
        def instruction(self):
            return "sort this episode"

    params = {"num_packages": 1, "barcode_down_prob": 0.0}
    run = run_episode(bundle, tmp_path, env=InstructedWorld(params))
    start = load_start_states(f"{run.episode_dir}:sc_0001", bundle)[0]
    trial = probe(bundle, [start], trials=1, env=InstructedWorld(params))[0]
    assert trial.status == "success"
