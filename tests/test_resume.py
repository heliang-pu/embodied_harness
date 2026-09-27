import pytest

from conftest import of_type, read_events, run_episode
from fmc3_harness.trace.reader import load_restored

WORLD = {"num_packages": 3, "barcode_down_prob": 0.0}


def test_resume_finishes_an_interrupted_episode(bundle, tmp_path):
    first = run_episode(bundle, tmp_path, world=WORLD, max_ticks=100)  # stops about 10 s in
    assert first.result is None
    assert not (first.episode_dir / "outcome.json").exists()
    before = read_events(first)
    sorted_before = sum(op["value"] for event in of_type(before, "ledger_update") for op in event["ops"])
    assert 0 < sorted_before < 3

    second = run_episode(bundle, tmp_path, world=WORLD, resume=True)
    assert second.result.status == "done"
    assert sum(second.result.ledger["sorted"].values()) == 3

    events = read_events(second)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    times = [event["t"] for event in events]
    assert times == sorted(times)  # the resumed part continues the original timeline
    assert len(of_type(events, "episode_resume")) == 1
    requests = of_type(events, "planner_request")
    assert len({event["call_id"] for event in requests}) == len(requests)
    assert [event["reason"] for event in requests].count("resume") == 1


def test_resume_refuses_a_finished_episode(bundle, tmp_path):
    run_episode(bundle, tmp_path, world=WORLD)
    with pytest.raises(RuntimeError, match="already finished"):
        run_episode(bundle, tmp_path, world=WORLD, resume=True)


def test_a_new_run_refuses_an_existing_trace(bundle, tmp_path):
    run_episode(bundle, tmp_path, world=WORLD, max_ticks=10)
    with pytest.raises(FileExistsError):
        run_episode(bundle, tmp_path, world=WORLD)


def test_a_torn_last_line_is_ignored(tmp_path):
    directory = tmp_path / "ep"
    directory.mkdir()
    (directory / "events.jsonl").write_text(
        '{"seq": 1, "t": 0.0, "type": "memory_update", "memory": "x"}\n{"seq": 2, "t": 0.1, "ty',
        encoding="utf-8",
    )
    restored = load_restored(directory)
    assert restored.memory == "x"
    assert restored.last_seq == 1
