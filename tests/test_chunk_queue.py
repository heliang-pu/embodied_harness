import json

import pytest

from conftest import make_bundle, of_type, read_events, run_episode, small_task
from embodied_harness.core.types import ActionChunk
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.policy.executor.chunk_queue import ChunkQueue
from embodied_harness.policy.executor.mock import MockPolicyExecutor


def chunk(version, *actions):
    return ActionChunk(version, list(actions))


def drain(queue):
    out = []
    while (item := queue.pop()) is not None:
        out.append((item.version, item.action))
    return out


def test_finish_chunk_runs_the_current_chunk_to_its_end():
    queue = ChunkQueue("finish_chunk")
    queue.switch(1)
    queue.push(chunk(1, "a1", "a2", "a3"))
    queue.push(chunk(1, "b1", "b2"))
    assert queue.pop().action == "a1"
    queue.switch(2)
    assert queue.pending() == 0
    queue.push(chunk(2, "c1"))
    assert drain(queue) == [(1, "a2"), (1, "a3"), (2, "c1")]


def test_hard_handoff_drops_old_actions():
    queue = ChunkQueue("hard")
    queue.switch(1)
    queue.push(chunk(1, "a1", "a2"))
    queue.pop()
    queue.switch(2)
    queue.push(chunk(2, "c1"))
    assert drain(queue) == [(2, "c1")]


def test_stale_chunks_are_dropped():
    queue = ChunkQueue()
    queue.switch(2)
    assert queue.push(chunk(1, "old")) is False
    assert len(queue) == 0


def test_pending_counts_one_version():
    queue = ChunkQueue()
    queue.switch(1)
    queue.push(chunk(1, "a", "b"))
    queue.switch(2)
    queue.push(chunk(2, "c"))
    assert queue.pending(1) == 2
    assert queue.pending() == 1
    assert len(queue) == 3


def test_rejects_unknown_handoff():
    with pytest.raises(ValueError):
        ChunkQueue("rtc")


def test_finishing_counts_the_rest_of_the_previous_instructions_chunk():
    queue = ChunkQueue("finish_chunk")
    queue.switch(1)
    queue.push(chunk(1, "a1", "a2", "a3"))
    queue.pop()
    assert queue.finishing() == 0
    queue.switch(2)
    assert (queue.finishing(), queue.pending()) == (2, 0)
    queue.pop()
    queue.pop()
    assert queue.finishing() == 0  # the new instruction's first chunk is due now
    hard = ChunkQueue("hard")
    hard.switch(1)
    hard.push(chunk(1, "a1", "a2"))
    hard.pop()
    hard.switch(2)
    assert hard.finishing() == 0


class TickedWorld(MockSortingWorld):
    """Counts ticks (one observe per tick) and records on which tick each version first acted."""

    def __init__(self, params):
        super().__init__(params)
        self.tick = 0
        self.first_act = {}

    def observe(self):
        self.tick += 1
        return super().observe()

    def act(self, action):
        self.first_act.setdefault(action["version"], self.tick)
        return super().act(action)


class TickedPolicy(MockPolicyExecutor):
    def __init__(self, library, world):
        super().__init__(library, chunk_size=20)
        self.world = world
        self.first_infer = {}

    def infer_chunk(self, obs):
        self.first_infer.setdefault(self._version, self.world.tick)
        return super().infer_chunk(obs)


def run_with(tmp_path, handoff):
    bundle = make_bundle(tmp_path / handoff, task=small_task(2), run={"executor": {"handoff": handoff}})
    world = TickedWorld({"num_packages": 2, "barcode_down_prob": 0.0})
    policy = TickedPolicy(bundle.library, world)
    run = run_episode(bundle, tmp_path / handoff, env=world, policy=policy)
    return run, world, policy


def test_finish_chunk_infers_a_calls_first_chunk_on_the_tick_it_starts(tmp_path):
    run, world, policy = run_with(tmp_path, "finish_chunk")
    assert run.result.status == "done"
    dispatched = {e["version"]: round(e["t"] * 10) + 1 for e in of_type(read_events(run), "skill_start")}
    # Some calls waited for the previous chunk's rest: those used to run a chunk inferred ticks earlier.
    assert any(world.first_act[version] > tick for version, tick in dispatched.items())
    # Inferred from the observation of the tick its first action went out on, never earlier.
    assert policy.first_infer == {version: world.first_act[version] for version in dispatched}


def test_finish_chunk_judges_a_call_from_its_own_first_action(tmp_path):
    # The verifier (timeout, min_duration_s, hold_s) starts with the call's own motion, so a call takes
    # as long by its verdict as with hard handoff, where nothing runs before it.
    def elapsed(handoff):
        run, _, _ = run_with(tmp_path, handoff)
        ends = of_type(read_events(run), "skill_end")
        return [(e["call"]["skill"], e["outcome"]["evidence"]["elapsed_s"]) for e in ends]

    assert elapsed("finish_chunk") == elapsed("hard")


class CountingWorld(TickedWorld):
    """Its saved state includes how many actions it has executed."""

    def __init__(self, params):
        super().__init__(params)
        self.acted = 0

    def act(self, action):
        self.acted += 1
        return super().act(action)

    def snapshot(self):
        return {**super().snapshot(), "acted": self.acted}

    def restore(self, state):
        super().restore(state)
        self.acted = state["acted"]


def test_finish_chunk_snapshots_a_call_where_it_starts(tmp_path):
    # trace.snapshots is the start state `probe` reruns a call from: after the previous chunk's rest.
    run_patch = {"executor": {"handoff": "finish_chunk"}, "trace": {"snapshots": True}}
    bundle = make_bundle(tmp_path, task=small_task(2), run=run_patch)
    world = CountingWorld({"num_packages": 2, "barcode_down_prob": 0.0})
    run = run_episode(bundle, tmp_path, env=world, policy=TickedPolicy(bundle.library, world))
    lines = (run.episode_dir / "actions.jsonl").read_text().splitlines()
    versions = [json.loads(line)["version"] for line in lines]
    for start in of_type(read_events(run), "skill_start"):
        snapshot = json.loads((run.episode_dir / "snapshots" / f"{start['call_id']}.json").read_text())
        assert snapshot["acted"] == versions.index(start["version"])  # every action before its own first
