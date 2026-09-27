import pytest

from fmc3_harness.core.types import ActionChunk
from fmc3_harness.executor.chunk_queue import ChunkQueue


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
