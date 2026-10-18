"""Loop pacing with the real clock: after an overrun the loop skips ticks instead of bursting."""

import pytest

from conftest import RUN_CONFIG
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.clock import next_deadline
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.policy.executor.mock import MockPolicyExecutor
from embodied_harness.runtime.commands import AutoCommands
from embodied_harness.runtime.runner import EpisodeRunner


def test_next_deadline_stays_on_the_grid():
    assert next_deadline(0.0, 0.1, now=0.05) == (0.1, 0)
    deadline, skipped = next_deadline(0.0, 0.1, now=0.35)  # the tick took 3.5 periods
    assert skipped == 3 and deadline == pytest.approx(0.4)


class WallClock:
    """A stand-in for the real clock: time moves where a test says, and sleeping waits for the deadline."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[tuple[float, float]] = []  # (time of the call, deadline)

    def now(self) -> float:
        return self.t

    def sleep_until(self, deadline: float) -> None:
        self.sleeps.append((self.t, deadline))
        self.t = max(self.t, deadline)


class SlowSecondChunk(MockPolicyExecutor):
    def __init__(self, library, clock: WallClock) -> None:
        super().__init__(library, chunk_size=5)
        self.clock = clock
        self.chunks = 0

    def infer_chunk(self, obs):
        self.chunks += 1
        if self.chunks == 2:
            self.clock.t += 0.35  # one slow inference: 3.5 control periods at 10 Hz
        return super().infer_chunk(obs)


def test_a_slow_inference_skips_ticks_instead_of_bursting(tmp_path):
    bundle = load_bundle(RUN_CONFIG)
    clock = WallClock()
    runner = EpisodeRunner(
        bundle,
        run_dir=tmp_path / "runs",
        episode_id="ep_0001",
        seed=0,
        commands=AutoCommands("abort"),
        env=MockSortingWorld({"num_packages": 2, "barcode_down_prob": 0.0}),
        policy=SlowSecondChunk(bundle.library, clock),
        clock=clock,
        max_ticks=40,
    )
    runner.run()
    # Never a deadline in the past: each tick waits for its slot, so no actions go out back to back.
    assert all(deadline >= called for called, deadline in clock.sleeps)
    gaps = [b[1] - a[1] for a, b in zip(clock.sleeps, clock.sleeps[1:], strict=False)]
    assert pytest.approx(0.4) in gaps  # the overrun tick moved the grid on by four periods
    assert all(gap == pytest.approx(0.1) or gap == pytest.approx(0.4) for gap in gaps)
