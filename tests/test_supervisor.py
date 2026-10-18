"""End-to-end episodes in the mock world: the main loop, recovery, planner faults, budgets, humans."""

import json
import threading

import pytest

from conftest import make_bundle, of_type, read_events, run_episode, small_task
from embodied_harness.core.types import ActionChunk
from embodied_harness.env.mock_sorting import MockSortingWorld
from embodied_harness.planner.base import PlannerOutput
from embodied_harness.planner.oracle import MockSortingOracle
from embodied_harness.policy.executor.mock import MockPolicyExecutor
from embodied_harness.runtime.commands import ScriptedCommands


def skills_started(events):
    return [event["call"]["skill"] for event in of_type(events, "skill_start")]


class FirstThenOracle:
    """Returns the given raw outputs first, then behaves like the oracle."""

    name = "first_then_oracle"

    def __init__(self, *first: str) -> None:
        self._first = list(first)
        self._oracle = MockSortingOracle()

    def plan(self, context):
        if self._first:
            return PlannerOutput(raw=self._first.pop(0))
        return self._oracle.plan(context)


# --- the main loop ------------------------------------------------------------------------------


def test_happy_path_sorts_everything(bundle, tmp_path):
    env = MockSortingWorld({"num_packages": 3, "barcode_down_prob": 0.0})
    run = run_episode(bundle, tmp_path, env=env)
    result = run.result
    assert result.status == "done"
    assert result.skill_calls == 6  # pick and place per package
    assert result.planner_calls == 6  # the last commit satisfies done_when, so no final planner call
    placed = {name: sum(p.location == name for p in env.packages) for name in "ABC"}
    assert result.ledger["sorted"] == placed
    assert sum(placed.values()) == 3

    events = read_events(run)
    assert events[0]["type"] == "episode_start"
    assert events[-1]["type"] == "episode_end"
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert json.loads((run.episode_dir / "outcome.json").read_text())["status"] == "done"


def test_barcode_down_packages_are_flipped_first(bundle, tmp_path):
    run = run_episode(bundle, tmp_path, world={"barcode_down_prob": 1.0})
    assert skills_started(read_events(run)) == ["pick", "flip", "place"] * 3


def test_memory_goes_back_to_the_planner(bundle, tmp_path):
    seen = []

    class Remembering(MockSortingOracle):
        def plan(self, context):
            seen.append((context.memory, sorted(context.proprio)))
            data = json.loads(super().plan(context).raw)
            data["memory"] = f"call {len(seen)}"
            return PlannerOutput(raw=json.dumps(data))

    run = run_episode(bundle, tmp_path, planner=Remembering())
    assert [memory for memory, _ in seen[:3]] == ["", "call 1", "call 2"]
    assert all(keys == ["gripper_closed", "gripper_width"] for _, keys in seen)
    assert run.result.memory == f"call {len(seen)}"


def test_memory_is_truncated_to_the_limit(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3), run={"planner": {"memory_max_chars": 8}})
    run = run_episode(bundle, tmp_path)
    memories = [event["memory"] for event in of_type(read_events(run), "memory_update")]
    assert memories and all(len(memory) <= 8 for memory in memories)


def test_trace_files_and_log_context(bundle, tmp_path):
    run = run_episode(bundle, tmp_path)
    directory = run.episode_dir
    for name in (
        "events.jsonl",
        "actions.jsonl",
        "episode.log",
        "outcome.json",
        "config.snapshot.yaml",
        "env_snapshot.json",
        "planner/pc_0001.json",
    ):
        assert (directory / name).is_file(), name

    call = json.loads((directory / "planner" / "pc_0001.json").read_text())
    assert call["input"]["memory"] == ""
    assert call["decision"]["skill"] == "pick"
    assert set(call["input"]["proprio"]) == {"gripper_width", "gripper_closed"}

    lines = [json.loads(line) for line in (directory / "episode.log").read_text().splitlines()]
    assert all(line["episode_id"] == "ep_0001" for line in lines)
    starts = [line for line in lines if line["msg"].startswith("start ")]
    assert starts and all(
        line["state"] == "EXECUTE" and line["skill_call_id"].startswith("sc_") for line in starts
    )

    boundaries = of_type(read_events(run), "boundary")
    assert boundaries[0]["previous"] is None
    assert boundaries[1]["previous"] == "pick"


# --- recovery (design doc §13) -------------------------------------------------------------------


def test_empty_grasp_retries_after_retract(bundle, tmp_path):
    run = run_episode(bundle, tmp_path, world={"scripted_outcomes": {"pick_next": ["empty_grasp"]}})
    events = read_events(run)
    starts = [(event["call"]["skill"], event["purpose"]) for event in of_type(events, "skill_start")]
    assert starts[:3] == [("pick", "main"), ("retract", "before_retry"), ("pick", "main")]
    assert of_type(events, "skill_end")[0]["outcome"]["reason"] == "empty_grasp"
    recovery = of_type(events, "recovery")[0]
    assert recovery["action"] == "retry" and recovery["before_retry"] == ["retract"]
    assert [e["current"] for e in of_type(events, "state_change")].count("RETRY") == 1
    assert run.result.status == "done"
    assert run.result.planner_calls == 6  # the local retry never asked the planner
    assert run.result.skill_calls == 8


def test_consecutive_failures_hold_and_abort(bundle, tmp_path):
    run = run_episode(bundle, tmp_path, world={"scripted_outcomes": {"pick_next": ["empty_grasp"] * 3}})
    assert run.result.status == "aborted"
    assert "failed 3 times in a row" in run.result.message
    assert len(of_type(read_events(run), "hold")) == 1


def test_drop_during_flip_goes_back_to_the_planner(bundle, tmp_path):
    run = run_episode(
        bundle, tmp_path, world={"barcode_down_prob": 1.0, "scripted_outcomes": {"flip": ["dropped"]}}
    )
    events = read_events(run)
    assert run.result.status == "done"
    assert skills_started(events)[:4] == ["pick", "flip", "pick", "flip"]
    recovery = of_type(events, "recovery")[0]
    assert recovery["reason"] == "dropped" and recovery["action"] == "planner"
    assert "skill_failed" in [event["reason"] for event in of_type(events, "planner_request")]


def test_place_timeout_retries_once_then_goes_to_the_planner(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(1))
    timeouts = {f"place_bin_{name}": ["timeout", "timeout"] for name in "abc"}
    run = run_episode(bundle, tmp_path, world={"num_packages": 1, "scripted_outcomes": timeouts})
    events = read_events(run)
    assert run.result.status == "done"
    assert [event["action"] for event in of_type(events, "recovery")] == ["retry", "planner"]
    place_reasons = [
        e["outcome"]["reason"] for e in of_type(events, "skill_end") if e["call"]["skill"] == "place"
    ]
    assert place_reasons == ["timeout", "timeout", None]


def test_runtime_exception_holds_with_an_internal_code(bundle, tmp_path):
    class Exploding(MockSortingWorld):
        def act(self, action):
            raise RuntimeError("driver crashed")

    run = run_episode(bundle, tmp_path, env=Exploding({"num_packages": 3, "barcode_down_prob": 0.0}))
    assert run.result.error_code == "E_INTERNAL"
    assert "driver crashed" in of_type(read_events(run), "error")[0]["error"]
    log = (run.episode_dir / "episode.log").read_text()
    assert "driver crashed" in log and "Traceback" in log


def test_executor_underrun_holds(bundle, tmp_path):
    class Silent(MockPolicyExecutor):
        def infer_chunk(self, obs):
            return ActionChunk(self._version, [])

    run = run_episode(bundle, tmp_path, policy=Silent(bundle.library))
    assert run.result.error_code == "E_EXEC_UNDERRUN"


# --- planner faults -----------------------------------------------------------------------------


def test_invalid_output_resamples_then_holds(bundle, tmp_path):
    class Broken:
        name = "broken"

        def plan(self, context):
            return PlannerOutput(raw="place C")

    run = run_episode(bundle, tmp_path, planner=Broken())
    assert run.result.status == "aborted"
    assert run.result.error_code == "E_PLANNER_INVALID"
    assert [event["resample"] for event in of_type(read_events(run), "decision_rejected")] == [0, 1, 2]
    assert run.result.resamples == 3


def test_precondition_violation_is_rejected(bundle, tmp_path):
    planner = FirstThenOracle('{"skill": "place", "args": {"bin": "A"}, "memory": ""}')
    run = run_episode(bundle, tmp_path, planner=planner)
    rejected = of_type(read_events(run), "decision_rejected")
    assert len(rejected) == 1 and "precondition not met" in rejected[0]["errors"][0]
    assert run.result.status == "done"


def test_skill_outside_the_task_is_rejected(bundle, tmp_path):
    run = run_episode(bundle, tmp_path, planner=FirstThenOracle('{"skill": "dance", "memory": ""}'))
    assert "not one of the task's skills" in of_type(read_events(run), "decision_rejected")[0]["errors"][0]
    assert run.result.status == "done"


def test_done_too_early_is_checked_against_the_ledger(bundle, tmp_path):
    run = run_episode(bundle, tmp_path, planner=FirstThenOracle('{"done": true, "memory": "all done?"}'))
    events = read_events(run)
    first_check = of_type(events, "check")[0]
    assert first_check["claimed_by_planner"] is True and first_check["passed"] is False
    assert "check_failed" in [event["reason"] for event in of_type(events, "planner_request")]
    assert run.result.status == "done"


class JudgingWorld(MockSortingWorld):
    """A simulator with its own verdict in info, like RoboMME (success only once its button is pressed)."""

    def __init__(self, params, success):
        super().__init__(params)
        self.success = success

    def observe(self):
        obs = super().observe()
        obs.info = {**obs.info, "success": self.success}
        return obs


@pytest.mark.parametrize("success", [False, True])
def test_a_done_episode_records_the_envs_own_verdict(tmp_path, success):
    # No done_when: the planner's done ends the episode, and only the env can say whether it was right.
    bundle = make_bundle(tmp_path, task=small_task(3, done_when=None))
    env = JudgingWorld({"num_packages": 3, "barcode_down_prob": 0.0}, success)
    run = run_episode(bundle, tmp_path, env=env)
    assert run.result.status == "done" and run.result.env_success is success
    assert json.loads((run.episode_dir / "outcome.json").read_text())["env_success"] is success
    assert of_type(read_events(run), "episode_end")[0]["env_success"] is success


def test_a_done_episode_without_an_env_verdict_has_none(bundle, tmp_path):
    run = run_episode(bundle, tmp_path)  # the mock world, like a real robot, reports no success flag
    assert run.result.status == "done" and run.result.env_success is None


def test_planner_errors_retry_with_backoff_then_hold(bundle, tmp_path):
    class Down:
        name = "down"

        def plan(self, context):
            raise ConnectionError("vLLM unreachable")

    run = run_episode(bundle, tmp_path, planner=Down())
    assert run.result.error_code == "E_PLANNER_TIMEOUT"
    assert len(of_type(read_events(run), "planner_error")) == 4  # the first call and 3 retries


def test_threaded_planner_timeout_holds(tmp_path):
    bundle = make_bundle(
        tmp_path,
        task=small_task(3),
        run={"planner": {"threaded": True, "timeout_s": 0.5, "retries": 1, "retry_backoff_s": 0.1}},
    )
    release = threading.Event()

    class Hung:
        name = "hung"

        def plan(self, context):
            release.wait(5)
            return PlannerOutput(raw="{}")

    try:
        run = run_episode(bundle, tmp_path, planner=Hung())
    finally:
        release.set()
    assert run.result.error_code == "E_PLANNER_TIMEOUT"


def test_threaded_planner_happy_path(tmp_path):
    # Simulated time runs far faster than the planner thread, so give it a generous timeout, and a
    # time budget to match: a busy test process can keep the thread waiting for thousands of ticks.
    bundle = make_bundle(
        tmp_path,
        task=small_task(3, budget={"max_time_s": 1e6}),
        run={"planner": {"threaded": True, "timeout_s": 10000}},
    )
    assert run_episode(bundle, tmp_path).result.status == "done"


# --- budgets and humans -----------------------------------------------------------------------------


def test_skill_call_budget(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3, budget={"max_skill_calls": 2}))
    run = run_episode(bundle, tmp_path)
    assert run.result.error_code == "E_BUDGET"
    assert "skill call budget" in run.result.message


def test_time_budget(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3, budget={"max_time_s": 5}))
    run = run_episode(bundle, tmp_path)
    assert run.result.error_code == "E_BUDGET"
    assert "time budget" in run.result.message


def test_human_skip_commits_the_failed_step(bundle, tmp_path):
    run = run_episode(
        bundle,
        tmp_path,
        world={"scripted_outcomes": {"pick_next": ["empty_grasp"] * 3}},
        commands=ScriptedCommands(["skip"]),
    )
    command = of_type(read_events(run), "human_command")[0]
    assert command["command"] == "skip" and command["failed_call"]["skill"] == "pick"
    assert run.result.human_commands == ["skip"]
    assert run.result.status == "done"


def test_human_retry_runs_the_failed_step_again(bundle, tmp_path):
    run = run_episode(
        bundle,
        tmp_path,
        world={"scripted_outcomes": {"pick_next": ["empty_grasp"] * 3}},
        commands=ScriptedCommands(["retry"]),
    )
    events = read_events(run)
    after_command = [e for e in events if e["seq"] > of_type(events, "human_command")[0]["seq"]]
    assert of_type(after_command, "skill_start")[0]["call"]["skill"] == "pick"
    assert run.result.status == "done"


def test_human_resume_asks_the_planner(bundle, tmp_path):
    run = run_episode(
        bundle,
        tmp_path,
        world={"scripted_outcomes": {"pick_next": ["empty_grasp"] * 3}},
        commands=ScriptedCommands(["resume"]),
    )
    assert "human_resume" in [event["reason"] for event in of_type(read_events(run), "planner_request")]
    assert run.result.status == "done"
