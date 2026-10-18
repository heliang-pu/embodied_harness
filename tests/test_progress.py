"""trace/progress.py: how far along a fixed plan an episode got, from its trace alone."""

from embodied_harness.runtime.trace.progress import plan_progress, reference_plan

PLAN = ["pick", "flip", "place(bin=A)", "pick", "flip", "place(bin=B)"]


class Trace:
    """Builds events the way the supervisor writes them."""

    def __init__(self, exact=True):
        self.events = []
        self.exact = exact
        self.calls = 0
        self.planner_call = None

    def propose(self, index, call):
        self.calls += 1
        self.planner_call = f"pc{self.calls}"
        meta = {"planner": "sequence", "sequence_index": index} if self.exact else {"planner": "vlm"}
        self.events.append(
            {"type": "planner_response", "call_id": self.planner_call, "meta": meta, "errors": []}
        )
        return self

    def run(self, call, status, reason=None, *, purpose="main"):
        skill, _, rest = call.partition("(")
        args = dict(item.split("=") for item in rest.rstrip(")").split(", ")) if rest else {}
        call_id = f"sc{len(self.events)}"
        self.events.append(
            {
                "type": "skill_start",
                "call_id": call_id,
                "planner_call_id": self.planner_call,
                "purpose": purpose,
                "call": {"skill": skill, "args": args},
            }
        )
        if status is not None:
            self.events.append(
                {
                    "type": "skill_end",
                    "call_id": call_id,
                    "purpose": purpose,
                    "outcome": {"status": status, "reason": reason},
                }
            )
        return self

    def step(self, index, status="success", reason=None):
        return self.propose(index, PLAN[index]).run(PLAN[index], status, reason)


def test_a_clean_run_completes_every_step():
    trace = Trace()
    for index in range(6):
        trace.step(index)
    progress = plan_progress(PLAN, trace.events, {"status": "done"})
    assert (progress.completed, progress.furthest, progress.first_failure, progress.rollbacks) == (6, 6, 6, 0)
    assert progress.exact and progress.first_try == [True] * 6 and progress.stopped_at is None


def test_a_rollback_lowers_completed_but_not_furthest():
    trace = Trace().step(0).step(1).step(2).step(3).step(4, "failure", "dropped")
    trace.propose(3, "pick").run("pick", None)  # the redone pick is running when the episode stops
    progress = plan_progress(PLAN, trace.events, {"status": "aborted"})
    assert (progress.completed, progress.furthest, progress.first_failure) == (3, 4, 4)
    assert progress.rollbacks == 1 and progress.rollbacks_from == [0, 0, 0, 0, 1, 0]
    assert progress.failures[4] == {"dropped": 1} and progress.stopped_at == 3
    assert progress.attempts == [1, 1, 1, 2, 1, 0]


def test_local_retries_count_for_the_first_failure():
    trace = Trace().step(0).propose(1, "flip")
    trace.run("flip", "failure", "timeout").run("flip", "success")  # a local retry, same planner call
    for index in range(2, 6):
        trace.step(index)
    progress = plan_progress(PLAN, trace.events, {"status": "done"})
    assert (progress.completed, progress.first_failure) == (6, 1)
    assert progress.first_try[1] is False and progress.attempts[1] == 2


def test_a_human_skip_completes_the_failed_step():
    trace = Trace().step(0).step(1, "failure", "dropped")
    trace.events.append(
        {"type": "human_command", "command": "skip", "failed_call": {"skill": "flip", "args": {}}}
    )
    progress = plan_progress(PLAN, trace.events, {"status": "aborted"})
    assert progress.completed == 2 and progress.skipped[1] == 1


def test_a_retry_of_a_later_step_under_a_rollback_proposal_is_placed_on_that_step():
    trace = Trace().step(0).step(1, "failure", "dropped").propose(0, "pick")  # rollback proposal
    trace.run("flip", "success")  # the human retried the flip under the rollback's planner call
    progress = plan_progress(PLAN, trace.events, {"status": "aborted"})
    assert progress.completed == 2 and progress.attempts[:2] == [1, 2] and progress.off_plan == 0


def test_a_successful_episode_counts_the_whole_plan():
    ended = Trace().step(0).propose(1, "flip").run("flip", None)
    ended.events.append({"type": "env_end", "success": True})
    assert plan_progress(PLAN, ended.events, {"status": "ended", "env_success": True}).completed == 6
    assert plan_progress(PLAN, Trace().step(0).events, {"status": "done"}).completed == 6


def test_a_done_episode_the_env_judged_a_failure_counts_only_the_steps_done():
    # The planner said done after the first pick, and the simulator disagreed: the run failed, as `run` and
    # `report` count it, so the plan is not complete either.
    trace = Trace(exact=False).propose(0, "pick").run("pick", "success")
    progress = plan_progress(PLAN, trace.events, {"status": "done", "env_success": False})
    assert (progress.completed, progress.furthest) == (1, 1)
    assert (progress.first_failure, progress.stopped_at) == (1, 1)
    # Without a verdict from the env (a real robot), finishing as done still completes the plan.
    assert plan_progress(PLAN, trace.events, {"status": "done", "env_success": None}).completed == 6


def test_without_sequence_meta_calls_are_matched_by_key():
    trace = Trace(exact=False)
    trace.propose(0, "pick").run("pick", "success")
    trace.propose(0, "flip").run("flip", "failure", "dropped")
    trace.propose(0, "pick").run("pick", "success")  # a redo of the pick: a rollback
    trace.propose(0, "flip").run("flip", "success")
    trace.propose(0, "place").run("place(bin=C)", "success")  # not in the plan
    progress = plan_progress(PLAN, trace.events, {"status": "aborted"})
    assert not progress.exact
    assert (progress.completed, progress.furthest, progress.rollbacks, progress.off_plan) == (2, 2, 1, 1)


def test_reference_plan_prefers_task_plan_then_sequence_calls():
    task_plan = {"task": {"plan": ["pick", {"skill": "place", "args": {"bin": "A"}}]}}
    assert reference_plan(task_plan) == ["pick", "place(bin=A)"]
    sequence = {
        "task": {"plan": None},
        "run": {"planner": {"type": "sequence", "params": {"calls": [{"skill": "pick"}]}}},
    }
    assert reference_plan(sequence) == ["pick"]
    other = {"task": {}, "run": {"planner": {"type": "mock_oracle", "params": {}}}}
    assert reference_plan(other) is None
