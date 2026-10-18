import numpy as np
import pytest

from conftest import RUN_CONFIG, make_bundle, small_task
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import OutcomeStatus
from embodied_harness.runtime.verifier import VerifierSession


def namespace(**values):
    base = {
        "gripper_width": 0.09,
        "gripper_closed": False,
        "holding": False,
        "empty_grasp": False,
        "released": True,
        "executor_done": False,
        "ledger": {},
        "task": {},
        "args": {},
    }
    base.update(values)
    return base


HOLDING = namespace(holding=True, gripper_width=0.05, gripper_closed=True, released=False)


@pytest.fixture(scope="module")
def registry():
    return load_bundle(RUN_CONFIG).registry


def session(registry, name):
    return VerifierSession(registry.skill(name), started_at=0.0, timeout_s=registry.timeout_for(name))


def test_success_needs_the_condition_to_hold(registry):
    verifier = session(registry, "pick")
    assert verifier.evaluate(3.0, HOLDING, {}) is None
    assert verifier.evaluate(3.4, HOLDING, {}) is None
    outcome = verifier.evaluate(3.5, HOLDING, {})
    assert outcome.status is OutcomeStatus.SUCCESS
    assert outcome.evidence["condition"] == "holding"


def test_a_flicker_restarts_the_hold_timer(registry):
    verifier = session(registry, "pick")
    assert verifier.evaluate(1.0, HOLDING, {}) is None
    assert verifier.evaluate(1.2, namespace(), {}) is None
    assert verifier.evaluate(1.6, HOLDING, {}) is None
    assert verifier.evaluate(2.0, HOLDING, {}) is None
    assert verifier.evaluate(2.1, HOLDING, {}).ok


def test_failure_reason_comes_from_the_predicate(registry):
    verifier = session(registry, "pick")
    empty = namespace(empty_grasp=True, gripper_closed=True, gripper_width=0.0, released=False)
    assert verifier.evaluate(3.0, empty, {}) is None
    outcome = verifier.evaluate(3.3, empty, {})
    assert outcome.status is OutcomeStatus.FAILURE
    assert outcome.reason == "empty_grasp"


def test_failure_beats_success(tmp_path):
    bundle = make_bundle(
        tmp_path,
        task=small_task(3),
        skills={
            "skills": {
                "pick": {
                    "verify": {
                        "success": [{"when": "holding"}],
                        "failure": [{"when": "gripper_closed", "reason": "closed"}],
                    }
                }
            }
        },
    )
    verifier = VerifierSession(bundle.registry.skill("pick"), started_at=0.0, timeout_s=10.0)
    outcome = verifier.evaluate(1.0, HOLDING, {})
    assert outcome.status is OutcomeStatus.FAILURE and outcome.reason == "closed"


def test_min_duration_delays_success(registry):
    verifier = session(registry, "flip")
    assert verifier.evaluate(1.0, HOLDING, {}) is None
    assert verifier.evaluate(3.9, HOLDING, {}) is None
    assert verifier.evaluate(4.0, HOLDING, {}).ok


def test_timeout(registry):
    verifier = session(registry, "pick")
    assert verifier.evaluate(6.9, namespace(), {}) is None
    assert verifier.evaluate(7.0, namespace(), {}).reason == "timeout"


def test_scripted_primitive_succeeds_when_the_executor_is_done(registry):
    verifier = session(registry, "retract")
    assert verifier.evaluate(0.3, namespace(), {}) is None
    assert verifier.evaluate(0.5, namespace(executor_done=True), {}).ok


def test_stall(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3), skills={"skills": {"pick": {"stall_s": 1.0}}})
    verifier = VerifierSession(bundle.registry.skill("pick"), started_at=0.0, timeout_s=10.0)
    still, moved = {"gripper_width": 0.09}, {"gripper_width": 0.05}
    assert verifier.evaluate(0.0, namespace(), still) is None
    assert verifier.evaluate(0.5, namespace(), still) is None
    assert verifier.evaluate(0.9, namespace(), moved) is None  # the change restarts the stall timer
    assert verifier.evaluate(1.8, namespace(), moved) is None
    assert verifier.evaluate(1.9, namespace(), moved).reason == "stall"


def test_stall_accepts_unchanged_numpy_state_vectors(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3), skills={"skills": {"pick": {"stall_s": 1.0}}})
    verifier = VerifierSession(bundle.registry.skill("pick"), started_at=0.0, timeout_s=10.0)

    def observation():
        return {"gripper_width": 0.09, "state": np.zeros(8, dtype=np.float32)}

    assert verifier.evaluate(0.0, namespace(), observation()) is None
    assert verifier.evaluate(0.9, namespace(), observation()) is None
    assert verifier.evaluate(1.0, namespace(), observation()).reason == "stall"


def test_a_numpy_state_vector_change_restarts_the_stall_timer(tmp_path):
    bundle = make_bundle(tmp_path, task=small_task(3), skills={"skills": {"pick": {"stall_s": 1.0}}})
    verifier = VerifierSession(bundle.registry.skill("pick"), started_at=0.0, timeout_s=10.0)
    still = {"gripper_width": 0.09, "state": np.zeros(8, dtype=np.float32)}
    moved_state = np.zeros(8, dtype=np.float32)
    moved_state[3] = 0.25
    moved = {"gripper_width": 0.09, "state": moved_state}

    assert verifier.evaluate(0.0, namespace(), still) is None
    assert verifier.evaluate(0.5, namespace(), still) is None
    assert verifier.evaluate(0.9, namespace(), moved) is None
    assert verifier.evaluate(1.8, namespace(), moved) is None
    assert verifier.evaluate(1.9, namespace(), moved).reason == "stall"
