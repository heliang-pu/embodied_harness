import pytest

from embodied_harness.config.schema import LedgerField
from embodied_harness.core.ledger import Ledger, LedgerError, LedgerOp

FIELDS = {
    "sorted": LedgerField(type="counter", keys=["A", "B"]),
    "count": LedgerField(type="int"),
    "holding": LedgerField(type="bool"),
    "mode": LedgerField(type="enum", values=["fast", "slow"], init="fast"),
}


def test_initial_values():
    assert Ledger(FIELDS).view() == {"sorted": {"A": 0, "B": 0}, "count": 0, "holding": False, "mode": "fast"}


def test_counter_and_int_ops():
    ledger = Ledger(FIELDS)
    ledger.apply(LedgerOp("add", "sorted", "A", 2))
    ledger.apply(LedgerOp("add", "count", None, 1))
    ledger.apply(LedgerOp("set", "count", None, 5))
    assert ledger.view()["sorted"] == {"A": 2, "B": 0}
    assert ledger.view()["count"] == 5


@pytest.mark.parametrize(
    "op",
    [
        LedgerOp("add", "sorted", "Z", 1),
        LedgerOp("add", "sorted", "A", 1.5),
        LedgerOp("add", "holding", None, 1),
        LedgerOp("set", "holding", None, "yes"),
        LedgerOp("set", "mode", None, "medium"),
        LedgerOp("set", "count", "A", 1),
        LedgerOp("set", "missing", None, 1),
    ],
)
def test_rejects_invalid_ops(op):
    with pytest.raises(LedgerError):
        Ledger(FIELDS).apply(op)


def test_view_is_a_copy():
    ledger = Ledger(FIELDS)
    ledger.view()["sorted"]["A"] = 99
    assert ledger.view()["sorted"]["A"] == 0


def test_replay_rebuilds_the_state():
    ops = [
        LedgerOp("add", "sorted", "B", 1),
        LedgerOp("add", "sorted", "B", 1),
        LedgerOp("set", "holding", None, True),
    ]
    ledger = Ledger.replay(FIELDS, ops)
    assert ledger.view()["sorted"] == {"A": 0, "B": 2}
    assert ledger.view()["holding"] is True
