import pytest

from embodied_harness.config.expressions import ExpressionError, compile_expression, compile_statement
from embodied_harness.core.ledger import LedgerOp


@pytest.mark.parametrize(
    ("source", "namespace", "expected"),
    [
        ("0.01 < w < 0.08", {"w": 0.05}, True),
        ("0.01 < w < 0.08", {"w": 0.09}, False),
        ("closed and w <= 0.01", {"closed": True, "w": 0.0}, True),
        ("not holding", {"holding": False}, True),
        (
            "sum(ledger.sorted) == task.expected_total",
            {"ledger": {"sorted": {"A": 1, "B": 2}}, "task": {"expected_total": 3}},
            True,
        ),
        ("ledger.sorted[args.bin] + 1", {"ledger": {"sorted": {"C": 4}}, "args": {"bin": "C"}}, 5),
        ("len(items) if flag else -1", {"items": [1, 2], "flag": True}, 2),
        ("x in ['A', 'B']", {"x": "B"}, True),
        ("true and not false", {}, True),
        ("a or b", {"a": 0, "b": "fallback"}, "fallback"),
    ],
)
def test_evaluates_allowed_expressions(source, namespace, expected):
    assert compile_expression(source)(namespace) == expected


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os')",
        "open('x')",
        "(lambda: 1)()",
        "ledger.__class__",
        "[x for x in items]",
        "x.y()",
        "(y := 1)",
        "sum(items, start=1)",
        "b'bytes'",
        "((",
    ],
)
def test_rejects_everything_else(source):
    with pytest.raises(ExpressionError):
        compile_expression(source)


def test_runtime_errors_name_the_source():
    with pytest.raises(ExpressionError, match=r"in '1 / x'"):
        compile_expression("1 / x")({"x": 0})


def test_unknown_names_fail_at_runtime():
    with pytest.raises(ExpressionError, match="unknown name 'missing'"):
        compile_expression("missing > 1")({})


def test_attribute_access_only_reads_namespaces():
    with pytest.raises(ExpressionError, match="cannot read"):
        compile_expression("x.real")({"x": 1})


def test_names_and_attribute_refs():
    expr = compile_expression("holding and sum(ledger.sorted) < task.expected_total")
    assert expr.names == {"holding", "ledger", "task"}
    assert expr.attribute_refs() == {("ledger", "sorted"), ("task", "expected_total")}


@pytest.mark.parametrize(
    ("source", "namespace", "expected"),
    [
        ("ledger.count = 3", {}, LedgerOp("set", "count", None, 3)),
        ("ledger.sorted[args.bin] += 1", {"args": {"bin": "C"}}, LedgerOp("add", "sorted", "C", 1)),
        ("ledger.count -= 2", {}, LedgerOp("add", "count", None, -2)),
        ("ledger.done = true", {}, LedgerOp("set", "done", None, True)),
    ],
)
def test_statements_produce_ledger_ops(source, namespace, expected):
    assert compile_statement(source).evaluate(namespace) == expected


@pytest.mark.parametrize(
    "source",
    [
        "count = 1",
        "ledger.a, ledger.b = 1, 2",
        "ledger.count *= 2",
        "task.x = 1",
        "ledger.count = open('x')",
        "ledger.a = 1; ledger.b = 2",
    ],
)
def test_statements_only_write_ledger_fields(source):
    with pytest.raises(ExpressionError):
        compile_statement(source)
