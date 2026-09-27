"""A small, safe expression language for config files.

Configs use expressions for predicates, verifier conditions, preconditions, effects and the
completion check. Only literals, names, dotted access into namespaces, indexing, arithmetic,
comparisons, boolean logic and a few whitelisted functions are accepted. Anything else is
rejected when the config loads; nothing is ever passed to eval().
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ..memory.ledger import LedgerOp


class ExpressionError(ValueError):
    pass


def _sum(values: Any) -> Any:
    # Counters are dicts; summing a counter should add up its counts, not its keys.
    if isinstance(values, Mapping):
        return sum(values.values())
    return sum(values)


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "all": all,
    "any": any,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sum": _sum,
}
CONSTANTS: dict[str, Any] = {"true": True, "false": False, "null": None}

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_STRUCTURAL = (
    ast.Expression,
    ast.Load,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.IfExp,
    ast.List,
    ast.Tuple,
    ast.Subscript,
)


def _check(tree: ast.AST, source: str) -> None:
    """Reject every node outside the allowed subset."""
    for node in ast.walk(tree):
        kind = type(node)
        if isinstance(node, _STRUCTURAL) or kind in _BINARY or kind in _COMPARE or kind in _UNARY:
            continue
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (bool, int, float, str)):
                continue
            raise ExpressionError(f"unsupported literal {node.value!r} in {source!r}")
        if isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ExpressionError(f"name {node.id!r} is not allowed in {source!r}")
            continue
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ExpressionError(f"attribute {node.attr!r} is not allowed in {source!r}")
            continue
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ExpressionError(f"only {', '.join(sorted(FUNCTIONS))} can be called, in {source!r}")
            continue
        raise ExpressionError(f"{kind.__name__} is not allowed in {source!r}")


@dataclass(frozen=True, eq=False)
class Expression:
    """A compiled expression. Call it with a namespace mapping to evaluate it."""

    source: str
    node: ast.expr

    @property
    def names(self) -> frozenset[str]:
        """Free names the expression reads, e.g. {"holding", "ledger"}."""
        found = {n.id for n in ast.walk(self.node) if isinstance(n, ast.Name)}
        return frozenset(found - FUNCTIONS.keys() - CONSTANTS.keys())

    def attribute_refs(self) -> set[tuple[str, str]]:
        """Pairs such as ("ledger", "sorted") for every `name.attr` in the expression."""
        return {
            (n.value.id, n.attr)
            for n in ast.walk(self.node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
        }

    def __call__(self, namespace: Mapping[str, Any]) -> Any:
        try:
            return _evaluate(self.node, namespace)
        except ExpressionError as exc:
            raise ExpressionError(f"{exc} (in {self.source!r})") from None
        except Exception as exc:
            raise ExpressionError(f"{type(exc).__name__}: {exc} (in {self.source!r})") from exc


def compile_expression(source: str) -> Expression:
    text = source.strip()
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse {source!r}: {exc.msg}") from None
    _check(tree, source)
    return Expression(source=text, node=tree.body)


def _evaluate(node: ast.AST, ns: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ns:
            return ns[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ExpressionError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Attribute):
        base = _evaluate(node.value, ns)
        if not isinstance(base, Mapping):
            raise ExpressionError(f"cannot read .{node.attr} from a {type(base).__name__}")
        if node.attr not in base:
            raise ExpressionError(f"{node.attr!r} not found")
        return base[node.attr]
    if isinstance(node, ast.Subscript):
        return _evaluate(node.value, ns)[_evaluate(node.slice, ns)]
    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        result: Any = is_and
        for value in node.values:
            result = _evaluate(value, ns)
            if bool(result) is not is_and:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_evaluate(node.operand, ns))
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_evaluate(node.left, ns), _evaluate(node.right, ns))
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, ns)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate(comparator, ns)
            if not _COMPARE[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return FUNCTIONS[node.func.id](*(_evaluate(arg, ns) for arg in node.args))
    if isinstance(node, ast.IfExp):
        return _evaluate(node.body, ns) if _evaluate(node.test, ns) else _evaluate(node.orelse, ns)
    if isinstance(node, ast.List):
        return [_evaluate(element, ns) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(element, ns) for element in node.elts)
    raise ExpressionError(f"cannot evaluate {type(node).__name__}")


@dataclass(frozen=True, eq=False)
class Statement:
    """An effect such as `ledger.sorted[args.bin] += 1`.

    Evaluating it only produces a LedgerOp; the supervisor applies the op and records it.
    """

    source: str
    field: str
    key: Expression | None
    op: Literal["set", "add", "sub"]
    value: Expression

    def evaluate(self, namespace: Mapping[str, Any]) -> LedgerOp:
        key = None if self.key is None else str(self.key(namespace))
        value = self.value(namespace)
        if self.op == "sub":
            return LedgerOp("add", self.field, key, -value)
        return LedgerOp(self.op, self.field, key, value)


def compile_statement(source: str) -> Statement:
    text = source.strip()
    try:
        module = ast.parse(text, mode="exec")
    except SyntaxError as exc:
        raise ExpressionError(f"cannot parse {source!r}: {exc.msg}") from None
    if len(module.body) != 1:
        raise ExpressionError(f"expected exactly one assignment in {source!r}")
    stmt = module.body[0]
    op: Literal["set", "add", "sub"]
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value, op = stmt.targets[0], stmt.value, "set"
    elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, (ast.Add, ast.Sub)):
        target, value = stmt.target, stmt.value
        op = "add" if isinstance(stmt.op, ast.Add) else "sub"
    else:
        raise ExpressionError(
            f"effects must look like `ledger.x = ...`, `ledger.x += ...` or `ledger.x -= ...`, got {source!r}"
        )
    key_node: ast.expr | None = None
    if isinstance(target, ast.Subscript):
        key_node, target = target.slice, target.value
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "ledger"
    ):
        raise ExpressionError(f"effects can only write to ledger fields, in {source!r}")
    return Statement(
        source=text,
        field=target.attr,
        key=None if key_node is None else _sub_expression(key_node, text),
        op=op,
        value=_sub_expression(value, text),
    )


def _sub_expression(node: ast.expr, source: str) -> Expression:
    _check(ast.Expression(body=node), source)
    return Expression(source=ast.unparse(node), node=node)
