"""Compiled view of the skills file: prompts, predicates, verifier conditions, effects and recovery.

Everything is compiled and cross-checked when the config loads, so mistakes surface in
`fmc3-harness validate` instead of halfway through an episode on the robot.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from ..config.errors import ConfigError
from ..config.schema import Condition, PromptLibrary, RecoveryAction, SkillsFile, SkillSpec, TaskSpec
from ..core.types import Prompt, SkillCall
from .expressions import Expression, ExpressionError, Statement, compile_expression, compile_statement

TASK_ATTRIBUTES = frozenset({"id", "goal", "rules", "expected_total"})
RUNTIME_NAMES = frozenset({"ledger", "args", "task", "executor_done"})


@dataclass(frozen=True)
class CompiledCondition:
    expr: Expression
    hold_s: float
    reason: str


@dataclass(frozen=True)
class CompiledSkill:
    name: str
    spec: SkillSpec
    preconditions: tuple[Expression, ...]
    success: tuple[CompiledCondition, ...]
    failure: tuple[CompiledCondition, ...]
    effects: tuple[Statement, ...]

    @property
    def scripted(self) -> bool:
        return self.spec.executor == "scripted"

    def expressions(self) -> Iterator[Expression]:
        yield from self.preconditions
        for condition in (*self.success, *self.failure):
            yield condition.expr
        for effect in self.effects:
            if effect.key is not None:
                yield effect.key
            yield effect.value


def _default_reason(when: str) -> str:
    """A failure condition that is a bare predicate name is its own reason, e.g. `empty_grasp`."""
    text = when.strip()
    return text if text.isidentifier() else "failure"


class SkillRegistry:
    def __init__(self, skills: SkillsFile, library: PromptLibrary, task: TaskSpec) -> None:
        self._file = skills
        self._library = library
        self._task = task
        problems: list[str] = []

        self._predicates: dict[str, Expression] = {}
        for name, source in skills.predicates.items():
            expr = self._compile(source, f"predicates.{name}", problems)
            if expr is not None:
                self._predicates[name] = expr

        self._skills: dict[str, CompiledSkill] = {}
        for name, spec in skills.skills.items():
            compiled = self._compile_skill(name, spec, problems)
            if compiled is not None:
                self._skills[name] = compiled

        self._done_when: Expression | None = None
        if task.done_when:
            self._done_when = self._compile(task.done_when, "task.done_when", problems)
            if self._done_when is not None:
                problems += self._check_attributes(self._done_when, "task.done_when", args=())
        if problems:
            raise ConfigError(problems)

    # --- queries used at runtime ----------------------------------------------------------------

    @property
    def library(self) -> PromptLibrary:
        return self._library

    @property
    def names(self) -> list[str]:
        return list(self._skills)

    @property
    def max_consecutive_failures(self) -> int:
        return self._file.defaults.max_consecutive_failures

    def skill(self, name: str) -> CompiledSkill:
        return self._skills[name]

    def resolve_prompt(self, call: SkillCall) -> Prompt | None:
        """The exact library text for a call; None for scripted primitives."""
        spec = self._skills[call.skill].spec
        if spec.prompt is None:
            return None
        if isinstance(spec.prompt, str):
            prompt_id = spec.prompt
        else:
            (arg_name,) = spec.args
            prompt_id = spec.prompt[str(call.args[arg_name])]
        entry = self._library.get(prompt_id)
        if entry is None:  # the loader already checked this; guard against later edits
            raise KeyError(f"prompt {prompt_id!r} is not in library {self._library.library!r}")
        return Prompt(entry.id, entry.text)

    def timeout_for(self, name: str) -> float:
        """Explicit timeout_s, else p95 duration from the library times the factor, else the default."""
        spec = self._skills[name].spec
        if spec.timeout_s is not None:
            return spec.timeout_s
        prompt_ids = [spec.prompt] if isinstance(spec.prompt, str) else list((spec.prompt or {}).values())
        p95s = []
        for prompt_id in prompt_ids:
            entry = self._library.get(prompt_id)
            if entry is not None and entry.duration_s is not None and entry.duration_s.p95:
                p95s.append(entry.duration_s.p95)
        if p95s:
            return max(p95s) * self._file.defaults.timeout_factor
        return self._file.defaults.timeout_s

    def recovery_for(self, name: str, reason: str | None) -> RecoveryAction:
        spec = self._skills[name].spec
        return spec.on_failure.get(reason or "", self._file.defaults.on_failure)

    def predicates(self, proprio: Mapping[str, Any]) -> dict[str, bool]:
        namespace = dict(proprio)
        values: dict[str, bool] = {}
        for name, expr in self._predicates.items():
            values[name] = namespace[name] = bool(expr(namespace))
        return values

    def check_call(self, call: SkillCall) -> list[str]:
        if call.skill not in self._skills:
            return [f"unknown skill {call.skill!r}"]
        spec = self._skills[call.skill].spec
        problems = []
        missing = [name for name in spec.args if name not in call.args]
        extra = [name for name in call.args if name not in spec.args]
        if missing:
            problems.append(f"{call.skill}: missing argument(s) {', '.join(missing)}")
        if extra:
            problems.append(f"{call.skill}: unexpected argument(s) {', '.join(extra)}")
        for name, arg in spec.args.items():
            if name in call.args and str(call.args[name]) not in arg.values:
                problems.append(f"{call.skill}: {name}={call.args[name]!r} is not one of {arg.values}")
        return problems

    def is_done(self, ledger: Mapping[str, Any], task: Mapping[str, Any]) -> bool | None:
        """None when the task has no done_when, so completion is up to the planner."""
        if self._done_when is None:
            return None
        return bool(self._done_when({"ledger": ledger, "task": task}))

    def check_names(self, proprio_keys: Iterable[str]) -> list[str]:
        """Every name an expression reads must exist once the env is known."""
        proprio = set(proprio_keys)
        problems = []
        known = set(proprio)
        for name, expr in self._predicates.items():
            unknown = expr.names - known
            if unknown:
                problems.append(
                    f"predicates.{name}: unknown name(s) {', '.join(sorted(unknown))}; "
                    f"the env provides {', '.join(sorted(proprio))}"
                )
            known.add(name)
        allowed = proprio | set(self._predicates) | RUNTIME_NAMES
        for skill_name, skill in self._skills.items():
            for expr in skill.expressions():
                unknown = expr.names - allowed
                if unknown:
                    names = ", ".join(sorted(unknown))
                    problems.append(f"skills.{skill_name}: unknown name(s) {names} in {expr.source!r}")
        if self._done_when is not None:
            unknown = self._done_when.names - {"ledger", "task"}
            if unknown:
                problems.append(
                    f"task.done_when: only ledger and task can be used, found {', '.join(sorted(unknown))}"
                )
        return problems

    # --- compilation --------------------------------------------------------------------------

    def _compile_skill(self, name: str, spec: SkillSpec, problems: list[str]) -> CompiledSkill | None:
        where = f"skills.{name}"
        before = len(problems)
        problems += self._check_prompt(where, spec)
        for reason, action in spec.on_failure.items():
            for primitive in action.before_retry:
                if primitive not in self._file.skills:
                    problems.append(
                        f"{where}.on_failure.{reason}: before_retry skill {primitive!r} is not defined"
                    )

        preconditions = []
        for source in spec.preconditions:
            expr = self._compile(source, f"{where}.preconditions", problems)
            if expr is not None:
                preconditions.append(expr)
        success = self._conditions(spec.verify.success, f"{where}.verify.success", problems)
        failure = self._conditions(spec.verify.failure, f"{where}.verify.failure", problems)
        effects = []
        for source in spec.effects.on_success:
            try:
                statement = compile_statement(source)
            except ExpressionError as exc:
                problems.append(f"{where}.effects: {exc}")
                continue
            problems += self._check_effect_target(statement, f"{where}.effects")
            effects.append(statement)

        compiled = CompiledSkill(name, spec, tuple(preconditions), success, failure, tuple(effects))
        for expr in compiled.expressions():
            problems += self._check_attributes(expr, where, args=spec.args.keys())
        return compiled if len(problems) == before else None

    def _check_prompt(self, where: str, spec: SkillSpec) -> list[str]:
        problems: list[str] = []
        if spec.prompt is None:
            if spec.executor == "vla":
                problems.append(
                    f"{where}: skills run by the VLA need a prompt from library {self._library.library!r}"
                )
            return problems
        if isinstance(spec.prompt, str):
            prompt_ids = [spec.prompt]
        else:
            if len(spec.args) != 1:
                problems.append(
                    f"{where}.prompt: a prompt map needs exactly one argument, found {len(spec.args)}"
                )
                return problems
            ((arg_name, arg),) = spec.args.items()
            missing = [value for value in arg.values if value not in spec.prompt]
            extra = [key for key in spec.prompt if key not in arg.values]
            if missing:
                problems.append(f"{where}.prompt: no prompt for {arg_name}={', '.join(missing)}")
            if extra:
                problems.append(f"{where}.prompt: {', '.join(extra)} is not a value of {arg_name}")
            prompt_ids = list(spec.prompt.values())
        for prompt_id in prompt_ids:
            if self._library.get(prompt_id) is None:
                problems.append(
                    f"{where}.prompt: {prompt_id!r} is not in prompt library {self._library.library!r}"
                )
        return problems

    def _conditions(
        self, conditions: list[Condition], where: str, problems: list[str]
    ) -> tuple[CompiledCondition, ...]:
        compiled = []
        for condition in conditions:
            expr = self._compile(condition.when, where, problems)
            if expr is not None:
                reason = condition.reason or _default_reason(condition.when)
                compiled.append(CompiledCondition(expr, condition.hold_s, reason))
        return tuple(compiled)

    def _check_effect_target(self, statement: Statement, where: str) -> list[str]:
        field = self._task.ledger.get(statement.field)
        if field is None:
            return [f"{where}: the ledger has no field {statement.field!r} ({statement.source!r})"]
        if field.type == "counter" and statement.key is None:
            return [f"{where}: counter {statement.field!r} needs a key, like ledger.{statement.field}[...]"]
        if field.type != "counter" and statement.key is not None:
            return [f"{where}: {statement.field!r} is a {field.type} field and takes no key"]
        if field.type in ("bool", "str", "enum") and statement.op != "set":
            return [f"{where}: {statement.field!r} can only be set, not incremented ({statement.source!r})"]
        return []

    def _check_attributes(self, expr: Expression, where: str, args: Iterable[str]) -> list[str]:
        problems = []
        allowed_args = set(args)
        for base, attr in sorted(expr.attribute_refs()):
            if base == "ledger" and attr not in self._task.ledger:
                problems.append(f"{where}: the ledger has no field {attr!r} ({expr.source!r})")
            elif base == "args" and attr not in allowed_args:
                problems.append(f"{where}: there is no argument {attr!r} ({expr.source!r})")
            elif base == "task" and attr not in TASK_ATTRIBUTES:
                problems.append(f"{where}: task has no attribute {attr!r} ({expr.source!r})")
        return problems

    @staticmethod
    def _compile(source: str, where: str, problems: list[str]) -> Expression | None:
        try:
            return compile_expression(source)
        except ExpressionError as exc:
            problems.append(f"{where}: {exc}")
            return None
