"""Compiled view of the skills file: prompts, predicates, verifier conditions, effects and recovery.

Everything is compiled and cross-checked when the config loads, so mistakes surface in
`embodied-harness validate` instead of halfway through an episode on the robot.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..core.ledger import Ledger, LedgerError
from ..core.types import Prompt, SkillCall
from .errors import ConfigError
from .expressions import Expression, ExpressionError, Statement, compile_expression, compile_statement
from .schema import (
    Condition,
    PlanStep,
    PromptLibrary,
    RecoveryAction,
    SkillsFile,
    SkillSpec,
    TaskSpec,
    TransitionStep,
)

TASK_ATTRIBUTES = frozenset({"id", "goal", "rules", "expected_total"})
RUNTIME_NAMES = frozenset({"ledger", "args", "task", "executor_done"})
TRANSITION_NAMES = frozenset({"previous", "next"})


@dataclass(frozen=True)
class CompiledCondition:
    expr: Expression
    hold_s: float
    reason: str
    kind: str | None = None  # failure conditions: readiness, grounding or control, for the report


@dataclass(frozen=True)
class CompiledSkill:
    name: str
    spec: SkillSpec
    preconditions: tuple[Expression, ...]
    success: tuple[CompiledCondition, ...]
    failure: tuple[CompiledCondition, ...]
    effects: tuple[Statement, ...]
    switch_when: tuple[Expression, ...] = ()

    @property
    def scripted(self) -> bool:
        return self.spec.executor == "scripted"

    def expressions(self) -> Iterator[Expression]:
        yield from self.preconditions
        yield from self.switch_when
        for condition in (*self.success, *self.failure):
            yield condition.expr
        for effect in self.effects:
            if effect.key is not None:
                yield effect.key
            yield effect.value


@dataclass(frozen=True)
class CompiledStep:
    """A primitive a transition rule runs; its arguments are evaluated when the rule fires."""

    skill: str
    args: tuple[tuple[str, Expression], ...] = ()

    def call(self, namespace: Mapping[str, Any]) -> SkillCall:
        return SkillCall(self.skill, {name: str(expr(namespace)) for name, expr in self.args})


@dataclass(frozen=True)
class CompiledTransition:
    name: str
    when: Expression
    run: tuple[CompiledStep, ...]

    def expressions(self) -> Iterator[Expression]:
        yield self.when
        for step in self.run:
            for _, expr in step.args:
                yield expr


def _default_reason(when: str) -> str:
    """A failure condition that is a bare predicate name is its own reason, e.g. `empty_grasp`."""
    text = when.strip()
    return text if text.isidentifier() else "failure"


def plan_call(step: str | PlanStep) -> SkillCall:
    """A task.plan step as the call it makes."""
    return SkillCall(step) if isinstance(step, str) else SkillCall(step.skill, dict(step.args))


def _repeatable(effect: Statement) -> bool:
    """Whether applying an effect twice leaves the ledger as applying it once does: a set that does not
    read the ledger. A rollback may redo only steps whose effects are all repeatable."""
    reads = effect.value.names | (effect.key.names if effect.key is not None else frozenset())
    return effect.op == "set" and "ledger" not in reads


def task_namespace(task: TaskSpec) -> dict[str, Any]:
    """What `task.<name>` reads in config expressions."""
    return {
        "id": task.id,
        "goal": task.goal,
        "rules": list(task.rules),
        "expected_total": task.expected_total,
    }


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

        self._transitions: list[CompiledTransition] = []
        for index, rule in enumerate(skills.transitions):
            where = f"transitions[{index}]"
            before = len(problems)
            expr = self._compile(rule.when, f"{where}.when", problems)
            steps = [self._compile_step(step, f"{where}.run", problems) for step in rule.run]
            if expr is not None and len(problems) == before:
                compiled_steps = tuple(step for step in steps if step is not None)
                self._transitions.append(CompiledTransition(rule.name or where, expr, compiled_steps))
        self._done_when: Expression | None = None
        if task.done_when:
            self._done_when = self._compile(task.done_when, "task.done_when", problems)
            if self._done_when is not None:
                problems += self._check_attributes(self._done_when, "task.done_when", args=())
        if skills.defaults.on_failure.rollback is not None:
            problems.append(
                "defaults.on_failure: rollback names the skill a failure undid, so it belongs in "
                "skills.<name>.on_failure.<reason>"
            )
        problems += self._check_escalation(skills.defaults.on_failure, "defaults.on_failure")
        self._plan: tuple[SkillCall, ...] | None = None
        if task.plan is not None:
            self._plan = tuple(plan_call(step) for step in task.plan)
            problems += self.check_plan(self._plan, "task.plan")
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

    @property
    def transitions(self) -> tuple[CompiledTransition, ...]:
        return tuple(self._transitions)

    @property
    def plan(self) -> tuple[SkillCall, ...] | None:
        """task.plan as calls; None when the task has no fixed order."""
        return self._plan

    def plan_rollbacks(self, calls: Sequence[SkillCall]) -> list[dict[str, int]]:
        """Per step of a fixed order, failure reason -> the step a rollback resumes from: the nearest earlier
        step calling the skill that the step's on_failure entry names. A reason without such a step is left
        out; that failure repeats the step."""
        table: list[dict[str, int]] = []
        for index, call in enumerate(calls):
            targets: dict[str, int] = {}
            skill = self._skills.get(call.skill)
            for reason, action in skill.spec.on_failure.items() if skill is not None else ():
                if action.rollback is None:
                    continue
                target = next(
                    (j for j in range(index - 1, -1, -1) if calls[j].skill == action.rollback), None
                )
                if target is not None:
                    targets[reason] = target
            table.append(targets)
        return table

    def check_plan(self, calls: Sequence[SkillCall], where: str) -> list[str]:
        """Load-time checks of a fixed order (task.plan, or a sequence planner's params.calls): valid calls
        of the task's skills, rollbacks that never redo a counting step, a skill-call budget that fits, and
        done_when false until the last step and true after it."""
        problems: list[str] = []
        for index, call in enumerate(calls):
            if call.skill not in self._task.skills:
                problems.append(
                    f"{where}[{index}]: {call.skill!r} is not one of the task's skills "
                    f"({', '.join(self._task.skills)})"
                )
            elif call.skill in self._skills:
                problems += [f"{where}[{index}]: {problem}" for problem in self.check_call(call)]
        if problems or any(call.skill not in self._skills for call in calls):
            return problems  # the checks below need valid calls of compiled skills
        for index, targets in enumerate(self.plan_rollbacks(calls)):
            for reason, target in targets.items():
                redone = next(
                    (
                        (step, effect)
                        for step in range(target, index)
                        for effect in self._skills[calls[step].skill].effects
                        if not _repeatable(effect)
                    ),
                    None,
                )
                if redone is not None:
                    step, effect = redone
                    problems.append(
                        f"{where}[{index}] {calls[index].key()}: a rollback on {reason!r} goes back to "
                        f"[{target}] {calls[target].key()} and would redo [{step}] {calls[step].key()}, "
                        f"whose effect {effect.source!r} would count twice"
                    )
        budget = self._task.budget.max_skill_calls
        if budget is not None and budget < len(calls):
            problems.append(
                f"task.budget.max_skill_calls ({budget}) is smaller than {where} ({len(calls)} steps)"
            )
        problems += self._check_done_when(calls, where)
        return problems

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

    @property
    def predicate_names(self) -> list[str]:
        return list(self._predicates)

    def predicate_values(
        self, proprio: Mapping[str, Any], names: Iterable[str] | None = None
    ) -> dict[str, bool]:
        """Only the named predicates (all when None) and the predicates they read, so proprio needs only
        the keys those use."""
        wanted = list(self._predicates) if names is None else list(names)
        needed: set[str] = set()
        pending = list(wanted)
        while pending:
            name = pending.pop()
            if name not in needed:
                needed.add(name)
                pending += [other for other in self._predicates[name].names if other in self._predicates]
        namespace = dict(proprio)
        values: dict[str, bool] = {}
        for name, expr in self._predicates.items():  # file order: a predicate reads only earlier ones
            if name in needed:
                values[name] = namespace[name] = bool(expr(namespace))
        return {name: values[name] for name in wanted}

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
        for rule in self._transitions:
            for expr in rule.expressions():
                unknown = expr.names - allowed - TRANSITION_NAMES
                if unknown:
                    problems.append(
                        f"{rule.name}: unknown name(s) {', '.join(sorted(unknown))} in {expr.source!r}"
                    )
        if self._done_when is not None:
            unknown = self._done_when.names - {"ledger", "task"}
            if unknown:
                problems.append(
                    f"task.done_when: only ledger and task can be used, found {', '.join(sorted(unknown))}"
                )
        return problems

    def _check_done_when(self, calls: Sequence[SkillCall], where: str) -> list[str]:
        """Apply every step's effects in order, the way the supervisor commits them: done_when must stay
        false until the last step and hold after it. Skipped when an effect reads robot state."""
        if self._done_when is None:
            return []
        effects = [self._skills[call.skill].effects for call in calls]
        allowed = {"ledger", "args", "task"}
        for statements in effects:
            for effect in statements:
                reads = effect.value.names | (effect.key.names if effect.key is not None else frozenset())
                if reads - allowed:
                    return []  # only a run can tell
        ledger = Ledger(self._task.ledger)
        task_ns = task_namespace(self._task)
        source = self._done_when.source
        try:
            for index, (call, statements) in enumerate(zip(calls, effects, strict=True)):
                if self.is_done(ledger.view(), task_ns):
                    return [
                        f"{where}: task.done_when {source!r} already holds before [{index}] {call.key()} "
                        f"(ledger {ledger.view()}), so the episode ends before the rest of the plan runs"
                    ]
                namespace = {"ledger": ledger.view(), "args": dict(call.args), "task": task_ns}
                for op in [effect.evaluate(namespace) for effect in statements]:
                    ledger.apply(op)
            if not self.is_done(ledger.view(), task_ns):
                return [
                    f"{where}: task.done_when {source!r} is still false after all {len(calls)} steps "
                    f"(ledger {ledger.view()})"
                ]
        except (ExpressionError, LedgerError) as exc:
            return [f"{where}: replaying the steps' effects failed: {exc}"]
        return []

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
            if action.rollback is not None:
                if action.rollback not in self._file.skills:
                    problems.append(
                        f"{where}.on_failure.{reason}: rollback skill {action.rollback!r} is not defined"
                    )
                elif action.rollback == name:
                    problems.append(
                        f"{where}.on_failure.{reason}: rollback names {name!r} itself; "
                        "leave it out to repeat the step"
                    )
                if action.escalate != "planner":
                    problems.append(
                        f"{where}.on_failure.{reason}: rollback needs escalate: planner "
                        "(a hold never reaches the planner)"
                    )
            problems += self._check_escalation(action, f"{where}.on_failure.{reason}")

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

        switch_when = []
        for source in spec.verify.switch_when:
            expr = self._compile(source, f"{where}.verify.switch_when", problems)
            if expr is not None:
                switch_when.append(expr)
        compiled = CompiledSkill(
            name, spec, tuple(preconditions), success, failure, tuple(effects), tuple(switch_when)
        )
        for expr in compiled.expressions():
            problems += self._check_attributes(expr, where, args=spec.args.keys())
        return compiled if len(problems) == before else None

    def _compile_step(
        self, step: str | TransitionStep, where: str, problems: list[str]
    ) -> CompiledStep | None:
        name, sources = (step, {}) if isinstance(step, str) else (step.skill, step.args)
        spec = self._file.skills.get(name)
        if spec is None:
            problems.append(f"{where}: {name!r} is not defined in the skills file")
            return None
        if spec.executor != "scripted":
            problems.append(f"{where}: {name!r} is not a scripted primitive")
            return None
        missing = [arg for arg in spec.args if arg not in sources]
        extra = [arg for arg in sources if arg not in spec.args]
        if missing:
            problems.append(f"{where}: {name} needs argument(s) {', '.join(missing)}")
        if extra:
            problems.append(f"{where}: {name} has no argument(s) {', '.join(extra)}")
        args = []
        for arg, source in sources.items():
            expr = self._compile(source, f"{where}.{name}.args.{arg}", problems)
            if expr is not None:
                args.append((arg, expr))
        return None if missing or extra else CompiledStep(name, tuple(args))

    def _check_escalation(self, action: RecoveryAction, where: str) -> list[str]:
        """A call's failure streak only resets when it succeeds (or a human resumes), so the retries count
        towards defaults.max_consecutive_failures: the failure after the last retry must still be under
        that cap, or the supervisor holds before escalate: planner is ever reached."""
        limit = self._file.defaults.max_consecutive_failures
        if action.escalate != "planner" or action.retry + 1 < limit:
            return []
        lower = f"lower retry to {limit - 2} or " if limit >= 2 else ""
        return [
            f"{where}: escalate: planner is never reached: after retry: {action.retry} the failure would go "
            f"to the planner at failure {action.retry + 1} in a row, but defaults.max_consecutive_failures "
            f"({limit}) holds at failure {limit}; {lower}raise max_consecutive_failures to {action.retry + 2}"
        ]

    def _check_prompt(self, where: str, spec: SkillSpec) -> list[str]:
        problems: list[str] = []
        if spec.prompt is None:
            if spec.executor == "policy":
                problems.append(
                    f"{where}: skills run by the policy need a prompt from library {self._library.library!r}"
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
            if condition.kind is not None and where.endswith(".success"):
                problems.append(
                    f"{where}: `kind` describes a failure; {condition.when!r} is a success condition"
                )
            if expr is not None:
                reason = condition.reason or _default_reason(condition.when)
                compiled.append(CompiledCondition(expr, condition.hold_s, reason, condition.kind))
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
