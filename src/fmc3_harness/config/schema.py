"""Schemas for every YAML file the harness reads (design doc §4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- prompt library: shipped next to the executor's weights -------------------------------


class DurationStats(_Model):
    p50: float | None = None
    p95: float | None = None


class TargetSlot(_Model):
    """How a grounded target reaches the executor; must match how the executor was trained."""

    type: Literal["point", "box"]
    render: Literal["overlay", "loc_tokens"]
    camera: str


class PromptEntry(_Model):
    id: str
    text: str
    episodes: int | None = None
    duration_s: DurationStats | None = None
    target: TargetSlot | None = None


class PromptLibrary(_Model):
    """Every subtask prompt the executor was trained on, word for word."""

    library: str
    source: str | None = None
    prompts: list[PromptEntry]

    @model_validator(mode="after")
    def _unique_ids(self) -> PromptLibrary:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for prompt in self.prompts:
            if prompt.id in seen:
                duplicates.add(prompt.id)
            seen.add(prompt.id)
        if duplicates:
            raise ValueError(f"duplicate prompt ids: {', '.join(sorted(duplicates))}")
        return self

    def get(self, prompt_id: str) -> PromptEntry | None:
        return next((p for p in self.prompts if p.id == prompt_id), None)


# --- skills file ----------------------------------------------------------------------------


class Condition(_Model):
    when: str
    hold_s: float = Field(0.0, ge=0)
    reason: str | None = None


class VerifySpec(_Model):
    success: list[Condition] = Field(default_factory=list)
    failure: list[Condition] = Field(default_factory=list)


class RecoveryAction(_Model):
    retry: int = Field(0, ge=0)
    before_retry: list[str] = Field(default_factory=list)
    escalate: Literal["planner", "hold"] = "planner"


class ArgSpec(_Model):
    type: Literal["enum"] = "enum"
    values: list[str]


class Effects(_Model):
    on_success: list[str] = Field(default_factory=list)


class SkillSpec(_Model):
    description: str | None = None
    executor: Literal["vla", "scripted"] = "vla"
    prompt: str | dict[str, str] | None = None
    args: dict[str, ArgSpec] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    verify: VerifySpec = Field(default_factory=VerifySpec)
    min_duration_s: float = Field(0.0, ge=0)
    timeout_s: float | None = Field(None, gt=0)
    stall_s: float | None = Field(None, gt=0)
    on_failure: dict[str, RecoveryAction] = Field(default_factory=dict)
    effects: Effects = Field(default_factory=Effects)


class SkillDefaults(_Model):
    on_failure: RecoveryAction = Field(default_factory=RecoveryAction)
    max_consecutive_failures: int = Field(3, ge=1)
    timeout_factor: float = Field(2.0, gt=0)
    timeout_s: float = Field(30.0, gt=0)


class SkillsFile(_Model):
    version: int = 1
    library: str | None = None
    predicates: dict[str, str] = Field(default_factory=dict)
    skills: dict[str, SkillSpec]
    defaults: SkillDefaults = Field(default_factory=SkillDefaults)


# --- task file --------------------------------------------------------------------------------


class LedgerField(_Model):
    type: Literal["counter", "int", "bool", "str", "enum"]
    keys: list[str] | None = None
    values: list[str] | None = None
    init: Any = None

    @model_validator(mode="after")
    def _check_shape(self) -> LedgerField:
        if self.type == "counter" and not self.keys:
            raise ValueError("counter fields need keys")
        if self.type == "enum" and not self.values:
            raise ValueError("enum fields need values")
        return self


class Budget(_Model):
    max_time_s: float | None = Field(None, gt=0)
    max_skill_calls: int | None = Field(None, ge=1)
    max_planner_calls: int | None = Field(None, ge=1)


class TaskSpec(_Model):
    id: str
    goal: str
    rules: list[str] = Field(default_factory=list)
    skills: list[str]
    expected_total: int | None = None
    ledger: dict[str, LedgerField] = Field(default_factory=dict)
    done_when: str | None = None
    budget: Budget = Field(default_factory=Budget)


class TaskFile(_Model):
    task: TaskSpec


# --- run config: which env, executor and planner to wire together ------------------------------


class EnvConfig(_Model):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ExecutorConfig(_Model):
    type: str
    library: str
    handoff: Literal["finish_chunk", "hard"] = "finish_chunk"
    params: dict[str, Any] = Field(default_factory=dict)


class PlannerConfig(_Model):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    threaded: bool = False
    timeout_s: float = Field(10.0, gt=0)
    retries: int = Field(3, ge=0)
    retry_backoff_s: float = Field(0.5, ge=0)
    max_resamples: int = Field(2, ge=0)
    memory_max_chars: int = Field(200, ge=1)
    trained_library: str | None = None


class RuntimeConfig(_Model):
    clock: Literal["sim", "real"] = "sim"
    control_hz: float = Field(10.0, gt=0)
    health_every_s: float = Field(10.0, gt=0)
    underrun_limit_s: float = Field(1.0, gt=0)
    primitive_steps: dict[str, int] = Field(default_factory=dict)
    default_primitive_steps: int = Field(5, ge=1)


class TraceConfig(_Model):
    root: str = "runs"


class LoggingConfig(_Model):
    console_level: str = "INFO"
    file_level: str = "DEBUG"


class RunConfig(_Model):
    name: str
    task: str
    skills: str
    env: EnvConfig
    executor: ExecutorConfig
    planner: PlannerConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
