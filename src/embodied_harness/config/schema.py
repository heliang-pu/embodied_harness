"""Schemas for every YAML file the harness reads (design doc §4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


FailureKind = Literal["readiness", "grounding", "control"]


class Condition(_Model):
    when: str
    hold_s: float = Field(0.0, ge=0)
    reason: str | None = None
    # Failure conditions only: what kind of failure this is, for `embodied-harness report`. readiness:
    # the skill could not start from the state it was given; grounding: it went for the wrong target;
    # control: it started right and failed while executing. Unset, the report infers it from timing.
    kind: FailureKind | None = None


class VerifySpec(_Model):
    success: list[Condition] = Field(default_factory=list)
    failure: list[Condition] = Field(default_factory=list)
    # With planner.call: periodic, the planner may switch away from this skill only while one of
    # these holds (e.g. `released` for a place). They never end the skill by themselves.
    switch_when: list[str] = Field(default_factory=list)


class RecoveryAction(_Model):
    retry: int = Field(0, ge=0)
    before_retry: list[str] = Field(default_factory=list)
    escalate: Literal["planner", "hold"] = "planner"
    # A skill name: this failure undid that skill's work (a drop undoes the pick). Once the retries are used
    # up and the failure reaches the planner, a planner that plays a fixed plan (sequence) resumes from the
    # nearest earlier plan step calling that skill instead of repeating the failed step. Other planners
    # ignore it. Needs escalate: planner; not allowed in defaults.on_failure.
    rollback: str | None = None


class ArgSpec(_Model):
    type: Literal["enum"] = "enum"
    values: list[str]


class Effects(_Model):
    on_success: list[str] = Field(default_factory=list)


class SkillSpec(_Model):
    description: str | None = None
    # policy: the learned policy (a VLA, a world action model, ...) runs the skill from a library prompt;
    # scripted: a harness primitive. "vla" is accepted as an older name for "policy".
    executor: Literal["policy", "scripted"] = "policy"
    prompt: str | dict[str, str] | None = None
    args: dict[str, ArgSpec] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    verify: VerifySpec = Field(default_factory=VerifySpec)
    min_duration_s: float = Field(0.0, ge=0)
    timeout_s: float | None = Field(None, gt=0)
    stall_s: float | None = Field(None, gt=0)
    on_failure: dict[str, RecoveryAction] = Field(default_factory=dict)
    effects: Effects = Field(default_factory=Effects)

    @field_validator("executor", mode="before")
    @classmethod
    def _old_executor_name(cls, value: Any) -> Any:
        return "policy" if value == "vla" else value


class SkillDefaults(_Model):
    on_failure: RecoveryAction = Field(default_factory=RecoveryAction)
    max_consecutive_failures: int = Field(3, ge=1)
    timeout_factor: float = Field(2.0, gt=0)
    timeout_s: float = Field(30.0, gt=0)


class TransitionStep(_Model):
    """A primitive with arguments, computed when the rule fires: e.g. {skill: approach, args: {color:
    "next.args.color"}} moves to the pre-grasp pose of the object the next call is about."""

    skill: str
    args: dict[str, str] = Field(default_factory=dict)  # argument -> expression, like `when`


class TransitionRule(_Model):
    """Primitives the harness runs between two skill calls when `when` holds, so that the next skill
    starts from a state its policy was trained on (e.g. back to the start pose before picking another
    object). The planner never sees them. `when` and the step arguments read `previous` and `next`
    (each {skill, args}; the first call's previous has skill null), proprio, predicates, `ledger` and
    `task`. A step is a scripted skill's name, or {skill, args} for one that takes arguments."""

    name: str | None = None
    when: str
    run: list[str | TransitionStep] = Field(min_length=1)


class SkillsFile(_Model):
    version: int = 1
    library: str | None = None
    predicates: dict[str, str] = Field(default_factory=dict)
    skills: dict[str, SkillSpec]
    transitions: list[TransitionRule] = Field(default_factory=list)
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


class PlanStep(_Model):
    """One step of a task's fixed order: a skill and literal values for its enum arguments."""

    skill: str
    args: dict[str, str] = Field(default_factory=dict)


class TaskSpec(_Model):
    id: str
    goal: str
    # "episode": each episode brings its own instruction (the env's instruction() at runtime, the
    # dataset's instruction column in data exports), which replaces `goal` for that episode.
    goal_source: Literal["config", "episode"] = "config"
    rules: list[str] = Field(default_factory=list)
    # Experience from earlier runs, shown to the planner after the rules. The reviewed entries of the
    # run config's `planner.lessons` file are appended here when the config loads.
    lessons: list[str] = Field(default_factory=list)
    skills: list[str]
    expected_total: int | None = None
    ledger: dict[str, LedgerField] = Field(default_factory=dict)
    done_when: str | None = None
    budget: Budget = Field(default_factory=Budget)
    # The task's fixed order of skill calls, for long tasks that always run the same way. The sequence
    # planner plays it (with the skills' rollback rules); `report` measures every planner against it. A step
    # is a skill name, or {skill, args} for a skill with arguments.
    plan: list[str | PlanStep] | None = Field(None, min_length=1)


class TaskFile(_Model):
    task: TaskSpec


# --- lessons file: experience from earlier runs, for the planner ----------------------------------


class Lesson(_Model):
    text: str = Field(min_length=1)
    reviewed: bool = False  # only reviewed lessons reach the planner
    source: str | None = None  # where it came from, e.g. the report finding that suggested it


class LessonsFile(_Model):
    version: int = 1
    lessons: list[Lesson] = Field(default_factory=list)


# --- suite file: a composition benchmark over run configs ---------------------------------------


class SuiteEntry(_Model):
    """One benchmark cell: a run config (with overrides) run for a number of episodes."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    config: str  # run config, relative to the suite file
    # A: an atomic task (one skill chain seen as a whole in training), AP: atomic with perturbations,
    # C: a composition of skills, CP: a composition with perturbations (RoboHiMan's four task types).
    category: Literal["A", "AP", "C", "CP"]
    split: Literal["seen", "unseen"] = "seen"  # whether this composition (or task) was in the training data
    episodes: int = Field(5, ge=1)
    seed: int | None = None  # the first episode's seed, later ones add 1 (default: the config's runtime.seed)
    overrides: dict[str, Any] = Field(default_factory=dict)  # merged into the run config
    task_overrides: dict[str, Any] = Field(default_factory=dict)  # merged into the task file's `task`
    note: str | None = None


class SuiteFile(_Model):
    suite: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    root: str = "runs/suites"  # results go to <root>/<suite>/<run id>/<entry>/, relative to the cwd
    entries: list[SuiteEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> SuiteFile:
        names = [entry.name for entry in self.entries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate entry names: {', '.join(duplicates)}")
        return self


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
    # boundary: ask the planner only when a skill ends (verifiers decide when).
    # periodic: also ask every `every_steps` control steps while a skill runs, at the executor's chunk
    # boundaries; a different answer preempts the running skill (RoboMME-style subgoal prediction).
    call: Literal["boundary", "periodic"] = "boundary"
    # When a periodic planner says done before the task's done_when is true, keep the active
    # skill and ask again instead of turning the false claim into an episode abort. The default
    # preserves the strict failure behavior used by normal runs.
    done_policy: Literal["resample", "continue"] = "resample"
    # Some simulator evaluations have no trustworthy planner-side completion signal; the
    # environment terminal verdict owns completion in that mode.
    allow_done: bool = True
    every_steps: int = Field(16, ge=1)
    threaded: bool = False
    timeout_s: float = Field(10.0, gt=0)
    retries: int = Field(3, ge=0)
    retry_backoff_s: float = Field(0.5, ge=0)
    max_resamples: int = Field(2, ge=0)
    memory_max_chars: int = Field(200, ge=1)
    trained_library: str | None = None
    # The manifest.json (or the directory) of the `data planner` export the planner was trained on,
    # relative to this config. Each run start (and `validate`) warns about every planner setting that
    # differs from it; a manifest that cannot be read is only a warning, and no other command reads it.
    trained_manifest: str | None = None
    # A lessons file (relative to this config). Its reviewed entries become task lessons; pin one file
    # per evaluation so every compared run sees the same experience.
    lessons: str | None = None


class RuntimeConfig(_Model):
    clock: Literal["sim", "real"] = "sim"
    control_hz: float = Field(10.0, gt=0)
    health_every_s: float = Field(10.0, gt=0)
    underrun_limit_s: float = Field(1.0, gt=0)
    primitive_steps: dict[str, int] = Field(default_factory=dict)
    default_primitive_steps: int = Field(5, ge=1)
    # The first episode's seed when neither `run --seed` nor a suite entry's seed is given; episode k gets
    # seed + k, so every episode has its own repeatable scene. An evaluation sets it to scenes its training
    # data never used.
    seed: int = 0


class TraceConfig(_Model):
    root: str = "runs"
    video: bool = False  # write episode.mp4 with every tick's camera images side by side
    video_fps: float = Field(10.0, gt=0)
    # Per-tick camera frames (one video per camera), proprio and actions sent, for `data from-runs`.
    frames: bool = False
    # The world state at the start of every skill call, snapshots/<call_id>.json, for `embodied-harness
    # probe` (simulated envs with snapshot() only).
    snapshots: bool = False


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
