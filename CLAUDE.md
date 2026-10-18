# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`embodied-harness` (import name `embodied_harness`; the repo directory is `embodied_harness`) runs a fine-tuned VLM planner and a VLA executor as one closed control loop. The models are swappable. Scheduling, memory, verification, recovery and tracing live in the harness.

The harness is robot-agnostic. It has an abstract env contract and ships these envs: a symbolic mock sorting world (`env/mock_sorting.py`), a RoboMME simulator adapter, gym-aloha's bimanual ALOHA simulator (`env/aloha_sim.py`), LIBERO (`env/libero.py`) and RoboTwin 2.0 (`env/robotwin.py`, native backend in `env/robotwin_native.py`).

README.md is written in Chinese and is the user-facing command reference. Module docstrings cite "design doc §N": that doc is the Claude artifact at https://claude.ai/code/artifact/539a719a-4856-4dec-a378-1c0629bf8947. Read it with the Artifact tool, not WebFetch.

## Environment

- **Which interpreter:** a Python 3.12 environment with the package installed editable with `.[dev]`.
  - Use the active environment's `python`, `ruff` and `embodied-harness` commands.
  - If the shell starts in a base environment, activate the development environment before running checks.
- **How to install:** run the env's own `python -m pip install -e ".[dev]"`.
  - A fresh conda env may have no pip. A bare `pip` then silently installs into base, so bootstrap with `python -m ensurepip` first.
  - Install `.[dev]` for the full test suite. The base deps are pydantic, PyYAML and numpy. The bare package's CLI starts, validates and runs mock episodes, because msgpack, websockets, Pillow and pyarrow are imported only where they are used (`tests/test_package.py` checks this).
- **Commands that need other environments:**
  - `embodied-harness data planner` decodes frames with LeRobot. Run it in the `lerobot` env, after `python -m pip install -e ".[data]"` there.
  - The `robomme` env needs Linux, an NVIDIA GPU and the `robomme` package. On this Mac only `validate` works for `configs/robomme/**`.
  - The `aloha_sim` env needs `gym-aloha` (MuJoCo); it sets `MUJOCO_GL=egl` unless the variable is set, for headless Linux.
  - RoboTwin: the native backend's planner `front` image is RoboTwin's `front_camera` (the demonstrations' `cam_front`), not the observer view `third_view_rgb`.

## Commands

```bash
python -m pytest -q                                   # full suite, seconds; no GPU, network or model needed
python -m pytest tests/test_periodic.py::test_switch_when_gates_planner_switches -q
python -m pytest -q -k resume
ruff check src tests scripts && ruff format --check src tests scripts

embodied-harness validate configs/runs/mock_sort.yaml     # load and cross-check every config piece
embodied-harness run configs/runs/mock_sort.yaml --episodes 3 --seed 0 --hold-policy abort
embodied-harness run configs/runs/mock_sort.yaml --resume runs/<run_id>/ep_0001
embodied-harness replay configs/runs/vlm_mock_sort.yaml runs/<run_id>/ep_0001 --out replay.json
embodied-harness data {labels,library,relabel,planner,eval} ...   # full invocations in README
embodied-harness report runs/<run_id> --by call                   # failure causes and kinds
embodied-harness probe configs/runs/mock_sort.yaml --at runs/<run_id>/ep_0001:sc_0003   # needs trace.snapshots
embodied-harness suite run configs/suites/mock_sort_composition.yaml
```

- **Path resolution:** `trace.root` (default `runs/`, gitignored) resolves against the current directory. A run config's `task`, `skills` and `executor.library` paths resolve against the config file's own directory.
- **Episode directories:** an episode dir that already has `events.jsonl` is refused. Use a new `--run-id` (the default is a timestamp) or `--resume`.
- **HOLD and exit codes:** `--hold-policy` defaults to `ask` on a TTY and `abort` otherwise. `run` exits 0 only when every episode succeeded by `runtime/trace/progress.episode_succeeded`: the env's `env_success` when it reported one, else status `done`. report, suite, plan progress and `data from-runs --success-only` use the same rule.
- **Seeds:** without `--seed`, `run` uses `runtime.seed` (default 0) plus the episode index; `--resume` keeps the recorded seed, and a suite entry without `seed` starts at its run config's `runtime.seed`. RoboMME, LIBERO and RoboTwin pick their scene by seed. `configs/robotwin/run_bottles_eval.yaml` sets `runtime.seed: 1000` because the training demonstrations used seeds 0–79.
- **Trained-manifest check:** when `planner.trained_manifest` names the planner's export (a directory or `manifest.json`, config-relative), `validate` and every episode start compare the bound planner's `settings()` with it and print warnings. It is never fatal.
- **Planner server:** `configs/runs/vlm_mock_sort.yaml` expects vLLM on `127.0.0.1:8000`; the `vllm serve` line is in README.
- **RoboMME configs:**
  - `configs/robomme/` is our PickXtimes setup.
  - `configs/robomme/mme/` reproduces RoboMME's own SimpleSG + QwenVL pipeline.

## Architecture

### Config bundle

A run config names a task file, a skills file, the executor's prompt library, and a `type` + `params` for each of env, executor and planner. Two steps load it:

1. `config/loader.load_bundle` validates each file against the pydantic models in `config/schema.py` (`extra="forbid"`, frozen) and compiles the skills into a `SkillRegistry`.
2. `runtime/factory.check_bundle` checks the names used in expressions against the env's `proprio_keys(params)`, and the planner's cameras against `camera_keys(params)`.

Problems are collected into one `ConfigError`, so `validate` reports all of them at once. Keep that pattern.

- **Prompt library = executor contract:**
  - Skills reference library prompt ids, never raw text. `SkillRegistry.resolve_prompt` returns the exact trained text for a call.
  - A prompt map (`prompt: {A: place_bin_a, ...}`) requires exactly one enum arg.
  - `skills.library` must equal the library name.
  - A different `planner.trained_library` raises `E_LIBRARY_MISMATCH`.
  - A skill's timeout defaults to its prompt's `duration_s.p95` times `defaults.timeout_factor`.
- **Config expressions** cover predicates, preconditions, verifier conditions, `switch_when`, effects and `done_when`.
  - They use the AST-whitelisted language in `config/expressions.py`, never `eval`.
  - They can see proprio keys, predicates, `ledger`, `args`, `task` and `executor_done`.
  - Effects are statements such as `ledger.sorted[args.bin] += 1` that yield `LedgerOp`s.

### Control loop and supervisor

`runtime/runner.EpisodeRunner` runs one tick per period of `runtime.control_hz`:

1. `env.observe()`
2. `Supervisor.tick(obs)`
3. While a skill runs, it pops one action from the `ChunkQueue` and sends it with `env.act()`. It infers a new chunk when none is queued for the current instruction version.
4. When no skill runs, it calls `env.hold()`.

Components read time only through `core/clock`. `SimClock` jumps forward instantly, so episodes are fast and repeatable.

`runtime/supervisor.py` is a deterministic state machine with the states PLAN, EXECUTE, RETRY, HOLD, DONE and ABORT. It is the only component that makes decisions:

- **Planner output:** the planner only proposes raw text.
  - The supervisor parses it (`planner/decision.py`) and rejects non-task skills, bad enum args and unmet preconditions.
  - It resamples up to `max_resamples` times at `resample_temperature`.
  - It retries planner timeouts and errors with backoff, then HOLDs.
- **Instruction versions:** each dispatch bumps the instruction version, and the queue drops chunks inferred for older versions. With `handoff: finish_chunk` the current chunk runs to its end; with `hard` it is dropped at once.
- **Verdicts:** `runtime/verifier.py` judges a running call.
  - Precedence is failure conditions, then stall, then success, then timeout.
  - Success only counts after `min_duration_s`. A skill with no success conditions succeeds on `executor_done`.
  - Every condition must hold for its `hold_s`.
- **Ledger:** only a verified success's `effects.on_success` can change it. It enters the planner prompt only through `openai_compat`'s `show_progress` (as `PlannerContext.ledger`). When `task.done_when` is set, it decides completion. A planner `done` that `done_when` contradicts triggers a replan. With `planner.allow_done: false` a planner `done` never passes, whatever the planner or output mode. Both refusals go through `Supervisor._done_refusal`: `check` passed=false, then resample/HOLD, or under `done_policy: continue` a `planner_done_ignored` event, after which the same call's next periodic check re-asks with reason check_failed and the refusal. A refused answer's memory is never committed.
- **Last result:** `Supervisor._last_result` is how the last main call ended, `{call, status: success|failure|preempted|skipped, reason}`. It is set in `_commit`, `_on_failure` and on periodic preemption, and passed as `PlannerContext.last` (rendered by `show_state`).
- **Failure recovery:**
  1. The skill's `on_failure[reason]` table retries, optionally after running `before_retry` scripted primitives.
  2. Once retries are used up, it escalates to the planner or HOLDs.
  3. Reaching `max_consecutive_failures` for the same call key also HOLDs.
  - **Transition rules** (`transitions` in the skills file): before dispatching a main call, the first rule whose `when` holds (names: `previous`, `next` as `{skill, args}`, proprio, predicates, `ledger`, `task`) queues scripted primitives that run first (purpose `transition`, event `transition`). The planner never sees them. The same `_retry_queue` machinery (a list of `SkillCall`s) runs `before_retry` primitives; the call they prepare is dispatched with `transitions=False`, and a failing prelude primitive HOLDs.
  - A rule step is a scripted skill's name or `{skill, args}` whose arg values are expressions over the same namespace (`registry.CompiledStep.call(namespace)`), e.g. `{skill: approach, args: {color: "next.args.color"}}`. A value outside the arg's enum HOLDs with `E_CONFIG`. `ScriptedExecutor` sends `{"primitive", "version"}` plus `"args"` only when the call has arguments.
  4. In HOLD the runner polls a `runtime/commands.CommandSource` for `resume`, `retry`, `skip` or `abort`.
- **When the planner is asked:**
  - `planner.call: boundary` asks only when a skill ends.
  - `periodic` also asks every `every_steps` ticks, at a chunk boundary.
  - A different periodic answer preempts the running skill, unless `verify.switch_when` blocks the switch or the skill's success is pending (a success condition holds but not yet for its `hold_s` / `min_duration_s`). The verifier runs before the periodic check on every tick, so a verified success always commits its effects. Blocked answers are `switch_blocked` events with `reason` `switch_when` or `success_pending`.
  - A periodic answer that confirms the running call skips that call's preconditions (the call may have made them false itself); the task-skill and enum checks still apply.
  - A periodic answer that is invalid, fails or times out leaves the running skill in place.
- **Threading:**
  - `planner.threaded: false` runs calls inline, so they cost no sim time.
  - On the robot, use `threaded: true` with `clock: real`. `ThreadPlannerRunner` then runs calls on a worker thread and abandons a hung call by swapping in a fresh thread pool.
- **Env-ended episodes:** an env can end an episode itself with `Observation.terminated`. The result has status `ended` plus `env_success`.
- **Actions sent:** `env.act()` / `env.hold()` may return the action actually sent (numeric). With `trace.frames: true`, `runtime/trace/frames.py` records it per tick with the camera frames, proprio and the source (`policy`/`primitive`/`hold`); `data from-runs` turns such runs into a LeRobot dataset.
- **Real-clock pacing:** `core/clock.next_deadline` re-anchors the tick grid after an overrun (skipped ticks go to health, `E_CTRL_OVERRUN`) instead of bursting the missed ticks.
- **Privileged state:** `Observation.info` is ground truth from the simulator. Only `mock_oracle` and that final `env_success` score may read it. Nothing on the real-robot path may depend on it.

### Traces are event-sourced

`runtime/trace/writer.TraceWriter` writes the episode directory; README lists the files. Resuming (`runtime/trace/reader.py`) rebuilds state from events:

- the ledger from `ledger_update` ops;
- memory from `memory_update`;
- the call counters from `skill_start` and `planner_request`;
- the main-call history and the last result from `skill_start` / `skill_end` (purpose `main`) and `human_command` skips;
- the sim world from `env_snapshot.json`, which envs with `snapshot()`/`restore()` write at every skill end.

With `trace.snapshots: true` the supervisor also writes `snapshots/<call_id>.json` at every dispatch (the world as the call started). `runtime/probe.py` (`embodied-harness probe`) restores those and reruns one call until its verdict, outside the supervisor (no planner, recovery or transitions), to compare clean and chained starts.

This has consequences:

- State that must survive a resume has to be emitted as an event.
- Renaming event types or fields breaks resume, replay, the `data` tooling and most tests, which assert on `events.jsonl`.
- Log lines and events pick up `episode_id`, `state`, `skill_call_id` and `planner_call_id` from contextvars (`runtime/logs/context.py`).
- Error codes live in `runtime/logs/codes.py` and are shared by logs, events and `outcome.json`.
- `runtime/trace/writer.to_jsonable` is the one JSON normalizer: numpy arrays and scalars become exact numbers (`tolist()`) in events, actions and outcome. Planner records (`planner/pc_*.json`) deliberately keep numpy proprio as the printed text the prompt showed. The JSONL torn-tail helpers `repair_tail` / `last_record` live in writer.py.

### Planner render ↔ training parity

- `planner/render.py` renders the prompt. Its text is Chinese on purpose.
- `OpenAICompatPlanner.render_request` is used both for live calls and by `data/export.py` to write SFT samples.
- `planner/schema.py` builds vLLM's structured-output schema from the skill vocabulary.

**Any change to the rendered output needs a new `RENDER_VERSION` and regenerated training data.** These couplings must stay in sync:

- `data/export.py` truncates memory exactly like `Supervisor._update_memory`.
- The export manifest records `planner.settings()`, and `data eval` warns if the live settings differ. `settings()` also records `no_done` (allow_done false), `call` (only when periodic) and 12-character digests `system` (the system prompt with the goal blanked) and `goal`. The comparison rules live in `planner/base.settings_differences` and are shared by `data eval`, `validate` and the run-start check against `planner.trained_manifest`; `system` and `goal` are compared only when the export recorded them.
- `scripts/train_planner_sft.py` splits the `<image>` placeholders back into the same message parts that `to_openai_messages` produces.
- `PlannerContext` is constructed in four places: the supervisor, `runtime/trace/replay.py`, `data/export.py` and `data/evaluate.py`. A new field has to be threaded through all four. Its `ledger` and `last` fields are recorded in the trace (`summary()`) and in `samples.jsonl`; `data/export.py` rebuilds the ledger by applying each earlier demonstrated step's effects (`commit_effects`) and treats every earlier step as a success.
- Opt-in render options leave the default layout alone, so they do not bump `RENDER_VERSION`; they are part of `settings()` instead: `show_history`, `show_state` (+ `state_predicates`), `show_progress` (+ `progress_fields`), `show_rejections` (why the previous answer of the round was refused, `PlannerContext.rejected`; demonstration exports never have rejections, from-runs exports keep the recorded ones), `extra_body` and a digest of `task.lessons`. `data eval` treats an option that is off the same as one an older manifest did not record, and refuses an export that lacks what an enabled option needs.
- `task.lessons` (inline, plus the reviewed entries of the run config's `planner.lessons` file appended by the loader) render in the system prompt after the rules. `embodied-harness lessons` (`runtime/trace/lessons.py`) drafts unreviewed entries from failure patterns seen at least 3 times.
- Thinking models: `extra_body` passes e.g. `chat_template_kwargs`; `planner/http.py` takes `reasoning_content` from the reply or strips a leading `<think>` block, and the reasoning goes to `meta.reasoning`.

`planner/robomme_subgoal.py` follows a separate contract:

- It reproduces RoboMME's own QwenVL prompt, including the subgoal history.
- It maps the answer onto a skill's single enum arg.
- It is meant for `call: periodic`.

### Data pipeline (`data/`)

The input is a LeRobot v3 dataset annotated by lerobot-annotate (`language_persistent` column). The steps:

1. `data labels` writes a label map from the free-form subtask labels to skill calls, with word-overlap suggestions.
2. A human reviews the map and sets `reviewed: true`. The later commands refuse unreviewed entries.
3. `library` fills in the prompt library's episode counts and p50/p95 durations.
4. `relabel` copies the dataset with the library texts as subtasks, for executor training.
5. `planner` exports ms-swift/LLaMA-Factory JSONL, holding out whole episodes for validation.
6. `eval` scores a served planner open-loop on those held-out episodes.

Consecutive spans that map to the same call merge into one step, the same way the runtime runs a call until its verifier fires.

- `data planner` drops done targets when `planner.allow_done` is false (manifest `done_dropped`).
- With `planner.call: periodic` it also adds in-step "keep the running call" samples every `every_steps * fps / control_hz` frames (`--periodic-limit` caps them; manifest `periodic` section). For policy skills without success conditions, the next decision becomes a periodic switch sample. Empty `--frame-offsets` is refused.

Later additions around that pipeline:

- `data annotate` runs lerobot-annotate as a subprocess on a copy of the dataset, steering its `plan_subtasks` / `plan_subtask_relabel` / `plan_memory` prompts through `LEROBOT_PROMPT_OVERRIDE_<name>` with the prompt library as the label vocabulary. `data labels --accept-exact` (and `annotate --labels`) auto-review labels that are exactly one call's library text.
- `data annotate --vlm-api openai` (`data/vlm_proxy.py`) starts a 127.0.0.1-only Chat Completions endpoint that adds the key from `OPENAI_API_KEY` (never from the command line) and forwards to OpenAI's Responses API (or Chat Completions with `--vlm-endpoint chat`), fixing what reasoning models reject (`max_tokens`, `temperature`), retrying 429/5xx and counting tokens into `.embodied_harness/usage.json`. `--episodes` maps to lerobot's `--only_episodes` for trial runs. `--vlm-api codex` answers each request with `codex exec` instead (`CodexExec`: images attached in order, read-only sandbox in an empty dir, a short annotator system prompt via `model_instructions_file`, optional features disabled, usage from the `turn.completed` event), so Codex's own provider and login are used and no key passes through the harness.
- `relabel --in-place` rewrites the annotated dataset itself, so one dataset serves the planner (task = episode goal, subtask + memory rows) and the policy (active subtask as prompt).
- `policy/integrations/lerobot.ActiveSubtaskAsTaskStep` is a LeRobot ProcessorStep that sets `task` to the active subtask from `language_persistent` during training (a no-op at inference). `embodied-harness policy subtask-checkpoint` inserts it into a checkpoint's `policy_preprocessor.json`. Pure logic lives in `policy/integrations/language.py`; only `policy/integrations/lerobot.py` imports lerobot.
- `data from-runs` (needs lerobot + PyAV) writes runs recorded with `trace.frames` as a LeRobot v3 dataset, with language rows written through lerobot-annotate's own writer.
- `embodied-harness report` (`runtime/trace/report.py`) reads traces only: per-skill and per-transition success, and one cause per failed episode (planner / transition / skill / harness / other; transition = failed right after a different call, sooner than the skill's median successful duration). Each failed call also gets a kind (`failure_kind`): the `kind` its verifier failure condition names (carried in the outcome evidence), else readiness (stall, or failed within `--early` of the typical duration), control (a later failure condition) or unknown (timeouts, env ends).
- `data coverage` (`data/coverage.py`) counts transitions (previous call -> call, START first) in the annotated dataset and looks up the transitions runs needed; `runtime/trace/report.composition_steps` folds retries into one step and is shared with the suite summary.

Evaluation tools around runs:

- `embodied-harness suite run|report` (`runtime/suite.py`, schema `SuiteFile` in `config/schema.py`): entries are run configs with `overrides` / `task_overrides`, tagged `category` A/AP/C/CP and `split` seen/unseen. Each entry directory keeps the merged `run_config.yaml` (absolute paths) and `task.yaml`; the summary lists transitions only unseen entries exercise.
- `embodied-harness probe` (`runtime/probe.py`): see traces above; it needs an env with `restore()`.
- `embodied-harness policy steer-check` (`runtime/steer.py`): sends the same observations (recorded `trace.frames` or env resets) with each call's prompt, `--repeats` times, and compares the separation of the prompts' mean chunks with the repeat noise (or with the action scale for a deterministic policy). Exit code 1 when a prompt pair is not distinguished. `read_video_frames` is the seam tests monkeypatch.

### Adding a backend

- **Registration:** add the class to `ENVS`, `EXECUTORS` or `PLANNERS` in `runtime/factory.py`.
- **Constructors:**
  - An env is built as `cls(params)`, with `params` as a dict.
  - An executor is built as `cls(library, **params)`.
  - A planner is built as `cls(**params)`.
  - A `TypeError` or `ValueError` raised in a constructor is reported as a config error.
- **Required hook:** envs need a classmethod `proprio_keys(params)`.
- **Optional hooks** are detected by duck typing (`getattr` + `callable`):
  - envs: `camera_keys(params)`, `instruction()` for `task.goal_source: episode`, and `snapshot()`/`restore()`;
  - executors: `begin_episode(task)` and `close()`;
  - planners: `bind(registry=, task=, planner_config=)`. `data planner` additionally needs `render_request`, `settings()`, `cameras`, `proprio_keys` and `image_extension`.
- **Imports:** the CLI imports every adapter through the factory. Import heavy dependencies (robomme, gym-aloha, websockets, msgpack, torch, lerobot, pyarrow, PIL) inside functions.
- **Policy implementations:** Model adapters live under `policy/`, grouped into `vla/`, `va/`, `rl/`, and `wam/`. OpenPI and LeRobot adapters currently live in `policy/vla/`; the other families are reserved packages. Shared WebSocket serving stays in `policy/serving/` and the client/codec stay in `policy/transport/`. `python -m embodied_harness.policy.vla.openpi` serves OpenPI; `python -m embodied_harness.policy.vla.lerobot` serves a LeRobot checkpoint. Model launch aliases live under `embodied_harness.policy.serving.*_policy`; the former top-level executor/serving/transport/integrations packages have been removed.
- **Proxies:** the planner HTTP client and the websocket client ignore proxy environment variables on purpose.
- **Policy server limits:** the websocket policy server caps request frames at `max_size` (64 MiB default, `--max-request-mb`), and both msgpack codecs decode only bool/int/uint/float dtypes.

## Tests

- **Helper imports:** `tests/` has no `__init__.py`. Tests import helpers directly, e.g. `from conftest import run_episode` or `from fake_openai import FakeChatServer`.
- **Config variants:** `conftest.make_bundle(tmp_path, run=..., task=..., skills=..., library=...)` deep-merges patches into temporary copies of the mock configs. A patch dict containing `"__replace__": True` replaces that section instead of merging.
- **Whole episodes:** `run_episode(...)` runs a full mock episode with three packages, all barcodes up. The env, planner, vla and commands can be injected. Assert on the trace with `read_events` and `of_type`.
- **Fakes:**
  - vLLM is faked by `FakeChatServer`, through the `servers` fixture.
  - Annotated LeRobot datasets come from `tests/lerobot_fixture.py`, which uses pyarrow only, no lerobot.
  - `data.commands.frame_source` is the seam tests monkeypatch to supply frames.

## Conventions

- **Python version:** code targets Python 3.10 (`target-version = "py310"`). Every non-`__init__` module starts with `from __future__ import annotations`.
- **Language:**
  - Code, comments and docstrings are in English.
  - README, planner prompts and task goals are in Chinese.
  - ruff ignores RUF001–003 so the Chinese full-width punctuation passes lint.
