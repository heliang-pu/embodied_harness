# 配置参考

一次运行由四个 YAML 文件描述，外加一个可选的经验文件。运行配置里写其他文件的路径，路径相对运行配置所在的目录。

```
run.yaml ──┬─ task:   任务文件      目标、规则、经验、可用技能、账本、完成条件、预算
           ├─ skills: 技能文件      技能词表、谓词、校验（含失败类别）、恢复、衔接规则、效果
           ├─ executor.library: 提示词库   策略训练过的提示词原文，随策略权重发布
           └─ planner.lessons:  经验文件   可选，从以往运行总结、审核过的经验
```

评测套件文件（[评测套件](#评测套件suite)）是另一种文件：它列出若干运行配置，按组合泛化的类别分开跑、分开汇总。

用 `embodied-harness validate run.yaml` 检查：四个文件一起加载，交叉校验（技能引用的提示词是否存在、表达式里的名字是否都有定义、规划器要的相机环境是否提供、规划器训练用的提示词库版本是否一致等），所有问题一次列出来。它不会连接策略服务或 vLLM，也不会实例化环境、执行器和规划器；`env.params`、`executor.params`、`planner.params` 中由具体后端解释的字段，要到实际构造或运行时才会检查。

Schema 定义的结构字段会拒绝未知名字；三类组件的 `params` 是开放字典，由各后端负责解释。当前相机静态检查只覆盖 `planner.params.cameras`，不会检查 `robomme_subgoal` 的单数 `camera` 或执行器的 `images` 映射。

## 运行配置

例子：[`configs/runs/mock_sort.yaml`](../configs/runs/mock_sort.yaml)、[`configs/robomme/mme/run_mme_pickx.yaml`](../configs/robomme/mme/run_mme_pickx.yaml)。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `name` | 必填 | 运行名 |
| `task` | 必填 | 任务文件路径 |
| `skills` | 必填 | 技能文件路径 |
| `env.type` / `env.params` | 必填 / `{}` | 环境类型和参数，见[内置组件](#内置组件) |
| `executor.type` | 必填 | `websocket`（策略服务）或 `mock` |
| `executor.library` | 必填 | 提示词库路径 |
| `executor.handoff` | `finish_chunk` | 切换技能时旧动作块怎么处理：`finish_chunk` 执行完再切，`hard` 立即丢弃 |
| `executor.params` | `{}` | 执行器参数 |
| `planner.type` / `planner.params` | 必填 / `{}` | 规划器类型和参数 |
| `planner.call` | `boundary` | `boundary`：技能结束时才问；`periodic`：执行中达到 `every_steps` 且动作块耗尽时也问，通过检查且门控允许的不同回答会触发切换；当前技能的成功正在判定时（成功条件已成立，还在等 `hold_s` 或 `min_duration_s`）先不问也不切，等它判成功、`effects` 记进账本后再问 |
| `planner.every_steps` | 16 | `periodic` 模式的间隔（控制步） |
| `planner.threaded` | `false` | 在后台线程调用规划器。真机上设为 `true` |
| `planner.timeout_s` | 10 | 调度器轮询中的规划超时，按运行时钟计算。同步调用不能被它中断，HTTP 等待还受 `planner.params.request_timeout_s` 限制 |
| `planner.retries` / `retry_backoff_s` | 3 / 0.5 | PLAN 状态超时或出错后的重试次数和初始退避，退避每次翻倍；用完进入 HOLD。执行中的 periodic 检查出错会保留当前技能 |
| `planner.max_resamples` | 2 | PLAN 状态回答不合法时的重采样次数；执行中的 periodic 无效回答保留当前技能，不使用此重采样循环 |
| `planner.done_policy` | `resample` | 执行中的 periodic 检查说 done、而这个 done 不成立（`done_when` 不成立，或 `allow_done: false`）时怎么办：`resample` 按被拒的回答处理（结束当前技能、重新规划，重采样用完进入 HOLD）；`continue` 保留当前技能，下次检查再问 |
| `planner.allow_done` | `true` | `false`：完成只由环境的终止信号决定，任何规划器说 done 都不算数（按 `done_policy` 处理）。`openai_compat` 的输出 schema 去掉 done，system prompt 改为「始终选择下一个技能，不要输出 done」，`data planner` 也不导出 done 样本。它是 `planner` 下的字段，写在 `planner.params` 里会报配置错误 |
| `planner.memory_max_chars` | 200 | 记忆的最大字符数，超出截断。训练数据导出也按这个截断 |
| `planner.trained_library` | 无 | 规划器训练时用的提示词库名。和执行器的提示词库不一致时拒绝启动（`E_LIBRARY_MISMATCH`） |
| `planner.trained_manifest` | 无 | 规划器训练数据的 `data planner` 导出目录或其中的 `manifest.json`，路径相对运行配置。每个 episode 开始时用 `data eval` 的规则对比规划器设置（`settings()`）和导出时记下的设置，每个不一致的设置记一条 WARNING，不阻止运行；`validate` 提前做同样的对比。只有这两处读它：文件不存在或读不了（比如导出在另一台机器上，或者还没导出）也只是一条警告，`data` 命令不读它 |
| `planner.lessons` | 无 | 经验文件路径，其中审核过的条目会追加到任务的 `lessons`，见[经验文件](#经验文件lessons) |
| `runtime.clock` | `sim` | `sim` 仿真时钟，`real` 真实时钟 |
| `runtime.control_hz` | 10 | 控制频率 |
| `runtime.health_every_s` | 10 | 每隔多少秒写一行健康摘要：实际控制频率、周期耗时 p95、超时次数、跳过的拍数（真机时钟）、动作队列深度、规划延迟 p50/p95、重采样次数 |
| `runtime.underrun_limit_s` | 1.0 | 执行中策略连续这么久拿不出动作就 HOLD（`E_EXEC_UNDERRUN`） |
| `runtime.primitive_steps` / `default_primitive_steps` | `{}` / 5 | 脚本原语（`executor: scripted` 的技能）各执行多少步 |
| `runtime.seed` | 0 | `run` 没给 `--seed` 时第一个 episode 的种子，之后每个 episode 加 1，所以各 episode 的场景不同、可复现；评测套件的项没写 `seed` 时也从这里开始。评测配置可以设成训练数据没用过的种子，比如 RoboTwin 评测配置设 1000（演示用的是 0–79）。RoboTwin 直接把 episode 的种子当场景种子，`params.episode_index` 只在 `reset()` 没收到种子时才用。`--resume` 仍用 episode 记录的种子 |
| `trace.root` | `runs` | 轨迹根目录，相对运行命令时的当前目录 |
| `trace.video` / `video_fps` | `false` / 10 | 把每个控制周期的相机画面拼成 `episode.mp4`（需要 imageio 和 ffmpeg） |
| `trace.frames` | `false` | 逐帧记录各路相机原始画面、本体、实际发出的动作和动作来源，用于 `data from-runs` 导出成 LeRobot 数据集（需要 imageio 或 PyAV） |
| `trace.snapshots` | `false` | 每个技能调用开始时保存一份世界状态（`snapshots/<call_id>.json`），供 `embodied-harness probe` 从同一起点重跑。只对提供 `snapshot()` 的仿真环境有效 |
| `logging.console_level` / `file_level` | `INFO` / `DEBUG` | 控制台和 `episode.log` 的日志级别 |

## 任务文件

例子：[`configs/tasks/sort_by_size.yaml`](../configs/tasks/sort_by_size.yaml)。字段都在 `task:` 下面。

| 字段 | 说明 |
| --- | --- |
| `id` | 任务 ID |
| `goal` | 任务目标，原样写进规划器的 prompt |
| `goal_source` | `config`（默认）：用上面的 `goal`；`episode`：每个 episode 用环境给的指令（`env.instruction()`）替换 `goal`，导出训练数据时用数据集的指令列 |
| `rules` | 规则列表，写进 `openai_compat` 的 prompt；RoboMME SimpleSG 的原始协议不渲染这个字段 |
| `lessons` | 经验列表，写在 prompt 的规则后面（「经验（以往运行的总结）」）。`planner.lessons` 文件里审核过的条目会追加到这里 |
| `skills` | 这个任务可用的技能，必须在技能文件里定义 |
| `expected_total` | 任务的目标数量，可以在 `done_when` 里用 `task.expected_total` |
| `ledger` | 账本字段：`{名字: {type, keys, values, init}}`，`type` 是 `counter`（需要 `keys`）、`int`、`bool`、`str`、`enum`（需要 `values`） |
| `done_when` | 任务完成条件，只能用 `ledger` 和 `task`。不写时相信规划器的 `done`；要是同时设了 `planner.allow_done: false`，规划器的 `done` 不算数，只有环境的终止信号能结束 episode |
| `budget` | `max_time_s`、`max_skill_calls`、`max_planner_calls`，用完进入 HOLD（`E_BUDGET`） |

## 技能文件

例子：[`configs/skills/sorting.yaml`](../configs/skills/sorting.yaml)、[`configs/robomme/composite/skills_composite.yaml`](../configs/robomme/composite/skills_composite.yaml)。下面是结构示例，任务引用的其他技能仍需完整定义。

```yaml
version: 1
library: mock_sort_v1            # 这份技能文件是给哪个提示词库写的；和执行器的库不一致时报错
predicates:                      # 每个控制周期由本体算出来的布尔量，可以在技能里引用
  holding: "0.01 < gripper_width < 0.08"
  released: "gripper_width > 0.08"
skills:
  place:
    description: 放进指定的箱子   # 可选
    executor: policy             # policy（默认，由策略执行）或 scripted（harness 原语）；旧写法 vla 等同 policy
    args:
      bin: {type: enum, values: [A, B, C]}
    prompt: {A: place_bin_a, B: place_bin_b, C: place_bin_c}   # 按参数值选提示词 id；无参数时直接写一个 id
    preconditions: ["holding"]   # 派发前检查，不满足就拒绝规划器的这个回答
    verify:
      success: [{when: released, hold_s: 0.3}]
      switch_when: [released]    # periodic 规划器只有在这些条件之一成立时才能把这个技能换掉；成功正在判定时也要等校验器的结论
    min_duration_s: 0
    timeout_s: 20                # 不写时：提示词库的 p95 时长 × defaults.timeout_factor，否则 defaults.timeout_s
    stall_s: 3                   # 本体这么久不变算卡住
    on_failure:                  # 按失败原因恢复；没列出的原因用 defaults.on_failure
      timeout: {retry: 1, before_retry: [retract], escalate: planner}
    effects:
      on_success: ["ledger.sorted[args.bin] += 1"]
  retract:
    executor: scripted
    timeout_s: 5
defaults:
  on_failure: {escalate: planner}   # retry: 0；escalate: planner 或 hold
  max_consecutive_failures: 3       # 当前按具体调用累计失败（本地重试也算），该调用成功或人工 retry/resume 后清零
  timeout_factor: 2.0
  timeout_s: 30
```

- `prompt` 的映射只支持一个参数。技能有多个维度时（比如颜色和次数），把它们合成一个参数，或者像 RoboMME 的配置那样直接用提示词原文当参数值。
- `executor: policy` 的技能必须有 `prompt`；`scripted` 的技能由 harness 自己执行，步数在 `runtime.primitive_steps` 里配置。
- `verify.failure` 与 `success` 使用相同的条件格式，可额外写 `reason`，不写时取 `when` 表达式原文。失败条件先判定；不要把 place 的正常松开同时配置成 `not holding` 失败，否则它会优先于成功触发。
- 失败条件还可以写 `kind`，告诉失败报表这是哪一类失败：`readiness`（没能从给定的起点开始）、`grounding`（目标没找对，比如一个检查目标颜色的视觉校验器）、`control`（开始正常，执行中失败）。不写时报表按失败的时间推断，见 [排错](debugging.md#失败归因报表)。成功条件不能写 `kind`。

  ```yaml
  verify:
    failure: [{reason: dropped, when: "not holding", hold_s: 0.3, kind: control}]
  ```

- `stall_s` 判的是「本体完全不动」：每一步所有本体字段的变化都不超过 1e-4，连续这么久才算卡住（失败原因 `stall`）。它能抓到策略停下来不动的情况，抓不到一直在小幅推动的卡住：ALOHA 仿真里插销顶在插座上插不进去时，关节每步仍抖动 2–4e-3 弧度，这种只能靠超时。
- 本地重试也计入 `max_consecutive_failures`：失败一次加一，只有这个调用成功（或人工 retry/resume）才清零。所以一条 `escalate: planner` 的恢复规则，`retry + 1` 必须小于 `max_consecutive_failures`，否则重试用完之前就已经 HOLD，规划器永远收不到这次失败。加载时会检查这一点（`defaults.on_failure` 也查），不满足就报配置错误，提示把 `retry` 调小或把上限调大。

### 衔接规则（`transitions`）

技能文件里还可以写衔接规则：从一个技能切到下一个技能时，如果条件成立，harness 先执行指定的脚本原语，再开始下一个技能。规划器看不到这些原语，它的决策和调用历史里只有它自己选的技能。用途是让下一个技能从策略训练时见过的状态开始，比如换一个物体之前先回到初始姿态。

```yaml
transitions:
  - name: home_before_new_object          # 可选，会出现在轨迹和日志里
    when: "previous.skill == 'place' and next.skill == 'pick' and next.args.color != previous.args.color"
    run: [go_home]                        # 依次执行的脚本原语（executor: scripted 的技能）
```

- `when` 能用的名字：`previous` 和 `next`（都是 `{skill, args}`；第一个技能之前 `previous.skill` 是 null），以及本体字段、谓词、`ledger`、`task`。表达式里的 `and` 会短路：先判断 `previous.skill`，再读它的参数，就不会因为参数不存在而出错。
- 规则在每次派发主技能时按顺序检查，第一条成立的生效。重试同一个调用时也会检查（这时 `previous` 和 `next` 相同）。规则或恢复表插入的原语跑完、再派发原来的技能时，不再重复检查。
- 原语失败就进入 HOLD。
- 该加哪些规则，看 `embodied-harness report --by call` 的衔接表：成功率低、而且失败发生在切换后不久的衔接，就是候选，见 [排错](debugging.md#失败归因报表)。

原语也可以带参数，参数值在规则生效时用同样的表达式算出来。这样一个原语能针对下一步的目标做准备，比如「移动到下一个要抓的物体上方」，而不只是回到固定姿态：

```yaml
skills:
  approach:
    executor: scripted
    args: {color: {type: enum, values: [red, green, blue]}}
    timeout_s: 5
transitions:
  - name: approach_next_object
    when: "previous.skill == 'place' and next.skill == 'pick'"
    run:
      - go_home                                          # 不带参数的原语照旧写名字
      - {skill: approach, args: {color: "next.args.color"}}
```

- 每个参数都必须给，名字必须是这个原语定义过的；算出来的值不在枚举里时 HOLD（`E_CONFIG`）。
- 环境收到的动作是 `{"primitive": "approach", "args": {"color": "green"}, "version": ...}`，怎么执行由环境决定，见 [接入](integration.md#环境--机器人)。移动到物体上方需要知道物体在哪：在真机上要来自感知模块，不能用仿真真值。
- 一个衔接到底是不是「起点不对」，可以用 `embodied-harness probe` 从同一个起点反复重跑下一个技能来验证，见 [排错](debugging.md#从同一个起点重跑probe)。

## 提示词库

例子：[`configs/libraries/mock_sort_v1.yaml`](../configs/libraries/mock_sort_v1.yaml)。

```yaml
library: mock_sort_v1          # 库名，也是版本号：改了提示词就换个名字
source: 从哪份数据、哪次训练导出的
prompts:
  - id: place_bin_a
    text: "place the package into bin A"     # 策略训练时见过的原文，一字不差
    episodes: 42                              # 以下由 `embodied-harness data library` 统计填写
    duration_s: {p50: 2.5, p95: 3.0}
```

提示词库是策略的一部分，放在权重旁边一起发布。`duration_s.p95` 用来推出技能的默认超时。

## 经验文件（lessons）

经验是写给规划器的、从以往运行里总结出来的话，比如「放完 A 区之后先 retract 再 pick」。它们出现在 prompt 的规则后面。可以直接写在任务文件的 `lessons` 里，也可以放在单独的文件里，由运行配置的 `planner.lessons` 引用：

```yaml
version: 1
lessons:
  - text: 放完之后先 retract 再 pick           # 规划器看到的原文
    reviewed: true                           # 只有审核过的条目会给规划器
    source: "report: place(bin=A) -> pick, empty_grasp, 4/5 failed, readiness"   # 从哪来的，可选
```

- `embodied-harness lessons RUNS --out lessons.yaml` 从运行记录里起草经验：同一个衔接、同一个失败原因出现至少 3 次（`--min-count`），或者规划器多次在任务没完成时声称完成，都会写成一条 `reviewed: false` 的草稿。草稿只陈述事实，把它改写成规划器能照着做的建议，再设 `reviewed: true`。再次运行时，已有条目和已起草过的模式不会重复添加。
- 评测时固定一个经验文件：同一组对比的运行都用同一份经验，结果才有可比性。`openai_compat` 的 `settings()` 记录经验的指纹（`lessons`），`data eval` 发现训练和评测时的经验不同会警告。
- 经验会改变 prompt。微调过的规划器如果训练时没见过经验，就不要临时加上；要用就把经验一起写进训练数据（`data planner` 用同一个运行配置导出即可）。
- `embodied-harness validate` 会列出有多少条经验给了规划器、多少条还没审核。

## 评测套件（suite）

组合泛化要分开测：单个任务链的成功率不能说明它能不能组合出新任务。套件文件把一组运行配置列成一张评测表，每一项标上类别和划分：

```yaml
suite: mock_sort_composition
root: runs/suites                 # 结果写到 <root>/<suite>/<run id>/<项名>/，相对当前目录
entries:
  - name: one_package
    config: ../runs/mock_sort.yaml       # 相对套件文件
    category: A                          # A 原子、AP 原子+扰动、C 组合、CP 组合+扰动
    split: seen                          # 这个组合在训练数据里有没有：seen / unseen
    episodes: 3
    seed: 0                              # 第一个 episode 的种子，之后每个加 1；不写时用运行配置的 runtime.seed
    overrides:                           # 合并进运行配置
      env: {params: {num_packages: 1}}
    task_overrides: {expected_total: 1}  # 合并进任务文件的 task
    note: 可选的说明
```

- `embodied-harness suite run SUITE [--run-id ID] [--only 项名 ...]` 依次跑每一项，每项目录里存着合并后的 `run_config.yaml`（和改过的 `task.yaml`），记录实际跑的是什么；`embodied-harness suite report SUITE RUN_DIR` 从已有结果重新汇总。
- 汇总三张表：每项的成功率和失败原因；类别 × 划分的成功率；只有 unseen 项才走过的衔接（上一步 → 下一步）及其成功率，也就是新组合要求而见过的任务都没有的步骤顺序。
- 用 `sequence` 规划器（给定正确的步骤顺序）跑，测的是策略和衔接；换成 VLM 规划器，测的是整个分层系统。只有策略、衔接规则、切换方式和预算相同时才适合对照；差值也包含固定序列与自适应规划器恢复行为的差异，不能直接当作纯规划器误差。例子见 [`configs/suites/mock_sort_composition.yaml`](../configs/suites/mock_sort_composition.yaml)。
- 类别和划分参考 RoboHiMan（arXiv 2510.13149）的 A/AP/C/CP 四类任务和规则规划器 / VLM 规划器分开评测的做法。

## 表达式

谓词、条件、前置条件、`switch_when`、`done_when` 都是受限表达式。加载配置时就检查语法和名字，运行时不会调用 `eval()`。

- 支持：字面量（数字、字符串、`true`/`false`/`null`）、名字、`a.b` 取字段、`a[b]` 下标、`+ - * / // %`、比较（含 `in`、`not in`、链式比较 `0.01 < x < 0.08`）、`and` / `or` / `not`、`x if c else y`、列表和元组。
- 函数：`abs`、`all`、`any`、`len`、`max`、`min`、`round`、`sum`（对 counter 求和时加的是计数）。
- 能用的名字：

| 位置 | 名字 |
| --- | --- |
| 谓词 | 环境的本体字段 |
| 技能里的条件 | 本体字段、谓词、`ledger`、`task`、`args`（当前调用的参数）、`executor_done` |
| 衔接规则的 `when` 和原语参数 | 上一行的名字（`args` 是下一个调用的参数），加上 `previous`、`next` |
| `done_when` | `ledger`、`task`（`task.id`、`task.goal`、`task.rules`、`task.expected_total`） |

- 效果（`effects.on_success`）只能写账本：`ledger.x = 值`、`ledger.x += 值`、`ledger.x -= 值`，counter 用 `ledger.x[键] += 1`。

## 内置组件

新的组件在 `src/embodied_harness/runtime/factory.py` 的三张表里注册，见 [接入新组件](integration.md)。

### 环境

**`mock_sorting`**：符号化的包裹分拣世界，不需要 GPU 和模型，用来调试流程。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `num_packages` | 10 | 包裹数 |
| `barcode_down_prob` | 0.3 | 条码朝下的概率（需要先翻面） |
| `fault_probs` | `{}` | 按提示词 id 注入失败，如 `{pick_next: {empty_grasp: 0.1}}` |
| `scripted_outcomes` | `{}` | 按提示词 id 依次强制指定结果，测试用 |
| `steps` | 内置 | 每个提示词执行多少步 |
| `seed` | 无 | `reset()` 没收到种子时用的随机种子。`run`、`suite` 和 `probe` 总会给种子，所以只在直接调用环境时起作用 |
| `cameras` / `image_size` | `[]` / `[96, 72]` | 合成相机画面，只用来走通图像链路 |

**`robomme`**：RoboMME 仿真（ManiSkill/SAPIEN，Linux + NVIDIA GPU）。详见 [RoboMME 实例](robomme.md)。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `task` | `PickXtimes` | RoboMME 任务名 |
| `split` | `test` | 数据划分；seed k 对应第 `first_episode + k` 个 episode |
| `first_episode` | 0 | 起始 episode |
| `max_steps` | 1300 | 最大步数，和 RoboMME 评测一致 |
| `difficulty` | 全部 | 只保留这些难度的 episode，如 `[medium, hard]` |
| `composite` | 无 | 组合任务，如 `[[red, 2], [green, 1]]`，只能用于 PickXtimes 场景 |

相机 `front`、`wrist`；本体 `gripper_width`（夹爪宽度）和 `state`（7 个关节加夹爪，8 维）。

**`aloha_sim`**：gym-aloha 的双臂 ALOHA 仿真（MuJoCo），需要 `gym-aloha`。没有显示器的 Linux 上默认用 EGL 渲染（`MUJOCO_GL=egl`，已设置时不覆盖）。seed k 用 k 重置仿真、摆放物体，所以 `run --seed 0 --episodes 50` 是固定的 50 个场景。任务成功（仿真给满分，比如插销插进插座）或走满 `max_steps` 步时，环境自己结束 episode，成功与否只用来评分。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `task` | `insertion` | `insertion`（插销）或 `transfer_cube`（递方块） |
| `max_steps` | 400 | 最大步数；50 Hz 下 400 步是 8 秒，和 scripted 演示一样长 |

相机 `top`；本体是 LeRobot ALOHA 数据集的 14 个值，逐个命名（`left_waist` … `left_gripper`、`right_waist` … `right_gripper`，关节单位是弧度，夹爪 1 张开、0 闭合），整体也放在 `state` 里给策略。动作是同样顺序的 14 个关节目标。`hold` 重发上一次的 14 个目标（刚重置时是开局姿态），夹爪指令也在里面，夹住的东西不会松开；它不按当前读数重设目标，否则手臂在重力下每步都会再往下沉一点。环境支持 `snapshot()`/`restore()`（MuJoCo 的完整积分状态加适配器自己的步数、判定和上一次目标），所以 `run --resume` 能回到中断时的场景，`trace.snapshots` 加 `probe` 也能用。例子见 [`configs/aloha_sim_insertion/`](../configs/aloha_sim_insertion/) 和 [ALOHA 仿真实例](aloha_sim.md)。

### 执行器

**`websocket`**：连接一个策略服务（openpi 协议），见 [策略](policy.md)。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `url` | 必填 | 如 `ws://127.0.0.1:8001` |
| `images` | 必填 | harness 相机名 → 请求里的键，如 `{front: observation/image}` |
| `state` | 无 | `{proprio: 本体字段, key: 请求里的键}`，把状态向量发过去 |
| `prompt_key` | `prompt` | 没有 `texts` 时提示词原文放在这个键 |
| `texts` | 无 | 请求键 → 模板，模板里可以用 `{prompt}`（提示词原文）和 `{goal}`（本 episode 的任务目标） |
| `actions_key` | `actions` | 回复里动作所在的键 |
| `n_action_steps` | 整块 | 每个动作块执行前多少步 |
| `extra` | `{}` | 每个请求都带上的固定字段 |
| `connect_timeout_s` / `request_timeout_s` | 120 / 30 | 连接等待时间 / 单次推理超时。JAX 模型第一次推理要编译，建议调到 300 |

**`mock`**：假策略，只配合 `mock_sorting` 使用。参数 `chunk_size`（默认 5）。

### 规划器

**`openai_compat`**：OpenAI 兼容接口的 VLM（vLLM 部署），用 harness 自己的 prompt 格式（`render_v1`），见 [VLM 规划器](planner_finetune.md)。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `base_url` | 必填 | 如 `http://127.0.0.1:8000/v1` |
| `model` | 必填 | `vllm serve --served-model-name` 的名字，或 LoRA 名 |
| `cameras` | 必填 | 发给模型的相机和顺序，必须和训练数据一致 |
| `proprio_keys` | 全部 | 写进 prompt 的本体字段，按这个顺序 |
| `proprio_digits` | 3 | 小数位数 |
| `image_size` | 原图 | `[宽, 高]`，发送前缩放 |
| `image_format` / `jpeg_quality` | `jpeg` / 90 | 图片编码 |
| `temperature` / `resample_temperature` | 0.0 / 0.7 | 首次采样温度 / 重采样温度 |
| `max_tokens` | 256 | |
| `request_timeout_s` | 30 | HTTP 超时 |
| `structured_output` | `response_format` | 约束解码方式：`response_format`（JSON schema），老版本 vLLM 用 `guided_json`，`none` 不约束 |
| `api_key_env` | 无 | 从这个环境变量读 API key（`vllm serve --api-key` 时用） |
| `use_env_proxy` | `false` | 默认不走 `HTTP(S)_PROXY`，避免本地 vLLM 的请求被代理截走 |
| `show_history` | `false` | prompt 里再列出已派发的主调用和这次被问的原因；历史不是成功清单。打开后训练数据也按这个格式导出，训练与部署应使用相同设置 |
| `show_state` | `false` | prompt 里加「当前状态」（技能文件的谓词，如 `holding=true`）和「上一步」（上一个主调用及结果，如 `pick 失败（empty_grasp）`） |
| `state_predicates` | 全部 | `show_state` 列出哪些谓词，按这个顺序 |
| `show_progress` | `false` | prompt 里加「进度」：账本的当前值，如 `sorted: A=1, B=0, C=0（共 1）；目标总数 3` |
| `progress_fields` | 全部 | `show_progress` 列出哪些账本字段 |
| `show_rejections` | `false` | 回答没通过检查而重采样时，prompt 里告诉规划器上一次为什么被拒（如 `pick: precondition not met: not holding`）。不打开时重采样只提高温度，零样本模型常常三次给出同一个错误答案。演示导出的样本里没有被拒的回答（运行记录导出的样本保留当时的拒绝原因），用演示微调的规划器一般不需要 |
| `extra_body` | 无 | 请求里再加的字段，如思考模型的 `{chat_template_kwargs: {enable_thinking: true}}`；不能覆盖 `model`、`messages` 和输出 schema |

`show_state` 和 `show_progress` 把规划器最容易看错的东西直接用文字给它：手里有没有东西、上一步成没成、已经完成了多少。分层 VLA 的系统性研究（arXiv 2606.10267）发现，任务越难 VLM 越不看图，给文字描述和接触信息后长任务成功率从 38.8% 升到 52.4%；而把历史原样堆进 prompt 基本没有帮助。这几个开关和 `show_history` 一样会改变 prompt，训练和部署要用同样的设置，`data planner` 会按设置导出（账本按演示里各步的效果重算）。`settings()` 除了上面这些参数，还记下 `planner.allow_done: false`（`no_done`）、`planner.call: periodic`（`call`）、system prompt 除任务目标外的指纹（`system`：规则、经验、技能和完成说明）和配置里任务目标的指纹（`goal`，`goal_source: episode` 时没有）。导出的 `manifest.json` 存下这些值，`data eval` 和 `planner.trained_manifest` 拿它们和部署时的设置对比；`data eval` 不比 `goal`，因为每条样本都按它记下的目标发送。旧导出没有记录的开关按默认值比较；没有 `system`、`goal` 指纹的旧导出比不了规则、经验、技能和目标，会多一条警告，提示重新导出。

思考模型（如 Qwen3-VL 的 Thinking 版本）：用 `vllm serve --reasoning-parser` 部署，思考内容会从回答里分开；没有分开时 harness 会把回答开头的 `<think>...</think>` 去掉。思考内容记在规划器调用记录（`planner/pc_XXXX.json`）的 `meta.reasoning` 里。思考要更多 token，把 `max_tokens` 和 `request_timeout_s` 调大。

**`robomme_subgoal`**：RoboMME 微调的 Qwen3-VL 子目标预测器，完全复现它的 prompt（任务目标、已派发的子目标历史、一张前视图）。输出的子目标文字映射到「唯一参数是枚举」的技能上；它返回的 harness 内部结果仍会包装成标准的 `skill/args/memory` 决策，但 `memory` 固定为空。配合 `call: periodic`、`every_steps: 16` 使用。参数：`base_url`、`model`（LoRA 名）、`camera`（默认 `front`）、`skill`（默认自动）、`image_format`（默认 `png`）、`max_tokens`（128）、`temperature`、`resample_temperature`、`request_timeout_s`（默认 60 秒）、`api_key_env`、`use_env_proxy`。

**`sequence`**：按固定顺序给出技能调用，失败后重复当前一步。HOLD 之后人工 resume，一般重发最近一次给出的那一步；如果从那次给出之后有主调用已经成功（比如预算用完、急停或 harness 出错发生在一步刚校验成功、下一次规划之前），就接着它的下一步，不会把做完的一步再做一遍。用作固定顺序基线；不具备自适应恢复能力，因此不是策略或任意规划器的严格上限。参数 `calls: [{skill: pick, args: {...}}, ...]`。

**`mock_oracle`**：`mock_sorting` 的上帝视角规划器，读仿真真值，只用于调试。参数 `size_to_bin`、`flip_barcode_down`。
