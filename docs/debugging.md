# 轨迹、日志与排错

## 轨迹目录

每次 `embodied-harness run` 写到 `<trace.root>/<run_id>/`（`--run-id` 不给时用时间戳）：

| 文件 | 内容 |
| --- | --- |
| `run.log` | 整次运行的日志：版本、Python、配置里和默认值不同的字段 |
| `ep_0001/events.jsonl` | 事件流，见下表 |
| `ep_0001/planner/pc_0001.json` | 每次规划调用：输入摘要、渲染后的 prompt、原始输出、解析结果、错误、延迟 |
| `ep_0001/planner/pc_0001/<相机>.<扩展名>` | 这次调用实际发给模型的图片，重放时原样再发 |
| `ep_0001/actions.jsonl` | 每个执行的动作和它的指令版本 |
| `ep_0001/episode.log` | JSON Lines 日志；记录可用时附带 episode、状态、技能调用 ID、规划调用 ID |
| `ep_0001/outcome.json` | 结果：状态、错误码、账本、记忆、调用次数、健康统计、`env_success` |
| `ep_0001/config.snapshot.yaml` | 解析后的完整配置 |
| `ep_0001/env_snapshot.json` | 环境实现 `snapshot()` 时，在技能边界保存的世界状态；续跑时只有同时实现 `restore()` 的环境才能恢复。没有快照的仿真环境（当前 RoboMME 适配器即如此）无法从物理状态续跑，`--resume` 会拒绝 |
| `ep_0001/terminal.json`、`ep_0001/terminal/*.png` | trace.frames 开启时保存的真实终态（无伪造动作），用于 planner done 样本 |
| `ep_0001/episode.mp4` | `trace.video: true` 时：每个控制周期的相机画面横向拼接。第 k 次 `--resume` 写到 `episode_resume_<k>.mp4`，不覆盖之前的视频 |
| `ep_0001/frames/<相机>.mp4`、`ep_0001/frames.jsonl` | `trace.frames: true` 时：每路相机的原始画面，以及每个控制周期的本体、实际发出的动作、动作来源、技能和提示词。`data from-runs` 用它们导出 LeRobot 数据集。第 k 次 `--resume` 的画面写到 `frames/resume_<k>/<相机>.mp4`；`frames.jsonl` 每行的 `segment`（0 是第一次运行，k 是第 k 次续跑）和 `frame`（在该段视频里的帧号，没有视频时为 null）指明它的画面在哪，`tick` 跨续跑连续编号 |
| `ep_0001/snapshots/sc_0001.json` | `trace.snapshots: true` 时：每个技能调用开始时的世界状态，`embodied-harness probe` 从这里重跑 |

`outcome.json` 里的 `status`：`done`（harness 判断任务完成）、`aborted`（放弃）、`ended`（环境自己结束了 episode，看 `env_success`）。`env_success` 是环境自己的结论（`info["success"]`）：环境结束 episode 时读一次；harness 以 `done` 结束时，环境报告了这个标志（仿真）也读一次。评分规则（`run` 的退出码、`report` 和它的计划进度、`suite`、`data from-runs --success-only`）：有 `env_success` 时按它，没有（真机、旧轨迹）时按 `status == done`。所以仿真里规划器声称完成、环境却说没完成的 episode 记为失败，计划进度也只算到它真正做完的步骤。

`events.jsonl` 和 `actions.jsonl` 里的 numpy 数组和标量（策略输出的动作、`boundary` 事件里本体的 `state`）记成精确的 JSON 数字和列表。之前录的轨迹里它们是 numpy 打印出的文本（最多 8 位小数，长向量还会换行），读旧轨迹的脚本两种都要认。规划记录 `planner/pc_*.json` 的输入本体里，numpy 值保留实时 prompt 显示的文本，所以 `replay` 和 `data from-runs` 从规划记录重建的输入，渲染出的本体和当时 prompt 里的一样。`data from-runs` 最后的 done 样本例外：它的本体取自 `terminal.json`，那里 numpy 值是数字列表。

## 事件

`events.jsonl` 每行一个事件，都有 `seq`、`t`（仿真或真实时间）、`wall`、`type`；`state`、`skill_call_id`、`planner_call_id` 会在对应的运行上下文存在时附上，启动和指令事件可能没有调用 ID：

| 事件 | 含义 |
| --- | --- |
| `episode_start` / `episode_end` / `episode_resume` | episode 开始、结束（带结果）、续跑 |
| `instruction` | 本 episode 的指令（`goal_source: episode`） |
| `state_change` | 调度器状态变化 |
| `planner_request` / `planner_response` | 规划请求（原因、第几次重采样）和结果（决策、延迟） |
| `decision_rejected` | 回答没通过检查，附原因 |
| `planner_error` | 规划调用出错或超时 |
| `planner_confirm` | `periodic` 模式下规划器给出和当前相同的技能 |
| `switch_blocked` | `periodic` 模式下想切换（或声称完成）被挡下：`reason: switch_when` 是门控不允许；`reason: success_pending` 是回答到达时当前技能的成功正在判定（成功条件已成立，还在等 `hold_s` 或 `min_duration_s`），等它成功后再问规划器。判定期间不发起新检查，所以只有 `threaded: true` 时晚到的回答会这样被挡 |
| `planner_done_ignored` | `done_policy: continue` 时执行中的 `done` 不成立，技能继续执行，下一次检查带着拒绝原因再问；`reason` 是 `done_when_false`，或 `done_not_allowed`（没写 `done_when` 且 `planner.allow_done: false`） |
| `memory_update` | 规划器写下的新记忆 |
| `boundary` / `skill_start` / `skill_end` | 技能切换（带切换时的本体）、开始（提示词原文、超时）、结束（结果、原因、证据）。每个 `skill_start` 都有一个 `skill_end`：`success`、`failure`，或 `preempted`（没有校验结论就停了；原因 `planner` / `planner_done` 是 `periodic` 规划器换掉了它，`hold` 是 HOLD 打断了它，证据里有 HOLD 的错误码和原因） |
| `recovery` | 失败后的处理：重试或交回规划器、HOLD |
| `transition` | 衔接规则生效：规则名、条件、要跑的原语（带参数时是 `approach(color=green)` 这样的调用）、前后两个调用 |
| `ledger_update` / `check` | 账本更新，完成条件检查（`claimed_by_planner` 为真表示检查的是规划器的 `done`） |
| `hold` / `human_command` | 进入 HOLD（原因、错误码），人工命令；`failed_call` 是命令作用的调用，即 HOLD 针对的调用（见 [核心概念](concepts.md#hold-针对哪个调用)），没有时为 null |
| `env_end` | 环境结束了 episode（状态、`success`） |
| `error` | harness 内部错误 |

几个常用的排查脚本（服务器上的 `/workspace/scripts/0927_robomme_pickx/`）：`inspect_episode.py` 打印一个 episode 的子目标时间线，`summarize_eval.py` 汇总一次评测的成功率和每个 episode，`summarize_seeds.py` 汇总多个策略种子。

## 失败归因报表

```bash
embodied-harness report runs/<run_id> [更多运行或 episode 目录] --by call --out report.json
```

只读轨迹，给出三张表：

- **每个技能**（`--by call` 时按带参数的调用分开）：开始了几次、成功、失败（按原因）、被抢占、成功率，以及成功时的典型耗时（中位数）。
- **每种衔接「前一个 → 下一个」**：下一个调用在这种衔接下的成功率。同一个技能在某种衔接后成功率明显偏低，就是衔接问题：可以补这种衔接的数据，也可以先加一条 [衔接规则](configuration.md#衔接规则transitions)。
- **每个失败 episode 的原因**：

| 原因 | 判定 |
| --- | --- |
| `success` | 环境判成功；环境不给结论时（真机、旧轨迹），以 `done` 结束 |
| `planner` | 规划器导致结束：输出一直不合法（包括反复声称完成而完成条件不成立）、调用超时、规划调用预算用完，或者它声称完成而环境判失败 |
| `transition` | 失败的调用紧跟在另一个不同的调用之后，而且比这个技能正常成功所需的时间还早就失败了（窗口取它成功时的典型耗时，至少 `--window` 秒，默认 3） |
| `skill` | 其他调用失败，包括同一个调用重试后仍然失败 |
| `harness` | 内部错误或配置错误 |
| `other` | 其他情况，包括完成条件成立而环境判失败（校验器放过了没做成的步骤） |

HOLD 打断的调用记为被抢占（`preempted`）。如果那次 HOLD 之后 episode 直接被放弃，或者环境在 HOLD 期间结束了 episode，它和运行到最后的调用一样记为 `aborted` 或 `ended_*`，失败归因到它身上。

每个失败的调用还分一个类别，出现在衔接表的「failure kinds」列、失败 episode 表的「kind」列，以及第一行的汇总里（如 `transition 4 (readiness 3, unknown 1)`）：

| 类别 | 意思 | 判定 |
| --- | --- | --- |
| `readiness` | 起点不对：上一个技能留下的状态，这个技能没法从那里开始 | 卡住（`stall`），或者在它典型耗时的前 1/3（`--early`）内就失败了 |
| `grounding` | 目标没找对 | 只能来自校验器：失败条件写了 `kind: grounding`（比如检查目标颜色的视觉校验器） |
| `control` | 开始正常，执行中失败 | 失败条件在更晚的时候触发，比如抓空、掉落 |
| `unknown` | 分不清 | 超时（起点不对的乱动和执行得慢都会超时）、环境在执行中结束了 episode |

失败条件写了 `kind` 时直接用它。这个分类参考了分层 VLA 衔接失败的分析（arXiv 2607.06256）：π0.5 的技能从干净起点单独跑成功率 77–100%，串起来就经常卡住，问题集中在「下一步能不能开始」「目标找没找对」「执行」三类。

局限：

- 这是只看轨迹的启发式判断。规划器选了一个合法但错误的技能，会被算成那个技能的失败；要分开两者，可以用固定顺序规划器（顺序一定对）跑一组对照。
- 本体校验器没发现的失败（比如方块放偏了，但夹爪已经松开），要到后面某一步才暴露，报表会把它算在暴露的那一步上。如果一个很简单的技能失败率很高，多半是它前面有没被发现的失败，考虑给前面的技能加更强的校验。
- 类别是推断的。要确认一个衔接是不是「起点不对」，用下面的 `probe` 做对照实验。

## 从同一个起点重跑（probe）

报表说某个衔接可疑时，可以直接做对照实验：从上一步留下的状态和从干净的起点，把下一个技能各跑若干次，比较成功率。需要仿真环境（提供 `snapshot()` 和 `restore()`），运行时打开 `trace.snapshots: true`，每个技能调用开始时的世界状态会存到 `snapshots/<call_id>.json`。

```bash
# 同一个 pick：干净起点（episode 的第一个调用）对比放完一个包裹之后
embodied-harness probe configs/my_task/run.yaml \
  --at runs/<run_id>/ep_0001:sc_0001 --at runs/<run_id>/ep_0003:sc_0007 --trials 10 --out probe.json

# 一个 episode 里每个主调用都从当时的起点重跑
embodied-harness probe configs/my_task/run.yaml --at runs/<run_id>/ep_0003 --trials 5

# 从同一个起点跑别的调用：比如换一个目标，看策略听不听指令
embodied-harness probe configs/my_task/run.yaml --at runs/<run_id>/ep_0003:sc_0007 \
  --call "pick(color=red)" --call "pick(color=green)" --trials 5
```

- 每次试验：`env.reset()`、恢复快照、只跑这一个调用直到校验器给出结果。不经过规划器、恢复表和衔接规则。账本恢复成当时的值，前置条件不满足的记为 not ready。
- 输出每个「起点 × 调用」的成功率、成功时的平均耗时和失败原因；`after` 列是这个起点之前的主调用（`START` 表示 episode 的第一个调用）。
- 干净起点成功、串联起点失败：这个技能没法从上一步留下的状态开始，加一条[衔接规则](configuration.md#衔接规则transitions)，或者补这种衔接的数据。两边都失败：是技能本身的问题。
- 试验之间只有环境或策略本身随机的地方会不同（flow matching 策略每次采样噪声不同）。快照是这个调用发出第一个动作时的世界：`handoff: finish_chunk` 下是上一个动作块剩下的几步执行完之后，和闭环里它真正开始的状态一致（还没开始动就被打断的调用保留派发时的快照）。
- 用不同调用比较目标时，结果由各技能的校验器判定，只能区分校验器分得出的目标。只看策略是否对不同提示词给出不同动作，用 `embodied-harness policy steer-check`，见[策略](policy.md#检查策略还听不听指令)。

## 从失败里起草经验

```bash
embodied-harness lessons runs/<run_id> [更多运行] --out configs/my_task/lessons.yaml
```

同一个衔接、同一个失败原因出现至少 3 次（`--min-count`）就起草一条经验；规划器多次在任务没完成时声称完成也会起草一条。草稿是 `reviewed: false` 的事实陈述，改写成建议、审核后，由运行配置的 `planner.lessons` 引用，见[经验文件](configuration.md#经验文件lessons)。

## 重放和续跑

```bash
# 用新的规划器重放历史调用，和当时的决策对比（开环，不会推进环境）
embodied-harness replay configs/my_task/run.yaml runs/<run_id>/ep_0001 --out replay.json

# 续跑中断的 episode：重建账本、memory 和调用计数；只有环境提供快照时才会恢复物理状态
embodied-harness run configs/my_task/run.yaml --resume runs/<run_id>/ep_0001
```

`replay` 使用记录的图像、本体、memory、动态 goal、`PlannerContext.history`、账本（`ledger`）和上一步结果（`last`）；旧轨迹若没有记录这些字段则回退到命令行配置，打开 `show_progress` 的规划器重放没有账本的旧记录会报错。它仍是开环比较，不会推进环境或重放执行器。`--resume` 先用 `episode_start` 里记录的种子调用 `env.reset()`（命令行的 `--seed` 被忽略，不同时日志里有警告；`episode_resume` 事件记下用的种子），再尝试恢复快照。声明了 `simulated = True` 却没有 `restore()` 的仿真环境（当前的 RoboMME 适配器）会被拒绝续跑：只 reset 会换一个场景，却接着用恢复出来的账本、记忆和历史，命令以退出码 2 结束，轨迹不被改动。对不支持快照的真实环境，继续运行前必须自行确认环境仍处于可接受状态。续跑的时间轴接在最后记录的事件、动作或帧之后；`frames.jsonl` 的未完成尾行会先截断，画面写到新的一段视频里，已录的视频不会被覆盖。`embodied-harness policy steer-check` 按每行的 `segment` / `frame` 取画面；在这之前续跑过的旧轨迹对不上号，会被拒绝。续跑从轨迹重建主调用历史、上一步结果及原 episode instruction；SequencePlanner 还会从 `planner_response` 的 `sequence_index` 和主技能结果恢复内部游标，在成功边界进入下一步，在执行中断、失败或衔接原语中断时重发当前步骤。旧轨迹没有 `sequence_index` 时，根据原始规划请求和响应重建；缺少必要事件或调用与配置不一致时会报错，避免误从头执行。这仍不是任意时刻的完整进程恢复。无快照环境的物理状态也不会恢复。events/actions 的未完成尾行会在追加前截断，完整但缺少换行的 JSON 会补上换行。

## 错误码

| 错误码 | 什么时候出现 | 怎么办 |
| --- | --- | --- |
| `E_CONFIG` | 配置加载或交叉校验失败；运行时表达式出错 | 按报错逐条改，`embodied-harness validate` 会一次列出所有问题 |
| `E_LIBRARY_MISMATCH` | 规划器训练用的提示词库（`planner.trained_library`）和执行器的库不一样 | 换成匹配的策略或规划器，或者重新训练规划器 |
| `E_PLANNER_INVALID` | 规划器的回答重采样 `max_resamples` 次后仍不合法，或者一轮规划里反复声称完成而 `done_when` 不成立（没写 `done_when` 而 `planner.allow_done: false` 时，规划器的 `done` 一律不成立） | 看 `decision_rejected` 事件的原因和 `check` 事件；常见的是前置条件不满足，或者没开约束解码 |
| `E_PLANNER_TIMEOUT` | 规划调用超时或出错，重试 `retries` 次后放弃 | 检查 vLLM 是否在运行，调大 `planner.timeout_s` |
| `E_EXEC_UNDERRUN` | 执行中策略连续 `underrun_limit_s` 秒拿不出动作 | 检查策略服务，调大 `request_timeout_s` |
| `E_BUDGET` | 时间、技能调用或规划调用的预算用完 | 调整 `task.budget`，或者查为什么反复调用 |
| `E_INTERNAL` | harness 内部异常 | 看 `episode.log` 里的 traceback |
| `E_CTRL_OVERRUN` | 真机时钟下某一拍超时（比如一次推理很慢），后面错过的几拍被跳过，不补发（DEBUG 日志；健康摘要里的 skipped） | 偶尔出现没关系；频繁出现就降低 `control_hz`，或者加大 `n_action_steps` 减少推理次数 |
| `E_ENV_DISCONNECT` | 预留，暂未使用 | |

## 常见问题

**非交互运行卡在 HOLD。** 终端上默认问人；在脚本里运行时加 `--hold-policy abort`（直接放弃）或 `resume`（让规划器重新规划，最多 3 次）。

**HOLD 时 `retry` 或 `skip` 没有反应。** 日志里有原因：这次 HOLD 不针对任何调用（比如技能成功后规划器一直输出不合法），只能 `resume` 或 `abort`；或者 `retry` 的调用前置条件现在不满足（比如 `pick` 被打断时已经抓住了包裹），改用 `skip`、`resume` 或 `abort`。HOLD 针对哪个调用，终端提示里那一行就是它。

**vLLM 请求走了代理。** 规划器默认不读 `HTTP(S)_PROXY`（`use_env_proxy: false`）。自己写脚本访问本地 vLLM 时加 `--noproxy '*'`。

**vLLM 报图片数超限。** `vllm serve --limit-mm-per-prompt '{"image": N}'` 里的 N 要不少于规划器的相机数。

**老版本 vLLM 不认 `response_format`。** 把 `structured_output` 改成 `guided_json`。

**第一次调用策略超时。** JAX（openpi）模型第一次推理要编译，把执行器的 `request_timeout_s` 调到 300。

**`data eval` 警告设置不一致。** 运行配置里规划器的相机、分辨率、本体字段、记忆长度、prompt 开关（`show_history`、`show_state`、`show_progress`）、经验或 `extra_body` 和导出时不同，模型看到的不是它训练时见过的东西。改回和 `manifest.json` 一致。关掉的开关和旧导出里没记录的字段视为相同。

**`data planner` 报 lerobot 找不到。** 它用 LeRobot 解码视频帧，要在装了 lerobot 的环境里运行。没有 torchcodec 时 LeRobot 会退回 pyav。

**`data annotate` 报缺 `openai`。** lerobot-annotate 的依赖没装，在 lerobot 的环境里 `pip install "lerobot[annotations]"`。

**`data annotate` 的标签很少对上提示词库。** 先 `--dry-run` 看 `.embodied_harness/prompts/plan_subtasks.txt` 里的词表对不对；再看演示里的动作是否真的是这些技能。VLM 太小时也会这样，换大一点的 `--vlm-model`。

**标签表缺条目。** 数据集里出现了标签表没有的子任务描述，重新运行 `embodied-harness data labels`，新条目会加进去，已审过的保留。

**记忆被截断。** 规划器写的记忆超过 `planner.memory_max_chars`，日志里有警告。导出训练数据时命令也会统计超长条数；超长的多就调大上限。
