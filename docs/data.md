# 数据：需要什么、怎么标、怎么导出

策略和规划器用同一份数据集训练，各自只用其中一部分字段（对照表见 [策略](policy.md#训练数据和规划器用同一份数据集)）。harness 在这份数据集上提供这些工具：

| 产物 | 命令 | 给谁用 |
| --- | --- | --- |
| 规划器微调样本 | `embodied-harness data planner` | VLM 规划器 SFT，见 [VLM 规划器](planner_finetune.md) |
| 子任务换成提示词库原文（原地改写，数据集直接用于策略训练） | `embodied-harness data relabel --in-place` | 策略（VLA / WAM）训练，见 [策略](policy.md#训练数据和规划器用同一份数据集) |
| 提示词库统计（每句提示词覆盖的 episode 数、时长 p50/p95） | `embodied-harness data library` | 技能默认超时 |
| 离线评测报告 | `embodied-harness data eval` | 规划器上线前的检查 |

## 1. 演示数据集

格式是 LeRobot v3 数据集（`meta/info.json`、`data/**/*.parquet`，视频或图片特征都行）。需要：

| 内容 | 默认键 | 说明 |
| --- | --- | --- |
| 相机 | `observation.images.*` | 规划器部署时看哪几路相机，数据里就要有哪几路，视角和分辨率尽量一致 |
| 状态向量 | `observation.state` | 用来算规划器看到的本体字段（例如夹爪宽度），可以在映射文件里换键名 |
| 子任务和记忆标注 | `language_persistent` | 见下一节。策略只用子任务，记忆只给规划器 |
| 每个 episode 的指令（可选） | 自定义，如 `instruction` | `goal_source: episode` 时需要 Parquet 中实际存在的字符串列；不自动解析 LeRobot 的 task_index |
| `episode_index`、`frame_index`、`timestamp` | — | LeRobot 标准列 |

录数据时就要想好部署时的相机和本体：规划器训练时看到什么，部署时就只能给它看什么。

## 2. 子任务和记忆标注

标注写在 `language_persistent` 列。每个 episode 的每一帧都带着同一份行列表，harness 用其中两种行（每行至少有 `style`、`timestamp`、`content`）：

- `subtask` 行：每个子任务片段开始的时刻一行，`content` 是这一段在做什么，比如 "pick up the red cube"。
- `memory` 行：除了第一个片段，每个片段开始时一行，`content` 是到这一刻为止值得记住的事情的压缩描述，比如 "red cube placed once; next: pick it again"。

harness 这样把它还原成规划器的决策序列：

```
片段开始时刻:   t0            t1                 t2                 结束
subtask 行:    pick red      place red          press button
memory 行:     -             "拿起了红色方块"      "红色已放 1 次"      （结束时刻最近的一行）
规划器第 i 步:  输入记忆 = 第 i-1 步写下的记忆，输出 = {第 i 段的技能, 第 i 段开始时的记忆}
最后一步:       输出 {"done": true, 结束时的记忆}（条件见下）
```

**完成标签的前提**：读取器为每个有 subtask 的 episode 在末帧补一个 done 决策，导出时再决定它能不能成为样本：

- 任务配了 `done_when` 时，导出按演示里各步的 `effects.on_success` 重算结束时的账本（和运行时 supervisor 提交一次成功的算法相同），`done_when` 不成立就不导出这个 done 样本：运行时 supervisor 会拒绝这种 done，提前停录的演示（比如插销还没插进去）不能教规划器过早说完成。这类 episode 的编号记在 manifest 的 `unfinished_episodes`，命令会打印一行警告；它们其余步骤的样本照常导出。effects 读本体字段或谓词时，映射文件要给出这些字段，即使没开 `show_progress`。
- 所有片段都映射为 null 的 episode（没有任何技能调用步）不产生样本，也不补 done（否则就是一条开局就说 done 的样本），计入 manifest 的 `spans.skipped_episodes`，也不占验证集名额。
- 任务没有 `done_when` 时，完成由规划器判断，导出沿用原来的做法：每个演示的结束都当作 done，不做检查。这时用于导出的 episode 必须确实完成了任务；失败、中断、超时的轨迹不能不经审核就混入。
- 运行配置写了 `planner.allow_done: false` 时，规划器不能说 done（输出 schema 没有 done，system prompt 要它不要输出 done），所以不导出任何 done 样本（演示和运行记录都一样），去掉的样本数（含 `--frame-offsets` 的各帧）记在 manifest 的 `done_dropped`。

memory 写的是当前决策时已知的历史，不是将要执行技能的未来结果。

标注方式有两种。

**用 `embodied-harness data annotate`（推荐）**：内部调用 LeRobot 自带的 lerobot-annotate，让一个 VLM 看视频、切片段、写子任务描述和记忆，再按 harness 的需要做了这些调整：

- **在副本上标注**：lerobot-annotate 会直接改写数据集的 parquet 文件，所以先把数据集复制到 `--out`（视频用软链接，`--copy-videos` 改为复制），只在副本上跑，原数据集不动。
- **按提示词库给标签**：把运行配置里提示词库的原文作为标签词表写进它「切分子任务」的 prompt：能对上的片段就逐字用库里的句子，对不上的才自己写。这样大部分标签不用人审就能对应到技能调用。`--free-labels` 关掉这一项。
- **记忆**：记忆的 prompt 里加上规划器的记忆长度上限（`planner.memory_max_chars`）和任务规则，提醒它保留进度（比如放了几次）。
- **只做需要的部分**：只生成子任务和记忆，关掉 plan 行、任务改写、插话和 VQA，省下 VLM 调用。

prompt 是通过 lerobot-annotate 自带的覆盖机制（环境变量 `LEROBOT_PROMPT_OVERRIDE_<模板名>`）传进去的：只在它原来的模板里加段落，不改原文。

```bash
embodied-harness data annotate configs/my_task/run.yaml /path/to/dataset --out /path/to/dataset_annotated \
  --vlm-model Qwen/Qwen3-VL-8B-Instruct --labels configs/my_task/labels.yaml
```

| 参数 | 说明 |
| --- | --- |
| `--out` | 标注副本的目录，必须是新的或空的 |
| `--labels` | 标完直接生成（或更新）标签表；和某个技能调用的库原文逐字一致的标签自动标为已审，其余留给人审 |
| `--vlm-model` | 标注用的 VLM，不写时用 lerobot-annotate 的默认模型。它默认会自己起一个 vLLM 服务 |
| `--vlm-url` | 用一个已经在运行的 OpenAI 兼容服务（比如自己起的 `vllm serve`），不再另起 |
| `--camera` | VLM 看哪一路相机（数据集的图像键），默认第一路 |
| `--free-labels` | 不给词表，让 VLM 自由描述 |
| `--copy-videos` | 复制视频而不是软链接 |
| `--dry-run` | 只生成改过的 prompt 并打印要执行的命令，不复制、不运行 |
| `--episodes 0,1` | 只标这几个 episode，先小规模试跑、看质量和费用 |
| `--concurrency N` | 同时发给 VLM 的请求数（lerobot-annotate 默认 16），托管 API 限流时调小 |
| `--vlm-api openai` | 用 OpenAI 的 API（或 `--vlm-upstream` 指定的兼容服务）标注，key 从环境变量读，见下文。同时要给 `--vlm-model` |
| `--vlm-api codex` | 用本机的 Codex CLI 标注（`codex exec`），用它自己配置的服务和登录，不需要 key，见下文。模型默认取 Codex 配置里的 |
| `--vlm-endpoint` | `--vlm-api openai` 时：`responses`（默认，当前 GPT-6 系列只走这个接口）或 `chat` |
| `--vlm-upstream` | `--vlm-api openai` 时：API 地址，默认 `https://api.openai.com/v1` |
| `--api-key-env` | 从哪个环境变量读 key，默认 `OPENAI_API_KEY` |
| `--reasoning-effort` | 推理强度，默认 `low`，标注不需要深度推理，调高会明显增加费用 |
| `--lerobot-arg=...` | 再传一个 lerobot-annotate 参数，可以重复，如 `--lerobot-arg=--plan.min_subtask_seconds=2` |

- 要在装了 lerobot 和它的标注依赖的环境里运行：`pip install "lerobot[annotations]"`（需要 `openai` 包）。
- 副本里的 `.embodied_harness/` 记录了这次用的 prompt（`prompts/*.txt`）和完整命令（`annotate.json`），方便复查。
- 跑完会打印统计：多少 episode 标上了、多少个片段、有几种不同的标签、多少标签逐字等于库原文。逐字率低，说明词表和演示里的动作对不上，或者 VLM 太弱。
- lerobot-annotate 常用的其他参数：`--plan.min_subtask_seconds`（最短片段，默认 1.5 秒）、`--plan.plan_max_steps`（每个 episode 最多几段，默认 8）、`--only_episodes`（只标一部分 episode）。

**用 OpenAI 的模型标注**：`--vlm-api openai` 让 harness 在本机起一个只监听 127.0.0.1 的转发端点，lerobot-annotate 连它，它再调 OpenAI 的 API：

```bash
export OPENAI_API_KEY=...        # 在自己的终端里设置；harness 从环境变量读，不接受命令行里的 key
# 先试两个 episode，看标签质量和花费
embodied-harness data annotate configs/my_task/run.yaml /path/to/dataset --out /path/to/trial \
  --vlm-api openai --vlm-model gpt-6-sol --episodes 0,1 --concurrency 4
```

- key 只在发给 OpenAI 的请求里出现，不会出现在命令行、日志、`annotate.json` 或 `usage.json` 里。
- 转发时做了这些转换：请求改成 Responses 接口（当前的 GPT-6 系列只走这个接口）；`max_tokens` 改成 `max_output_tokens`，至少 4096，因为推理 token 也算在里面；去掉推理模型不接受的 `temperature`；设 `store: false`，不在 OpenAI 那边保存数据集的图片。遇到 429 限流或服务端错误会按 Retry-After 退避重试。
- lerobot-annotate 把视频采样成带时间戳的拼图（contact sheet）按图片发送，托管 API 能直接收；原样的视频（`video_url`）会被拒绝，不会发出去。
- 跑完打印 token 用量和按官方价目估算的费用，并写到副本的 `.embodied_harness/usage.json`。价目（每百万 token，输入/输出，2026-09）：`gpt-6-luna` $0.1/$0.5，`gpt-6-sol` $2/$10，`gpt-6-astra` $10/$50。先用 `--episodes` 试跑两三个，按用量乘上 episode 数估算全量费用。
- 本机访问 API 要走代理时，转发端点会沿用环境变量里的 `HTTPS_PROXY`；lerobot-annotate 到转发端点走本机，不经过代理。

**用 Codex CLI 标注**：`--vlm-api codex` 仍由 lerobot-annotate 切片段、写子任务和记忆，只是它的每一次提问都交给本机的 `codex exec` 回答。Codex 用自己配置的服务（`~/.codex/config.toml` 里的 provider，比如中转服务）和登录，harness 不接触任何 key：

```bash
embodied-harness data annotate configs/my_task/run.yaml /path/to/dataset --out /path/to/trial \
  --vlm-api codex --vlm-model gpt-6-astra --episodes 0,1 --concurrency 4
```

- 每次提问起一个 `codex exec`：图片按原顺序附上（文字里留 `[image N]` 标记），只读沙箱、空目录、不保存会话，用一句标注员的系统提示代替编程助手的长提示，并关掉浏览器、插件、图像生成等用不到的功能。
- 每次调用仍有约 1.2 万输入 token 的固定开销（Codex 自身的工具说明，其中约 3 千可缓存），比直接调 API 贵。账号能直接用 API 时，优先用 `--vlm-api openai`。
- 每次调用十几到二三十秒，`--concurrency` 就是同时运行的 `codex exec` 个数。用量从 Codex 的事件流里读，同样写进 `usage.json`；费用按 API 价目估算，按订阅计费的账号以实际账单为准。

**程序生成**：仿真数据往往自带子目标标签，直接按标签变化的时刻写 `subtask` 行，记忆按模板生成，不需要 VLM。

## 3. 标签表：把自由描述映射到技能调用

VLM 写的子任务描述往往是自由文本（尤其是 `--free-labels` 或词表对不上的时候），同一个技能会有很多种说法。而规划器只能输出技能词表里的调用，策略只认提示词库里的原文，所以每种说法都要由人映射一次：

```bash
embodied-harness data labels configs/my_task/run.yaml /path/to/dataset --out configs/my_task/labels.yaml
```

生成的文件按出现次数排序，每条的注释里是按词重叠算出的候选，只作参考：

```yaml
labels:
  - label: "place the blue cube onto the target"
    count: 107
    call: "place(color=blue)"    # closest: place(color=blue) 1.00, ...
    reviewed: false
```

用 `data annotate --labels` 生成、或者给 `data labels` 加 `--accept-exact` 时，和某个技能调用的库原文逐字一致的标签会自动映射并标为已审，只剩对不上的要人看。

逐条检查 `call`：技能调用写成 `pick`、`place(color=blue)`；写 `null` 表示这类片段不进规划器数据（策略训练集里保留原文）。检查完把 `reviewed` 改成 `true`。有没审过的条目，后面的命令会拒绝运行（快速试验可以加 `--allow-unreviewed`）。数据集里新增了说法就再跑一次，已审过的条目会保留。

同一个技能被连续标成几段时（比如 "grasp the cube" 和 "lift the cube" 都映射到 `pick`），导出时会合成一步，因为 harness 里一次调用会一直执行到校验器判定结束。

## 4. 数据映射：数据集怎么对应规划器的输入

```yaml
# configs/data/mapping.example.yaml
state_key: observation.state
cameras:                        # 规划器相机名 -> 数据集图像键
  head: observation.images.head
  wrist: observation.images.wrist
proprio:                        # 本体字段名 -> 基于状态向量 state 的表达式
  gripper_width: "state[7]"
  gripper_closed: "state[7] < 0.01"
instruction_key: instruction    # 只有 goal_source: episode 的任务需要
```

机器人或仿真的环境适配器必须用同样的方式算本体字段，否则训练和部署看到的数字不一样。

## 5. 导出

所有命令的第一个参数都是**部署时用的那份运行配置**：规划器的相机、分辨率、本体字段、记忆长度和技能词表都从里面读，保证训练样本和线上调用走同一套渲染代码。

```bash
RUN=configs/my_task/run.yaml
DATA=/path/to/dataset
LABELS=configs/my_task/labels.yaml

# 提示词库统计：补上每句提示词覆盖的 episode 数和时长 p50/p95
embodied-harness data library "$RUN" "$DATA" --labels "$LABELS" --out configs/my_task/library.yaml

# 规划器微调样本：按 episode 留出 10% 做验证
embodied-harness data planner "$RUN" "$DATA" --labels "$LABELS" --mapping configs/my_task/mapping.yaml --out exports/planner_v1

# 规划器导出和 library 统计完成后，再把副本中的子任务改成策略训练用的库原文。
# 不要在同一份数据上先 relabel 再 planner：relabel 会把 label 文本换掉，而旧 labels.yaml 的 key 仍是自由描述。
embodied-harness data relabel "$RUN" "$DATA" --labels "$LABELS" --in-place
```

`data planner` 用 LeRobot 解码视频帧，要在装了 lerobot 的环境里运行（在那个环境里 `pip install -e ".[data]"`）。常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--val-fraction` | 0.1 | 按 episode 留出做验证的比例 |
| `--seed` | 0 | 划分用的随机种子 |
| `--frame-offsets` | `0` | 数据增强：每个决策点前后再取几帧，如 `0,2,4`。线上规划器是在校验器判定时被调用的，不一定正好在标注的边界帧。至少给一个，空的会报错 |
| `--periodic-limit` | 每次检查一条 | 只用于 `planner.call: periodic` 的演示导出（其他情况报错）：每个步骤内部最多取几条定期检查样本，在它的各次检查里均匀挑（含第一次和最后一次）。`0` 表示不要步骤内部的样本。继续当前调用的样本每次检查一条，切换到下一步的决策每步只有一次（再乘 `--frame-offsets` 的帧数），步骤一长前者就占绝大多数；用它限制每步条数，让两者平衡些 |
| `--absolute-paths` | 否 | 图片写绝对路径。给 ms-swift、LLaMA-Factory 用时建议打开 |
| `--repo-id` | `local/<目录名>` | 传给 LeRobotDataset 的 repo id |

导出目录：

| 文件 | 内容 |
| --- | --- |
| `train.jsonl`、`val.jsonl` | messages + images 格式的 SFT 样本，ms-swift、LLaMA-Factory 能直接读 |
| `samples.jsonl` | 每条样本的输入（本体、记忆、图片、目标）和期望输出，`data eval` 用 |
| `images/` | 按部署设置编码好的图片 |
| `manifest.json` | 渲染版本、规划器设置、各调用的样本数、片段统计（`spans`，含全部映射为 null 的 `skipped_episodes`）、任务的 `done_when` 和结束时没达到它的 episode（`unfinished_episodes`）、因 `allow_done: false` 去掉的 done 样本数（`done_dropped`）、记忆超长统计；periodic 导出还有 `periodic`（检查间隔、上限、步骤内部样本数、只能由规划器切换结束的步骤数） |

**periodic 规划器的样本**：运行配置是 `planner.call: periodic` 时，规划器在技能执行中也会被定期问（每 `every_steps` 个控制步、动作块用完时），这时「已执行的步骤」里已经有正在执行的调用，原因是「执行中的定期检查」，记忆是选它时写下的那条。演示里没有这种决策，导出会补上，和 supervisor 的做法一致：

- 每个步骤内部按 `every_steps ÷ runtime.control_hz × 数据集 fps`（`meta/info.json`）帧的间隔取样，只取这一步自己的片段（后面映射为 null 的片段里演示已经不在做它），并停在下一步的第一条样本之前。目标是继续当前调用、记忆不变；上一步结果和账本是这一步开始时的值（这时它还没提交效果）。实际检查还要等动作块用完，动作块比 `every_steps` 长时线上会问得更稀。
- 技能没有 success 条件、又是策略技能（策略执行器不会报告完成）时，校验器结束不了它，只有规划器在定期检查里换掉它。这种步骤的下一步决策也是一次定期检查：已执行的步骤里还有它，上一步结果和账本都是它执行时的值；它以「被打断」结束，不提交效果。有 success 条件的技能照常在边界导出「上一步已完成」的样本。
- 用运行记录导出（`data from-runs`）时，记录里已经有当时真实的定期检查，不再补。

记忆超过 `planner.memory_max_chars` 时按运行时的方式截断，命令会提示超长的条数和最长长度，超长的多就调大这个上限。

## 6. 要多少数据

每个 episode 的样本数取决于标注片段映射：连续映射到同一调用的片段会合并，映射为 null 的片段会跳过，最后通常再有一个 done 样本（结束时 `done_when` 不成立、`allow_done: false`、或 episode 没有任何调用步时没有，见第 2 节）；每个 step 再乘以 `--frame-offsets` 的去重后帧数。periodic 配置再加上步骤内部的定期检查样本（见第 5 节，`--periodic-limit` 限制每步条数）。不要用“子任务段数 + 1”估算最终样本数，直接看导出的 manifest。

我们还没有在真实数据上把规划器训完，还给不出可靠的数据量结论。参考一下：RoboMME 每个任务 100 条演示，它的子目标预测器就是在这个规模的数据上微调的。上线前一定要跑 `data eval`，再在真机或仿真里跑闭环。

### 组合要覆盖的是步骤顺序

想让系统组合出新任务，数据要覆盖的是「哪个技能接在哪个技能后面」，而不是所有组合。组合泛化的诊断研究（arXiv 2607.29687）发现，新组合失败多半是指令引导不了策略，而不是缺底层技能；覆盖了动作之间依赖关系的一个结构化子集（约四分之一的任务空间）就能恢复大部分泛化，每个任务补一条演示就能把失败的组合从 0.4% 提到 54.7%。

```bash
# 训练数据里有哪些衔接（上一步 → 下一步，第一个技能之前是 START）
embodied-harness data coverage configs/my_task/run.yaml /path/to/dataset --labels configs/my_task/labels.yaml
# 再给一些运行记录（比如评测套件的 unseen 项）：列出它们走过的每个衔接在训练数据里出现几次、运行时成功率多少
embodied-harness data coverage configs/my_task/run.yaml /path/to/dataset --labels configs/my_task/labels.yaml \
  --runs runs/suites/<suite>/<run id>/<项名>
```

训练里出现 0 次、运行时成功率又低的衔接，就是该补数据的地方：采几条从那个衔接开始的演示，比穷举所有组合划算。评测套件见 [配置](configuration.md#评测套件suite)。

## 7. 运行记录变成训练数据

运行记录同时保留两种数据：**策略学习的动作条件**，以及**规划器当时实际看到的上下文和决策**。两者不能再用同一组 skill_start 时间推导。

先在 run 配置里打开：

```yaml
trace:
  frames: true
```

每个真实控制步保存相机视频、状态、实际动作、harness.source 和实际动作所属的 call_id/prompt。finish_chunk 下即使新技能已经派发，未执行完的旧动作仍归属旧调用。完成或环境终止后，额外写 terminal.json 和 terminal/ 相机图片；这份终态没有伪造 action，不增加策略训练帧。

在安装了 LeRobot/PyAV 的环境导出：

```bash
embodied-harness data from-runs runs/my_run --out /path/to/runs_dataset --success-only
embodied-harness data labels configs/robomme/run_pickx.yaml /path/to/runs_dataset \
  --out configs/robomme/labels_from_runs.yaml --accept-exact
# 审核 labels_from_runs.yaml 后：
embodied-harness data planner configs/robomme/run_pickx.yaml /path/to/runs_dataset \
  --labels configs/robomme/labels_from_runs.yaml \
  --mapping configs/robomme/data_mapping.yaml --out exports/from_runs_v2
```

上面的 run config 要与采集任务的技能和提示词库匹配。输出包含标准 LeRobot 数据、language_persistent 标注，以及版本化的 `.embodied_harness/from_runs.json` 和其引用的图片。复制/移动数据集时保留整个 `.embodied_harness` 目录。

| 使用方 | 使用的数据 |
| --- | --- |
| 策略训练 / relabel | 根据每帧实际 call_id 变化生成的 subtask 段，包括失败尝试、重试、恢复和 transition 原语；不会把失败动作贴到前一成功调用 |
| planner 训练 / data eval | sidecar 中被接受的 planner 请求/响应，保留实际 memory、history、ledger、last、reason、resample 和拒绝原因，不把每个动作段假定为成功 |
| data library | 有明确 success 结果的 main 调用及其实际持续时间，不使用 periodic 规划间隔作为技能时长 |
| done 样本 | 只有任务判成功且存在真实 terminal 观测时才生成，规则见下；失败和缺终态的旧轨迹不自动补 done |

运行中的失败调用不会再被重新执行成功 effects：例如一次 place 失败、重规划后 place 成功，训练输入保留当时真实账本，只累计真正成功的那次。连续选择同一技能的不同规划请求也不会合并；被拒绝或被 switch_when 阻止的回答不作为接受决策。

终态 done 样本的规则：

- 环境自己结束 episode 时，只有它判了成功（`env_end` 的 success 为 true）才生成；判失败或没给判定（success 为空）都不补 done。
- 环境在某个 main 调用执行中途判成功时（比如插销已经插好，insert 的校验器还没来得及判定），这个调用视为成功：终态样本的账本加上它的 `effects.on_success`，「上一步」是它的成功，和 supervisor 提交一次成功之后一样。已经有判定结果的调用（失败、被打断）和脚本原语不算。
- 规划器自己说 done 结束的 episode（任务没有 done_when 时），终态样本的输入是它说 done 那一刻实际看到的记忆、调用原因、历史、账本和上一步，目标记忆是它当时写下的记忆；done 回答自己写的记忆、以及周期调用时它造成的打断，都不会出现在输入里。
- 导出时，任务配了 done_when 的话，终态账本同样要满足它，否则不导出这条 done，episode 记入 `unfinished_episodes`。

Python API 默认 `read_annotated_episodes(...)` 读取动作语言段；`planner=True` 读取 sidecar 规划决策。CLI 的 data planner 已选择后者。普通人工示教仍沿用技能段合并与成功演示重算逻辑。

动态 goal 也保存在 sidecar 中：即使 LeRobot 将 task 存成 task_index，`goal_source: episode` 的 planner 导出仍能恢复原目标，无须手工补 instruction 列。如果同时提供了指令字符串列，读取器会核对其与 sidecar 一致。

其他约束：

- 新 sidecar 使用 schema_version=2。旧版 from-runs 导出缺少上下文时会明确要求从原始轨迹重新导出，避免静默当成功演示。
- labels 中的 null 可以排除指定 planner 标签；后续样本仍使用其真实记录的状态，不受排除操作影响。将标签改映射到另一调用会报错，需要另行准备人工纠正数据。
- runtime planner 导出要求 frame_offsets 为默认的 0。真实决策绑定当时的观测，不能任意挪到邻近帧；示教数据仍支持偏移增强。
- 没有 call_id 的旧帧可以按事件生成兼容语言段，但无法保证 finish_chunk 的动作边界准确；需要准确训练数据时重新采集。
- harness.source 区分 policy、primitive、hold。策略训练应按目标筛选来源；hold 帧被保留以维持时间连续性，不代表策略在执行前一技能。
- 当前 LeRobot 导出要求整数控制频率；非整数频率明确拒绝，不会默默四舍五入。时间戳按帧序号/控制频率生成，真实时钟下错过的控制拍不会补帧。
- resumed 轨迹的 JSONL 与单组视频不能可靠拼接，from-runs 会明确拒绝。SequencePlanner 可以续跑，但这不等于续跑录像可以直接拼成训练集。
- --success-only 和 `run`、`report`、`suite` 用同一条评分规则筛 episode：环境给了结论（env_success）时按它，没给（真机、旧轨迹）时按 status=done。所以仿真里以 done 结束、环境却判失败的 episode 不会导出。它不是独立的专家审核；被接受的规划决策和模型 memory 也不保证语义正确，训练前仍需检查。

## 检查清单

- [ ] 相机的路数、顺序、分辨率和部署一致；
- [ ] 映射文件里的本体字段和机器人适配器的计算方式一致；
- [ ] 标签表全部审过，`null` 的片段确实不该让规划器学；
- [ ] 提示词库就是策略训练时用的那一版，运行配置里写了 `planner.trained_library`；
- [ ] 验证集按 episode 划分（默认如此），没有和训练集共享 episode；
- [ ] 要评测的组合里的每个「上一步 → 下一步」在训练数据里都出现过（`data coverage`），没出现的先补演示。
