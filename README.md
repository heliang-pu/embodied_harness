# Embodied Harness

把 **VLM 规划器**和**机器人策略**（VLA、WAM 等）组合成一个闭环的运行时。

规划器决定下一步做哪个子任务，策略把子任务做出来，harness 负责中间的一切：

- 什么时候问规划器，检查它的回答；
- 把子任务换成策略训练时见过的那句提示词；
- 根据本体判断子任务做完没有，失败了怎么恢复；
- 记下完整的轨迹。

模型可以随时换，这些逻辑不用跟着改。

- pip 包名和命令：`embodied-harness`；import 名：`embodied_harness`
- 设计文档：<https://claude.ai/code/artifact/539a719a-4856-4dec-a378-1c0629bf8947>

```
 相机、本体 ──► 调度器（确定性状态机）──► 规划器（VLM，vLLM 部署）：下一步 {技能, 参数, 记忆}
                  │          ▲
       提示词原文  ▼          │ 技能结束/失败（校验器看本体）
              策略（VLA / WAM，websocket 服务）──► 动作块 ──► 环境 / 机器人
```

## 特点

- **任务即配置**：技能词表、提示词库、任务目标、校验条件、恢复规则都写在 YAML 里，换任务不改代码。
- **模型无关**：规划器走 OpenAI 兼容接口（vLLM），策略走 websocket（OpenPI 或 XPolicyLab 协议），各自跑在自己的进程和环境里。
- **提示词库约束**：技能引用库里的 id，派发时使用库原文。配置了 `planner.trained_library`
  时，库名不匹配会拒绝启动；库名由发布者维护，harness 不会自动核验服务端权重的训练词表。配置了
  `planner.trained_manifest`（规划器训练导出的目录或 manifest.json）时，`validate` 和每个 episode
  开始都会比对规划器设置和训练时是否一致，不一致只告警。
- **RoboMME 规划不依赖仿真真值**：规划器使用图像、本体及历史，技能校验使用本体；环境 verdict
  用于终止 episode 和评分。mock oracle 为验证调度流程会读取模拟内部状态。
- **可追溯**：事件流、规划输入摘要和输出、下发动作默认落盘；VLM 后端还保存实际输入图片。
  视频需打开 `trace.video`；环境快照续跑需要适配器实现 `snapshot` / `restore`。
- **长任务和组合泛化的诊断**：失败报表按规划 / 衔接 / 技能归因，失败再分起点不对 / 目标没找对 / 执行失败；仿真里可以从记录的起点重跑某一步做对照；评测套件把原子 / 组合、见过 / 没见过分开统计；`data coverage` 检查训练数据覆盖了哪些步骤衔接；`policy steer-check` 检查策略还听不听指令。
- **数据闭环**：规划器和策略可以从同一份 LeRobot 数据集取各自需要的字段。演示数据用
  lerobot-annotate 标注；开启 `trace.frames: true` 的运行记录也能通过 `data from-runs` 导出为
  LeRobot 数据集；sidecar 保留动态目标、每次规划的原始状态和实际调用结果，规划器与策略分别读取。

## 进展

- 调度、校验、恢复、轨迹、日志：在 mock 世界里完整跑通，不依赖模型和机器人。
- VLM 规划器（vLLM + Qwen3-VL）、websocket 策略、数据工具：已实现，有测试。自己的规划器还没有在真实数据上训完。
- **RoboMME PickXtimes**：用 RoboMME 发布的 π0.5 和子目标预测器在 harness 里复现，3 个策略种子共 150 个 episode，成功 144 个（96.0%）；论文报告 95.33%。
- **零样本规划器**：在 mock 分拣（4 个包裹，含随机故障）上，不微调的 Qwen3-VL-4B 只看图和本体时 0/8；给结构化的「当前状态」「进度」并把记忆压到 10 字后 7/8（唯一的失败来自执行器）。记忆留得长时它会坚持自己写下的旧计划，同样 0/8。详见 [VLM 规划器](docs/planner_finetune.md#不微调时)。
- **RoboMME 组合任务**：用同样的模型组合出 RoboMME 里没有的多颜色任务，不重新训练。规划正确时（固定顺序），加上 harness 的 `go_home` 原语和只发子目标，π0.5 能做到 50%（红×2→绿×1）和 17%（蓝→绿→红），瓶颈是叠放；RoboMME 的原样流程是 0%，不微调的 Qwen3-VL-4B 规划器也是 0%。详见 [RoboMME 实例](docs/robomme.md#组合任务)。

## 安装

需要 Python ≥ 3.10。

```bash
conda create -n embodied-harness python=3.12 -y
conda activate embodied-harness
python -m pip install -e ".[dev]"
```

新建的 conda 环境里可能没有 pip，这时 `pip` 会装到 base 环境里去。先运行 `python -m ensurepip`，之后都用 `python -m pip`。

基础依赖是 pydantic、PyYAML 和 numpy，不装可选依赖也能用 CLI 校验配置、跑 mock episode。可选依赖：`planner`（numpy、Pillow，给规划器编码图片）、`policy`（websocket 策略客户端和服务端）、`data`（导出训练数据，另外需要 lerobot）、`dev`（以上全部加 pytest、ruff）。

在新环境中运行 `python -m pip install -e ".[dev]"`，即可使用命令和包。
新数据使用 `.embodied_harness/` sidecar 目录、`embodied_harness` 版本字段和新的 checkpoint
处理器类路径；旧格式数据需要重新导出或重新准备。

## 五分钟上手

mock 分拣世界不需要 GPU 和模型：一个符号化的桌面、一个假策略、一个上帝视角的规划器，用来看清整个流程。

```bash
embodied-harness validate configs/runs/mock_sort.yaml
embodied-harness run configs/runs/mock_sort.yaml --episodes 3 --seed 0 --hold-policy abort
```

每个 episode 写到 `runs/<run_id>/ep_0001/`：`events.jsonl` 是事件流，`planner/` 是每次规划调用，`outcome.json` 是结果，`episode.log` 是日志。各文件的说明见 [轨迹、日志与排错](docs/debugging.md)。

## 接自己的机器人和模型

1. **写配置**：提示词库（策略训练过的每句提示词）、技能文件（技能、校验、恢复）、任务文件、运行配置。见 [核心概念](docs/concepts.md) 和 [配置参考](docs/configuration.md)。
2. **准备数据**：录完整长任务的 LeRobot v3 数据集，`embodied-harness data annotate` 标出子任务和记忆（内部调用 lerobot-annotate），人审标签表里对不上提示词库的条目。见 [数据](docs/data.md)。
3. **微调规划器**：先用 `data planner` 导出 SFT 数据，训练 Qwen3-VL，vLLM 部署，`data eval` 离线评测。部署用 `planner.call: periodic` 时，要用同一份 periodic 运行配置导出，导出里才有技能执行中途「继续当前步」的样本（`--periodic-limit` 可限量）。见 [VLM 规划器](docs/planner_finetune.md)。
4. **训练策略**：再用 `data relabel --in-place` 把子任务换成库原文，`policy subtask-checkpoint` 准备 checkpoint，用 `lerobot-train` 直接在同一份数据集上微调，再起成 websocket 服务。不要用 relabel 后的旧自由文本 labels.yaml 再运行 `data planner`。见 [策略](docs/policy.md)。
5. **接环境**：实现一个 Env 类（reset、observe、act、hold、stop、close），登记到工厂表里。见 [接入新组件](docs/integration.md)。
6. **跑闭环**：`embodied-harness run`，用 `report` 看失败出在规划、衔接还是技能上，失败再按起点不对 / 目标没找对 / 执行失败分类；仿真里用 `probe` 从同一个起点重跑，确认是不是衔接的问题，然后加 [衔接规则](docs/configuration.md#衔接规则transitions)（原语可以带参数，比如移动到下一个目标上方）；`replay` 对比新旧规划器。
7. **测组合泛化**：用 [评测套件](docs/configuration.md#评测套件suite) 把原子 / 组合、见过 / 没见过的任务分开评测，`data coverage` 看新组合要求的衔接在训练数据里有没有；反复出现的失败用 `lessons` 起草成经验给规划器；微调策略前后用 `policy steer-check` 检查它还听不听指令。

## 文档

| 文档 | 内容 |
| --- | --- |
| [核心概念](docs/concepts.md) | 分工、一个 episode 怎么跑、状态机、校验、恢复、记忆、动作块 |
| [配置参考](docs/configuration.md) | 运行配置、任务、技能、提示词库，以及经验文件和评测套件文件的全部字段；表达式语法；内置组件的参数 |
| [数据](docs/data.md) | 需要什么数据、怎么标注、标签表、数据映射、导出命令 |
| [VLM 规划器](docs/planner_finetune.md) | 规划器的输入输出、数据准备、微调、部署、离线评测、SimpleSG 数据 |
| [策略](docs/policy.md) | 提示词库、训练数据、websocket 协议、现成和自写的服务端、接入 WAM |
| [OpenPI 集成](docs/openpi.md) | 独立模型服务、WebSocket 通信、LeRobot v3.0 校验与训练输入映射 |
| [接入新组件](docs/integration.md) | 新的环境/机器人、执行器、规划器怎么写 |
| [RoboMME 实例](docs/robomme.md) | 复现 RoboMME、组合任务、结果和失败分析 |
| [ALOHA 仿真实例](docs/aloha_sim.md) | gym-aloha 双臂插销：Codex 标注、微调规划器和 pi05、闭环成功率和失败分析 |
| [轨迹、日志与排错](docs/debugging.md) | 轨迹文件、事件、失败归因报表、同起点重跑（probe）、经验起草、错误码、常见问题 |

## 命令一览

| 命令 | 作用 |
| --- | --- |
| `embodied-harness validate <run.yaml>` | 加载并交叉校验配置 |
| `embodied-harness run <run.yaml> [--episodes N] [--seed S] [--run-id ID] [--hold-policy ask\|abort\|resume] [--resume EP_DIR]` | 运行 episode，写轨迹。不传 `--seed` 时从运行配置的 `runtime.seed`（默认 0）开始，每个 episode 加 1；`--resume` 沿用记录的种子 |
| `embodied-harness replay <run.yaml> <ep_dir>... [--out F]` | 用配置里的规划器重放历史调用，对比决策 |
| `embodied-harness report <run_dir>... [--by call] [--out F]` | 失败归因：每个技能、每种衔接的成功率，每个失败 episode 归到规划 / 衔接 / 技能，失败再分起点 / 目标 / 执行三类，见 [排错](docs/debugging.md#失败归因报表) |
| `embodied-harness probe <run.yaml> --at EP_DIR[:CALL_ID]... [--call CALL...] [--trials N]` | 仿真里从记录的起点重跑技能调用，比较干净起点和串联起点（需要 `trace.snapshots`），见 [排错](docs/debugging.md#从同一个起点重跑probe) |
| `embodied-harness lessons <run_dir>... --out F` | 从反复出现的失败里起草经验，审核后给规划器，见 [配置](docs/configuration.md#经验文件lessons) |
| `embodied-harness suite run\|report <suite.yaml> ...` | 组合评测套件：按原子 / 组合、见过 / 没见过汇总成功率，见 [配置](docs/configuration.md#评测套件suite) |
| `embodied-harness data annotate <run.yaml> <dataset> --out DIR [--labels F] [--vlm-api openai --vlm-model M]` | 用 lerobot-annotate 标注子任务和记忆（在副本上，按提示词库给标签）；可以用 OpenAI 的托管模型，key 从环境变量读，见 [数据](docs/data.md#2-子任务和记忆标注) |
| `embodied-harness data labels\|library\|relabel\|planner\|eval\|coverage ...` | 其余数据工具，见 [数据](docs/data.md)；`coverage` 列出训练数据覆盖了哪些「上一步 → 下一步」 |
| `embodied-harness data from-runs <run_dir>... --out DIR` | 把打开 `trace.frames` 的运行记录导出成 LeRobot 数据集（带子任务和记忆标注），见 [数据](docs/data.md#7-运行记录变成训练数据) |
| `embodied-harness policy subtask-checkpoint <ckpt> --out DIR` | 准备 LeRobot checkpoint，让策略直接在标注数据集上按子任务提示词微调，见 [策略](docs/policy.md) |
| `embodied-harness policy steer-check <run.yaml> [--frames RUN...] [--call CALL...]` | 检查策略对不同提示词给不给出不同的动作（微调前后对比），见 [策略](docs/policy.md#检查策略还听不听指令) |

## 代码结构

| 目录 | 内容 |
| --- | --- |
| `src/embodied_harness/config/` | YAML schema、加载校验、技能注册表和受限表达式 |
| `src/embodied_harness/core/` | 共享数据类型、时钟、ID 和账本 |
| `src/embodied_harness/env/` | 机器人/环境接口与 mock、RoboMME、ALOHA 仿真适配 |
| `src/embodied_harness/planner/` | VLM、固定顺序等规划器，决策解析、prompt 渲染和调用管理 |
| `src/embodied_harness/policy/` | VLA/VA/RL/WAM 策略；共享执行器、WebSocket 服务、传输协议和训练适配 |
| `src/embodied_harness/runtime/` | 控制循环、监督器、校验器、评测工具，以及 `logs/` 和 `trace/` |
| `src/embodied_harness/data/` | 数据标注、重标注、LeRobot v3 读取/导出、规划器数据和覆盖分析 |

`scripts/train_planner_sft.py` 是规划器的 SFT 训练脚本（全参数或 LoRA）。`configs/` 下是 mock 分拣、RoboMME 复现和组合任务的配置，`configs/suites/` 是评测套件的例子。

## 测试

```bash
python -m pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
```
