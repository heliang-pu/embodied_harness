# fmc3-harness

把「自己微调的小 VLM 规划器」和「VLA 执行器」组合成一个闭环的运行时。模型可以换，调度、记忆、接口、校验和日志留在 harness 里。
设计文档：<https://claude.ai/code/artifact/539a719a-4856-4dec-a378-1c0629bf8947>

当前进度：
- **M0 已完成**：用 mock 分拣世界和 oracle 规划器跑通调度、记账、校验、错误恢复、轨迹和日志，不依赖任何模型和机器人。
- **M1 规划器接入已完成**：vLLM 部署的 Qwen3-VL-2B（OpenAI 兼容接口）、多路相机、和训练共用的 prompt 渲染器、由技能词表生成的约束解码 schema、规划器重放。
- **数据工具已完成**：从 lerobot-annotate 标注的数据集生成规划器微调样本、执行器训练集和提示词库统计，并在留出的演示上离线评测规划器（`fmc3-harness data`）。还差用真实数据训练，过 M1 的门槛。

## 安装

```bash
conda create -n fmc3-harness python=3.12 -y
conda activate fmc3-harness
python -m pip install -e ".[dev]"
```

新建的 conda 环境里可能没有 pip，这时 `pip` 会落到 base 环境。先用 `python -m ensurepip` 装上，再用 `python -m pip` 安装。

## 跑 mock 闭环

```bash
fmc3-harness validate configs/runs/mock_sort.yaml
fmc3-harness run configs/runs/mock_sort.yaml --episodes 3 --seed 0 --hold-policy abort
```

每个 episode 写到 `runs/<run_id>/ep_0001/`：

| 文件 | 内容 |
| --- | --- |
| `events.jsonl` | 事件流：状态切换、规划请求和结果、技能开始和结束、恢复、账本和记忆更新、保持、人工操作 |
| `planner/pc_0001.json` | 每次规划调用的输入摘要、渲染后的 prompt、原始输出、解析结果和延迟 |
| `planner/pc_0001/<camera>.jpg` | 这次调用实际发给模型的图片，重放时原样再发 |
| `actions.jsonl` | 每个下发的动作和它所属的指令版本 |
| `episode.log` | 运行日志，JSON Lines，每行带 episode、状态、技能调用和规划调用的 ID |
| `outcome.json` | 最终结果、错误码、账本、记忆和健康统计 |
| `config.snapshot.yaml` | 解析后的完整配置 |
| `env_snapshot.json` | 仅仿真环境：最近一次技能边界的世界状态 |

中断的 episode 可以续跑：`fmc3-harness run configs/runs/mock_sort.yaml --resume runs/<run_id>/ep_0001`。

## 接 vLLM 规划器（M1）

先用 vLLM 起微调好的 Qwen3-VL-2B。每条请求允许的图片数要不少于相机数（旧版本 vLLM 写成 `image=3`）：

```bash
vllm serve /path/to/qwen3-vl-2b-planner --served-model-name fmc3-planner --port 8000 --limit-mm-per-prompt '{"image": 3}'
```

`configs/runs/vlm_mock_sort.yaml` 用 mock 世界的三路合成画面走通「画面 → vLLM → 决策」整条链路：

```bash
fmc3-harness run configs/runs/vlm_mock_sort.yaml --episodes 1 --hold-policy abort
```

合成画面只用来验证链路；用真实画面训练的模型在上面规划不好。规划器配置里的 `cameras`、`image_size`、`proprio_keys` 必须和训练数据一致。

**重放**：训了新模型后，先在历史轨迹上离线对比决策，再上机器人：

```bash
fmc3-harness replay configs/runs/vlm_mock_sort.yaml runs/<run_id>/ep_0001 --out replay.json
```

改了渲染格式（`fmc3_harness.planner.render`）就要升 `RENDER_VERSION`，并重新生成训练数据。

## 从标注数据生成训练数据

输入是用 lerobot-annotate 标过子任务和记忆的 LeRobot v3 数据集，标注在 `language_persistent` 列。它写的子任务是自由描述，同一个技能会有很多种说法，所以第一步是把说法映射到技能调用，由人审一遍：

```bash
fmc3-harness data labels configs/runs/vlm_mock_sort.yaml /path/to/dataset --out configs/data/labels.yaml
```

打开 `labels.yaml` 逐条检查 `call`：技能调用写成 `pick`、`place(bin=A)` 这样；写 `null` 表示这类片段不进规划器数据，执行器训练集里保留原文。检查完把 `reviewed` 改成 `true`，没审完的条目后面的命令会拒绝。注释里是按词重叠算的候选，只作参考。数据集新增了说法就再跑一次，审过的条目会保留。

```bash
LABELS="--labels configs/data/labels.yaml"
# 提示词库：用标注片段的时长补上片段数和 p50/p95，技能超时由此推出
fmc3-harness data library configs/runs/vlm_mock_sort.yaml /path/to/dataset $LABELS --out configs/libraries/my_library.yaml
# 执行器训练集：子任务文字换成提示词库原文，记忆行不动，视频默认软链接（--copy-videos 复制）
fmc3-harness data relabel configs/runs/vlm_mock_sort.yaml /path/to/dataset $LABELS --out /path/to/dataset_relabeled
# 规划器微调样本：和部署同一套渲染和图像编码，按 episode 留出 10% 做验证
fmc3-harness data planner configs/runs/vlm_mock_sort.yaml /path/to/dataset $LABELS \
  --mapping configs/data/mapping.example.yaml --out exports/planner_v1
# 训好的模型用 vLLM 起来之后，在验证集上离线评测
fmc3-harness data eval configs/runs/vlm_mock_sort.yaml exports/planner_v1 --out exports/planner_v1/eval.json
```

- 运行配置要用部署时的那份：规划器的相机、分辨率、本体字段和记忆上限都从里面读。
- `--mapping` 写规划器相机对应数据集的哪个图像键，以及本体状态怎么从 `observation.state` 算出来，例子见 `configs/data/mapping.example.yaml`。机器人上的环境适配器要用同一份映射。
- `data planner` 用 LeRobot 解码视频帧，要在装了 lerobot 的环境里运行：激活 lerobot 的 conda 环境，在本仓库目录下执行 `python -m pip install -e ".[data]"`。
- 导出目录里有：`train.jsonl` 和 `val.jsonl`，是 ms-swift、LLaMA-Factory 能直接读的 messages + images 格式，图片路径相对导出目录；`samples.jsonl` 存每条样本的输入和目标，给 `data eval` 用；`images/`；`manifest.json` 记渲染版本、规划器设置、各调用的样本数和记忆超长统计。
- 连续几个片段映射到同一调用时合成一步，因为 harness 里一次调用会一直执行到校验器判定。每一步的输入记忆是上一步写下的记忆，和运行时一致。
- 记忆超过 `planner.memory_max_chars` 时按运行时的方式截断，命令会提示条数和最长长度；超长的多就调大上限。
- `data eval` 是开环评测：每一步都从演示里的状态和记忆出发，统计调用准确率、done 召回、提前 done 次数和记忆相似度。

## 测试

```bash
python -m pytest -q
ruff check src tests && ruff format --check src tests
```

## 代码结构

| 目录 | 内容 |
| --- | --- |
| `config/` | YAML 的 schema 和加载器，加载时做交叉校验 |
| `skills/` | 受限表达式和技能注册表：提示词解析、谓词、校验条件、效果、恢复规则 |
| `supervisor/` | 调度器状态机：规划、派发、校验、提交、恢复、保持 |
| `planner/` | 规划器接口、输出解析、同步和线程两种调用方式、共用渲染器、约束解码 schema、图像编码、vLLM 后端、mock oracle、固定序列 |
| `executor/` | 执行器接口、动作块队列、mock VLA、脚本原语 |
| `verifiers/` | 单次技能调用的判定：本体规则、去抖、卡住、超时 |
| `memory/` | 账本，可以从轨迹重放出来 |
| `env/` | 环境接口和 mock 分拣世界（可选合成相机画面） |
| `trace/` | 轨迹写入、续跑用的读取和状态重建、规划器重放 |
| `data/` | lerobot-annotate 数据集读取、标签表、提示词库统计、规划器样本导出、执行器训练集重标注、离线评测 |
| `logs/` | 日志上下文、错误码、健康摘要 |
| `runtime/` | 组件工厂、人工指令来源、控制循环 |
