# RoboMME 实例

[RoboMME](https://robomme.github.io)（arXiv 2603.04639）是一个考察机器人记忆的仿真基准，基于 ManiSkill/SAPIEN，有 16 个任务。我们用它的 **PickXtimes** 验证 harness：把指定颜色的方块拿起来放到目标上，重复 N 次（1–5 次），然后按按钮。一共 2N+1 个子目标，必须记住已经放了几次，按早了或者多放了都算失败。

这一页分两部分：先在 harness 里复现 RoboMME 论文的结果，再用同样的模型做一个 RoboMME 里没有的**组合任务**。

## 环境准备

RoboMME 只能在 Linux + NVIDIA GPU 上跑。三个进程各用自己的 Python 环境，靠网络通信：

| 进程 | 环境 | 内容 |
| --- | --- | --- |
| 仿真 + harness | RoboMME 要求的环境（numpy 1.26、gymnasium 0.29），加上 `pip install -e ".[policy]"` | `embodied-harness run` |
| 策略服务 | RoboMME 的 openpi 仓库自带的 `.venv`（JAX） | RoboMME 的 π0.5，websocket，端口 8001 |
| 规划器服务 | vLLM | Qwen3-VL-4B，加 RoboMME 的 LoRA，端口 8000 |

模型（Hugging Face）：

- 策略：`Yinpei/mme_vla_suite` 里的 `symbolic-simple-subgoal`，第 79999 步。这是用「任务目标 + 子目标」做提示词训练的 π0.5，16 个任务一起训练。
- 子目标预测器：`Yinpei/vlm_subgoal_predictor` 里的 `simple_subgoal` LoRA，基座 `Qwen/Qwen3-VL-4B-Instruct`。

a100-2 上的完整环境在 `/workspace`，脚本在 `/workspace/scripts/0927_robomme_pickx/`：`serve_mme.sh` 起两个服务，`eval_mme.sh` 把 50 个 episode 分到几张卡上跑，`eval_seeds.sh` 依次跑几个策略种子，`summarize_*.py` 汇总结果。这些是服务器本地脚本，不包含在本仓库中。起服务的核心命令如下；路径占位符需要替换成实际下载位置，后面的 `embodied-harness` 命令要回到本仓库根目录运行：

```bash
# 策略：RoboMME 的 openpi 服务端，JAX 只占 35% 显存，给 vLLM 和渲染留空间
cd robomme_policy_learning
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 .venv/bin/python scripts/serve_policy.py \
  --seed=7 --port=8001 policy:checkpoint --policy.dir=<mme_vla_suite>/symbolic-simple-subgoal/79999 \
  --policy.config=mme_vla_suite

# 规划器：SimpleSG 只看 front 一张图；保留 image: 2 也供后面的双相机 JSON 规划器使用
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-VL-4B-Instruct --served-model-name qwen3-vl-4b --port 8000 \
  --enable-lora --lora-modules simple_subgoal=<vlm_subgoal_predictor>/qwenvl/simple_subgoal/checkpoint-1400 \
  --max-lora-rank 16 --max-model-len 8192 --gpu-memory-utilization 0.30 \
  --limit-mm-per-prompt '{"image": 2, "video": 0}'
```

vLLM 第一次启动要编译算子，`PATH` 里需要有它的 venv（ninja）和 `/usr/local/cuda/bin`。

## 复现 RoboMME：SimpleSG + QwenVL

配置：[`configs/robomme/mme/run_mme_pickx.yaml`](../configs/robomme/mme/run_mme_pickx.yaml)，完全按 RoboMME 自己的评测方式跑：

- 规划器 `robomme_subgoal`：达到 16 步且动作块耗尽时问一次子目标预测器（`planner.call: periodic`），输入任务目标、已派发的主子目标历史和一张前视图；连续相同调用去重，词表外的回答不会进入历史。适配器复用 RoboMME 的提示文本；
- 执行器：它们的 π0.5，提示词是任务目标加当前子目标，每个动作块执行前 16 步；
- 技能文件里只有一个 `subgoal(text)` 技能，参数就是 19 句子目标原文；没有校验器，切换完全由预测器决定；
- RoboMME 测试集 50 个 episode，关节角动作，最多 1300 步。

```bash
embodied-harness run configs/robomme/mme/run_mme_pickx.yaml --seed 0 --episodes 50 --hold-policy abort
```

结果（每个策略种子 50 个 episode，seed 是策略服务的 `--seed`）：

| 策略种子 | 成功率 |
| --- | --- |
| 7 | 48/50 = 96.0% |
| 42 | 48/50 = 96.0% |
| 0 | 48/50 = 96.0% |
| 合计 | 144/150 = 96.0% |

论文里是 95.33%（3 个 checkpoint × 3 个种子的平均）。注意这个数只是 PickXtimes 一个任务；同一方法在 16 个任务上的平均只有 29.00%。

按重复次数：

| 重复次数 | episode 数 | 成功 |
| --- | --- | --- |
| 1 | 48 | 100% |
| 2 | 30 | 100% |
| 3 | 36 | 100% |
| 4 | 18 | 16/18 |
| 5 | 18 | 14/18 |

6 个失败 episode 的原因：

- 3 个是预测器切换过早：方块还在夹爪里就跳到了下一次「抓」，或者还没抓住就跳到了「放」，环境里的实际次数就少了一次。技能上的 `switch_when` 门控（夹爪没松开不许离开 `place`，没夹住不许离开 `pick`）就是针对这种情况的，见 [`configs/robomme/mme/run_mme_pickx_gated.yaml`](../configs/robomme/mme/run_mme_pickx_gated.yaml)，还没跑完评测。
- 2 个是策略一直抓不起来（同一个 episode，两个种子下都是）：夹爪反复合上又张开，1300 步用完。预测器一直正确地给出「第一次抓红色」。
- 1 个在夹爪层面看顺序是对的（5 次抓放），但按按钮时环境判失败，原因还没查清。

## 组合任务

RoboMME 的 PickXtimes 每个 episode 只用一种颜色。场景里其实有红绿蓝三块方块（medium 和 hard 难度），所以我们可以用同样的子目标组合出新任务，比如「红色放两次，再放一次绿色，然后按按钮」。模型一个都不重新训练，只改配置，看 harness 能不能把已有的能力组合起来。

环境参数 `composite: [[red, 2], [green, 1]]` 生成这样的任务：

- 指令沿用 RoboMME 的句式："pick up the red cube and place it on the target, repeating this action two times, then pick up the green cube and place it on the target, then press the button to stop"；
- 判分用 RoboMME 自己的检查函数，按新的子目标序列检查：拿起别的颜色的方块，或者提前按按钮，都立即判失败；
- 只用测试集里有三种颜色的 24 个 episode（`difficulty: [medium, hard]`）。

配置在 [`configs/robomme/composite/`](../configs/robomme/composite/)，两个组合 × 三种规划器：

| 规划器 | 配置 | 说明 |
| --- | --- | --- |
| 固定顺序 | `run_sequence_*.yaml` | 按预设顺序给子目标，技能结束由夹爪宽度判断。作为固定顺序与校验规则下的执行基线 |
| 零样本 VLM | `run_vlm_*.yaml` | Qwen3-VL-4B-Instruct 基座，不微调。看两路相机、夹爪宽度、已派发的步骤（`show_history`）和自己的记忆，在技能结束时选下一步 |
| RoboMME 预测器 | `run_mme_*.yaml` | RoboMME 微调的预测器，原样使用（每 16 步问一次），它只在单色任务上训练过 |

```bash
embodied-harness run configs/robomme/composite/run_sequence_red2_green1.yaml --seed 0 --episodes 24 --hold-policy abort
```

### 执行器的三个问题，和 harness 的处理

π0.5 只在单色的 PickXtimes 上训练过。用固定顺序规划器（顺序一定对）试下来，组合任务暴露出执行器的三个问题：

1. **换颜色时抓回刚放下的方块。** 放完红色后，机械臂还停在红色方块正上方，这时收到「第一次抓绿色」，π0.5 会就近抓回红色（4/4 次都是这样，看视频确认过）。它在训练里从没换过颜色，而每一次「第一次抓」都是从 episode 的初始姿态开始的。只是把切换时机往后推（松开后多等几秒）没有用，放置阶段拖长了它反而会去碰别的方块。

   处理：加一个脚本原语 `go_home`，把关节插值回本 episode 的初始姿态，同时张开夹爪。换颜色前先执行它，「第一次抓新颜色」就回到了 π0.5 熟悉的情形。

2. **整句的多色任务目标会让它抓错颜色。** RoboMME 的 π0.5 同时接收任务目标和子目标。给它三种颜色的组合指令，它在第一步就可能去抓指令里的另一种颜色。

   处理：只把当前子目标发给它，任务目标那一栏也填子目标原文（`texts: {prompt: "{prompt}", ...}`）。全局任务由规划器负责，执行器只管当前这一步。

3. **目标上已经放着方块。** 第一种颜色放完后方块一直留在目标上，下一种颜色只能叠上去或挤在旁边。RoboMME 判「放到目标上」看的是离目标中心水平距离 ≤ 5 cm，叠歪了或者把下面的方块推开，这一次就不算，最后按按钮时判失败。这是组合任务本身的物理冲突，判分没有放宽。

前两项使用环境适配器中已实现的 `go_home` 原语并调整配置，模型没有任何改动。4 个 episode 的小实验里，「红×2 → 绿×1」从 0/4 提高到 4/4。

### 结果

每个组合 24 个 episode（测试集里三种颜色都有的 medium 和 hard），RoboMME 的步数上限 1300，策略种子 7：

| 规划器 | 红×2 → 绿×1 | 蓝 → 绿 → 红 |
| --- | --- | --- |
| 固定顺序基线，`go_home` + 只发子目标 | **12/24 = 50%** | **4/24 = 17%** |
| RoboMME 预测器，原样流程（完整任务目标，没有 `go_home`） | 0/24 | 0/24 |
| 零样本 Qwen3-VL-4B，不给步骤历史 | 0/24 | 0/24 |
| 零样本 Qwen3-VL-4B，`show_history: true` | 0/24 | 0/24 |

这些行同时改变了规划器、切换方式、是否执行 `go_home` 和策略是否接收完整任务目标，不是只替换规划器的一组消融。固定顺序的结果也不是任意自适应规划器的严格上限：它不会在掉落等异常后插入新的恢复步骤。当前 `run_vlm_*.yaml` 对应 `show_history: true`；要复现实验中的“不给步骤历史”，需在配置副本中设为 `false`。

失败原因：

- **固定顺序**：「红×2 → 绿×1」的 12 个失败里，11 个是按按钮时环境判失败，也就是有一次「放到目标上」没被环境记上，主要是第 3 个问题（叠放）；另 1 个是方块在抓起后滑落，固定顺序规划器只会重复「放」，被前置条件连续拒绝后放弃。「蓝 → 绿 → 红」要把三块叠在同一个目标上，更难：20 个失败里，12 个在第三次抓取时抓错了方块，4 个按按钮时判失败，3 个超出 1300 步，1 个方块滑落后放弃。
- **RoboMME 预测器**：它在单色任务上是 96%，换成组合指令后一个都没成功。按失败时正在执行的子目标分：换颜色后的第一次抓取 23 次，是抓回了原来的方块（第 1 个问题）；第一步就抓错颜色 11 次（第 2 个问题）；「放红色」11 次；按按钮时判失败 3 次。「放红色」这 11 次是同一个模式：第二次放完红色后，预测器反复输出词表外的 "pick up the green cube"（不带序号，是 RoboMME 其他任务里的写法），harness 每次都拒绝，已经过时的「放红色」就一直执行下去，最后被环境判失败。
- **零样本 4B，不给步骤历史**：抓起红色后，它自己写的记忆明明是「下一步：把红色放到目标上」，输出的却还是「第一次抓红色」，被前置条件（手里没拿东西才能抓）连续拒绝后放弃（35 次），其余超时（13 次）。prompt 里只有画面、夹爪宽度和记忆，它不知道刚才做了什么、做完没有。
- **零样本 4B，给步骤历史**：prompt 里加上「已执行的步骤」和「这次调用的原因」之后，不再被拒绝，但换来两种新失败：48 个 episode 里有 44 个在 `go_home` 之后反复选 `go_home`，直到超时，而它自己写的记忆是「下一步：抓绿色」；另外 4 个红色只放了一次就转去抓绿色，数错了次数。

结论：

- 当前固定顺序与校验规则下，π0.5 在这两种组合上的成功率为 50% 和 17%，执行与技能切换仍有明显损失。`go_home` 和只发子目标的处理改善了实验结果，但还需要控制变量的评测分别测量两者收益。
- RoboMME 微调的预测器加它的原样流程，在这里测试的两个组合与配置上均为 0%。这表明当前流程没有迁移成功，尚不能据此断定所有新任务都必须重新收集数据和训练。
- 当前零样本 4B 配置没有完成这两个组合：即使给了步骤历史，也出现重复选择 `go_home` 和计数错误。微调、改进输入与切换规则、或比较更大模型，都是下一步实验方向。这里的失败和后来在 mock 分拣上看到的一样：模型写下的记忆和它的选择对不上。mock 上给结构化的「当前状态」「进度」并把记忆压到 10 字后，零样本 4B 从 0/8 升到 7/8（见 [VLM 规划器](planner_finetune.md#不微调时)）；这个设置还没有在 RoboMME 组合任务上重跑。
- 规划器输出不受约束时，拒绝词表外的回答还不够，当前子目标会过时。RoboMME 预测器也可以像 `openai_compat` 那样在解码时限定只能选词表里的子目标，这是下一步可以做的。
