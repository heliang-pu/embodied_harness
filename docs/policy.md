# 策略（VLA / WAM）

harness 里的策略负责执行一个子任务：输入当前观测（各相机图像、状态向量）和一句提示词，输出一段动作（动作块，`[T, 动作维度]`，已经反归一化到环境的动作空间）。VLA、WAM（world action model），或者任何按这个方式工作的模型都可以接进来。

策略跑在自己的进程和 Python 环境里，通过 websocket 和 harness 通信。harness 和仿真器不需要导入模型的依赖，换模型也不用改 harness。

## 提示词库

策略只认它训练时见过的提示词，所以每个策略都要带一份提示词库：列出训练数据里出现过的每一句子任务提示词，一字不差，每句一个 id。技能只引用 id，harness 派发技能时把原文发给策略。

- 提示词库和权重一起发布，库名就是版本号。提示词有任何改动，就换一个库名。
- 训练数据里的子任务文字要和库里的原文完全一致。`embodied-harness data relabel` 会把标注里的自由描述统一换成库里的原文，见下一节。
- `embodied-harness data library` 用标注片段统计每句提示词覆盖的 episode 数和时长 p50/p95，技能的默认超时就是 p95 × `timeout_factor`。
- 规划器的运行配置里写 `planner.trained_library`。规划器训练用的库和策略带的库不一致时，harness 拒绝启动。

## 训练数据：和规划器用同一份数据集

策略和规划器用同一份 LeRobot 数据集，各取所需：

| 字段 | 策略 | 规划器 |
| --- | --- | --- |
| 每帧图像、状态 | ✓ | ✓（只用子任务交界的帧） |
| 动作 | ✓ | — |
| LeRobot `task` / `task_index` | 无 subtask 时作为回退语言条件 | 不自动解码 task_index；动态目标要有 mapping.instruction_key 指定的字符串列 |
| `language_persistent` 里的子任务行（提示词库原文） | ✓（每帧的提示词） | ✓（下一步做什么） |
| `language_persistent` 里的记忆行 | — | ✓ |

以下是策略训练分支。若同一份数据还要训练规划器，先完成 data planner 和 library 统计，再原地 relabel；或为策略使用 relabel --out 副本。原地 relabel 后继续用旧自由文本 labels.yaml 会报缺少标签，应重新生成并审核映射。

流程：

```bash
# 1. 标注（在副本上），人审标签表里对不上提示词库的条目
embodied-harness data annotate configs/my_task/run.yaml /path/to/raw --out /path/to/dataset --labels configs/my_task/labels.yaml
# 2. 把还不是库原文的子任务标签原地换成库原文（改的就是第 1 步生成的副本）
embodied-harness data relabel configs/my_task/run.yaml /path/to/dataset --labels configs/my_task/labels.yaml --in-place
# 3. 准备要微调的 checkpoint：复制一份（权重用软链接），在预处理里插入「按帧取子任务」的步骤
embodied-harness policy subtask-checkpoint /path/to/pi05_base --out /path/to/pi05_base_subtask
# 4. 用 LeRobot 直接在这份数据集上微调
lerobot-train --policy.path=/path/to/pi05_base_subtask --dataset.root=/path/to/dataset ...
```

第 3 步插进去的步骤在训练时按每一帧的时间戳，从 `language_persistent` 里找出当时所在的子任务，替换掉 `task`，然后策略自己的预处理再生成提示词（π0.5 是 `Task: <子任务>, State: …`）。代码位于 `embodied_harness.policy.integrations.lerobot.ActiveSubtaskAsTaskStep`，checkpoint 也保存这个类路径。

- 数据集本身不用改：`task` 还是 episode 的指令。LeRobot 在加载样本时解析它；当前 planner reader 直接读 Parquet，不会解析 task_index。普通示教的动态目标要另有 instruction 等字符串列；schema v2 的 from-runs 数据可以从随附 sidecar 恢复目标。
- 推理时没有标注列，这一步什么都不做。harness 发的提示词（库原文）就是 task，训练和部署看到的提示词完全一样，已在 π0.5 上用真实 tokenizer 核对过。
- LeRobot 按类路径加载这一步，所以跑 `lerobot-train` 的环境里要装上 embodied_harness（`pip install -e <本仓库>`）。训练中保存的 checkpoint 会带着这一步。
- 支持在生成提示词前读 `task` 的策略：π0.5，以及直接对 `task` 分词的 π0、SmolVLA 等。只适用于从已有 checkpoint 微调（比如 pi05_base），从头训练还需要别的接法。
- 某一帧不在任何子任务片段里时，保持原来的 `task`。

上述子任务条件训练不使用 memory；仍需要同步的图像、状态、动作和子任务标注。已训练成“总目标 + 子目标”协议的策略（如 RoboMME 原版）应保持其对应的输入约定。

`data relabel --out <新目录>` 仍然可以另外复制一份，不改原数据集。

## 检查策略还听不听指令

分层系统能组合出新任务，前提是策略按提示词做事。在窄的数据上微调，可能让策略不管提示词是什么都做同一件事：分层 VLA 的系统性研究（arXiv 2606.10267）里，一个策略在仿真数据上微调后，长任务成功率从 41.3% 掉到 7.5%，原因就是听不进指令了。微调前后各跑一次这个检查：

```bash
embodied-harness policy steer-check configs/my_task/run.yaml \
  --frames runs/<run_id> --during pick --call "pick(color=red)" --call "pick(color=green)" --out steer.json
```

- 同一批观测，分别配上每个调用的提示词发给策略，每个提示词重复问几次（`--repeats`，默认 3）。比较两个提示词平均动作块之间的距离（separation）和同一个提示词重复采样之间的距离（noise）。距离超过噪声 `--min-ratio` 倍（默认 2）算分得开；确定性的策略没有噪声，改用所有动作的整体幅度做尺子，超过 `--min-relative`（默认 5%）算分得开。
- 观测来自 `trace.frames` 记录的运行（真机也行，不需要仿真和真值），`--samples` 条（默认 8）均匀取；`--during pick` 只取 pick 执行期间的帧，在「该决定抓哪个」的状态下比较不同目标。不给 `--frames` 时用环境 reset 后的第一帧（`--seed` 起的若干个种子）。
- 若策略请求模板包含 {goal}，检查时也会传入总任务。goal_source: episode 的观测从环境 instruction() 或轨迹 instruction 事件恢复各自目标；缺失目标会报错，避免使用空目标进行比较。
- 不给 `--call` 时比较任务里所有由策略执行的调用。有提示词对分不开时退出码为 1，可以放进微调后的回归检查。
- 只要求策略对不同提示词给出不同动作，不判断动作对不对；动作对不对要靠 `probe` 或闭环评测。视频解码需要 imageio（带 ffmpeg）或 PyAV。

## 策略服务：协议

默认使用 OpenPI 的 websocket 协议（msgpack，numpy 数组按 OpenPI 的方式编码）：

1. 客户端连上后，服务端先发一条元信息（字典，比如动作块长度、需要的键）。
   仓库自带服务端若 policy 提供 `reset()`，会在每个新连接建立时调用它清空 episode 状态；外部 openpi 服务需自行保证每个连接或任务开始时重置时序状态。
2. 之后每条消息：客户端发观测字典（numpy 数组和字符串），服务端回一个字典，至少包含 `actions`，另外带 `server_timing`（推理耗时）。
   数组只能是布尔、整数或浮点类型。其他 dtype（object、结构体、复数、字符串等）在解码时直接报错：object 数组会把收到的字节当成对象指针，一条消息就能让服务端崩溃。
3. 服务端出错时发一条文本帧（traceback）并断开，客户端把它当作错误抛出。
4. `GET /healthz` 返回 200，用于健康检查。

请求里各个键叫什么由执行器配置决定，所以同一个 harness 可以连不同约定的服务端：

```yaml
executor:
  type: websocket
  library: library.yaml
  handoff: finish_chunk
  params:
    url: ws://127.0.0.1:8001
    images: {front: observation.images.front, wrist: observation.images.wrist}   # 相机 -> 请求键
    state: {proprio: state, key: observation.state}   # 本体字段 state（状态向量）-> 请求键
    texts: {prompt: "{prompt}"}      # 请求键 -> 模板；{prompt} 是库里的原文，{goal} 是本 episode 的任务目标
    n_action_steps: 10               # 每个动作块执行前 10 步，然后重新推理
    request_timeout_s: 30
```

不写 `texts` 时，库里的原文放在 `prompt_key`（默认 `prompt`）。有的策略同时需要任务目标和子任务，比如 RoboMME 的 π0.5 要 `{prompt: "{goal}", simple_subgoal: "{prompt}"}`。

## 现成的服务端

**LeRobot checkpoint**（pi05 等）：在装了 lerobot 的环境里运行

```bash
python -m embodied_harness.policy.vla.lerobot --checkpoint outputs/train/my_run/checkpoints/last/pretrained_model --port 8001
```

它接收策略配置里的每个图像特征（如 `observation.images.front`，HxWx3 uint8）、`observation.state`（float32）和 `prompt`（文字），返回 `{"actions": [chunk_size, action_dim]}`。预处理和后处理用 checkpoint 自带的 processor。元信息里有它需要的键、`chunk_size` 和动作维度，连上后会打印在日志里。

**OpenPI checkpoint**：新增 `python -m embodied_harness.policy.vla.openpi --config ... --checkpoint ... --mapping ...`，
在独立 OpenPI 环境加载模型，接收 LeRobot 特征名，通过共享映射转换为模型原生输入。
LeRobot v3.0 数据校验、动作块读取和完整配置见 [OpenPI 集成](openpi.md)。

也可以直接连 OpenPI 自带的服务端，例如 RoboMME 的 `scripts/serve_policy.py`，按它的键名配置
`images`、`state`、`texts`，见 [`configs/robomme/mme/run_mme_pickx.yaml`](../configs/robomme/mme/run_mme_pickx.yaml)。

仓库自带的两个服务端（`policy.vla.lerobot` 和 `policy.vla.openpi`）默认监听 `--host 0.0.0.0`（所有网卡）。协议没有鉴权，连得上端口就能调用策略，所以只在可信网络里开放：只给本机用时写 `--host 127.0.0.1`，跨机器时绑定局域网或 tailnet 的地址。单个请求最大 `--max-request-mb` MiB（默认 64；四路 480×640 RGB 图像加状态不到 4 MiB），更大的请求会让服务端以 1009（message too big）关闭这条连接，并在日志里提示。

## 自己写服务端

任何有 `infer(obs) -> dict` 方法的对象都可以直接服务起来：

```python
import numpy as np
from embodied_harness.policy.serving.websocket_server import serve


class MyPolicy:
    def infer(self, obs: dict) -> dict:
        front = obs["observation.images.front"]      # HxWx3 uint8
        state = obs["observation.state"]             # float32[n]
        text = obs["prompt"]                         # 提示词库原文
        actions = ...                                # float32[T, action_dim]，环境的动作空间
        return {"actions": np.asarray(actions, dtype=np.float32)}


serve(MyPolicy(), host="0.0.0.0", port=8001, metadata={"chunk_size": 50})
```

需要 `policy` 这个 extra（`pip install -e ".[policy]"`：msgpack、numpy、websockets）。服务端和 harness 可以在不同的机器上。`serve` 的 `max_size` 是单个请求的字节上限，默认 64 MiB。

### 接入 XPolicyLab

XPolicyLab 也使用 WebSocket，但它规定了另一套标准线格式：`hello`、`reset`、`infer`、`trial_end`
请求/响应帧，带 `message_id`、`evaluation_id`、`trial_id` 和 `step`，观测放在 `vision`、`state`、
`instruction` 等标准字段中。harness 同时支持两种协议；已有 OpenPI 配置无需修改，XPolicyLab
服务把执行器参数改成 `protocol: xpolicy` 即可：

```yaml
executor:
  type: websocket
  library: library.yaml
  params:
    protocol: xpolicy
    url: ws://127.0.0.1:19000
    images:
      head: vision/cam_head/color
      left_wrist: vision/cam_left_wrist/color
      right_wrist: vision/cam_right_wrist/color
    state: {proprio: state, key: state/left_arm_joint_state}
    texts: {instruction: "{prompt}"}
    nested_keys: true
    n_action_steps: 16
```

这里的 `images`、`state` 和 `texts` 只负责把 harness 观测映射到 XPolicyLab 的标准字典；动作由
服务端返回 `payload.actions`，可以是数值动作块，也可以是标准动作字典。RoboTwin adapter 会把
双臂关节动作字典转换为其 14 维动作顺序。XPolicyLab 提供统一的 policy-server WebSocket 层，仿真器或真实机器
人仍各自需要一个 environment adapter，把相机、状态和动作映射到这套字段。RoboTwin adapter 位于
`embodied_harness.env.robotwin`，只把 RGB、14 维关节状态发给策略，仿真器的成功判定只在终止时用于评分。

## 接入 WAM

WAM 一般同时预测未来画面和动作，接入方式和 VLA 相同：回复里放 `actions`，其他输出（预测的画面、价值估计等）可以放在别的键里，执行器目前只读 `actions`，其余的忽略。需要注意：

- **推理慢**：可以加大 `n_action_steps`（一次推理多执行几步），调大 `request_timeout_s`。执行器现在是同步推理：动作队列空了才推理，推理期间控制循环在等。仿真里没关系；真机上慢模型会造成停顿，需要异步推理，这个还没实现。
- **历史帧**：执行器每次只发当前一帧。模型需要历史帧时由服务端自己缓存。协议里还没有「新 episode 开始」的消息，有状态的服务端要靠别的信号清空缓存，比如提示词或任务目标变了。真要接有状态的模型时，我们再在协议里加一个 reset 消息。
- **用预测画面做校验**：WAM 预测的未来画面可以用来判断子任务是否会成功，这需要新的校验器类型，目前的校验器只看本体。

## 切换和动作块

- `n_action_steps`：每个动作块执行多少步。越小越灵敏，推理越频繁。RoboMME 的 π0.5 用 16（动作块长度 20）。
- `handoff`：规划器换技能时，`finish_chunk` 把旧动作块剩下的动作执行完再切换，`hard` 立即丢弃。子目标切换频繁（`planner.call: periodic`）时用 `hard`。
