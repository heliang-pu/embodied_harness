# OpenPI、WebSocket 与 LeRobot v3.0

运行时通过 `executor.type: websocket` 调策略。OpenPI 在自己的 Python/GPU 环境中加载模型，
训练数据保持 LeRobot v3.0；harness 的调度器不导入 OpenPI、JAX、Torch 或 LeRobot。

```text
Env → WebsocketPolicyExecutor → OpenPIClient → WebSocket → OpenPIPolicy → OpenPI checkpoint
      观测与提示词映射           连接/编解码/校验            字段与图像布局适配

LeRobot v3.0 → OpenPILeRobotDataset → OpenPIDatasetSpec → OpenPI 机器人变换/归一化/模型
               官方 v3 解码与分块     与服务端共享字段映射
```

`policy/transport/` 负责 openpi 的 metadata 握手、msgpack/numpy 编解码、连接生命周期与动作回复检查；
`policy/executor/` 负责提示词和指令版本；`policy/vla/openpi.py` 负责加载模型；
`policy/integrations/openpi.py` 负责输入映射。动作单位、关节顺序、归一化和机器人变换仍由训练配置与
checkpoint 决定，不根据数据集名字猜测。OpenPI 输出必须已经回到环境的动作空间。

## 1. 数据格式和映射

数据集的 `meta/info.json` 必须声明 `codebase_version: v3.0`，以及相机、状态和动作特征。
LeRobot v3 是磁盘数据格式，不是 WebSocket 消息格式：线上传解码后的数组，不传 Parquet 或视频文件。

示例映射是 [`openpi_mapping.yaml`](../configs/aloha_sim_insertion/openpi_mapping.yaml)：

```yaml
images:
  images/cam_high: observation.images.top
state_key: observation.state
action_key: action
task_key: task
image_layout: chw
subtask_style: subtask
```

左边 `images/cam_high` 表示 OpenPI 的嵌套输入 `images["cam_high"]`；右边是数据集或请求中的
字面特征名，点号不会被拆开。这个 ALOHA 示例用 CHW RGB uint8；其他模型按其输入变换设置
`image_layout: hwc` 或不同目标键。现代 LeRobot 解码的 CHW float[0,1] 和服务端收到的 HWC uint8
都会转换到映射声明的布局。映射不会改变状态与动作的数值单位。

```bash
embodied-harness policy openpi-check /path/to/dataset_v3 \
  --mapping configs/aloha_sim_insertion/openpi_mapping.yaml

# 在安装现代 LeRobot 的环境中，进一步验证真实解码及动作块形状。
embodied-harness policy openpi-check /path/to/dataset_v3 \
  --mapping configs/aloha_sim_insertion/openpi_mapping.yaml \
  --sample 0 --action-horizon 50 --video-backend pyav
```

不带 `--sample` 只检查元数据，不证明视频完整或模型可训练。旧版数据需要用 LeRobot 的官方
迁移工具转换；不能只改版本字符串。

## 2. 启动 OpenPI 模型服务

按 [OpenPI 官方安装说明](https://github.com/Physical-Intelligence/openpi) 准备独立模型环境，
再在该环境安装本项目的 `policy` extra。无需在 harness 环境安装 OpenPI。

```bash
python -m pip install -e '/path/to/embodied-harness[policy]'

python -m embodied_harness.policy.vla.openpi \
  --config YOUR_OPENPI_TRAIN_CONFIG \
  --checkpoint /path/to/openpi/checkpoint \
  --mapping /path/to/embodied-harness/configs/aloha_sim_insertion/openpi_mapping.yaml \
  --host 127.0.0.1 --port 8001
```

`--config` 是 OpenPI 注册的训练配置名，必须与 checkpoint 一致。服务端调用 OpenPI 官方
`get_config()` 和 `create_trained_policy()`，由 checkpoint 自带的统计量完成归一化/反归一化。
`--default-prompt` 只用于缺省语言输入；正常运行时使用 harness 发来的提示词库原文。
`--max-request-mb` 是单个请求的上限（默认 64 MiB），相机多或分辨率高时调大；监听地址的建议见 [策略](policy.md)。

服务端请求采用 LeRobot 特征名：图像是 HWC RGB uint8，`observation.state` 是状态向量，
`prompt` 是当前子任务。映射层把这些字段转换为模型所需的嵌套键和布局。
回复遵循 openpi 协议：`actions` 是 `[T, action_dim]` 的有限数值数组。
客户端会拒绝空块、维度错误、非数值和 NaN/Inf；服务端声明 `action_dim` 时还会核对维度。

也可以直接连接 OpenPI 自带的 `scripts/serve_policy.py`，此时执行器的请求键及图像布局必须
匹配该服务器的原生输入。现有 RoboMME 中含 `/` 的平面请求键保持原样；只有显式开启
`nested_keys` 才把路径拆成嵌套字典。

## 3. 运行与回流

[`run_insertion_openpi.yaml`](../configs/aloha_sim_insertion/run_insertion_openpi.yaml) 使用现有
ALOHA 仿真、固定技能顺序和 WebSocket 策略，先验证策略集成。它要求 checkpoint 学过该库的
`pick up the socket and the peg` / `insert the peg into the socket` 提示词；公开整任务 checkpoint
不能被当作子任务策略的等价替代。

```bash
embodied-harness validate configs/aloha_sim_insertion/run_insertion_openpi.yaml
embodied-harness run configs/aloha_sim_insertion/run_insertion_openpi.yaml \
  --episodes 3 --seed 1000 --hold-policy abort
```

配置开启 `trace.frames`，因此在安装 LeRobot 和视频依赖的环境可复用现有导出：

```bash
embodied-harness data from-runs runs/<run_id> --out /path/to/new_dataset_v3 --success-only
```

要加入 VLM，替换示例的 `planner` 配置即可，策略服务和数据映射不需要修改。

## 4. 用 v3 数据训练 OpenPI

本项目提供 `OpenPILeRobotDataset`，使用现代 `lerobot.datasets.lerobot_dataset.LeRobotDataset`，
由 LeRobot 处理 v3 分片、视频、`task_index` 解码，以及不跨 episode 的动作块采样/末帧填充：

```python
from embodied_harness.data.openpi_dataset import OpenPILeRobotDataset
from embodied_harness.policy.integrations.openpi import load_openpi_spec

dataset = OpenPILeRobotDataset(
    "/path/to/dataset_v3",
    load_openpi_spec("configs/aloha_sim_insertion/openpi_mapping.yaml"),
    repo_id="local/insertion",
    action_horizon=50,
    video_backend="pyav",
)
sample = dataset[0]  # images, state, actions, prompt；尚未归一化
```

有 `language_persistent` 时，按样本时间选当前子任务，回退到解码后的 `task`。先用现有
`data relabel` 将标签换成提示词库原文。设置 `subtask_style: null` 则只使用整任务。

训练侧应把该 dataset 接到 OpenPI 的机器人/模型变换和 DataLoader 之前：映射已经完成，
**不要再重复执行原始的 repack 或 `PromptFromLeRobotTask`**。例如，在兼容现代 LeRobot 的
OpenPI 环境中：

```python
import dataclasses
from openpi import transforms
from openpi.training import data_loader

# data_config 来自所选的训练配置；归一化统计也必须用同一个 v3 dataset 计算。
mapped_config = dataclasses.replace(data_config, repack_transforms=transforms.Group())
dataset = data_loader.transform_dataset(dataset, mapped_config)
# 然后交给训练器使用的批处理/DataLoader。
```

本次核对的 OpenPI 上游 `main` 仍引用 `lerobot.common.datasets` 并锁定旧 LeRobot，
因此默认训练脚本不能直接读取 v3。这里提供了 v3 reader 和映射入口，没有改写上游源码或
全局 monkey-patch；训练环境需要完成相应的现代 LeRobot 兼容，或使用已迁移的 OpenPI 分支。
模型服务与这个训练加载器版本问题独立，已有 OpenPI checkpoint 可以直接通过 WebSocket 使用。

本地回归测试覆盖映射、v3 reader 调用契约、动作块、模型加载接口和 WebSocket 通信；
实际 GPU 权重、训练收敛和仿真成功率需要在相应模型环境验证。
