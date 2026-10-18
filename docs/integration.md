# 接入新的环境、策略和规划器

源码顶层按七个职责组织：`config`、`core`、`data`、`env`、`planner`、`policy`、`runtime`。
新增模型适配器放进 `policy/vla`、`policy/va`、`policy/rl` 或 `policy/wam`；共享的执行器、
WebSocket 服务、传输协议和训练接入分别在 `policy/executor`、`policy/serving`、
`policy/transport`、`policy/integrations`。
控制流放在 `runtime/supervisor.py`，技能判定放在 `runtime/verifier.py`，账本放在 `core/ledger.py`。
技能注册表和表达式分别在 `config/skills.py`、`config/expressions.py`，不再单独建顶层目录。

三类组件都是普通的 Python 类，在 `src/embodied_harness/runtime/factory.py` 的三张表里登记名字，运行配置里用 `type` 选择：

```python
ENVS = {"mock_sorting": MockSortingWorld, "robomme": RoboMMEEnv, "aloha_sim": AlohaSimEnv}
EXECUTORS = {"mock": MockPolicyExecutor, "websocket": WebsocketPolicyExecutor}
PLANNERS = {"mock_oracle": ..., "sequence": ..., "openai_compat": ..., "robomme_subgoal": ...}
```

重的依赖（仿真器、websockets、msgpack 等）在构造实例或收发消息时才导入，没装这些依赖的环境里也能加载配置、跑测试；基础依赖只有 pydantic、PyYAML 和 numpy。

## 环境 / 机器人

```python
from typing import Any

from embodied_harness.core.types import Observation


class MyRobot:
    def __init__(self, params: dict): ...            # env.params 原样传进来

    @classmethod
    def proprio_keys(cls, params) -> list[str]:     # 本体字段名，谓词和条件里可以用；validate 用它检查名字
        return ["gripper_width", "state"]

    @classmethod
    def camera_keys(cls, params) -> list[str]:      # 可选；validate 用它检查 planner.params.cameras
        return ["head", "wrist"]

    def reset(self, seed=None) -> Observation: ...
    def observe(self) -> Observation: ...
    def act(self, action) -> Any: ...                # 执行一个控制步，可返回实际发出的数值动作
    def hold(self) -> Any: ...                       # 没有技能在执行时每个周期调用，保持当前姿态
    def stop(self) -> None: ...                      # 急停
    def close(self) -> None: ...

    # 可选
    def instruction(self) -> str | None: ...         # 每个 episode 的指令，配合 task.goal_source: episode
    def snapshot(self) -> dict | None: ...           # 仅仿真：技能边界处保存世界状态，续跑时恢复；
    def restore(self, state: dict) -> None: ...      # trace.snapshots 时每个调用开始都存一份，probe 用它重跑
```

`Observation` 的字段：

| 字段 | 说明 |
| --- | --- |
| `t` | 时间 |
| `images` | 相机名 → HxWx3 uint8 图像 |
| `proprio` | 本体字段名 → 值（数字、布尔或数组），和 `proprio_keys` 对应 |
| `info` | 特权信息，只有仿真填。闭环里不读，只在环境结束 episode 时读一次 `info["success"]` 用来评分 |
| `terminated` | 环境自己结束了 episode（任务完成或失败、步数用完）。harness 看到后立即停下 |

`act()` 和 `hold()` 可以返回实际发给机器人的数值动作（经过限幅以后的、或者由原语换算出来的），和 LeRobot `Robot.send_action` 的约定一样。打开 `trace.frames` 时会记下这个返回值，运行记录才能用 `data from-runs` 变成训练数据；返回 `None` 表示不知道。

`act()` 收到两种动作：

- 策略的动作：执行器返回的数组，直接发给机器人；
- 脚本原语：`{"primitive": 名字, "version": 版本号}`，由环境自己解释成动作；原语带参数时还有 `"args": {参数: 值}`（比如衔接规则里的 `approach(color=green)`）。原语技能执行 `runtime.primitive_steps[名字]` 步，技能文件里没有成功条件时，步数走完就算成功。例子是 RoboMME 的 `go_home`：每一步把关节朝本 episode 的初始姿态移动一点，同时张开夹爪。带目标的原语（移动到某个物体上方）在真机上要用感知算出目标位置，不能读仿真真值。

上真机时：

- 本体字段的计算方式必须和训练数据的映射文件（`mapping.yaml`）一致；
- `runtime.clock: real`，`planner.threaded: true`，规划期间控制循环不停；
- `hold()` 必须随时可以安全调用：规划中、等人（HOLD）时都会调用。

## 策略

首选做法是把模型放进一个 websocket 服务，执行器用 `websocket`，不用写代码，见 [策略](policy.md)。只有在不能走网络的情况下才需要写执行器类：

```python
from embodied_harness.core.types import ActionChunk
from embodied_harness.policy.executor.base import ExecutorCaps


class MyExecutor:
    def __init__(self, library, **params): ...       # 提示词库 + executor.params

    def prompt_library(self): return self._library
    def capabilities(self) -> ExecutorCaps: return ExecutorCaps(chunk_size=10)
    def reset(self) -> None: ...                     # 每个 episode 开始时清状态
    def set_instruction(self, call, prompt, version) -> None: ...   # prompt.text 是库里的原文
    def infer_chunk(self, obs) -> ActionChunk:
        return ActionChunk(self._version, actions)   # 带上 set_instruction 给的版本号
    def is_done(self) -> bool: return False          # 只有有限长度的脚本原语才会返回 True

    # 可选
    def begin_episode(self, task) -> None: ...       # 拿到本 episode 的任务目标
    def close(self) -> None: ...
```

## 规划器

```python
from embodied_harness.planner.base import PlannerContext, PlannerOutput


class MyPlanner:
    name = "my_planner"

    def __init__(self, **params): ...                # planner.params

    def bind(self, *, registry, task, planner_config) -> None:   # 可选：拿到技能词表，生成约束解码的 schema
        ...

    def plan(self, context: PlannerContext) -> PlannerOutput:
        # context: task（目标已按 episode 替换）、images、proprio、memory、reason、skills、resample、history、
        #          ledger（账本，任务进度）、last（上一个主调用及结果）
        return PlannerOutput(raw='{"skill": "pick", "args": {}, "memory": "..."}',
                             meta={"model": "..."}, attachments={"head.jpg": jpeg_bytes})
```

- `raw` 必须是 `{"skill", "args", "memory"}` 或 `{"done": true, "memory"}` 形式的 JSON 字符串。解析、合法性检查、重采样、超时都由调度器处理，规划器只管生成。
- `attachments` 是这次调用实际给模型看的图片，会存进轨迹，重放时原样再发。
- `context.resample > 0` 表示上一次回答被拒绝，可以换个温度重采样。
- 想让 `data planner` 能给你的规划器导出训练数据，要实现 `render_request(context)`、`settings()`、`cameras`、`proprio_keys`、`image_extension`，可参考 `OpenAICompatPlanner`。

## 测试

新组件在 `tests/` 里加测试，照着现有的写：`tests/test_policy_server.py` 起一个本地策略服务测执行器，`tests/fake_openai.py` 是一个假的 OpenAI 接口，用来测规划器。
