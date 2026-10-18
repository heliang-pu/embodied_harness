# VLM 规划器微调：数据契约与检查细节

[返回首页](../README.md) · [训练与部署命令](planner_finetune.md) · [数据工具](data.md)

本文保留原微调指南中的数据准备细节。训练参数与服务命令统一放在 planner_finetune.md，避免两份命令说明随代码更新而分叉。

## 两条训练路线

| 项目 | JSON 技能规划器 | RoboMME SimpleSG |
| --- | --- | --- |
| 后端 | openai_compat | robomme_subgoal |
| 模型输入 | 任务、规则、技能签名、多相机、本体、memory；可选 history/reason | 任务目标、front 图像、已派发的子目标历史 |
| 模型输出 | skill/args/memory 或 done/memory JSON | 如 pick up the red cube for the second time |
| 记忆 | 模型每次重写短文本 | harness 维护的主调用序列，连续重复项去重 |
| 调度 | 通常在技能校验边界规划；call: periodic 时执行中也定期检查，导出随之补上步骤内部的检查样本 | 每隔 every_steps 且动作块耗尽时重新预测 |
| 训练入口 | data planner + scripts/train_planner_sft.py | 使用上游 SimpleSG 数据/训练流程；仓库没有专用导出器 |

SimpleSG 的模型输出在适配器里被包装为内部技能调用，memory 为空。其数据应覆盖“继续同一子目标”和“切换”时刻，不能直接把 JSON 技能边界样本作为训练目标。

[library_pickx_simplesg.yaml](../configs/robomme/mme/library_pickx_simplesg.yaml) 的 19 句 prompt 是三种颜色 × 五次抓取、三种颜色放置和一个按钮。它只是有限指令集合，不含图像、机器人动作或记忆，不能单独拿它训练出规划器或执行器。

## 原始演示需要什么

| 数据 | 要求 |
| --- | --- |
| 图像 | 与部署相同的相机语义、顺序、颜色通道与预处理 |
| 本体状态 | 与图像同步，明确向量维度、单位、夹爪量程 |
| 总任务指令 | 固定任务可在 task.goal 中定义；变化任务需要每个 episode 的目标 |
| 技能边界 | 每段技能的开始时间及自由文本标注 |
| 记忆标注 | 截至当前边界已经发生的事实及后续必要上下文 |
| 结束结果 | 成功、失败、超时和人工中断分开保存与审核 |
| action | VLM 不以它为输出标签；同时训练 VLA 时必须保存同步动作 |

动作数据还要明确关节顺序、绝对或增量控制、控制频率、单位、夹爪命令及归一化方式。仅有视频或动作数组，没有技能边界与目标，不能直接进入当前规划器导出流程。

目前 data planner 使用 LeRobotDataset 解码相机帧，需要有效的 LeRobot v3 数据及元数据，支持 video 或 image feature。并非只要创建下面几个同名空文件就够了：

```text
dataset/
├── meta/info.json
├── meta/episodes/…
├── data/chunk-000/file-000.parquet
└── videos/<camera-feature>/chunk-000/file-000.mp4
```

读取器直接读取 Parquet 的 episode_index、frame_index、timestamp、language_persistent；配置了 proprio 映射时还需要 state_key。frame_index 应是 episode 内从 0 开始的连续帧号，timestamp 为 episode 内秒数。

## 动态目标的存储不能混淆

goal_source: episode 时，线上从环境 instruction() 取目标；数据导出从 mapping.instruction_key 指定的实际字符串列取目标。该字符串在同一 episode 内应一致且非空。

```yaml
state_key: observation.state
instruction_key: instruction
cameras:
  front: observation.images.front
  wrist: observation.images.wrist
proprio:
  gripper_width: "state[7]"
```

这个 mapping 对应 JSON PickXtimes。state[7] 只有在第 8 项确实是夹爪宽度时才可复用。

**LeRobot 样本中的 task 不一定是 Parquet 的 task 字符串列。** 普通示教的 reader 仍需要 instruction_key 指定的实际字符串列；schema v2 的 from-runs 数据则从 sidecar 读取原 episode goal，即使没有 instruction 字符串列也可直接导出。复制数据集时应保留 .embodied_harness 目录。

## language_persistent 如何表示标注

同一 episode 每帧携带同一组完整语言行，读取器只取其中一份。以下例子表示第一次放置已经完成，在 12.4 秒开始第二次抓红色方块：

```json
[
  {"role": "assistant", "style": "subtask", "timestamp": 12.4, "content": "pick up the red cube"},
  {"role": "assistant", "style": "memory", "timestamp": 12.4, "content": "红色已完成1次放置，目标3次；现在开始第2次抓取。"}
]
```

解析器使用 style、timestamp、content。lerobot-annotate 可能还带 camera、tool_calls 等字段，应保留原有 Arrow 类型。subtask 自由文本经 labels.yaml 映射为具体调用，如 pick(color=red)。

## memory_in 与 memory_out 的时间关系

memory_out 在规划决策被接受时就保存，此时所选技能尚未执行。它描述截至当前的历史，不能把将来的动作结果记为已经成功。

| 当前观测 | memory_in | 目标调用 | memory_out 示例 |
| --- | --- | --- | --- |
| 初始 | 空字符串 | pick(color=red) | 空字符串，或初始目标与进度 |
| 第一次抓取完成 | 上次 pick 决策写下的文字 | place(color=red) | 已抓住红色，放置完成0次；准备第1次放置 |
| 第一次放置完成 | 上次 place 决策写下的文字 | pick(color=red) | 已完成放置1次；开始第2次抓取 |
| 第二次抓取完成 | 上次 pick 决策写下的文字 | place(color=red) | 已完成放置1次；准备第2次放置 |

[annotations.py](../src/embodied_harness/data/annotations.py) 和 [steps.py](../src/embodied_harness/data/steps.py) 的规则是：

1. 在边界 t 取不晚于 t 的最新 memory 行作为 memory_out，没有则为空。
2. memory_in 来自上一个实际导出的 planner step，第一步固定为空。
3. 普通示教的连续同调用标注段合并，保留首次记忆；运行 sidecar 中的重复规划请求不合并。
4. call:null 的片段被跳过，不单独推进记忆链，并阻断两侧同调用的合并。
5. memory 按 planner.memory_max_chars 截断，默认 200 字符，不是 200 token。

长任务中应专门检查“画面相似、历史次数不同”的样本。不要把终局计数提前复制到每帧，也不要用模型自己生成、未经核验的记忆直接充当正确标签。

## 自动 done 与数据边界

普通示教数据有 subtask 时在末帧追加 done，因此仍应先审核其已成功完成。schema v2 的运行数据使用独立 planner sidecar：只在 episode 判成功且有真实 terminal 观测时生成 done，不从动作数据末帧猜测完成。

不需要再标一个名为 done 的 subtask。若最后一帧需要准确终局记忆，可在该时间补 memory 行，否则会沿用最近的旧记忆。PickXtimes 次数够了仍应先执行 press_button，不能用模型 done 代替环境要求的按钮。

trace.frames 只记录真实动作帧；完成或环境 terminated 后，采集器另外写 terminal.json 和终态相机图片。schema v2 的 from-runs planner 导出用它生成 done，不把终态动作伪造进 VLA 帧。旧轨迹没有 terminal sidecar 时不会自动生成可靠 done。

## 数据覆盖与导出检查

没有经过本项目验证的通用“至少 N 条”门槛。先用少量完整演示验证整个导出、训练和部署链路，再按验证集错误扩充颜色、目标、次数、布局、遮挡和失败恢复场景。

按完整 episode 留出验证数据；跨采集批次或场景的更严格划分需自行整理。普通示教可用 --frame-offsets 做少量增强，仍需核对目标语义；运行 sidecar 的 planner 决策使用精确观测，只接受默认 offset=0。

导出后检查：

- train.jsonl 非空；有验证 episode 时才生成 val.jsonl。
- manifest 中相机、尺寸、编码、本体字段、记忆上限、prompt 开关（show_history、show_state、show_progress、show_rejections）、经验指纹（lessons）、extra_body、allow_done（no_done）、调用方式（periodic 时的 call）、system prompt 除目标外的指纹（system）和任务目标指纹（goal）与部署一致。部署配置写上 planner.trained_manifest，validate 和 run 开始时会逐项对比并警告。
- images 数量、顺序与 user 中的图像占位符一致；默认路径相对导出目录。
- 每个技能与参数、press_button/done 的样本覆盖合理。
- 样本数已考虑合并、null 跳过和 offset 去重。
- 先 data planner / library，再 relabel --in-place；否则旧 labels.yaml 可能不认识改写后的标签。

## show_history、评测和重放

JSON planner 的 show_history 默认 false；打开后会渲染已派发主调用及本次调用原因。历史不是成功清单，也不包含 before_retry 等恢复原语。

当前导出将 history 存入 samples.jsonl，data eval 恢复同一历史。旧导出没有该字段时，show_history=true 的评测会明确要求重新导出。离线评测仍使用标注 memory，不会把上一条预测的错误记忆传到下一条，所以不能替代闭环实验。

新 planner 轨迹记录动态 goal；replay 恢复记录的 goal 和 history。旧记录缺 goal 时使用运行配置的目标，缺 history 时使用空列表，因此旧动态任务轨迹仍需补齐上下文后再比较。

show_state、show_progress 与 show_history 同理，都会改变 prompt，训练和部署要一致：

- show_state 渲染谓词（由映射后的本体算出，映射要提供谓词用到的本体字段）和「上一步」。演示里的每一步都算成功，所以训练样本里的「上一步」都是「成功」；运行时才会出现失败或被打断。例外是 periodic 导出里没有 success 条件的策略技能：它只能被定期检查换掉，所以导出按「被打断」记，也不提交它的效果。
- show_progress 渲染账本。导出时按前面各步技能的 effects.on_success 重算（演示里的步骤都算成功完成），存入 samples.jsonl 的 ledger；打开它的 data eval 遇到没有 ledger 的旧导出会要求重新导出。
- show_rejections 只在重采样时出现。演示导出的样本里没有被拒的回答（运行记录导出的样本保留当时的拒绝原因），用演示微调的规划器一般不开。
- 经验（task.lessons 及 planner.lessons 中审核过的条目）写在 system prompt 里；换了经验文件就等于换了 prompt，manifest 里的 lessons 指纹会变。

轨迹里的规划器调用记录同时保存 ledger、last 和 rejected，replay 按记录重建。

## SFT 实现边界

训练脚本使用支持 Qwen3-VL 的 Transformers、PyTorch 和 Pillow；默认 BF16、SDPA、梯度检查点、冻结 visual 编码器，训练其余参数。它是单设备基础 SFT，不是 LoRA，没有 DDP、梯度累积、定期 checkpoint 或优化器断点续训。输出是最终完整 Hugging Face 模型目录，不自动选择最佳验证 checkpoint。

只对 assistant JSON 和结束 token 计算 loss，屏蔽输入和 padding。held_out_decoding 是普通贪心生成后解析对象并比较字段；省略 args 和 args:{} 对无参数技能等价，数组/null 等非对象响应记为无效。data eval 则通过实际模型服务，可能启用了约束解码，两者不是同一种测试。

显存需求取决于图像 token、模型、batch 和序列长度，仓库没有实测硬件门槛。先小 batch 验证 loss、显存和样本，再扩大；固定模型 revision、依赖版本、run/mapping/labels、导出 manifest 和训练命令，便于复现实验。
