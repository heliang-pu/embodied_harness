# VLM 规划器：微调、部署、评测

默认的规划器后端是 `openai_compat`：用 vLLM 部署一个小 VLM（我们选的是 Qwen3-VL-2B/4B），harness 通过 OpenAI 兼容接口调用它。这一页讲怎么把它微调成你的任务规划器。数据怎么准备见 [数据](data.md)；更详细的原始数据契约、记忆时间示例和训练检查见 [微调数据细节](vlm_finetuning.md)。

```
data annotate 标注 ─► 标签表(人审) ─► data planner 导出 ─► SFT 训练 ─► vllm serve ─► data eval 离线评测 ─► replay 重放 ─► run 闭环
```

## 规划器看到什么、输出什么

prompt 由 `src/embodied_harness/planner/render.py` 生成，当前版本 `render_v1`。线上调用和训练数据导出用的是同一个函数，所以训练时模型看到的和部署时一模一样。下面是 mock 分拣任务的格式示例（三路相机，不是附带了对应图像的实测样本）：

```json
{
  "messages": [
    {"role": "system", "content": "你是机器人的任务规划器。根据任务、各相机画面、本体状态和你上一轮写下的记忆，决定下一步做什么，并改写记忆。\n任务：把桌上所有包裹按尺寸分拣：小的放 A，中的放 B，大的放 C\n规则：\n- 条码朝下的包裹先翻面再放\n可用技能：\n- pick\n- flip\n- place(bin: A|B|C)\n- retract\n只输出一个 JSON 对象：{\"skill\": 技能名, \"args\": {参数}, \"memory\": 新记忆}，任务完成时输出 {\"done\": true, \"memory\": 新记忆}。记忆不超过 200 字，只保留对后续有用的信息。"},
    {"role": "user", "content": "[head] <image>\n[wrist_left] <image>\n[wrist_right] <image>\n本体状态：gripper_width=0.045, gripper_closed=false\n记忆：已放 A 1 个、B 2 个；当前包裹条码朝下"},
    {"role": "assistant", "content": "{\"skill\": \"flip\", \"memory\": \"已放 A 1 个、B 2 个；当前包裹条码朝下，准备翻面\"}"}
  ],
  "images": ["images/episode_000003/frame_000120_head.jpg", "images/episode_000003/frame_000120_wrist_left.jpg", "images/episode_000003/frame_000120_wrist_right.jpg"]
}
```

- system 里是任务目标、规则、技能签名和输出格式，都来自配置文件。
- user 里按配置的相机顺序放图片，然后是本体字段和上一步的记忆。
- assistant 是目标输出。记忆由规划器自己改写，用来记进度（放了几个、下一步做什么）。
- `show_history: true` 时，记忆前面还有「已执行的步骤」和「这次调用的原因」两行。导出训练数据时也按同样的方式生成，保证训练和部署一致。
- `show_state: true` 时，本体状态后面加「当前状态」（谓词，如 `holding=true, released=false`）和「上一步」（如 `place(bin=A) 成功`）两行；`show_progress: true` 时再加「进度」一行（如 `sorted: A=1, B=0, C=0（共 1）；目标总数 10`）。导出时谓词由映射后的本体算出，上一步取演示里的前一步（演示里都算成功），账本按前面各步技能的 `effects.on_success` 重算。
- 任务有经验（`task.lessons` 或 `planner.lessons`）时，system 里规则后面加「经验（以往运行的总结）」。
- 这些开关都会改变 prompt，所以训练和部署要用同一组设置；`manifest.json` 记下导出时的设置，`data eval` 发现不一致会警告。`planner.allow_done: false`（`no_done`）、`planner.call: periodic`（`call`）、system prompt 除任务目标外的指纹（`system`，含规则、经验、技能和完成说明）和任务目标的指纹（`goal`）也记在里面。不微调、直接用基座模型时，建议打开 `show_state` 和 `show_progress` 并把记忆压短，见 [不微调时](#不微调时)。
- `planner.call: periodic` 时，规划器在技能执行中也被问，看到的「已执行的步骤」里已经有正在执行的调用。要用 periodic 部署，就用 periodic 的运行配置导出：导出会补上步骤内部的定期检查样本（目标是继续当前调用），见 [数据](data.md#5-导出)。只用边界样本训练的模型在定期检查时会提前换到下一步。

部署时 vLLM 用由技能词表生成的 JSON schema 做约束解码，服务正常完成生成时可以限制语法与枚举；网络错误、长度截断和技能前置条件仍由 harness 检查。

## 1. 导出训练数据

```bash
embodied-harness data planner configs/my_task/run.yaml /path/to/dataset \
  --labels configs/my_task/labels.yaml --mapping configs/my_task/mapping.yaml \
  --out exports/planner_v1 --frame-offsets 0,2,4
```

要在装了 lerobot 的环境里运行。输出 `train.jsonl`、`val.jsonl`（上面的格式）、`samples.jsonl`、`images/`、`manifest.json`，各参数见 [数据](data.md#5-导出)。

## 2. 训练

### 用仓库里的脚本

```bash
python scripts/train_planner_sft.py --model Qwen/Qwen3-VL-2B-Instruct \
  --data exports/planner_v1 --out outputs/planner_v1
```

- 默认对语言模型做全参数微调。加 `--lora-rank` 改用 LoRA：只训练语言模型各投影层（q/k/v/o、gate/up/down）上的低秩适配器，训练完合并回权重。
- 视觉编码器默认冻结（`--train-vision` 解冻，只能配合全参数微调）。梯度检查点。
- 精度：模型按 bf16 加载。全参数微调时，要训练的权重转成 fp32 主权重，AdamW 在 fp32 上更新；LoRA 时冻结的基座保持 bf16，适配器是 fp32。CUDA 上前向和反向在 bf16 autocast 下计算。保存前权重转回 bf16，和基座 checkpoint、vLLM 部署时的精度一致。不能直接在 bf16 权重上更新：lr 1e-5 时 AdamW 每步只改约 1e-5，对 |w| ≥ 2^-8 的权重（大部分语言模型权重和全部 RMSNorm 权重）不到半个 bf16 ulp，更新会被舍入掉。
- 按 harness 发给 vLLM 的方式重建 user 消息（文字和图片交错），用模型自己的 chat template，训练和部署的 token 序列一致。
- 只在 assistant 那一段（JSON 和结束符）上算 loss。
- AdamW，余弦学习率，预热 3%，梯度裁剪 1.0。每个 epoch 结束算一次验证 loss；训练完在验证集上贪心解码，统计 JSON 能否解析、调用是否一致、记忆是否一致。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 基座模型，Hugging Face ID 或本地路径 |
| `--data` | 必填 | `data planner` 的导出目录 |
| `--out` | 必填 | 输出目录 |
| `--epochs` | 3 | 可以是小数 |
| `--lr` | 1e-5，LoRA 时 2e-4 | |
| `--batch-size` | 8 | 显存不够就减小 |
| `--warmup` | 0.03 | 预热占总步数的比例 |
| `--train-vision` | 否 | 同时训练视觉编码器 |
| `--lora-rank` | 0 | 大于 0 时用这个秩的 LoRA；0 是全参数微调 |
| `--lora-alpha` | 秩的两倍 | 适配器的缩放是 alpha / 秩 |
| `--lora-dropout` | 0.05 | |
| `--eval-samples` | 200 | 训练完解码评测的验证样本数 |
| `--seed` / `--device` | 0 / `cuda` | |

脚本是单卡的，需要 torch、支持 Qwen3-VL 的 transformers 和 Pillow。输出目录就是标准的 Hugging Face checkpoint，`vllm serve` 可以直接加载；LoRA 训练的适配器已经合并进权重，输出也一样。另外有 `train_log.jsonl`（loss、学习率、验证 loss）、`held_out_decoding.json` 和记录训练参数的 `train_args.json`。`held_out_decoding.json` 评的是转回 bf16 之后、实际保存的权重。

全参数微调的显存：每个训练参数约 16 字节（fp32 权重 4、梯度 4、AdamW 两个状态 8）。Qwen3-VL-2B 的语言模型约 1.7B 参数，约 28 GB；加 `--train-vision` 后全部约 2.1B 参数，约 34 GB。激活另算，随 `--batch-size` 和图片数增长。24 GB 的卡（如 4090）放不下 2B 模型的全参数微调，要用 `--lora-rank`：冻结的基座按 bf16 算，每参数 2 字节（2B 约 4 GB），适配器很小。

### 用 ms-swift 或 LLaMA-Factory

导出的 `train.jsonl` / `val.jsonl` 是这两个框架都支持的 messages + images 格式，想用多卡或别的训练技巧时可以直接用它们。导出时加 `--absolute-paths`，图片路径就不依赖工作目录。下面的命令只作参考，参数名以你所用版本的文档为准：

```bash
# ms-swift
swift sft --model Qwen/Qwen3-VL-2B-Instruct --train_type lora --freeze_vit true \
  --dataset exports/planner_v1/train.jsonl --val_dataset exports/planner_v1/val.jsonl \
  --num_train_epochs 3 --output_dir outputs/planner_v1_lora
```

LLaMA-Factory 在 `dataset_info.json` 里登记为 sharegpt 格式：

```json
"planner_v1": {
  "file_name": "/abs/path/exports/planner_v1/train.jsonl",
  "formatting": "sharegpt",
  "columns": {"messages": "messages", "images": "images"},
  "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
           "assistant_tag": "assistant", "system_tag": "system"}
}
```

模板要选 Qwen3-VL 自己的模板。换了模板，训练时的 token 序列就和 vLLM 部署时不一样了。

torch 2.9 配 cuDNN 9.10 时，Qwen3-VL 视觉编码器第一层的 bf16 Conv3d 极慢：A100 上一个 batch 要 70 秒，换成等价的矩阵乘只要 0.3 毫秒。仓库里的脚本已经按矩阵乘计算，用别的框架时要么换 torch 版本，要么同样处理（vLLM 自己已经这样做了）。

## 3. 部署

```bash
vllm serve outputs/planner_v1 --served-model-name vlm-planner --port 8000 \
  --limit-mm-per-prompt '{"image": 3, "video": 0}'
```

`--limit-mm-per-prompt` 里的图片数不能少于规划器的相机数，否则请求会被拒绝。LoRA 权重这样部署：

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct --enable-lora --lora-modules vlm-planner=outputs/planner_v1_lora \
  --max-lora-rank 16 --port 8000 --limit-mm-per-prompt '{"image": 3, "video": 0}'
```

运行配置里的规划器部分：

```yaml
planner:
  type: openai_compat
  trained_library: my_library_v1        # 训练数据用的提示词库；和执行器的库不一致时拒绝启动
  params:
    base_url: http://127.0.0.1:8000/v1
    model: vlm-planner                   # --served-model-name 或 LoRA 名
    cameras: [head, wrist_left, wrist_right]   # 和训练数据一致：相机、顺序
    image_size: [448, 336]                     # 和训练数据一致
    proprio_keys: [gripper_width, gripper_closed]
    structured_output: response_format        # 老版本 vLLM 用 guided_json
```

`cameras`、`image_size`、`image_format`、`jpeg_quality`、`proprio_keys`、`proprio_digits`、`planner.memory_max_chars` 都会写进导出的 `manifest.json`。部署时任何一项不一样，模型看到的就不是它训练时见过的东西。

## 4. 离线评测

模型起来之后，先在留出的验证集上开环评测：每一步都从演示里的状态、记忆和已派发步骤历史出发，看模型的决策对不对。`show_history: true` 时，导出文件的 `samples.jsonl` 也保存这段 history，`data eval` 会恢复它。旧版导出缺少 history 时需重新导出，当前评测会明确拒绝。

```bash
embodied-harness data eval configs/my_task/run.yaml exports/planner_v1 --out exports/planner_v1/eval.json
```

| 指标 | 含义 |
| --- | --- |
| same call | 技能和参数都和标注一致的比例 |
| same skill | 技能一致（不管参数）的比例 |
| done recall | 该结束时说了 done 的比例 |
| premature done | 不该结束时说了 done 的次数 |
| memory similarity | 写出的记忆和标注记忆的相似度（1.0 为相同） |
| latency p50/p95 | 单次调用延迟 |

如果运行配置里的规划器设置和导出时不一样，命令会先打印警告；任务目标不比，每条样本按它记下的目标发送。导出早于 `system` 指纹时，规则、经验和技能改没改看不出来，命令也会提示重新导出。`--split train|val|all` 选数据，`--limit` 限制条数，最常见的错误组合会列在最后。样本有不止一种决策时，再按类型分开给出同调用数：`start`、`skill_done`、`periodic keep`（定期检查，应继续当前调用）、`periodic switch`（定期检查，应换到下一步）。periodic 导出里继续当前调用的样本占多数，总的同调用率会掩盖从不切换的模型，要看 `periodic switch` 那一项。

**开环分数高，不等于模型在看图。** 演示只有一种顺序、没有失败时，进度行（`show_progress`）和记忆就足以推出下一步，模型会走这条文字捷径。可以复制一份导出目录做反事实检查：把 `images/` 里的图换成黑图，或者把 `samples.jsonl` 里的 `ledger`、`proprio` 改成和画面矛盾，再跑一次 `data eval`。分数不变，说明模型没用这项输入；分数跟着改动走，说明它信的是这一项。

ALOHA 咖啡的实测（`configs/aloha_coffee/`，命令在 `run_planner.yaml` 开头）：lerobot-annotate + Codex 标注，32 条干净 episode（26 条训练、6 条验证），两步任务 `place_capsule` → `place_cup` → done，Qwen3-VL-2B 加 LoRA（秩 16，3 个 epoch），单张 A100 约 1 分钟训完。本体试了两种：只给两个夹爪，和给全部 14 维状态（两臂各 6 个关节 + 夹爪，现在的配置）。

| 验证集 42 个样本 | 同调用 | done 召回 | 记忆相似度 |
| --- | --- | --- | --- |
| 零样本 Qwen3-VL-4B，同样的 prompt（本体只有夹爪） | 18/42 | 0/6 | 0.10 |
| 微调，本体只有夹爪 | 42/42 | 6/6 | 0.85 |
| 微调，本体是全部状态 | 42/42 | 6/6 | 0.90 |
| 全部状态，图片全换成黑图 | 42/42 | 6/6 | 0.85 |
| 全部状态，进度改成「都没放」 | 18/42 | 0/6 | 0.56 |
| 全部状态，状态换成开局时的值 | 42/42 | 6/6 | 0.89 |

只有夹爪时，黑图和进度矛盾的结果和全部状态时一样（42/42、18/42）。零样本模型不理会进度行，几乎总选第一步，从不结束。微调教会了它按进度和记忆排步骤、按时结束、按标注的写法写记忆，但它的决策只跟着进度行走：图片和关节状态都没用上，进度错了也不会被它们纠正。这份数据里没有失败，画面、状态和进度从来不矛盾，模型没有理由去看后两者。要让它用图片（或状态）发现错误，训练数据里得有错误，比如失败和重试的演示、闭环运行里记录的失败。

## 5. 重放和闭环

训练了新模型以后，先在历史轨迹上离线对比它和旧模型的决策，再上机器人或仿真：

```bash
embodied-harness replay configs/my_task/run.yaml runs/<run_id>/ep_0001 runs/<run_id>/ep_0002 --out replay.json
embodied-harness run configs/my_task/run.yaml --episodes 20 --seed 0 --hold-policy abort
```

重放会原样重发记录的图片（planner/pc_XXXX/），并恢复新轨迹记录的动态 episode goal、history、账本和上一步结果；旧轨迹缺字段时回退到命令行配置。它仍是开环比较，不会推进环境或重放执行器，见 [重放与续跑限制](debugging.md#重放和续跑)。

## 不微调时

- **零样本**：把 `model` 换成基座模型（比如 Qwen3-VL-4B-Instruct），打开 `show_state: true` 和 `show_progress: true`，并把 `planner.memory_max_chars` 压到 10 左右（理由见下一条的实测）；`show_history`、`show_rejections` 可选。约束解码保证输出合法，但选哪一步全靠模型本身。在 RoboMME 组合任务上，4B 基座模型只看图和步骤历史时会原地打转、数错次数，见 [RoboMME](robomme.md#组合任务)。分层 VLA 的系统性研究（arXiv 2606.10267）也发现：原样堆历史基本没用，给结构化的文字状态才有用；开思考比换大模型更有用。
- **零样本时把记忆压短**：自由记忆会让模型坚持自己写下的计划，哪怕它和「当前状态」「进度」矛盾。实测（mock 分拣 4 个包裹，一路 448×336 合成画面，抓空、翻面掉落等随机故障，Qwen3-VL-4B-Instruct 零样本，每组 8 个 episode）：

  | prompt 设置 | 完成 |
  | --- | --- |
  | 只有图、本体、记忆（200 字） | 0/8（第一步后就原地重复 pick） |
  | `show_state` + `show_progress`，记忆 200 字 | 0/8 |
  | `show_history`，记忆 10 字 | 2/8 |
  | `show_state` + `show_progress`，记忆 10 字 | 7/8（唯一的失败是执行器连续抓空 3 次） |

  没有故障时，「`show_state` + `show_progress`，记忆 10 字」用最少的 9 步完成，只压短记忆、不给状态则一步都走不下去。所以不微调时建议打开 `show_state`、`show_progress`，并把 `planner.memory_max_chars` 调到 10 左右（约束解码会限制记忆长度）。`show_rejections` 在这组实验里没有明显作用：4B 模型被告知前置条件不满足后仍给出同一个回答。微调过的规划器按训练数据的设置来。
- **思考模型**：部署 Qwen3-VL 的 Thinking 版本，`vllm serve` 加 `--reasoning-parser`，把 `max_tokens` 调到 2048 左右、`request_timeout_s` 和 `planner.timeout_s` 相应调大。混合思考的模型用 `extra_body: {chat_template_kwargs: {enable_thinking: true}}` 打开。思考内容记在 `planner/pc_XXXX.json` 的 `meta.reasoning`。
- **经验**：多次运行后用 `embodied-harness lessons` 从反复出现的失败里起草经验，审核后用 `planner.lessons` 给规划器，见 [经验文件](configuration.md#经验文件lessons)。
- **RoboMME 式子目标预测器**：`robomme_subgoal` 后端复现了 RoboMME 微调的 Qwen3-VL 的 prompt，用它们发布的 LoRA 权重，训练用它们自己的代码。它看任务目标、子目标历史和一张前视图，每 16 步预测一次。

## 保持训练和部署一致

- 改了 `planner/render.py` 的输出格式，就要升 `RENDER_VERSION`，重新导出数据、重新训练。
- `data eval` 会对比 `manifest.json` 里的规划器设置和当前配置，不一致就警告。
- 运行配置写上 `planner.trained_manifest: <导出目录>`（相对运行配置）。每个 episode 开始时按同样的规则对比，每个不一致的设置在日志里记一条 WARNING（不阻止运行），比如训练时开着 `show_progress`、部署时关了，或者任务目标、规则、技能列表改过；`validate` 会提前打印同样的对比。目标、规则和技能只有导出记下了 `goal`、`system` 指纹才比得出来，更早的导出没有，这时会多一条警告，提示重新导出。导出不在这台机器上（或还没导出）时也只警告一条，不做对比。
- 运行配置写上 `planner.trained_library`。策略换了提示词库而规划器没有重训时，启动会报 `E_LIBRARY_MISMATCH`。
