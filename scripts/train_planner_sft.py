"""Supervised fine-tuning of the planner VLM (Qwen3-VL) on an `embodied-harness data planner` export.

Every record holds the system prompt, the user turn with `<image>` placeholders, the target JSON
and the image paths. The user turn is rebuilt as the same list of text and image parts that
vLLM receives from the harness, so the chat template produces the same tokens in training and
at serving time. The loss covers only the assistant turn (the JSON and its end token).

    python scripts/train_planner_sft.py --model Qwen/Qwen3-VL-2B-Instruct \
        --data exports/planner_v1 --out outputs/planner_v1

By default the language model is fine-tuned in full, with fp32 master weights. With `--lora-rank`
only low-rank fp32 adapters on its projection layers train over the frozen bf16 weights, and they
are merged into the weights before saving. On CUDA the forward and backward compute in bf16 under
autocast. The output directory is a regular bf16 Hugging Face checkpoint that `vllm serve` loads
directly. Needs torch, transformers (with Qwen3-VL) and Pillow.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

IMAGE_PLACEHOLDER = "<image>"
ASSISTANT_START = "<|im_start|>assistant\n"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(torch.nn.Module):
    """A frozen linear layer plus a trainable low-rank update, kept in fp32 and merged after training."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = torch.nn.Dropout(dropout)
        device = base.weight.device
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, device=device))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=device))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x).to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + (self.scale * update).to(x.dtype)

    @torch.no_grad()
    def merge(self) -> torch.nn.Linear:
        weight = self.base.weight
        weight.copy_((weight.float() + self.scale * (self.lora_b @ self.lora_a)).to(weight.dtype))
        return self.base


def add_lora(model: Any, rank: int, alpha: float, dropout: float) -> int:
    """Freeze the model and wrap the language model's projection layers; returns how many."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in LORA_TARGETS
        and ".visual." not in f".{name}."
    ]
    for name in names:
        parent, _, child = name.rpartition(".")
        owner = model.get_submodule(parent)
        setattr(owner, child, LoRALinear(getattr(owner, child), rank, alpha, dropout))
    return len(names)


def merge_lora(model: Any) -> None:
    """Fold every adapter into its weight, leaving the base model's layout and parameter names."""
    names = [name for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    for name in names:
        parent, _, child = name.rpartition(".")
        owner = model.get_submodule(parent)
        setattr(owner, child, getattr(owner, child).merge())


def full_fine_tuning(model: Any, train_vision: bool) -> list[torch.nn.Parameter]:
    """Freeze the vision encoder unless it trains, and keep fp32 master weights for what trains.

    The model loads in bf16. An AdamW step at lr 1e-5 moves a weight by about 1e-5, less than half
    a bf16 ulp once |w| >= 2^-8 (most language model weights, every RMSNorm weight), so bf16 weights
    would round the updates away. Frozen weights stay bf16, and `autocast` still computes in bf16.
    """
    if not train_vision:
        for name, parameter in model.named_parameters():
            if ".visual." in f".{name}.":
                parameter.requires_grad = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    cast_parameters(trainable, torch.float32)
    return trainable


def cast_parameters(parameters: Iterable[torch.nn.Parameter], dtype: torch.dtype) -> None:
    """Change the parameters' dtype in place; tied weights stay tied.

    Buffers such as the RoPE frequencies keep fp32, which `model.to(dtype)` would round as well.
    """
    for parameter in parameters:
        parameter.data = parameter.data.to(dtype)


def autocast(device: str) -> contextlib.AbstractContextManager[Any]:
    """Compute in bf16 on CUDA; fp32 weights are cast per operation and get fp32 gradients.

    On other devices the model computes in the dtypes of its weights.
    """
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def conv_patches_as_matmul(model: Any) -> int:
    """Run patch-embedding Conv3d layers (kernel equal to stride) as the matrix product they are.

    Qwen3-VL embeds image patches with such a Conv3d. cuDNN 9.10 (torch 2.9) runs it in bf16 about
    10^5 times slower than the matmul, a minute per training step. Parameter names do not change.
    """
    count = 0
    for module in model.modules():
        if (
            isinstance(module, torch.nn.Conv3d)
            and tuple(module.kernel_size) == tuple(module.stride)
            and tuple(module.padding) == (0, 0, 0)
            and tuple(module.dilation) == (1, 1, 1)
            and module.groups == 1
        ):
            module.forward = _patch_matmul(module)
            count += 1
    return count


def _patch_matmul(conv: torch.nn.Conv3d) -> Any:
    convolve = conv.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[2:]) != tuple(conv.kernel_size):  # more than one patch per input
            return convolve(x)
        out = torch.nn.functional.linear(x.flatten(1), conv.weight.flatten(1), conv.bias)
        return out.view(x.shape[0], conv.out_channels, 1, 1, 1)

    return forward


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def chat(record: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Messages as the harness sends them to vLLM, and the target text."""
    system, user, assistant = record["messages"]
    content: list[dict[str, Any]] = []
    pieces = user["content"].split(IMAGE_PLACEHOLDER)
    for index, text in enumerate(pieces):
        if text:
            content.append({"type": "text", "text": text})
        if index < len(pieces) - 1:
            content.append({"type": "image"})
    messages = [{"role": "system", "content": system["content"]}, {"role": "user", "content": content}]
    return messages, assistant["content"]


class Batcher:
    def __init__(self, processor: Any, root: Path) -> None:
        self.processor = processor
        self.root = root
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "right"
        self.assistant_ids = tokenizer.encode(ASSISTANT_START, add_special_tokens=False)
        self.end = "<|im_end|>\n"

    def images(self, record: dict[str, Any]) -> list[Image.Image]:
        return [Image.open(self.root / path).convert("RGB") for path in record["images"]]

    def prompt(self, record: dict[str, Any]) -> str:
        messages, _ = chat(record)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [self.prompt(record) + chat(record)[1] + self.end for record in records]
        images = [image for record in records for image in self.images(record)]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        width = len(self.assistant_ids)
        for row in range(labels.shape[0]):
            ids = batch["input_ids"][row].tolist()
            starts = [i for i in range(len(ids) - width + 1) if ids[i : i + width] == self.assistant_ids]
            if not starts:
                raise ValueError("assistant turn not found in a training sample")
            labels[row, : starts[-1] + width] = -100
        batch["labels"] = labels
        return batch


@torch.no_grad()
def evaluate_loss(
    model: Any, batcher: Batcher, records: list[dict[str, Any]], batch_size: int, device: str
) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(records), batch_size):
        batch = {k: v.to(device) for k, v in batcher(records[start : start + batch_size]).items()}
        with autocast(device):
            loss = model(**batch).loss
        total += float(loss) * len(records[start : start + batch_size])
        count += len(records[start : start + batch_size])
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def evaluate_decisions(
    model: Any, batcher: Batcher, records: list[dict[str, Any]], device: str
) -> dict[str, Any]:
    """Greedy decoding on held-out samples: does the JSON parse and name the same call and memory?"""
    model.eval()
    processor = batcher.processor
    same_call = same_memory = parsed = 0
    for record in records:
        inputs = processor(
            text=[batcher.prompt(record)], images=[batcher.images(record)], return_tensors="pt"
        ).to(device)
        with autocast(device):
            output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        target = json.loads(chat(record)[1])
        try:
            predicted = json.loads(text)
            if not isinstance(predicted, dict):
                continue
            parsed += 1
        except json.JSONDecodeError:
            continue
        same_call += (
            predicted.get("done", False) == target.get("done", False)
            and predicted.get("skill") == target.get("skill")
            and (predicted.get("args") or {}) == (target.get("args") or {})
        )
        same_memory += predicted.get("memory") == target.get("memory")
    model.train()
    n = max(len(records), 1)
    return {
        "samples": len(records),
        "parsed": parsed / n,
        "same_call": same_call / n,
        "same_memory": same_memory / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="directory written by `embodied-harness data planner`"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=None, help="default 1e-5, or 2e-4 with --lora-rank")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=float, default=0.03, help="fraction of steps")
    parser.add_argument("--train-vision", action="store_true", help="also train the vision encoder")
    parser.add_argument(
        "--lora-rank", type=int, default=0, help="train LoRA adapters of this rank (0: full fine-tuning)"
    )
    parser.add_argument("--lora-alpha", type=float, default=None, help="default twice the rank")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-samples", type=int, default=200, help="held-out samples decoded at the end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.lora_rank and args.train_vision:
        parser.error("--train-vision trains the vision encoder in full; it does not combine with --lora-rank")
    if args.lr is None:
        args.lr = 2e-4 if args.lora_rank else 1e-5
    if args.lora_alpha is None:
        args.lora_alpha = 2.0 * args.lora_rank

    from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_records(args.data / "train.jsonl")
    val = load_records(args.data / "val.jsonl")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(args.device)
    conv_patches_as_matmul(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if args.lora_rank:
        wrapped = add_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
        print(f"LoRA rank {args.lora_rank}, alpha {args.lora_alpha:g} on {wrapped} projection layers")
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = full_fine_tuning(model, args.train_vision)
    millions = sum(p.numel() for p in trainable) / 1e6
    print(f"train {len(train)} samples, val {len(val)}; trainable {millions:.1f}M params, lr {args.lr:g}")

    batcher = Batcher(processor, args.data)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup), total_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "train_log.jsonl").open("a", encoding="utf-8")
    model.train()
    step, started = 0, time.time()
    while step < total_steps:
        order = list(range(len(train)))
        random.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            if step >= total_steps:
                break
            batch = {
                k: v.to(args.device)
                for k, v in batcher([train[i] for i in order[start : start + args.batch_size]]).items()
            }
            with autocast(args.device):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                entry = {
                    "step": step,
                    "loss": round(float(loss), 4),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(time.time() - started),
                }
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
            if val and (step % steps_per_epoch == 0 or step == total_steps):
                entry = {
                    "step": step,
                    "epoch": round(step / steps_per_epoch, 2),
                    "val_loss": round(evaluate_loss(model, batcher, val, args.batch_size, args.device), 4),
                }
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
    model.config.use_cache = True
    if args.lora_rank:
        merge_lora(model)
    # The fp32 master weights go back to the base checkpoint's bf16, the dtype vLLM serves; the
    # held-out decoding below scores these rounded weights.
    cast_parameters(model.parameters(), torch.bfloat16)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    (args.out / "train_args.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )
    if val:
        sample = random.Random(args.seed).sample(val, min(args.eval_samples, len(val)))
        result = evaluate_decisions(model, batcher, sample, args.device)
        print(json.dumps({"held_out_decoding": result}), flush=True)
        (args.out / "held_out_decoding.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    log.close()


if __name__ == "__main__":
    main()
