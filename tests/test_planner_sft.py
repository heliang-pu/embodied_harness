"""Held-out SFT scoring uses CPU stubs, without loading a model or processor checkpoint."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import warnings
from pathlib import Path

import pytest


@pytest.fixture
def sft():
    pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_planner_sft.py"
    spec = importlib.util.spec_from_file_location("train_planner_sft", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(sft, raw, target):
    """Run the real scorer with one deterministic generated response."""

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Processor:
        tokenizer = None

        def __init__(self):
            self.tokenizer = self

        def __call__(self, **kwargs):
            return Inputs(input_ids=sft.torch.tensor([[1, 2]]))

        def decode(self, tokens, **kwargs):
            return raw

    class Batcher:
        processor = Processor()

        def prompt(self, record):
            return "rendered prompt"

        def images(self, record):
            return []

    class Model:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def generate(self, **kwargs):
            assert not self.training
            return sft.torch.tensor([[1, 2, 3]])

    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "images": [],
    }
    model = Model()
    result = sft.evaluate_decisions(model, Batcher(), [record], "cpu")
    assert model.training  # scoring restores training mode
    return result


def test_no_argument_calls_accept_an_explicit_empty_object(sft):
    target = {"skill": "press_button", "memory": "ready"}
    result = score(sft, json.dumps({**target, "args": {}}), target)
    assert result == {"samples": 1, "parsed": 1.0, "same_call": 1.0, "same_memory": 1.0}


@pytest.mark.parametrize("raw", ["[]", "null", '"pick"', "not JSON"])
def test_invalid_decisions_are_scored_without_crashing(sft, raw):
    result = score(sft, raw, {"skill": "pick", "memory": ""})
    assert result == {"samples": 1, "parsed": 0.0, "same_call": 0.0, "same_memory": 0.0}


def test_same_skill_with_wrong_arguments_is_not_the_same_call(sft):
    target = {"skill": "pick", "args": {"color": "red"}, "memory": "ready"}
    result = score(sft, json.dumps({**target, "args": {"color": "blue"}}), target)
    assert result == {"samples": 1, "parsed": 1.0, "same_call": 0.0, "same_memory": 1.0}


def test_lora_trains_only_adapters_and_merges_into_the_base_layout(sft):
    torch = sft.torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4)
            self.q_proj = torch.nn.Linear(4, 6)
            self.down_proj = torch.nn.Linear(6, 4, bias=False)

        def forward(self, x):
            return self.down_proj(self.q_proj(self.norm(x)))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Module()
            self.visual.q_proj = torch.nn.Linear(4, 4)
            self.language = Block()

        def forward(self, x):
            return self.language(x)

    torch.manual_seed(0)
    model = Model()
    names = set(model.state_dict())
    x = torch.randn(3, 4)
    before = model(x)
    assert sft.add_lora(model, rank=2, alpha=4.0, dropout=0.0) == 2  # the vision tower is left alone
    assert torch.allclose(model(x), before)  # B starts at zero
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        f"language.{layer}.lora_{part}" for layer in ("q_proj", "down_proj") for part in ("a", "b")
    }
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_b"):
                parameter.normal_()
    tuned = model(x)
    assert not torch.allclose(tuned, before)
    sft.merge_lora(model)
    assert set(model.state_dict()) == names
    assert torch.allclose(model(x), tuned, atol=1e-5)


def test_patch_convolutions_run_as_a_matrix_product(sft):
    torch = sft.torch
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv3d(3, 8, kernel_size=(2, 4, 4), stride=(2, 4, 4)), torch.nn.Conv3d(8, 8, 3)
    )
    names = set(model.state_dict())
    patches, video = torch.randn(5, 3, 2, 4, 4), torch.randn(1, 3, 4, 8, 8)
    expected = model[0](patches), model[0](video)
    assert sft.conv_patches_as_matmul(model) == 1  # a 3x3x3 convolution is not a patch embedding
    assert torch.allclose(model[0](patches), expected[0], atol=1e-5)
    assert torch.allclose(model[0](video), expected[1], atol=1e-5)  # several patches: plain convolution
    assert set(model.state_dict()) == names


@pytest.mark.parametrize("train_vision", [False, True])
def test_full_fine_tuning_steps_fp32_master_weights_and_saves_bf16(sft, train_vision):
    torch = sft.torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(4, 4)
            self.embed = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4, bias=False)
            self.norm = torch.nn.Parameter(torch.ones(4))  # an RMSNorm weight
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.lm_head.weight = self.embed.weight  # tied, as in Qwen3-VL-2B
            self.register_buffer("inv_freq", torch.linspace(0.1, 1.0, 4), persistent=False)

        def forward(self, ids, pixels):  # vision features are cast to the text dtype, as in Qwen3-VL
            image = self.visual(pixels.to(self.visual.weight.dtype)).to(self.embed.weight.dtype)
            return self.lm_head(self.norm * self.proj(self.embed(ids) + image))

    torch.manual_seed(0)
    model = Model()
    sft.cast_parameters(model.parameters(), torch.bfloat16)  # as from_pretrained(dtype=bfloat16) loads it
    names = set(model.state_dict())
    trainable = sft.full_fine_tuning(model, train_vision)
    dtypes = {name: parameter.dtype for name, parameter in model.named_parameters()}
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert frozen == (set() if train_vision else {"visual.weight", "visual.bias"})
    assert len(trainable) == len(dtypes) - len(frozen)
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    assert all(dtypes[name] == torch.bfloat16 for name in frozen)

    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.0)  # the full fine-tuning default
    model(torch.tensor([[1, 2, 3]]), torch.randn(1, 3, 4)).square().mean().backward()
    optimizer.step()
    for parameter, old in zip(trainable, before, strict=True):
        assert (parameter != old).all()  # every weight moved by about 1e-5, the RMSNorm's 1.0 included
    # The script used to step the bf16 weights themselves, which rounds such an update away.
    bf16 = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    bf16.grad = torch.ones_like(bf16)
    torch.optim.AdamW([bf16], lr=1e-5, weight_decay=0.0).step()
    assert torch.equal(bf16, torch.ones_like(bf16))

    sft.cast_parameters(model.parameters(), torch.bfloat16)  # before saving: a bf16 checkpoint for vLLM
    assert {parameter.dtype for parameter in model.parameters()} == {torch.bfloat16}
    assert model.inv_freq.dtype == torch.float32  # buffers keep their precision
    assert model.lm_head.weight is model.embed.weight
    assert set(model.state_dict()) == names


def test_autocast_computes_in_bf16_on_cuda_only(sft):
    torch = sft.torch
    assert isinstance(sft.autocast("cpu"), contextlib.nullcontext)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # without CUDA the context disables itself; it is only inspected
        context = sft.autocast("cuda:0")
    assert isinstance(context, torch.autocast)
    assert context.fast_dtype == torch.bfloat16
