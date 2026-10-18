"""Training a LeRobot policy on the harness's dataset: subtask prompts from `language_persistent`."""

import json

import pytest

from embodied_harness.cli import main
from embodied_harness.policy.integrations.checkpoint import STEP_CLASS, add_subtask_step, prepare_checkpoint
from embodied_harness.policy.integrations.language import active_text, subtask_prompts


def row(style, t, content):
    return {"role": "assistant", "content": content, "style": style, "timestamp": t, "camera": None}


ROWS = [
    row("subtask", 0.0, "pick up the red cube"),
    row("subtask", 1.2, "place the red cube onto the target"),
    row("memory", 1.2, "I picked up the red cube."),
    row("subtask", 2.5, "press the button to stop"),
]


def test_the_active_subtask_is_the_latest_one_started():
    assert active_text(ROWS, 0.0) == "pick up the red cube"
    assert active_text(ROWS, 1.19) == "pick up the red cube"
    assert active_text(ROWS, 1.2) == "place the red cube onto the target"
    assert active_text(ROWS, 1.1999999) == "place the red cube onto the target"  # float32 timestamps
    assert active_text(ROWS, 9.0) == "press the button to stop"
    assert active_text(ROWS, 1.3, style="memory") == "I picked up the red cube."
    assert active_text([row("subtask", 1.0, "later")], 0.5) is None
    assert active_text([row("subtask", 1.0, "a"), row("subtask", 1.0, "b")], 1.0) == "b"


def test_samples_without_a_subtask_keep_their_task():
    prompts = subtask_prompts(["goal A", "goal B"], [ROWS, []], [1.5, 1.5], "subtask")
    assert prompts == ["place the red cube onto the target", "goal B"]


def pi05_pipeline():
    return {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
            {"registry_name": "normalizer_processor", "config": {}},
            {"registry_name": "pi05_prepare_state_tokenizer_processor_step", "config": {}},
            {"registry_name": "tokenizer_processor", "config": {"task_key": "task"}},
            {"registry_name": "device_processor", "config": {"device": "cuda"}},
        ],
    }


def test_the_step_goes_before_the_prompt_step():
    pipeline = pi05_pipeline()
    assert add_subtask_step(pipeline)
    names = [step.get("registry_name") or step.get("class") for step in pipeline["steps"]]
    assert names.index(STEP_CLASS) == names.index("pi05_prepare_state_tokenizer_processor_step") - 1
    assert not add_subtask_step(pipeline)  # already there
    pi0 = {"steps": [{"registry_name": "normalizer_processor"}, {"registry_name": "tokenizer_processor"}]}
    add_subtask_step(pi0)
    assert pi0["steps"][1]["class"] == STEP_CLASS
    with pytest.raises(ValueError, match="no prompt step"):
        add_subtask_step({"steps": [{"registry_name": "normalizer_processor"}]})


def test_prepared_checkpoints_link_the_weights(tmp_path, capsys):
    source = tmp_path / "pi05_base"
    (source / "tokenizer").mkdir(parents=True)
    (source / "config.json").write_text('{"type": "pi05"}')
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer" / "tokenizer.json").write_text("{}")
    (source / "policy_preprocessor.json").write_text(json.dumps(pi05_pipeline()))
    out = tmp_path / "pi05_base_subtask"
    assert main(["policy", "subtask-checkpoint", str(source), "--out", str(out)]) == 0
    assert (out / "model.safetensors").is_symlink()
    assert (out / "tokenizer" / "tokenizer.json").is_file() and not (out / "tokenizer").is_symlink()
    steps = json.loads((out / "policy_preprocessor.json").read_text())["steps"]
    assert steps[2] == {"class": STEP_CLASS, "config": {"style": "subtask"}}
    assert (
        json.loads((source / "policy_preprocessor.json").read_text()) == pi05_pipeline()
    )  # source untouched
    assert "lerobot-train --policy.path=" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        prepare_checkpoint(source, out)


def test_preparing_a_checkpoint_preserves_an_existing_subtask_step(tmp_path):
    pipeline = pi05_pipeline()
    pipeline["steps"].insert(2, {"class": STEP_CLASS, "config": {"style": "custom"}})
    source = tmp_path / "source_checkpoint"
    source.mkdir()
    (source / "policy_preprocessor.json").write_text(json.dumps(pipeline))
    out = tmp_path / "prepared_checkpoint"

    prepare_checkpoint(source, out)

    prepared = json.loads((out / "policy_preprocessor.json").read_text())
    assert len(prepared["steps"]) == len(pipeline["steps"])
    assert prepared["steps"][2] == {"class": STEP_CLASS, "config": {"style": "custom"}}
    assert json.loads((source / "policy_preprocessor.json").read_text()) == pipeline


def test_the_lerobot_step_rewrites_task_during_training_only():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lerobot.processor")
    from lerobot.processor import TransitionKey

    from embodied_harness.policy.integrations.lerobot import ActiveSubtaskAsTaskStep

    step = ActiveSubtaskAsTaskStep()
    batch = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["episode goal", "episode goal"],
            "timestamp": torch.tensor([0.5, 2.6]),
            "language_persistent": [ROWS, ROWS],
        }
    }
    assert step(batch)[TransitionKey.COMPLEMENTARY_DATA]["task"] == [
        "pick up the red cube",
        "press the button to stop",
    ]
    single = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "goal",
            "timestamp": torch.tensor(1.5),
            "language_persistent": ROWS,
        }
    }
    assert step(single)[TransitionKey.COMPLEMENTARY_DATA]["task"] == "place the red cube onto the target"
    inference = {TransitionKey.COMPLEMENTARY_DATA: {"task": "place the red cube onto the target"}}
    assert step(inference) is inference  # no language rows at inference: the harness's prompt stays
    assert step.get_config() == {"style": "subtask"}
