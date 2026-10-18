"""OpenPI checkpoint serving adapter tests without OpenPI or JAX installed."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from embodied_harness.policy.integrations.openpi import OpenPIDatasetSpec
from embodied_harness.policy.vla.openpi import OpenPIPolicy, main


class FakePolicy:
    def __init__(self) -> None:
        self.seen: list[dict] = []
        self.reset_count = 0

    def metadata(self):
        return {"upstream": "fake", "chunk_size": 99}

    def infer(self, inputs):
        self.seen.append(inputs)
        return {"actions": np.zeros((2, 7), dtype=np.float32), "upstream_field": "kept"}

    def reset(self):
        self.reset_count += 1


def make_spec() -> OpenPIDatasetSpec:
    return OpenPIDatasetSpec(
        images={"images/cam_high": "observation.images.top"},
        state_key="observation.state",
        image_layout="chw",
    )


def wire_observation(*, prompt: str | None = "pick") -> dict:
    result = {
        "observation.images.top": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "observation.state": np.array([1.0, 2.0], dtype=np.float32),
    }
    if prompt is not None:
        result["prompt"] = prompt
    return result


def test_inference_reuses_mapping_and_keeps_policy_output():
    fake = FakePolicy()
    server = OpenPIPolicy(
        "pi05",
        "/tmp/checkpoint",
        mapping=make_spec(),
        policy=fake,
    )

    reply = server.infer(wire_observation())

    assert reply["upstream_field"] == "kept"
    assert reply["actions"].shape == (2, 7)
    native = fake.seen[0]
    assert set(native) == {"images", "state", "prompt"}
    assert native["images"]["cam_high"].shape == (3, 2, 4)
    assert native["images"]["cam_high"].dtype == np.uint8
    np.testing.assert_array_equal(native["state"], [1.0, 2.0])
    assert native["prompt"] == "pick"


def test_inference_uses_wire_prompt_without_training_annotations():
    fake = FakePolicy()
    server = OpenPIPolicy("pi05", "/tmp/checkpoint", mapping=make_spec(), policy=fake)
    obs = wire_observation()
    obs.update(
        task="episode goal",
        language_persistent=[{"style": "subtask", "timestamp": 0.0, "content": "old subtask"}],
    )

    server.infer(obs)

    assert fake.seen[0]["prompt"] == "pick"
    assert server.spec.task_key == "task"
    assert server.spec.image_input_layout == "auto"


def test_default_prompt_is_used_when_wire_prompt_is_absent():
    fake = FakePolicy()
    server = OpenPIPolicy(
        "pi05",
        "/tmp/checkpoint",
        mapping=make_spec(),
        default_prompt="sort the parcel",
        policy=fake,
    )

    server.infer(wire_observation(prompt=None))

    assert fake.seen[0]["prompt"] == "sort the parcel"


def test_metadata_preserves_upstream_and_does_not_report_padded_action_dim():
    fake = FakePolicy()
    server = OpenPIPolicy("pi05", "/tmp/checkpoint", mapping=make_spec(), policy=fake)
    metadata = server.metadata()

    assert metadata["upstream"] == "fake"
    assert metadata["chunk_size"] == 99
    assert metadata["config"] == "pi05"
    assert metadata["checkpoint"] == "/tmp/checkpoint"
    assert metadata["image_keys"] == ["observation.images.top"]
    assert metadata["state_key"] == "observation.state"
    assert "action_dim" not in metadata

    fake.metadata = {"action_dim": 7}
    assert server.metadata()["action_dim"] == 7


def test_invalid_observation_fails_before_inference():
    fake = FakePolicy()
    server = OpenPIPolicy("pi05", "/tmp/checkpoint", mapping=make_spec(), policy=fake)
    with pytest.raises(ValueError, match="prompt"):
        server.infer(wire_observation(prompt=None))
    with pytest.raises(ValueError, match="missing feature"):
        server.infer({"prompt": "pick"})
    assert not fake.seen


def test_reset_delegates_to_policy():
    fake = FakePolicy()
    server = OpenPIPolicy("pi05", "/tmp/checkpoint", mapping=make_spec(), policy=fake)
    server.reset()
    assert fake.reset_count == 1


def test_mapping_can_be_loaded_from_yaml(tmp_path: Path):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "images": {"images/cam_high": "observation.images.top"},
                "state_key": "observation.state",
                "image_layout": "chw",
            }
        ),
        encoding="utf-8",
    )
    fake = FakePolicy()
    server = OpenPIPolicy("pi05", "/tmp/checkpoint", mapping=mapping, policy=fake)
    server.infer(wire_observation())
    assert fake.seen[0]["prompt"] == "pick"


def test_serving_preserves_shared_model_input_targets():
    fake = FakePolicy()
    spec = OpenPIDatasetSpec(
        images={"images/cam_high": "observation.images.top"},
        state_target="robot_state",
        prompt_target="language",
        actions_target="robot_actions",
        image_layout="chw",
    )
    server = OpenPIPolicy("pi05", "/tmp/checkpoint", mapping=spec, policy=fake)
    server.infer(wire_observation())

    assert set(fake.seen[0]) == {"images", "robot_state", "language"}
    assert fake.seen[0]["language"] == "pick"


def test_openpi_is_imported_lazily(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []
    config_module = types.ModuleType("openpi.training.config")
    policy_module = types.ModuleType("openpi.policies.policy_config")

    def get_config(name):
        calls.append(("config", name))
        return types.SimpleNamespace(model=types.SimpleNamespace(action_horizon=16, action_dim=32))

    def create_trained_policy(config, checkpoint, default_prompt=None):
        calls.append(("policy", config, checkpoint, default_prompt))
        fake = FakePolicy()
        fake.metadata = {"upstream": "kept"}
        return fake

    config_module.get_config = get_config
    policy_module.create_trained_policy = create_trained_policy
    monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
    monkeypatch.setitem(sys.modules, "openpi.training", types.ModuleType("openpi.training"))
    monkeypatch.setitem(sys.modules, "openpi.training.config", config_module)
    monkeypatch.setitem(sys.modules, "openpi.policies", types.ModuleType("openpi.policies"))
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", policy_module)

    server = OpenPIPolicy(
        "pi05",
        "/tmp/checkpoint",
        mapping=make_spec(),
        default_prompt="default",
    )

    assert calls[0] == ("config", "pi05")
    assert calls[1][0:3] == ("policy", server.train_config, "/tmp/checkpoint")
    assert calls[1][3] == "default"
    assert server.metadata()["chunk_size"] == 16
    assert "action_dim" not in server.metadata()


def test_cli_passes_openpi_policy_to_existing_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    class FakeServerPolicy:
        def __init__(self, config, checkpoint, *, mapping, default_prompt):
            captured["policy"] = (config, checkpoint, mapping, default_prompt)

        def metadata(self):
            return {"fake": True}

    def serve(policy, *, host, port, metadata, max_size):
        captured["serve"] = (policy, host, port, metadata, max_size)

    monkeypatch.setattr("embodied_harness.policy.vla.openpi.OpenPIPolicy", FakeServerPolicy)
    monkeypatch.setattr("embodied_harness.policy.serving.websocket_server.serve", serve)
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(yaml.safe_dump({"images": {"images/cam": "observation.images.top"}}), encoding="utf-8")

    main(
        [
            "--config",
            "pi05",
            "--checkpoint",
            "/tmp/ckpt",
            "--mapping",
            str(mapping),
            "--default-prompt",
            "pick",
            "--host",
            "127.0.0.1",
            "--port",
            "8123",
        ]
    )

    assert captured["policy"][:2] == ("pi05", "/tmp/ckpt")
    assert captured["policy"][3] == "pick"
    assert captured["serve"][1:3] == ("127.0.0.1", 8123)
    assert captured["serve"][4] == 64 * 2**20  # requests are capped by default

    required = ["--config", "pi05", "--checkpoint", "/tmp/ckpt", "--mapping", str(mapping)]
    main([*required, "--max-request-mb", "8"])
    assert captured["serve"][4] == 8 * 2**20
    with pytest.raises(SystemExit):
        main([*required, "--max-request-mb", "0"])
