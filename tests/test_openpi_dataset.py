"""The LeRobot v3/OpenPI boundary uses episode-aware chunk sampling and preserves prompts."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from embodied_harness.cli import main
from embodied_harness.data.openpi_dataset import OpenPILeRobotDataset
from embodied_harness.policy.integrations.openpi import OpenPIDatasetSpec


def dataset_root(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 20,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    "action": {"dtype": "float32", "shape": [2]},
                    "observation.images.top": {"dtype": "video", "shape": [4, 6, 3]},
                },
            }
        )
    )
    return tmp_path


def fake_lerobot(monkeypatch, sample):
    calls = []

    class Dataset:
        def __init__(self, repo_id, **kwargs):
            calls.append((repo_id, kwargs))

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return sample

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = Dataset
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
    return calls


def sample():
    return {
        "observation.state": np.array([0.2, 0.5], np.float32),
        "observation.images.top": np.ones((3, 4, 6), np.float32),
        "action": np.array([[1, 2], [3, 4], [3, 4]], np.float32),
        "task": "sort the parcels",
        "timestamp": 1.0,
        "language_persistent": [
            {"style": "subtask", "timestamp": 0, "content": "pick"},
            {"style": "subtask", "timestamp": 1, "content": "place"},
        ],
    }


def test_v3_loader_requests_action_horizon_and_keeps_subtask(tmp_path, monkeypatch):
    calls = fake_lerobot(monkeypatch, sample())
    dataset = OpenPILeRobotDataset(
        dataset_root(tmp_path),
        OpenPIDatasetSpec(images={"images/cam_high": "observation.images.top"}),
        action_horizon=3,
        repo_id="local/test",
        episodes=[2],
        video_backend="pyav",
    )
    assert len(dataset) == 1
    assert calls[0][0] == "local/test"
    assert calls[0][1]["delta_timestamps"] == {"action": [0, 0.05, 0.1]}
    assert calls[0][1]["episodes"] == [2]
    assert calls[0][1]["video_backend"] == "pyav"
    item = dataset[0]
    assert item["prompt"] == "place"
    assert item["images"]["cam_high"].dtype == np.uint8
    np.testing.assert_array_equal(item["actions"], sample()["action"])


def test_loader_rejects_invalid_action_shape(tmp_path, monkeypatch):
    item = sample()
    item["action"] = np.zeros((2, 2), np.float32)
    fake_lerobot(monkeypatch, item)
    dataset = OpenPILeRobotDataset(
        dataset_root(tmp_path),
        OpenPIDatasetSpec(images={"images/cam_high": "observation.images.top"}),
        action_horizon=3,
    )
    with pytest.raises(ValueError, match="action chunk shape"):
        dataset[0]


def test_openpi_check_cli_metadata_and_decoded_sample(tmp_path, monkeypatch, capsys):
    root = dataset_root(tmp_path)
    mapping = root / "mapping.yaml"
    mapping.write_text("images: {images/cam_high: observation.images.top}\n")
    command = ["policy", "openpi-check", str(root), "--mapping", str(mapping)]
    assert main(command) == 0
    assert "metadata only" in capsys.readouterr().out
    fake_lerobot(monkeypatch, sample())
    assert main([*command, "--sample", "0", "--action-horizon", "3"]) == 0
    assert "decoded sample 0" in capsys.readouterr().out
    assert main([*command, "--sample", "2"]) == 2
    assert "--sample must be" in capsys.readouterr().err
