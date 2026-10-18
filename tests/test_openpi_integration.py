"""OpenPI input mapping over LeRobot v3 metadata, without OpenPI or LeRobot installed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodied_harness.policy.integrations.openpi import (
    OpenPIDatasetSpec,
    load_openpi_spec,
    validate_lerobot_v3_dataset,
)


def metadata(root: Path, *, version: str = "v3.0") -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"placeholder")
    info = {
        "codebase_version": version,
        "fps": 10,
        "features": {
            "observation.images.top": {"dtype": "video", "shape": [48, 64, 3]},
            "observation.images.wrist": {"dtype": "image", "shape": [32, 32, 3]},
            "observation.state": {"dtype": "float32", "shape": [3], "names": ["x", "y", "gripper"]},
            "action": {"dtype": "float32", "shape": [3]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return root


def spec() -> OpenPIDatasetSpec:
    return OpenPIDatasetSpec(
        images={"images/cam_high": "observation.images.top", "images/wrist": "observation.images.wrist"},
        image_layout="hwc",
        image_input_layout="chw",
    )


def sample() -> dict:
    return {
        "observation.images.top": np.full((3, 2, 4), 0.5, dtype=np.float32),
        "observation.images.wrist": np.zeros((3, 2, 4), dtype=np.uint8),
        "observation.state": np.array([1, 2, 3], dtype=np.float32),
        "action": np.array([4, 5, 6], dtype=np.float32),
        "timestamp": np.float32(1.0),
        "task": "sort everything",
        "language_persistent": [
            {"style": "subtask", "timestamp": 0.0, "content": "pick the red cube"},
            {"style": "subtask", "timestamp": 1.0, "content": "place the red cube"},
        ],
    }


def test_v3_metadata_and_decoded_sample_map_to_openpi(tmp_path: Path):
    report = validate_lerobot_v3_dataset(metadata(tmp_path), spec())
    assert report.fps == 10
    assert report.state_dim == report.action_dim == 3
    assert set(report.image_features) == {"observation.images.top", "observation.images.wrist"}

    output = spec().repack(sample())
    assert set(output) == {"images", "state", "actions", "prompt"}
    assert set(output["images"]) == {"cam_high", "wrist"}
    assert output["images"]["cam_high"].shape == (2, 4, 3)
    assert output["images"]["cam_high"].dtype == np.uint8
    assert output["images"]["cam_high"][0, 0, 0] == 128
    assert output["prompt"] == "place the red cube"


def test_repack_inference_does_not_require_action():
    item = sample()
    del item["action"]
    output = spec().repack(item, training=False)
    assert "actions" not in output
    assert output["prompt"] == "place the red cube"


def test_prompt_falls_back_to_task_and_task_index_is_not_text():
    item = sample()
    item.pop("language_persistent")
    assert spec().prompt(item) == "sort everything"
    item.pop("task")
    with pytest.raises(ValueError, match="task_index alone"):
        spec().prompt({**item, "task_index": 0})


@pytest.mark.parametrize(
    "version, message",
    [("v2.1", "codebase_version"), ("v3.0", "feature 'action'")],
)
def test_metadata_rejects_old_or_incomplete_datasets(tmp_path: Path, version: str, message: str):
    root = metadata(tmp_path, version=version)
    if version == "v3.0":
        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        del info["features"]["action"]
        info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=message):
        validate_lerobot_v3_dataset(root, spec())


def test_paths_and_image_layout_are_validated():
    with pytest.raises(ValueError, match=r"assigned more than once|paths conflict"):
        OpenPIDatasetSpec(images={"images/cam": "observation.images.top"}, state_target="images")
    with pytest.raises(ValueError, match="image_layout"):
        OpenPIDatasetSpec(images={"images/cam": "observation.images.top"}, image_layout="hwc3")


def test_float_image_range_and_shape_are_checked():
    item = sample()
    item["observation.images.top"] = np.full((3, 2, 4), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        spec().repack(item)
    item["observation.images.top"] = np.zeros((2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="3 nonempty dimensions"):
        spec().repack(item)


@pytest.mark.parametrize(
    "key, feature, message",
    [
        ("observation.state", {"dtype": "float32", "shape": [2, 3]}, "one-dimensional"),
        ("action", {"dtype": "string", "shape": [3]}, "numeric vector"),
        ("action", {"dtype": "float32", "shape": [0]}, "positive integers"),
        ("action", {"dtype": "float32", "shape": [3], "names": ["x"]}, "names"),
        ("observation.images.top", {"dtype": "video", "shape": [48, 64, 4]}, "RGB"),
    ],
)
def test_metadata_rejects_invalid_feature_schema(tmp_path, key, feature, message):
    root = metadata(tmp_path)
    path = root / "meta" / "info.json"
    info = json.loads(path.read_text())
    info["features"][key] = feature
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match=message):
        validate_lerobot_v3_dataset(root, spec())


def test_yaml_loader_reads_one_mapping_and_rejects_unknown_fields(tmp_path):
    path = tmp_path / "mapping.yaml"
    path.write_text("images: {images/cam_high: observation.images.top}\nimage_layout: hwc\n")
    mapping = load_openpi_spec(path)
    assert mapping.images == {"images/cam_high": "observation.images.top"}
    assert mapping.image_layout == "hwc"
    path.write_text("images: {images/cam_high: observation.images.top}\nstate_keey: wrong\n")
    with pytest.raises(ValueError, match="state_keey"):
        load_openpi_spec(path)


def test_repacking_preserves_action_chunk_and_state_units():
    item = sample()
    item["action"] = np.arange(15, dtype=np.float32).reshape(5, 3) / 10
    result = spec().repack(item)
    assert result["actions"] is item["action"]
    assert result["state"] is item["observation.state"]
    assert item["observation.images.top"].dtype == np.float32
    assert item["task"] == "sort everything"
