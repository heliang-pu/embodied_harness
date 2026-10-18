"""The data boundary between LeRobot v3.0 datasets and an OpenPI training configuration.

This module imports neither OpenPI nor LeRobot. It validates the dataset's declared features and
repacks a *decoded, unbatched* LeRobot sample into the keys a particular OpenPI data transform
expects. Video decoding, action-chunk sampling, robot-specific transforms and normalization remain
with the training process. In particular, repacking does not make an older OpenPI/LeRobot loader
understand the v3.0 storage format.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .language import SUBTASK_STYLE, active_text

_NUMERIC_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)


def _nonempty_key(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _shape(feature: Mapping[str, Any], key: str) -> tuple[int, ...]:
    shape = feature.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or not shape
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise ValueError(f"feature {key!r} must declare a nonempty shape of positive integers")
    return tuple(shape)


def _vector(feature: Mapping[str, Any], key: str) -> int:
    shape = _shape(feature, key)
    if len(shape) != 1 or feature.get("dtype") not in _NUMERIC_DTYPES:
        raise ValueError(f"feature {key!r} must be a one-dimensional numeric vector")
    names = feature.get("names")
    if isinstance(names, list) and len(names) != shape[0]:
        raise ValueError(f"feature {key!r}: {len(names)} names for {shape[0]} vector elements")
    return shape[0]


def _stamp(value: Any) -> float:
    # LeRobot yields scalar torch tensors; numpy and Arrow scalars work here too, without imports.
    if hasattr(value, "item"):
        value = value.item()
    elif hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("a sample with language_persistent needs a scalar numeric timestamp")
    stamp = float(value)
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    return stamp


def _put_path(result: dict[str, Any], path: str, value: Any) -> None:
    """Put a value at a slash-delimited OpenPI key (e.g. ``images/cam_high``)."""
    parts = path.split("/")
    if any(not part for part in parts):
        raise ValueError(f"OpenPI target {path!r} has an empty path component")
    current = result
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"OpenPI target paths conflict at {part!r}")
        current = existing
    if parts[-1] in current:
        raise ValueError(f"OpenPI target path {path!r} is assigned more than once")
    current[parts[-1]] = value


def _image_uint8(image: Any, input_layout: str, output_layout: str) -> Any:
    """Convert decoded CHW floats or websocket HWC bytes to a configured RGB layout."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenPI image repacking needs numpy; install embodied-harness[policy]") from exc
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError(f"image sample must have 3 nonempty dimensions, got shape {array.shape}")
    if input_layout == "auto":
        if array.shape[0] == 3 and array.shape[-1] != 3:
            input_layout = "chw"
        elif array.shape[-1] == 3 and array.shape[0] != 3:
            input_layout = "hwc"
        else:
            raise ValueError(f"cannot infer image layout from shape {array.shape}; set image_input_layout")
    if input_layout == "chw" and array.shape[0] != 3:
        raise ValueError(f"CHW image has invalid channel dimension in shape {array.shape}")
    if input_layout == "hwc" and array.shape[-1] != 3:
        raise ValueError(f"HWC image has invalid channel dimension in shape {array.shape}")
    if array.dtype.kind == "f":
        if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError("floating-point images must contain finite values in [0, 1]")
        array = np.rint(array * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        raise ValueError("images must be uint8 bytes or floating-point values in [0, 1]")
    if input_layout != output_layout:
        array = np.transpose(array, (2, 0, 1) if output_layout == "chw" else (1, 2, 0))
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class OpenPIDatasetSpec:
    """Explicit dataset-to-model key mapping, suitable as an OpenPI callable repack transform.

    ``images`` maps OpenPI slash paths, such as ``images/cam_high``, to literal LeRobot feature keys,
    such as ``observation.images.top``. Images become RGB uint8 in ``image_layout``; state/action
    objects and their units are preserved. The training config must put this transform before its
    robot/model transforms, and must use a LeRobot v3-capable reader. Action semantics and joint order
    are never inferred from a dataset name.
    """

    images: Mapping[str, str]
    state_key: str = "observation.state"
    action_key: str = "action"
    task_key: str = "task"
    state_target: str = "state"
    actions_target: str = "actions"
    prompt_target: str = "prompt"
    subtask_style: str | None = SUBTASK_STYLE
    image_layout: str = "chw"
    image_input_layout: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.images, Mapping) or not self.images:
            raise ValueError("images must map at least one OpenPI input key to a LeRobot feature")
        for target, source in self.images.items():
            _nonempty_key(target, "image target")
            _nonempty_key(source, "image feature")
        for field in (
            "state_key",
            "action_key",
            "task_key",
            "state_target",
            "actions_target",
            "prompt_target",
        ):
            _nonempty_key(getattr(self, field), field)
        if self.image_layout not in {"chw", "hwc"}:
            raise ValueError("image_layout must be 'chw' or 'hwc'")
        if self.image_input_layout not in {"auto", "chw", "hwc"}:
            raise ValueError("image_input_layout must be 'auto', 'chw' or 'hwc'")
        targets = [*self.images, self.state_target, self.actions_target, self.prompt_target]
        if len(set(targets)) != len(targets):
            raise ValueError("OpenPI target keys must be distinct")
        structure: dict[str, Any] = {}
        for target in targets:
            _put_path(structure, target, object())
        if self.subtask_style is not None:
            _nonempty_key(self.subtask_style, "subtask_style")
        object.__setattr__(self, "images", dict(self.images))

    def prompt(self, sample: Mapping[str, Any]) -> str:
        """Use the active annotation, falling back to the decoded episode task."""
        rows = sample.get("language_persistent")
        if self.subtask_style is not None and rows is not None:
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise ValueError("language_persistent must be a sequence of annotation rows")
            if any(not isinstance(row, Mapping) for row in rows):
                raise ValueError("language_persistent must contain annotation mappings")
            if rows:
                text = active_text(rows, _stamp(sample.get("timestamp")), self.subtask_style)
                if text:
                    return text
        task = sample.get(self.task_key)
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                f"sample has no nonempty {self.task_key!r} string or active subtask; use a decoded "
                "LeRobotDataset sample (task_index alone is not an instruction)"
            )
        return task

    def repack(self, sample: Mapping[str, Any], *, training: bool = True) -> dict[str, Any]:
        """Map one decoded sample to OpenPI's nested keys without changing robot units."""
        keys = [*self.images.values(), self.state_key]
        if training:
            keys.append(self.action_key)
        missing = list(dict.fromkeys(key for key in keys if key not in sample))
        if missing:
            raise ValueError(f"decoded LeRobot sample is missing feature(s): {', '.join(missing)}")
        result: dict[str, Any] = {}
        for target, source in self.images.items():
            _put_path(
                result, target, _image_uint8(sample[source], self.image_input_layout, self.image_layout)
            )
        _put_path(result, self.state_target, sample[self.state_key])
        if training:
            _put_path(result, self.actions_target, sample[self.action_key])
        _put_path(result, self.prompt_target, self.prompt(sample))
        return result

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return self.repack(sample)


def load_openpi_spec(path: str | Path) -> OpenPIDatasetSpec:
    """Load a small YAML key mapping without importing OpenPI or LeRobot.

    The document root is the spec itself, for example ``images: {images/cam_high:
    observation.images.top}``. An optional ``openpi:`` wrapper is accepted for consistency with run
    bundles. Unknown fields are rejected so a typo cannot silently change a training input.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("loading an OpenPI spec needs PyYAML; install embodied-harness") from exc
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"OpenPI spec not found: {source}") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: OpenPI spec must be a YAML mapping")
    if "openpi" in value:
        wrapper = value
        value = wrapper["openpi"]
        extra = set(wrapper) - {"openpi"}
        if extra:
            raise ValueError(f"{source}: unknown OpenPI wrapper field(s): {', '.join(map(str, extra))}")
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: openpi must be a YAML mapping")
    fields = {
        "images",
        "state_key",
        "action_key",
        "task_key",
        "state_target",
        "actions_target",
        "prompt_target",
        "subtask_style",
        "image_layout",
        "image_input_layout",
    }
    unknown = sorted(str(key) for key in set(value) - fields)
    if unknown:
        raise ValueError(f"{source}: unknown OpenPI spec field(s): {', '.join(map(str, unknown))}")
    if "images" not in value or not isinstance(value["images"], Mapping):
        raise ValueError(f"{source}: images must be a mapping of OpenPI paths to LeRobot feature keys")
    return OpenPIDatasetSpec(**dict(value))


@dataclass(frozen=True)
class LeRobotV3Metadata:
    """Validated declarations, not a guarantee that video decoding or model training will succeed."""

    root: Path
    fps: float
    state_dim: int
    action_dim: int
    image_features: Mapping[str, Mapping[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "codebase_version": "v3.0",
            "fps": self.fps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "image_features": dict(self.image_features),
        }


def validate_lerobot_v3_dataset(root: Path, spec: OpenPIDatasetSpec) -> LeRobotV3Metadata:
    """Validate v3.0 metadata and the state/action/image features required by ``spec``.

    This reads ``meta/info.json`` only; it does not decode media or import model dependencies. The
    dataset remains untouched. A v2 dataset must be converted with LeRobot's own migration tooling,
    rather than merely changing its version string.
    """
    root = Path(root).resolve()
    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{info_path} not found; expected a LeRobot v3.0 dataset") from None
    if not isinstance(info, dict):
        raise ValueError(f"{info_path}: metadata must be a JSON object")
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"{info_path}: expected codebase_version 'v3.0', got {info.get('codebase_version')!r}; "
            "use LeRobot's dataset migration tooling for older formats"
        )
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{info_path}: fps must be a finite positive number")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{info_path}: features must be an object")

    def feature(key: str) -> Mapping[str, Any]:
        value = features.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"{info_path}: missing or invalid feature {key!r}")
        return value

    state_dim = _vector(feature(spec.state_key), spec.state_key)
    action_dim = _vector(feature(spec.action_key), spec.action_key)
    image_features = {}
    for key in spec.images.values():
        image = feature(key)
        shape = _shape(image, key)
        if image.get("dtype") not in {"image", "video"} or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"feature {key!r} must be an RGB image/video with HxWx3 metadata shape")
        image_features[key] = image
    return LeRobotV3Metadata(root, float(fps), state_dim, action_dim, image_features)
