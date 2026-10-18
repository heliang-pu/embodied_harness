"""Encodes camera images for the planner. Needs the `planner` extra (numpy and Pillow)."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

FORMATS = {"jpeg": ("JPEG", "jpg"), "png": ("PNG", "png")}


def file_extension(fmt: str) -> str:
    return FORMATS[fmt][1]


def encode_image(
    image: Any, *, size: Sequence[int] | None = None, fmt: str = "jpeg", quality: int = 90
) -> bytes:
    """Encode an HxWx3 array (uint8, or float in [0, 1]) or a PIL image; bytes pass through untouched.

    Bytes are images that were already encoded with the same settings, e.g. read back from a
    trace for a planner replay, so they are sent exactly as the original call sent them.
    """
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if fmt not in FORMATS:
        raise ValueError(f"image format must be one of {', '.join(FORMATS)}, got {fmt!r}")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "encoding camera images needs numpy and Pillow: pip install 'embodied-harness[planner]'"
        ) from exc

    if not isinstance(image, Image.Image):
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255).round().astype(np.uint8)
        if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        image = Image.fromarray(array)
    image = image.convert("RGB")
    if size is not None and image.size != (int(size[0]), int(size[1])):
        image = image.resize((int(size[0]), int(size[1])), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    pil_format = FORMATS[fmt][0]
    if pil_format == "JPEG":
        image.save(buffer, format="JPEG", quality=quality)
    else:
        image.save(buffer, format=pil_format)
    return buffer.getvalue()


def data_url(data: bytes, fmt: str = "jpeg") -> str:
    return f"data:image/{fmt};base64,{base64.b64encode(data).decode('ascii')}"
