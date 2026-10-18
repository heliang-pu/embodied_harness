"""The msgpack codec used by the openpi websocket protocol.

The wire representation intentionally mirrors ``openpi_client.msgpack_numpy``.  In particular,
array metadata uses byte keys (the original openpi implementation does this) while ordinary
application keys remain strings.  ``unpackb`` accepts both byte and string metadata keys so it can
also talk to clients built with a newer msgpack configuration.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np

# Requests and replies carry images, state vectors and action chunks, so the decoder only builds bool,
# integer and floating-point arrays.  Every other dtype is refused: an object dtype would turn the sender's
# bytes into object pointers, and structured or subarray dtypes can embed objects.  The decoder is
# deliberately narrower than ``pack_array``, which refuses only what openpi_client's encoder refuses.
_WIRE_KINDS = frozenset("biuf")


def pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: dict[Any, Any]) -> Any:
    # openpi's first codec used bytes keys; accepting str keys makes the decoder tolerant of
    # msgpack clients that encode those keys as raw strings instead of ``bin`` values.
    marker = _field(obj, "__ndarray__")
    if marker is not None:
        try:
            data = _field(obj, "data")
            dtype = _wire_dtype(_field(obj, "dtype"))
            shape = _field(obj, "shape")
            if not isinstance(data, (bytes, bytearray, memoryview)):
                raise TypeError("array data is not bytes")
            return np.ndarray(buffer=data, dtype=dtype, shape=shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid ndarray payload: {exc}") from exc
    if _field(obj, "__npgeneric__") is not None:
        try:
            return _wire_dtype(_field(obj, "dtype")).type(_field(obj, "data"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid numpy scalar payload: {exc}") from exc
    return obj


def _wire_dtype(spec: Any) -> np.dtype:
    dtype = np.dtype(spec)
    if dtype.kind not in _WIRE_KINDS:
        raise ValueError(f"unsupported dtype: {dtype}")
    return dtype


def _field(obj: dict[Any, Any], name: str) -> Any:
    """Read a codec field whose key may be encoded as str or bytes."""

    if name in obj:
        return obj[name]
    return obj.get(name.encode("ascii"))


# Set these explicitly instead of relying on msgpack's version-dependent defaults.  ``use_bin_type``
# preserves ndarray bytes while ``raw=False`` decodes regular text fields as Python strings.
Packer = functools.partial(msgpack.Packer, default=pack_array, use_bin_type=True)
packb = functools.partial(msgpack.packb, default=pack_array, use_bin_type=True)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array, raw=False)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array, raw=False)
