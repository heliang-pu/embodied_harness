"""Small synchronous client and validators for the openpi websocket protocol.

This module deliberately contains no model or LeRobot imports.  It is the boundary between a
runtime executor and a policy process: metadata is received once per connection and each request
is a msgpack map containing numpy arrays.  Keeping this transport independent lets openpi, a
LeRobot checkpoint server, and a test double share the same adapter.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


class OpenPIProtocolError(RuntimeError):
    """The policy server returned a malformed response or an inference error."""


def _mapping_field(mapping: dict[Any, Any], name: str) -> Any:
    if name in mapping:
        return mapping[name]
    return mapping.get(name.encode("ascii"))


def validate_metadata(metadata: Any) -> dict[str, Any]:
    """Validate the optional handshake contract and return the original map.

    ``chunk_size`` and ``action_dim`` are optional for compatibility with older openpi servers,
    but when advertised they must be positive integers.  Other metadata is intentionally opaque.
    """

    if not isinstance(metadata, dict):
        raise OpenPIProtocolError("metadata handshake must be a map")
    for name in ("chunk_size", "action_dim"):
        value = _mapping_field(metadata, name)
        if value is None:
            continue
        if isinstance(value, bool):
            raise OpenPIProtocolError(f"metadata.{name} must be a positive integer, got {value!r}")
        if not isinstance(value, (int, np.integer)):
            raise OpenPIProtocolError(f"metadata.{name} must be a positive integer, got {value!r}")
        if value < 1:
            raise OpenPIProtocolError(f"metadata.{name} must be a positive integer, got {value!r}")
    return metadata


def set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    """Set a value using an openpi-style slash path, preserving flat keys for compatibility.

    ``set_path(payload, "observation/image", frame)`` creates nested dictionaries.  Existing
    harness configurations use keys such as ``observation.images.front`` and remain flat because
    only ``/`` has path semantics.
    """

    if not path:
        raise ValueError("request key must not be empty")
    parts = path.split("/")
    if any(not part for part in parts):
        raise ValueError(f"request key has an empty path component: {path!r}")
    current = mapping
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        elif not isinstance(existing, dict):
            raise ValueError(f"request key {path!r} conflicts with an existing value at {part!r}")
        current = existing
    if parts[-1] in current:
        raise ValueError(f"request key {path!r} is assigned more than once")
    current[parts[-1]] = value


def validate_actions(
    reply: dict[str, Any],
    *,
    key: str = "actions",
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Validate and normalize a policy action chunk returned by openpi.

    Openpi action chunks are finite, two-dimensional numeric arrays (time, action dimension).
    A one-dimensional response is rejected rather than silently interpreted as one timestep; this
    catches a common server/client shape mismatch before an action reaches a robot.
    """

    if not isinstance(reply, dict):
        raise OpenPIProtocolError(f"reply must be a map, got {type(reply).__name__}")
    if key not in reply:
        raise OpenPIProtocolError(f"reply has no {key!r}; keys: {sorted(reply)}")
    try:
        raw = np.asarray(reply[key])
    except (TypeError, ValueError) as exc:
        raise OpenPIProtocolError(f"reply field {key!r} is not an array: {exc}") from exc
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise OpenPIProtocolError(f"reply field {key!r} must have shape [time, action_dim], got {raw.shape}")
    if raw.dtype.kind not in "uif":
        raise OpenPIProtocolError(f"reply field {key!r} has non-numeric dtype {raw.dtype}")
    actions = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(actions).all():
        raise OpenPIProtocolError(f"reply field {key!r} contains NaN or Inf")
    expected = _mapping_field(validate_metadata(metadata or {}), "action_dim")
    if expected is not None:
        try:
            expected = int(expected)
        except (TypeError, ValueError) as exc:
            raise OpenPIProtocolError(f"metadata.action_dim must be an integer, got {expected!r}") from exc
        if expected < 1 or actions.shape[1] != expected:
            raise OpenPIProtocolError(
                f"reply field {key!r} has action_dim {actions.shape[1]}, expected {expected}"
            )
    return actions


class OpenPIClient:
    """A reconnectable, blocking openpi websocket client.

    Imports ``websockets`` lazily so importing the harness/configuration does not require policy
    dependencies.  ``connect`` receives and validates the metadata handshake; ``call`` returns a
    decoded map and surfaces text-frame tracebacks as :class:`OpenPIProtocolError`.
    """

    def __init__(self, url: str, *, connect_timeout_s: float = 120.0, request_timeout_s: float = 30.0):
        if not math.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be finite and positive")
        if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive")
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self.request_timeout_s = request_timeout_s
        self.metadata: dict[str, Any] = {}
        self._ws: Any = None
        self._context: Any = None

    def connect(self) -> dict[str, Any]:
        from websockets.sync.client import connect

        deadline = time.monotonic() + self.connect_timeout_s
        while True:
            context = None
            try:
                try:
                    context = connect(
                        self.url,
                        compression=None,
                        max_size=None,
                        open_timeout=min(10.0, self.request_timeout_s),
                        ping_interval=None,
                        proxy=None,
                    )
                except TypeError:
                    # websockets < 15: the threading client has no proxy support and no keepalive pings,
                    # and passes unknown keyword arguments to socket.create_connection, which rejects them.
                    context = connect(
                        self.url,
                        compression=None,
                        max_size=None,
                        open_timeout=min(10.0, self.request_timeout_s),
                    )
                ws = context.__enter__()
                from . import msgpack_numpy

                handshake = ws.recv(timeout=self.request_timeout_s)
                if isinstance(handshake, str):
                    raise OpenPIProtocolError(f"policy server handshake error:\n{handshake}")
                try:
                    metadata = msgpack_numpy.unpackb(handshake)
                except Exception as exc:
                    raise OpenPIProtocolError(f"invalid metadata handshake: {exc}") from exc
                metadata = validate_metadata(metadata)
                self._context, self._ws, self.metadata = context, ws, metadata
                return metadata
            except (OSError, TimeoutError) as exc:
                if context is not None:
                    context.__exit__(type(exc), exc, exc.__traceback__)
                if time.monotonic() >= deadline:
                    raise OpenPIProtocolError(f"policy server {self.url} is not reachable: {exc}") from exc
                time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))
            except Exception:
                if context is not None:
                    context.__exit__(None, None, None)
                raise

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        from websockets.exceptions import ConnectionClosed

        if self._ws is None:
            self.connect()
        from . import msgpack_numpy

        payload = msgpack_numpy.packb(request)
        for attempt in (1, 2):
            try:
                self._ws.send(payload)
                reply = self._ws.recv(timeout=self.request_timeout_s)
                if isinstance(reply, str):
                    self.close()
                    raise OpenPIProtocolError(f"policy server error:\n{reply}")
                try:
                    decoded = msgpack_numpy.unpackb(reply)
                except Exception as exc:
                    self.close()
                    raise OpenPIProtocolError(f"invalid policy reply: {exc}") from exc
                if not isinstance(decoded, dict):
                    self.close()
                    raise OpenPIProtocolError("policy reply must be a map")
                return decoded
            except ConnectionClosed:
                if attempt == 2:
                    raise
                self.close()
                self.connect()
            except (TimeoutError, OSError) as exc:
                # A timed-out request may still be in flight on the server.  Drop this socket so
                # its late response cannot be mistaken for the next request's response.
                self.close()
                raise OpenPIProtocolError(f"policy request to {self.url} timed out or failed: {exc}") from exc
        raise AssertionError("unreachable")

    def close(self) -> None:
        context, ws = self._context, self._ws
        self._context = self._ws = None
        if ws is not None:
            ws.close()
        if context is not None:
            context.__exit__(None, None, None)
