"""Synchronous client for the XPolicyLab WebSocket protocol.

XPolicyLab and the harness both use WebSocket + MessagePack, but their wire
contracts are deliberately different.  OpenPI sends one metadata map at
connect time and then flat observation maps.  XPolicyLab sends request/response
frames (``hello``, ``reset``, ``infer``) with an ``observation`` payload and
explicit evaluation/trial identifiers.  This module keeps that difference at
the transport boundary so the runtime executor can support either server.

The codec below mirrors :mod:`msgpack-numpy`, which is the codec used by the
official XPolicyLab client.  It is kept local so importing the harness does not
make the optional XPolicyLab package a runtime dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any
from uuid import uuid4

import numpy as np

# As in the openpi codec (msgpack_numpy.py), only bool, integer and floating-point arrays are decoded.  An
# object or structured dtype would turn the sender's bytes into object pointers.
_WIRE_KINDS = frozenset("biuf")


class XPolicyProtocolError(RuntimeError):
    """The XPolicyLab server returned an invalid frame or an RPC error."""


def validate_actions(reply: Mapping[str, Any], *, key: str = "actions") -> np.ndarray | list[dict[str, Any]]:
    """Validate the standard XPolicyLab action chunk.

    XPolicyLab models commonly return a list of action dictionaries (for example,
    ``left_arm_joint_state`` plus ``left_ee_joint_state``), while a few adapters
    return a numeric ``[time, action_dim]`` array.  Preserve either representation
    so the environment adapter, which owns robot-specific action ordering, can
    perform the final conversion.
    """

    if not isinstance(reply, Mapping) or key not in reply:
        raise XPolicyProtocolError(f"reply has no {key!r}")
    raw = reply[key]
    if isinstance(raw, np.ndarray):
        actions = np.asarray(raw)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
            raise XPolicyProtocolError(f"reply field {key!r} must have shape [time, action_dim]")
        if actions.dtype.kind not in "uif" or not np.isfinite(actions).all():
            raise XPolicyProtocolError(f"reply field {key!r} must contain finite numeric values")
        return np.asarray(actions, dtype=np.float32)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or not raw:
        raise XPolicyProtocolError(f"reply field {key!r} must be an action sequence")
    if all(isinstance(item, Mapping) for item in raw):
        result: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            action = dict(item)
            if not action:
                raise XPolicyProtocolError(f"reply action {index} is empty")
            for name, value in action.items():
                array = np.asarray(value)
                if array.ndim != 1 or array.size < 1 or array.dtype.kind not in "uif":
                    raise XPolicyProtocolError(f"reply action {index}.{name} must be a numeric vector")
                if not np.isfinite(array).all():
                    raise XPolicyProtocolError(f"reply action {index}.{name} contains NaN or Inf")
                action[name] = np.asarray(array, dtype=np.float32)
            result.append(action)
        return result
    raise XPolicyProtocolError(f"reply field {key!r} must be numeric or a list of action maps")


def _encode_numpy(obj: Any) -> Any:
    """Encode numpy values in the ``msgpack-numpy`` map representation."""

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in {"O", "V"}:
            raise TypeError(f"unsupported numpy dtype for XPolicyLab: {obj.dtype}")
        return {
            b"nd": True,
            b"type": obj.dtype.str,
            b"kind": b"",
            b"shape": obj.shape,
            b"data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        if obj.dtype.kind in {"O", "V"}:
            raise TypeError(f"unsupported numpy dtype for XPolicyLab: {obj.dtype}")
        return {b"nd": False, b"type": obj.dtype.str, b"data": obj.tobytes()}
    return obj


def _field(mapping: Mapping[Any, Any], name: str) -> Any:
    if name in mapping:
        return mapping[name]
    return mapping.get(name.encode("ascii"))


def _decode_numpy(obj: dict[Any, Any]) -> Any:
    marker = _field(obj, "nd")
    if marker is None:
        return obj
    data = _field(obj, "data")
    dtype = _field(obj, "type")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("XPolicyLab ndarray data is not bytes")
    try:
        dtype = np.dtype(dtype)
        if dtype.kind not in _WIRE_KINDS:
            raise ValueError(f"unsupported dtype: {dtype}")
        if marker is True:
            shape = tuple(_field(obj, "shape") or ())
            return np.ndarray(buffer=data, dtype=dtype, shape=shape)
        return np.frombuffer(data, dtype=dtype)[0]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid XPolicyLab ndarray payload: {exc}") from exc


def packb(value: Any) -> bytes:
    """Encode one XPolicyLab frame."""

    # Imported here, like websockets: the CLI loads this module and must start without the `policy` extra.
    import msgpack

    return msgpack.packb(value, default=_encode_numpy, use_bin_type=True)


def unpackb(value: bytes) -> Any:
    """Decode one XPolicyLab frame, accepting both byte and string metadata keys."""

    import msgpack

    return msgpack.unpackb(value, object_hook=_decode_numpy, raw=False, strict_map_key=False)


def _as_map(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise XPolicyProtocolError(f"{label} must be a map, got {type(value).__name__}")
    return value


class XPolicyClient:
    """Blocking client for an XPolicyLab policy server.

    ``connect`` performs the protocol ``hello`` exchange.  ``reset`` clears
    model state for a new episode, and ``infer`` atomically sends one standard
    observation and returns the server's ``payload`` map (normally containing
    ``actions``).  Observation/action schema conversion remains the caller's
    responsibility, as required by XPolicyLab.
    """

    def __init__(
        self,
        url: str,
        *,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        connect_timeout_s: float = 120.0,
        request_timeout_s: float = 120.0,
    ) -> None:
        for name, value in (
            ("connect_timeout_s", connect_timeout_s),
            ("request_timeout_s", request_timeout_s),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.url = url
        self.evaluation_id = evaluation_id or f"embodied-harness-{uuid4().hex}"
        self.trial_id = trial_id
        self.connect_timeout_s = connect_timeout_s
        self.request_timeout_s = request_timeout_s
        self.metadata: dict[str, Any] = {}
        self._context: Any = None
        self._ws: Any = None
        self._step = 0

    def connect(self) -> dict[str, Any]:
        from websockets.sync.client import connect

        try:
            try:
                context = connect(
                    self.url,
                    compression=None,
                    max_size=None,
                    open_timeout=self.connect_timeout_s,
                    proxy=None,
                )
            except TypeError:  # websockets < 15 has no proxy argument
                context = connect(
                    self.url,
                    compression=None,
                    max_size=None,
                    open_timeout=self.connect_timeout_s,
                )
            ws = context.__enter__()
            self._context, self._ws = context, ws
            reply = self._request("hello", "hello_ack", {"client": "embodied_harness"})
            self.metadata = dict(reply)
            return self.metadata
        except (OSError, TimeoutError) as exc:
            self.close()
            raise XPolicyProtocolError(f"policy server {self.url} is not reachable: {exc}") from exc
        except Exception:
            self.close()
            raise

    def reset(self) -> dict[str, Any]:
        if self._ws is None:
            self.connect()
        # The official client carries the trial identifier in the reset
        # payload as well as the frame envelope.  Keeping it there matters
        # for servers that use the payload to initialise per-trial state.
        if self.trial_id is None:
            self.trial_id = f"trial-{uuid4().hex}"
        payload = {"trial_id": self.trial_id} if self.trial_id is not None else {}
        reply = self._request("reset", "reset_result", payload)
        self._step = 0
        return reply

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            self.connect()
        if not isinstance(observation, Mapping):
            raise XPolicyProtocolError("infer observation must be a map")
        reply = self._request("infer", "infer_result", {"observation": dict(observation)})
        self._step += 1
        return reply

    def close(self) -> None:
        context, ws = self._context, self._ws
        self._context = self._ws = None
        if ws is not None:
            with suppress(Exception):
                ws.close()
        if context is not None:
            with suppress(Exception):
                context.__exit__(None, None, None)

    def _request(self, method: str, expected: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            raise XPolicyProtocolError("policy server is not connected")
        from websockets.exceptions import ConnectionClosed

        request_id = uuid4().hex
        frame = {
            "message_type": method,
            "message_id": request_id,
            "evaluation_id": self.evaluation_id,
            "trial_id": self.trial_id,
            "step": self._step,
            "payload": payload,
        }
        try:
            self._ws.send(packb(frame))
            raw = self._ws.recv(timeout=self.request_timeout_s)
        except (ConnectionClosed, TimeoutError, OSError) as exc:
            self.close()
            raise XPolicyProtocolError(f"policy request {method!r} timed out or failed: {exc}") from exc
        if isinstance(raw, str):
            self.close()
            raise XPolicyProtocolError(f"policy server error: {raw}")
        try:
            reply = _as_map(unpackb(raw), "policy reply")
        except Exception as exc:
            self.close()
            if isinstance(exc, XPolicyProtocolError):
                raise
            raise XPolicyProtocolError(f"invalid policy reply: {exc}") from exc
        if reply.get("message_id") != request_id:
            raise XPolicyProtocolError(
                f"policy reply message_id {reply.get('message_id')!r} does not match {request_id!r}"
            )
        if reply.get("evaluation_id") != self.evaluation_id:
            raise XPolicyProtocolError("policy reply evaluation_id does not match this client")
        message_type = reply.get("message_type")
        if message_type == "error":
            error = _as_map(reply.get("payload") or {}, "policy error payload")
            raise XPolicyProtocolError(str(error.get("message") or error.get("code") or "policy error"))
        if message_type != expected:
            raise XPolicyProtocolError(f"expected {expected!r}, got {message_type!r}")
        return _as_map(reply.get("payload") or {}, "policy reply payload")
