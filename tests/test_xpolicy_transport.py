"""XPolicyLab WebSocket frames stay separate from the existing OpenPI wire."""

import inspect
import socket

import msgpack
import numpy as np
import pytest

from conftest import RUN_CONFIG
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import Observation, Prompt, SkillCall
from embodied_harness.policy.executor.websocket_policy import WebsocketPolicyExecutor
from embodied_harness.policy.transport import xpolicy

LIBRARY = load_bundle(RUN_CONFIG).library
PROMPT = Prompt("pick_next", "pick up the next package")


class _Context:
    def __init__(self, socket):
        self.socket = socket

    def __enter__(self):
        return self.socket

    def __exit__(self, *_):
        self.socket.closed = True


class _XPolicySocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, data):
        self.sent.append(xpolicy.unpackb(data))

    def recv(self, timeout):
        request = self.sent[-1]
        method = request["message_type"]
        if method == "hello":
            payload = {"ok": True, "server": "test-xpolicylab", "chunk_size": 3, "action_dim": 8}
            response_type = "hello_ack"
        elif method == "reset":
            payload = {"ok": True}
            response_type = "reset_result"
        else:
            payload = {"actions": np.arange(24, dtype=np.float32).reshape(3, 8)}
            response_type = "infer_result"
        return xpolicy.packb(
            {
                "message_type": response_type,
                "message_id": request["message_id"],
                "evaluation_id": request["evaluation_id"],
                "trial_id": request.get("trial_id"),
                "step": request.get("step", 0),
                "payload": payload,
            }
        )

    def close(self):
        self.closed = True


def test_xpolicy_codec_matches_msgpack_numpy_shape_and_dtype():
    value = {"frame": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)}
    decoded = xpolicy.unpackb(xpolicy.packb(value))
    np.testing.assert_array_equal(decoded["frame"], value["frame"])
    assert decoded["frame"].dtype == np.uint8


@pytest.mark.parametrize(
    "dtype",
    [
        "|O",
        "(2,)O",
        {"names": ["a", "b"], "formats": ["<f4", "|O"], "offsets": [0, 8], "itemsize": 16},
        "|V8",
        "<c8",
        "<U2",
    ],
    ids=["object", "object-subarray", "struct-with-object", "void", "complex", "str"],
)
def test_xpolicy_codec_refuses_dtypes_other_than_bool_integer_and_float(dtype):
    # A reply's object array would turn the server's bytes into object pointers in the harness.
    for marker in (True, False):
        frame = msgpack.packb(
            {"x": {b"nd": marker, b"type": dtype, b"kind": b"", b"shape": [2], b"data": bytes(64)}},
            use_bin_type=True,
        )
        with pytest.raises(ValueError, match="unsupported dtype"):
            xpolicy.unpackb(frame)


def test_xpolicy_client_connects_through_the_websockets_14_threading_api(monkeypatch):
    # Before websockets 15 the threading connect() had no proxy or ping_interval parameter and passed its
    # other keyword arguments on to socket.create_connection, which rejects unknown ones.
    import websockets.sync.client

    xsocket = _XPolicySocket()

    def connect_14(
        uri, *, compression="deflate", open_timeout=10, close_timeout=10, max_size=2**20, **kwargs
    ):
        inspect.signature(socket.create_connection).bind(("test", 80), timeout=open_timeout, **kwargs)
        return _Context(xsocket)

    monkeypatch.setattr(websockets.sync.client, "connect", connect_14)
    client = xpolicy.XPolicyClient("ws://test", evaluation_id="eval-1")
    assert client.connect()["server"] == "test-xpolicylab"
    client.close()


def test_xpolicy_action_maps_are_preserved_for_environment_adapters():
    reply = {
        "actions": [
            {
                "left_arm_joint_state": np.zeros(6),
                "left_ee_joint_state": np.zeros(1),
                "right_arm_joint_state": np.ones(6),
                "right_ee_joint_state": np.ones(1),
            }
        ]
    }
    actions = xpolicy.validate_actions(reply)
    assert isinstance(actions, list) and actions[0]["right_arm_joint_state"].shape == (6,)


def test_xpolicy_client_uses_hello_reset_and_infer_frames(monkeypatch):
    import websockets.sync.client

    socket = _XPolicySocket()
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *args, **kwargs: _Context(socket))

    client = xpolicy.XPolicyClient("ws://test", evaluation_id="eval-1", trial_id="trial-2")
    assert client.connect()["server"] == "test-xpolicylab"
    client.reset()
    result = client.infer({"vision": {"cam_head": {"color": np.zeros((2, 2, 3), np.uint8)}}})
    assert result["actions"].shape == (3, 8)
    assert [frame["message_type"] for frame in socket.sent] == ["hello", "reset", "infer"]
    assert socket.sent[-1]["payload"]["observation"]["vision"]["cam_head"]["color"].shape == (2, 2, 3)
    assert all(frame["evaluation_id"] == "eval-1" for frame in socket.sent)
    client.close()
    assert socket.closed


def test_websocket_executor_can_target_xpolicy_without_changing_openpi_mapping(monkeypatch):
    import websockets.sync.client

    socket = _XPolicySocket()
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *args, **kwargs: _Context(socket))
    client = WebsocketPolicyExecutor(
        LIBRARY,
        url="ws://test",
        protocol="xpolicy",
        evaluation_id="eval-1",
        images={"front": "vision/cam_head/color"},
        state={"proprio": "state", "key": "state/left_arm_joint_states"},
        texts={"task": "{prompt}"},
        nested_keys=True,
        n_action_steps=2,
    )
    client.reset()
    client.set_instruction(SkillCall("pick"), PROMPT, version=5)
    chunk = client.infer_chunk(
        Observation(
            images={"front": np.zeros((2, 2, 3), np.uint8)},
            proprio={"state": np.zeros(8, dtype=np.float32)},
        )
    )
    assert chunk.version == 5 and len(chunk.actions) == 2
    infer = socket.sent[-1]
    observation = infer["payload"]["observation"]
    assert observation["task"] == PROMPT.text
    assert observation["vision"]["cam_head"]["color"].shape == (2, 2, 3)
    assert observation["state"]["left_arm_joint_states"].shape == (8,)
    client.close()
