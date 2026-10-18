"""Websocket policy serving: the openpi-compatible codec, the server and the executor client."""

import asyncio
import logging
import threading
import time

import numpy as np
import pytest

from conftest import RUN_CONFIG
from embodied_harness.config.loader import load_bundle
from embodied_harness.core.types import Observation, Prompt, SkillCall
from embodied_harness.policy.executor.websocket_policy import PolicyServerError, WebsocketPolicyExecutor
from embodied_harness.policy.serving import msgpack_numpy
from embodied_harness.policy.serving.websocket_server import serve_async
from embodied_harness.policy.transport.openpi import (
    OpenPIClient,
    OpenPIProtocolError,
    set_path,
    validate_actions,
    validate_metadata,
)

LIBRARY = load_bundle(RUN_CONFIG).library
PROMPT = Prompt("pick_next", "pick up the next package")


class EchoPolicy:
    """Actions encode what the request carried, so the test can check the wire format."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.requests = []

    def infer(self, obs):
        self.requests.append(obs)
        if self.fail:
            raise ValueError("model exploded")
        value = float(obs["observation.images.front"].mean()) + float(obs["observation.state"].sum())
        actions = np.full((10, 8), value, dtype=np.float32)
        actions[:, 0] = np.arange(10)
        return {"actions": actions, "prompt_seen": obs["prompt"]}


@pytest.fixture
def policy_server():
    started = []

    def start(policy, metadata=None, **options):
        ready = threading.Event()
        port = []
        loop = asyncio.new_event_loop()
        stop = asyncio.Event()

        def run():
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    serve_async(
                        policy,
                        host="127.0.0.1",
                        port=0,
                        metadata=metadata or {"chunk_size": 10},
                        ready=lambda bound: (port.append(bound), ready.set()),
                        stop=stop,
                        **options,
                    )
                )
            finally:
                ready.set()  # a server that failed to start must not leave the test waiting

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert ready.wait(10)
        assert port, "the policy server did not start"
        started.append((loop, stop, thread))
        return f"ws://127.0.0.1:{port[0]}"

    yield start
    for loop, stop, thread in started:
        loop.call_soon_threadsafe(stop.set)
        thread.join(10)


def executor(url, **kwargs):
    return WebsocketPolicyExecutor(
        LIBRARY,
        url=url,
        images={"front": "observation.images.front"},
        state={"proprio": "state", "key": "observation.state"},
        **kwargs,
    )


def observation(pixel=10):
    return Observation(
        images={"front": np.full((4, 6, 3), pixel, dtype=np.uint8)},
        proprio={"state": np.arange(8, dtype=np.float32), "gripper_width": 0.04},
    )


def test_codec_round_trip_matches_openpi_layout():
    payload = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.array([0.5, -1.0], dtype=np.float32),
        "scalar": np.float32(2.5),
        "nested": {"ids": np.array([1, 2, 3], dtype=np.int64)},
        "prompt": "pick",
    }
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    assert decoded["prompt"] == "pick"
    np.testing.assert_array_equal(decoded["image"], payload["image"])
    assert decoded["state"].dtype == np.float32
    assert decoded["scalar"] == 2.5 and isinstance(decoded["scalar"], np.float32)
    np.testing.assert_array_equal(decoded["nested"]["ids"], [1, 2, 3])
    # openpi_client encodes arrays exactly this way; keep the wire format byte-compatible.
    assert msgpack_numpy.pack_array(np.zeros(2, np.int16)) == {
        b"__ndarray__": True,
        b"data": b"\x00\x00\x00\x00",
        b"dtype": "<i2",
        b"shape": (2,),
    }
    with pytest.raises(ValueError):
        msgpack_numpy.packb({"bad": np.array([object()])})


def test_codec_accepts_string_metadata_keys_from_newer_msgpack_clients():
    # Some msgpack clients encode the ndarray envelope's keys as strings rather than bin values.
    envelope = msgpack_numpy.packb(
        {"x": {"__ndarray__": True, "data": b"\x01\x02", "dtype": "|u1", "shape": [2]}}
    )
    decoded = msgpack_numpy.unpackb(envelope)
    np.testing.assert_array_equal(decoded["x"], np.array([1, 2], dtype=np.uint8))


# Dtypes no request or reply needs. An object dtype (also inside a subarray or a struct) makes numpy read
# the sender's bytes as object pointers: one such request crashed a LeRobot policy server.
NON_NUMERIC_DTYPES = [
    pytest.param("|O", id="object"),
    pytest.param("(2,)O", id="object-subarray"),
    pytest.param(
        {"names": ["a", "b"], "formats": ["<f4", "|O"], "offsets": [0, 8], "itemsize": 16},
        id="struct-with-object",
    ),
    pytest.param("f4,f4", id="struct"),
    pytest.param("|V8", id="void"),
    pytest.param("<c8", id="complex"),
    pytest.param("<U2", id="str"),
    pytest.param("|S4", id="bytes"),
    pytest.param("<M8[s]", id="datetime"),
]


@pytest.mark.parametrize("dtype", NON_NUMERIC_DTYPES)
def test_codec_refuses_dtypes_other_than_bool_integer_and_float(dtype):
    array = {b"__ndarray__": True, b"data": bytes(64), b"dtype": dtype, b"shape": [2]}
    with pytest.raises(ValueError, match="unsupported dtype"):
        msgpack_numpy.unpackb(msgpack_numpy.packb({"x": array}))
    scalar = {b"__npgeneric__": True, b"data": 0, b"dtype": dtype}
    with pytest.raises(ValueError, match="unsupported dtype"):
        msgpack_numpy.unpackb(msgpack_numpy.packb({"x": scalar}))


@pytest.mark.parametrize("dtype", ["|b1", "|i1", "<u2", "<i8", "<f2", "<f4", "<f8", ">f4"])
def test_codec_round_trips_bool_integer_and_float_dtypes(dtype):
    array = np.arange(6).reshape(2, 3).astype(dtype)
    scalar = array.flat[1]
    decoded = msgpack_numpy.unpackb(msgpack_numpy.packb({"array": array, "scalar": scalar}))
    np.testing.assert_array_equal(decoded["array"], array)
    assert decoded["array"].dtype == array.dtype
    # np.float64 is a Python float, which msgpack packs natively.
    assert decoded["scalar"] == scalar and np.asarray(decoded["scalar"]).dtype == scalar.dtype


class ZeroPolicy:
    """Answers without reading the request."""

    def infer(self, obs):
        return {"actions": np.zeros((2, 8), dtype=np.float32)}


def test_server_answers_an_object_array_with_an_error_and_keeps_serving(policy_server):
    client = OpenPIClient(policy_server(ZeroPolicy()))
    pointers = {b"__ndarray__": True, b"data": bytes(16), b"dtype": "|O", b"shape": [2]}
    with pytest.raises(OpenPIProtocolError, match="unsupported dtype"):
        client.call({"observation.state": pointers})
    assert client.call({"observation.state": np.zeros(2, np.float32)})["actions"].shape == (2, 8)
    client.close()


def test_server_closes_a_connection_whose_request_is_over_max_size(policy_server, caplog):
    from websockets.exceptions import ConnectionClosed

    client = OpenPIClient(policy_server(ZeroPolicy(), max_size=4096))
    with caplog.at_level(logging.WARNING, logger="embodied_harness.serving"):
        assert client.call({"observation.state": np.zeros(16, np.float32)})["actions"].shape == (2, 8)
        with pytest.raises(ConnectionClosed, match="message too big"):
            client.call({"observation.state": np.zeros(4096, np.float32)})
        # Only that connection was closed: the next request reconnects.
        assert client.call({"observation.state": np.zeros(16, np.float32)})["actions"].shape == (2, 8)
        deadline = time.monotonic() + 5
        while "--max-request-mb" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)  # the server logs from its own thread
    assert "--max-request-mb" in caplog.text
    client.close()


def test_server_limits_request_size_by_default(monkeypatch):
    import websockets.asyncio.server

    options = {}

    class Server:
        sockets = ()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

    def serve(handler, host, port, **kwargs):
        options.update(kwargs)
        return Server()

    monkeypatch.setattr(websockets.asyncio.server, "serve", serve)
    stop = asyncio.Event()
    stop.set()
    asyncio.run(serve_async(ZeroPolicy(), port=0, stop=stop))
    assert options["max_size"] == 64 * 2**20
    with pytest.raises(ValueError, match="max_size"):
        asyncio.run(serve_async(ZeroPolicy(), port=0, stop=stop, max_size=0))


def test_lerobot_server_command_sets_the_request_size_limit(monkeypatch):
    from embodied_harness.policy.vla import lerobot

    served = []

    class FakePolicy:
        def __init__(self, checkpoint, *, device):
            pass

        def metadata(self):
            return {"chunk_size": 2}

    monkeypatch.setattr(lerobot, "LeRobotPolicy", FakePolicy)
    monkeypatch.setattr(
        "embodied_harness.policy.serving.websocket_server.serve",
        lambda policy, **kwargs: served.append(kwargs),
    )
    lerobot.main(["--checkpoint", "/unused"])
    lerobot.main(["--checkpoint", "/unused", "--max-request-mb", "8"])
    assert [kwargs["max_size"] for kwargs in served] == [64 * 2**20, 8 * 2**20]
    with pytest.raises(SystemExit):
        lerobot.main(["--checkpoint", "/unused", "--max-request-mb", "0"])


def test_openpi_action_validation_rejects_bad_shapes_and_values():
    with pytest.raises(OpenPIProtocolError, match="shape"):
        validate_actions({"actions": np.zeros(3, dtype=np.float32)})
    with pytest.raises(OpenPIProtocolError, match="NaN"):
        validate_actions({"actions": np.array([[np.nan]], dtype=np.float32)})
    with pytest.raises(OpenPIProtocolError, match="action_dim"):
        validate_actions({"actions": np.zeros((2, 3), dtype=np.float32)}, metadata={"action_dim": 2})


def test_openpi_nested_request_paths():
    payload = {}
    set_path(payload, "observation/image", np.zeros((2, 3, 3), dtype=np.uint8))
    set_path(payload, "observation/state", np.zeros(2, dtype=np.float32))
    assert set(payload) == {"observation"}
    assert set(payload["observation"]) == {"image", "state"}
    with pytest.raises(ValueError, match="conflicts"):
        set_path(payload, "observation/image/rgb", 1)


def test_executor_gets_action_chunks_from_the_server(policy_server):
    policy = EchoPolicy()
    url = policy_server(policy, {"chunk_size": 10, "policy": "echo"})
    client = executor(url, n_action_steps=4)
    client.reset()
    assert client.metadata == {"chunk_size": 10, "policy": "echo"}
    assert client.capabilities().chunk_size == 4
    client.set_instruction(SkillCall("pick"), PROMPT, version=7)
    chunk = client.infer_chunk(observation(pixel=10))
    assert chunk.version == 7
    assert len(chunk.actions) == 4
    np.testing.assert_allclose([a[0] for a in chunk.actions], [0, 1, 2, 3])
    assert chunk.actions[0][1] == pytest.approx(10 + 28)
    request = policy.requests[0]
    assert set(request) == {"observation.images.front", "observation.state", "prompt"}
    assert request["prompt"] == "pick up the next package"
    np.testing.assert_array_equal(request["observation.images.front"], observation(pixel=10).images["front"])
    client.close()


def test_openpi_wrapper_and_v3_repack_deliver_the_same_model_inputs(policy_server):
    from dataclasses import replace

    from embodied_harness.policy.integrations.openpi import OpenPIDatasetSpec
    from embodied_harness.policy.vla.openpi import OpenPIPolicy

    requests = []

    class NativePolicy:
        def infer(self, sample):
            requests.append(sample)
            assert sample["images"]["cam_high"].shape == (3, 4, 6)
            return {"actions": np.tile(sample["state"], (3, 1))}

    spec = OpenPIDatasetSpec(images={"images/cam_high": "observation.images.front"})
    wrapper = OpenPIPolicy("test", "/unused", mapping=spec, policy=NativePolicy())
    client = executor(policy_server(wrapper, wrapper.metadata()), n_action_steps=2)
    obs = observation()
    try:
        client.reset()
        client.set_instruction(SkillCall("pick"), PROMPT, version=12)
        chunk = client.infer_chunk(obs)
        assert chunk.version == 12 and len(chunk.actions) == 2
        np.testing.assert_array_equal(chunk.actions[0], obs.proprio["state"])
        train_sample = {
            "observation.images.front": obs.images["front"].transpose(2, 0, 1).astype(np.float32) / 255,
            "observation.state": obs.proprio["state"],
            "action": np.tile(obs.proprio["state"], (3, 1)),
            "task": PROMPT.text,
        }
        training = replace(spec, image_input_layout="chw").repack(train_sample)
        np.testing.assert_array_equal(training["images"]["cam_high"], requests[0]["images"]["cam_high"])
        np.testing.assert_array_equal(training["state"], requests[0]["state"])
        assert training["prompt"] == requests[0]["prompt"] == PROMPT.text
    finally:
        client.close()


def test_server_errors_surface_in_the_harness(policy_server):
    policy = EchoPolicy(fail=True)
    client = executor(policy_server(policy))
    client.reset()
    client.set_instruction(SkillCall("pick"), PROMPT, version=1)
    with pytest.raises(PolicyServerError, match="model exploded"):
        client.infer_chunk(observation())
    policy.fail = False  # the server closed that connection; the client reconnects
    assert len(client.infer_chunk(observation()).actions) == 10
    client.close()


def test_unreachable_server_times_out():
    client = executor("ws://127.0.0.1:9", connect_timeout_s=0.5)
    with pytest.raises(PolicyServerError, match="not reachable"):
        client.reset()


def test_executor_only_runs_library_prompts(policy_server):
    client = executor(policy_server(EchoPolicy()))
    with pytest.raises(ValueError, match="not in library"):
        client.set_instruction(SkillCall("pick"), Prompt("made_up", "dance"), version=1)
    with pytest.raises(ValueError, match="has no prompt"):
        client.set_instruction(SkillCall("retract"), None, version=1)
    with pytest.raises(ValueError, match="state must be"):
        WebsocketPolicyExecutor(LIBRARY, url="ws://x", images={"front": "a"}, state={"key": "b"})


def test_request_texts_carry_the_goal_and_the_subgoal(policy_server):
    policy = EchoPolicy()
    client = WebsocketPolicyExecutor(
        LIBRARY,
        url=policy_server(policy),
        images={"front": "observation/image"},
        state={"proprio": "state", "key": "observation/state"},
        texts={"prompt": "{goal}", "simple_subgoal": "{prompt}"},
    )
    request_keys = {"observation/image", "observation/state", "prompt", "simple_subgoal"}
    policy.infer = lambda obs: (policy.requests.append(obs), {"actions": np.zeros((2, 8), np.float32)})[1]
    client.reset()
    client.begin_episode(load_bundle(RUN_CONFIG).task.model_copy(update={"goal": "sort two parcels"}))
    client.set_instruction(SkillCall("pick"), PROMPT, version=1)
    client.infer_chunk(
        Observation(images={"front": np.zeros((2, 2, 3), np.uint8)}, proprio={"state": np.zeros(8)})
    )
    request = policy.requests[0]
    assert set(request) == request_keys
    assert request["prompt"] == "sort two parcels"
    assert request["simple_subgoal"] == "pick up the next package"
    client.close()
    with pytest.raises(ValueError, match="only"):
        WebsocketPolicyExecutor(LIBRARY, url="ws://x", images={"front": "a"}, texts={"prompt": "{task}"})


def test_executor_can_build_native_openpi_nested_chw_inputs(policy_server):
    policy = EchoPolicy()
    policy.infer = lambda obs: (policy.requests.append(obs), {"actions": np.zeros((2, 8), np.float32)})[1]
    client = WebsocketPolicyExecutor(
        LIBRARY,
        url=policy_server(policy),
        images={"front": "observation/image"},
        state={"proprio": "state", "key": "observation/state"},
        nested_keys=True,
        image_layout="chw",
    )
    client.reset()
    client.set_instruction(SkillCall("pick"), PROMPT, version=1)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    client.infer_chunk(Observation(images={"front": image}, proprio={"state": np.zeros(8)}))
    request = policy.requests[0]
    np.testing.assert_array_equal(request["observation"]["image"], image.transpose(2, 0, 1))
    np.testing.assert_array_equal(request["observation"]["state"], np.zeros(8))
    client.close()


class StubConnection:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.closed = False
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True

    def send(self, payload):
        self.sent.append(payload)

    def recv(self, timeout):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self):
        self.closed = True


def test_timeout_drops_connection_and_next_call_uses_a_new_handshake(monkeypatch):
    import websockets.sync.client

    first = StubConnection([msgpack_numpy.packb({"action_dim": 2}), TimeoutError("late inference")])
    second = StubConnection(
        [msgpack_numpy.packb({"action_dim": 3}), msgpack_numpy.packb({"actions": np.zeros((1, 3))})]
    )
    sockets = iter([first, second])
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *args, **kwargs: next(sockets))
    client = OpenPIClient("ws://test")
    with pytest.raises(OpenPIProtocolError, match="late inference"):
        client.call({"prompt": "pick"})
    assert first.closed
    assert client.call({"prompt": "place"})["actions"].shape == (1, 3)
    assert client.metadata["action_dim"] == 3
    client.close()


def test_openpi_client_connects_through_the_websockets_14_threading_api(policy_server, monkeypatch):
    # pyproject allows websockets>=13. Before 15 the threading connect() had no proxy or ping_interval
    # parameter and passed its other keyword arguments on to socket.create_connection, which rejects them.
    import socket
    from urllib.parse import urlsplit

    import websockets.sync.client

    installed = websockets.sync.client.connect

    def connect_14(
        uri, *, compression="deflate", open_timeout=10, close_timeout=10, max_size=2**20, **kwargs
    ):
        url = urlsplit(uri)
        kwargs.setdefault("timeout", open_timeout)
        sock = socket.create_connection((url.hostname, url.port), **kwargs)
        sock.settimeout(None)
        # The handshake runs on the installed version, without keepalive, which 14 did not have.
        return installed(
            uri,
            sock=sock,
            compression=compression,
            open_timeout=open_timeout,
            close_timeout=close_timeout,
            max_size=max_size,
            ping_interval=None,
            proxy=None,
        )

    monkeypatch.setattr(websockets.sync.client, "connect", connect_14)
    client = OpenPIClient(policy_server(EchoPolicy()))
    assert client.connect() == {"chunk_size": 10}
    request = {
        "observation.images.front": np.zeros((4, 6, 3), np.uint8),
        "observation.state": np.zeros(8, np.float32),
        "prompt": "pick",
    }
    assert client.call(request)["actions"].shape == (10, 8)
    client.close()


def test_client_refuses_an_object_array_in_a_reply(monkeypatch):
    import websockets.sync.client

    pointers = {b"__ndarray__": True, b"data": bytes(16), b"dtype": "|O", b"shape": [2]}
    socket = StubConnection(
        [msgpack_numpy.packb({"action_dim": 2}), msgpack_numpy.packb({"actions": pointers})]
    )
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *args, **kwargs: socket)
    client = OpenPIClient("ws://test")
    with pytest.raises(OpenPIProtocolError, match="unsupported dtype"):
        client.call({"prompt": "pick"})
    assert socket.closed


def test_invalid_handshake_closes_the_connection(monkeypatch):
    import websockets.sync.client

    socket = StubConnection([msgpack_numpy.packb({"chunk_size": 0})])
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *args, **kwargs: socket)
    with pytest.raises(OpenPIProtocolError, match="chunk_size"):
        OpenPIClient("ws://test").connect()
    assert socket.closed


def test_implicit_connection_keeps_the_instruction(monkeypatch):
    import websockets.sync.client

    socket = StubConnection(
        [msgpack_numpy.packb({"action_dim": 8}), msgpack_numpy.packb({"actions": np.zeros((2, 8))})]
    )
    monkeypatch.setattr(websockets.sync.client, "connect", lambda *args, **kwargs: socket)
    client = executor("ws://test")
    client.set_instruction(SkillCall("pick"), PROMPT, version=4)
    chunk = client.infer_chunk(observation())
    assert chunk.version == 4
    assert msgpack_numpy.unpackb(socket.sent[0])["prompt"] == PROMPT.text
    client.close()


@pytest.mark.parametrize("value", [True, 1.5, "3", -1, float("inf")])
def test_invalid_metadata_dimensions_are_rejected(value):
    with pytest.raises(OpenPIProtocolError, match="action_dim"):
        validate_metadata({"action_dim": value})


@pytest.mark.parametrize("actions", [[[True]], [["1"]], [[complex(1, 2)]], [[float("inf")]]])
def test_non_real_or_nonfinite_actions_are_rejected(actions):
    with pytest.raises(OpenPIProtocolError):
        validate_actions({"actions": actions})
