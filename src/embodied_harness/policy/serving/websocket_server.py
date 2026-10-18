"""Websocket policy server with openpi's protocol.

On connect the server sends its metadata. Then, per message: the client sends an observation
dict, the server replies with the policy's result dict (at least "actions") plus
"server_timing". On an exception the server sends the traceback as a text frame and closes the
connection, which the client turns into an error. A request larger than `max_size` bytes closes
the connection with code 1009. GET /healthz answers 200 for health checks.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any, Protocol

from ..transport import msgpack_numpy
from ..transport.openpi import OpenPIProtocolError, validate_actions, validate_metadata

logger = logging.getLogger("embodied_harness.serving")

# Largest request accepted, in bytes. A request with four 480x640 RGB uint8 images and a state vector is
# under 4 MiB; 64 MiB leaves room for larger cameras while keeping any client from making the server
# buffer an unbounded frame.
DEFAULT_MAX_SIZE = 64 * 2**20


class ServedPolicy(Protocol):
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]: ...


async def _handle(policy: ServedPolicy, metadata: dict[str, Any], websocket: Any) -> None:
    import websockets

    peer = getattr(websocket, "remote_address", None)
    logger.info("client connected: %s", peer)
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    packer = msgpack_numpy.Packer()
    try:
        metadata = validate_metadata(metadata)
    except OpenPIProtocolError:
        logger.exception("invalid policy metadata")
        await websocket.close(code=1011, reason="invalid policy metadata")
        return
    await websocket.send(packer.pack(metadata))
    previous_total: float | None = None
    while True:
        try:
            started = time.monotonic()
            obs = msgpack_numpy.unpackb(await websocket.recv())
            infer_started = time.monotonic()
            result = dict(policy.infer(obs))
            # Validate at the boundary so a malformed model response never reaches a robot.  The
            # client performs the same check, but validating here gives remote clients a useful
            # protocol error and keeps every server implementation consistent.
            validate_actions(result, metadata=metadata)
            timing = {"infer_ms": (time.monotonic() - infer_started) * 1000}
            if previous_total is not None:
                timing["prev_total_ms"] = previous_total * 1000
            result["server_timing"] = timing
            await websocket.send(packer.pack(result))
            previous_total = time.monotonic() - started
        except websockets.ConnectionClosed as exc:
            if exc.sent is not None and exc.sent.code == 1009:  # message too big: over max_size
                logger.warning("closed %s: request %s (see --max-request-mb)", peer, exc.sent.reason)
            else:
                logger.info("client disconnected: %s", peer)
            return
        except Exception:
            logger.exception("policy inference failed")
            await websocket.send(traceback.format_exc())
            await websocket.close(code=1011, reason="policy inference failed, see the traceback message")
            return


def _health(connection: Any, request: Any) -> Any:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def serve_async(
    policy: ServedPolicy,
    *,
    host: str = "0.0.0.0",
    port: int = 8001,
    metadata: dict[str, Any] | None = None,
    ready: Callable[[int], None] | None = None,
    stop: asyncio.Event | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
) -> None:
    """Serve until `stop` is set (or forever). `ready(port)` is called once listening; port 0 picks one.

    `max_size` is the largest request in bytes; a larger one closes its connection (code 1009).
    """
    from websockets.asyncio.server import serve

    if isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < 1:
        raise ValueError(f"max_size must be a positive number of bytes, got {max_size!r}")

    async def handler(websocket: Any) -> None:
        await _handle(policy, metadata or {}, websocket)

    # A synchronous policy inference can legitimately occupy the event loop for longer than the
    # websocket keepalive interval (for example a CPU fallback for a large VLA). Disable transport
    # pings here; request_timeout_s remains the inference deadline and prevents an infinite call.
    async with serve(
        handler, host, port, compression=None, max_size=max_size, process_request=_health, ping_interval=None
    ) as server:
        bound = server.sockets[0].getsockname()[1] if server.sockets else port
        logger.info("serving policy on ws://%s:%s", host, bound)
        if ready is not None:
            ready(bound)
        if stop is None:
            await server.serve_forever()
        else:
            await stop.wait()


def serve(
    policy: ServedPolicy,
    *,
    host: str = "0.0.0.0",
    port: int = 8001,
    metadata: dict[str, Any] | None = None,
    max_size: int = DEFAULT_MAX_SIZE,
) -> None:
    asyncio.run(serve_async(policy, host=host, port=port, metadata=metadata or {}, max_size=max_size))
