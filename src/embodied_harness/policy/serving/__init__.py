"""Policy serving over websockets, wire-compatible with openpi's policy server.

A policy runs in its own process and Python environment; the harness talks to it through
`executor.websocket_policy.WebsocketPolicyExecutor`. Needs the `policy` extra (websockets, msgpack).
Model-specific adapters live under :mod:`embodied_harness.policy`; old module paths remain importable here.
"""
