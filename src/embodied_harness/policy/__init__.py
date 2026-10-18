"""Robot policy implementations, grouped by model family.

The policy packages contain model-specific checkpoint adapters.  Runtime execution and
WebSocket transport remain in :mod:`embodied_harness.policy.executor` and
:mod:`embodied_harness.policy.transport` so model dependencies stay outside the harness process.
"""
