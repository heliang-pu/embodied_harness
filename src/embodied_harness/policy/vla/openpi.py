"""Serve an OpenPI checkpoint through the harness websocket protocol.

The harness sends observations using LeRobot feature names (for example
``observation.images.top`` and ``observation.state``).  OpenPI policies consume the
native nested mapping (``images``, ``state`` and ``prompt``), so this module keeps that
adapter at the process boundary.  OpenPI is imported only when a checkpoint is loaded;
the wrapper can therefore be used with a small fake policy in tests or in tooling that
does not have JAX installed.

Run in the OpenPI environment with::

    python -m embodied_harness.policy.vla.openpi \
        --config pi05 --checkpoint /path/to/checkpoint --mapping mapping.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..integrations.openpi import OpenPIDatasetSpec, load_openpi_spec

logger = logging.getLogger("embodied_harness.policy.vla.openpi")


def _upstream_metadata(policy: Any) -> dict[str, Any]:
    """Read optional metadata without requiring a particular OpenPI policy version."""

    value = getattr(policy, "metadata", None)
    if callable(value):
        value = value()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _action_horizon(config: Any) -> int | None:
    """Return the action chunk length, when the config declares one.

    ``model.action_dim`` in OpenPI can describe a padded model output (often 32), and
    must not be advertised as the robot action dimension.  The temporal horizon is the
    only model field used here; the action dimension is left to the actual policy output.
    """

    model = getattr(config, "model", None)
    value = getattr(model, "action_horizon", None)
    if isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


class OpenPIPolicy:
    """Adapt LeRobot-style websocket observations to a trained OpenPI policy.

    ``policy`` is an optional already-created policy object.  Supplying it is useful for
    tests and for embedding the websocket server in another process; with no object, the
    OpenPI imports remain lazy until :meth:`_load_policy` is called.
    """

    def __init__(
        self,
        config: str,
        checkpoint: str | Path,
        *,
        mapping: OpenPIDatasetSpec | str | Path,
        default_prompt: str | None = None,
        policy: Any | None = None,
    ) -> None:
        self.config_name = config
        self.checkpoint = str(checkpoint)
        self.default_prompt = default_prompt
        self.spec = mapping if isinstance(mapping, OpenPIDatasetSpec) else load_openpi_spec(mapping)
        # Wire images are HWC uint8 arrays and prompt is supplied by the executor.  The
        # native OpenPI transform still receives the configured output layout (normally CHW).
        self.wire_spec = dataclasses.replace(
            self.spec,
            task_key="prompt",
            subtask_style=None,
            image_input_layout="hwc",
        )
        self.train_config: Any | None = None
        self.policy = policy if policy is not None else self._load_policy(config)

    def _load_policy(self, config: str) -> Any:
        """Load OpenPI lazily, keeping JAX/OpenPI out of the harness import path."""

        from openpi.policies.policy_config import create_trained_policy
        from openpi.training.config import get_config

        train_config = get_config(config)
        self.train_config = train_config
        return create_trained_policy(train_config, self.checkpoint, default_prompt=self.default_prompt)

    def metadata(self) -> dict[str, Any]:
        """Return the handshake metadata, preserving policy metadata when available."""

        result = _upstream_metadata(self.policy)
        result.setdefault("policy", "openpi")
        result.setdefault("config", self.config_name)
        result.setdefault("checkpoint", self.checkpoint)
        horizon = _action_horizon(self.train_config)
        if horizon is not None:
            result.setdefault("chunk_size", horizon)
        # These are wire keys, not model action dimensions.  Deliberately do not infer
        # action_dim from config.model.action_dim: OpenPI models may pad that dimension.
        result.setdefault(
            "wire",
            {
                "images": dict(self.spec.images),
                "state_key": self.spec.state_key,
                "prompt_key": "prompt",
            },
        )
        result.setdefault("image_keys", list(self.spec.images.values()))
        result.setdefault("state_key", self.spec.state_key)
        result.setdefault("prompt_key", "prompt")
        return result

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Repack one wire observation and return OpenPI's unmodified inference result."""

        if not isinstance(obs, Mapping):
            raise TypeError(f"policy observation must be a mapping, got {type(obs).__name__}")
        sample = dict(obs)
        if self.default_prompt is not None and (
            not isinstance(sample.get("prompt"), str) or not sample["prompt"].strip()
        ):
            sample["prompt"] = self.default_prompt
        native = self.wire_spec.repack(sample, training=False)
        output = self.policy.infer(native)
        if not isinstance(output, Mapping):
            raise TypeError(f"OpenPI policy returned {type(output).__name__}; expected a mapping")
        if "actions" not in output:
            raise ValueError("OpenPI policy response has no 'actions' field")
        # Keep OpenPI's action post-processing and dtype untouched.  The websocket
        # server's msgpack codec serializes numpy arrays directly.
        return dict(output)

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()


def main(argv: list[str] | None = None) -> None:
    from ..serving.websocket_server import DEFAULT_MAX_SIZE, serve

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="OpenPI training config name")
    parser.add_argument("--checkpoint", required=True, help="OpenPI checkpoint directory")
    parser.add_argument("--mapping", required=True, type=Path, help="OpenPI/LeRobot mapping YAML")
    parser.add_argument("--default-prompt", default=None, help="prompt used when a request has no prompt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--max-request-mb",
        type=int,
        default=DEFAULT_MAX_SIZE // 2**20,
        help="largest request in MiB; a larger one closes the connection (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.max_request_mb < 1:
        parser.error("--max-request-mb must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    policy = OpenPIPolicy(
        args.config,
        args.checkpoint,
        mapping=args.mapping,
        default_prompt=args.default_prompt,
    )
    metadata = policy.metadata()
    logger.info("loaded OpenPI policy: %s", metadata)
    serve(policy, host=args.host, port=args.port, metadata=metadata, max_size=args.max_request_mb * 2**20)


if __name__ == "__main__":
    main()
