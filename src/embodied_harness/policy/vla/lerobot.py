"""Serve a LeRobot policy checkpoint (e.g. a fine-tuned pi05) over the websocket protocol.

Run it in the environment that has lerobot, not in the harness or simulator environment:

    python -m embodied_harness.policy.vla.lerobot --checkpoint <train_dir>/pretrained_model --port 8001

Request: one entry per image feature of the policy (e.g. "observation.images.front": HxWx3 uint8),
"observation.state": float32[n] and "prompt": the instruction text.
Reply: {"actions": float32[chunk_size, action_dim]}, unnormalized, in the env's action space.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

PROMPT_KEY = "prompt"


class LeRobotPolicy:
    def __init__(self, checkpoint: str | Path, *, device: str = "cuda") -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self._torch = torch
        self.checkpoint = str(checkpoint)
        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.device = device
        self.config = config
        self.policy = get_policy_class(config.type).from_pretrained(self.checkpoint, config=config)
        self.policy.to(device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.image_keys = list(config.image_features)
        state = config.robot_state_feature
        self.state_key = "observation.state" if state is not None else None

    def metadata(self) -> dict[str, Any]:
        action = self.config.action_feature
        return {
            "policy": self.config.type,
            "checkpoint": self.checkpoint,
            "image_keys": self.image_keys,
            "state_key": self.state_key,
            "prompt_key": PROMPT_KEY,
            "chunk_size": getattr(self.config, "chunk_size", None),
            "action_dim": action.shape[0] if action is not None else None,
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        torch = self._torch
        missing = [key for key in [*self.image_keys, self.state_key] if key and key not in obs]
        if missing:
            raise KeyError(f"request is missing {', '.join(missing)}; the policy expects {self.metadata()}")
        batch: dict[str, Any] = {}
        for key in self.image_keys:
            # Same conversion as lerobot's preprocess_observation: HWC uint8 -> CHW float in [0, 1].
            image = torch.from_numpy(np.array(obs[key], copy=True, order="C"))
            batch[key] = image.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if self.state_key:
            batch[self.state_key] = torch.from_numpy(
                np.array(obs[self.state_key], dtype=np.float32, copy=True)
            ).unsqueeze(0)
        batch["task"] = [str(obs.get(PROMPT_KEY, ""))]
        with torch.inference_mode():
            batch = self.preprocessor(batch)
            actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(actions)
        return {"actions": actions[0].float().cpu().numpy()}

    def reset(self) -> None:
        self.policy.reset()


def main(argv: list[str] | None = None) -> None:
    from ..serving.websocket_server import DEFAULT_MAX_SIZE, serve

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="directory with config.json, model.safetensors and processors"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--max-request-mb",
        type=int,
        default=DEFAULT_MAX_SIZE // 2**20,
        help="largest request in MiB; a larger one closes the connection (default: %(default)s)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.max_request_mb < 1:
        parser.error("--max-request-mb must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    policy = LeRobotPolicy(args.checkpoint, device=args.device)
    logging.getLogger("embodied_harness.policy.vla").info("loaded %s", policy.metadata())
    serve(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata(),
        max_size=args.max_request_mb * 2**20,
    )


if __name__ == "__main__":
    main()
