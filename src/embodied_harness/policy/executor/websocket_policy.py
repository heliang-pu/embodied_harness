"""Executor behind a websocket policy server (openpi protocol: msgpack with numpy arrays).

The policy runs in its own process and Python environment, so the harness and the simulator never
import the model's dependencies. Each inference sends the current camera images, the state vector
and the prompt-library text, and executes the first `n_action_steps` actions of the chunk it
gets back. Works with `python -m embodied_harness.policy.vla.lerobot` and with openpi servers;
the request keys are configurable to match either.
"""

from __future__ import annotations

import logging
import string
from typing import Any

import numpy as np

from ...config.schema import PromptLibrary, TaskSpec
from ...core.types import ActionChunk, Observation, Prompt, SkillCall
from ..transport.openpi import OpenPIClient, OpenPIProtocolError, set_path, validate_actions
from ..transport.xpolicy import XPolicyClient, XPolicyProtocolError
from ..transport.xpolicy import validate_actions as validate_xpolicy_actions
from .base import ExecutorCaps

logger = logging.getLogger("embodied_harness.executor")


class PolicyServerError(RuntimeError):
    pass


class WebsocketPolicyExecutor:
    name = "websocket"

    def __init__(
        self,
        library: PromptLibrary,
        *,
        url: str,
        images: dict[str, str],
        state: dict[str, str] | None = None,
        prompt_key: str = "prompt",
        texts: dict[str, str] | None = None,
        actions_key: str = "actions",
        n_action_steps: int | None = None,
        extra: dict[str, Any] | None = None,
        nested_keys: bool = False,
        image_layout: str = "hwc",
        connect_timeout_s: float = 120.0,
        request_timeout_s: float = 30.0,
        protocol: str = "openpi",
        handover_release: bool = False,
        handover_left_threshold: float = 0.5,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
    ) -> None:
        """
        images: harness camera -> request key, e.g. {front: observation.images.front}.
        state: {proprio: <proprio key holding the state vector>, key: <request key>}.
        texts: request key -> template with {prompt} (the library text) and {goal} (the episode's
            task goal), e.g. RoboMME's {prompt: "{goal}", simple_subgoal: "{prompt}"}. Without it the
            library text goes into `prompt_key`.
        extra: constant fields added to every request.
        nested_keys: interpret slash-separated request keys as nested paths instead of flat keys.
        protocol: ``openpi`` (the default legacy wire) or ``xpolicy`` (XPolicyLab frames).
        For ``xpolicy``, set request keys such as ``vision/cam_head/color`` and
        ``state/left_arm_joint_state`` with ``nested_keys: true``.  The adapter wraps that
        standard observation in an XPolicyLab ``infer`` frame; it does not invent missing
        simulator fields.
        """
        if not images:
            raise ValueError("images must map at least one camera to a request key")
        if state is not None and set(state) != {"proprio", "key"}:
            raise ValueError("state must be {proprio: <proprio key>, key: <request key>}")
        if n_action_steps is not None and n_action_steps < 1:
            raise ValueError("n_action_steps must be at least 1")
        if protocol not in {"openpi", "xpolicy"}:
            raise ValueError("protocol must be 'openpi' or 'xpolicy'")
        self._library = library
        self._url = url
        self._images = dict(images)
        self._state = dict(state) if state is not None else None
        self._texts = dict(texts) if texts else {prompt_key: "{prompt}"}
        for key, template in self._texts.items():
            names = {name for _, name, _, _ in string.Formatter().parse(template) if name}
            if names - {"prompt", "goal"}:
                raise ValueError(
                    f"texts.{key}: only {{prompt}} and {{goal}} can be used, found {sorted(names)}"
                )
        self._goal = ""
        self._actions_key = actions_key
        self._n_action_steps = n_action_steps
        self._extra = dict(extra or {})
        self._nested_keys = nested_keys
        if image_layout not in {"hwc", "chw"}:
            raise ValueError("image_layout must be 'hwc' or 'chw'")
        self._image_layout = image_layout
        self._connect_timeout = connect_timeout_s
        self._request_timeout = request_timeout_s
        self._protocol = protocol
        self._handover_release = bool(handover_release)
        if not 0.0 < handover_left_threshold < 1.0:
            raise ValueError("handover_left_threshold must be between 0 and 1")
        self._handover_left_threshold = float(handover_left_threshold)
        self._evaluation_id = evaluation_id
        self._trial_id = trial_id
        self._client: OpenPIClient | XPolicyClient | None = None
        self.metadata: dict[str, Any] = {}
        self._prompt: Prompt | None = None
        self._version = 0

    # --- executor contract ---------------------------------------------------------------------

    def prompt_library(self) -> PromptLibrary:
        return self._library

    def capabilities(self) -> ExecutorCaps:
        advertised_value = self.metadata.get("chunk_size", self.metadata.get(b"chunk_size"))
        try:
            advertised = int(advertised_value or 1)
        except (TypeError, ValueError) as exc:
            raise PolicyServerError(
                f"policy metadata chunk_size must be an integer: {advertised_value!r}"
            ) from exc
        if advertised < 1:
            raise PolicyServerError(f"policy metadata chunk_size must be positive: {advertised}")
        return ExecutorCaps(chunk_size=self._n_action_steps or advertised)

    def infer_chunk(self, obs: Observation) -> ActionChunk:
        if self._prompt is None:
            raise RuntimeError("infer_chunk called before set_instruction")
        request: dict[str, Any] = {}
        for camera, key in self._images.items():
            try:
                image = np.asarray(obs.images[camera])
            except KeyError as exc:
                raise PolicyServerError(f"observation has no camera {camera!r}") from exc
            if image.ndim != 3 or image.shape[2] != 3:
                raise PolicyServerError(f"observation camera {camera!r} must be HxWx3 RGB; got {image.shape}")
            if self._image_layout == "chw":
                image = image.transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
            try:
                self._set_field(request, key, image)
            except ValueError as exc:
                raise PolicyServerError(str(exc)) from exc
        if self._state is not None:
            try:
                state = np.asarray(obs.proprio[self._state["proprio"]], dtype=np.float32)
            except KeyError as exc:
                raise PolicyServerError(
                    f"observation has no proprio field {self._state['proprio']!r}"
                ) from exc
            try:
                self._set_field(request, self._state["key"], state)
            except ValueError as exc:
                raise PolicyServerError(str(exc)) from exc
        for key, template in self._texts.items():
            try:
                self._set_field(request, key, template.format(prompt=self._prompt.text, goal=self._goal))
            except ValueError as exc:
                raise PolicyServerError(str(exc)) from exc
        for key, value in self._extra.items():
            try:
                self._set_field(request, key, value)
            except ValueError as exc:
                raise PolicyServerError(str(exc)) from exc
        reply = self._call(request)
        try:
            if self._protocol == "xpolicy":
                actions = validate_xpolicy_actions(reply, key=self._actions_key)
            else:
                actions = validate_actions(reply, key=self._actions_key, metadata=self.metadata)
        except (OpenPIProtocolError, XPolicyProtocolError) as exc:
            if self._client is not None:
                self._client.close()
            raise PolicyServerError(str(exc)) from exc
        actions = self._apply_handover_release(actions)
        steps = self._n_action_steps or len(actions)
        timing = reply.get("server_timing") or {}
        logger.debug(
            "policy chunk %s for %r, infer %.0f ms",
            getattr(actions, "shape", (len(actions), "dict")),
            self._prompt.id,
            timing.get("infer_ms", -1),
        )
        return ActionChunk(self._version, list(actions[:steps]))

    def _apply_handover_release(self, actions: Any) -> Any:
        """Open the right gripper once the learned handover closes the left gripper.

        RoboTwin's handover contract requires a visible right-to-left transfer.  Some Pi0.5
        chunks close the left gripper correctly but keep the right gripper closed, which leaves
        the supervisor's handover gate blocked forever.  This opt-in adapter only applies to
        prompts from the handover family and preserves the model's joint targets; it is disabled
        for every other task and protocol by default.
        """
        if not self._handover_release or self._prompt is None or not self._prompt.id.startswith("handover_"):
            return actions
        array = np.asarray(actions)
        if array.ndim != 2 or array.shape[1] != 14 or not np.issubdtype(array.dtype, np.number):
            return actions
        updated = np.array(array, copy=True)
        closed = np.flatnonzero(updated[:, 6] < self._handover_left_threshold)
        if closed.size:
            updated[closed[0] :, 13] = 1.0
        return updated

    def _set_field(self, request: dict[str, Any], key: str, value: Any) -> None:
        if self._nested_keys:
            set_path(request, key, value)
        else:
            request[key] = value

    def reset(self) -> None:
        self._prompt = None
        self._version = 0
        self._ensure_connected()
        if self._protocol == "xpolicy":
            assert isinstance(self._client, XPolicyClient)
            try:
                self._client.reset()
            except XPolicyProtocolError as exc:
                self._client.close()
                self._client = None
                raise PolicyServerError(str(exc)) from exc

    def begin_episode(self, task: TaskSpec) -> None:
        """The episode's task goal, for request templates that use {goal}."""
        self._goal = task.goal

    def set_instruction(self, call: SkillCall, prompt: Prompt | None, version: int) -> None:
        if prompt is None:
            raise ValueError(f"{call.skill} has no prompt; the policy only runs prompts from its library")
        if self._library.get(prompt.id) is None:
            raise ValueError(f"prompt {prompt.id!r} is not in library {self._library.library!r}")
        self._prompt = prompt
        self._version = version

    def is_done(self) -> bool:
        return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        assert self._client is not None
        try:
            if self._protocol == "xpolicy":
                assert isinstance(self._client, XPolicyClient)
                reply = self._client.infer(request)
            else:
                assert isinstance(self._client, OpenPIClient)
                reply = self._client.call(request)
            self.metadata = self._client.metadata
            return reply
        except (OpenPIProtocolError, XPolicyProtocolError) as exc:
            raise PolicyServerError(str(exc)) from exc

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        if self._protocol == "xpolicy":
            self._client = XPolicyClient(
                self._url,
                evaluation_id=self._evaluation_id,
                trial_id=self._trial_id,
                connect_timeout_s=self._connect_timeout,
                request_timeout_s=self._request_timeout,
            )
        else:
            self._client = OpenPIClient(
                self._url,
                connect_timeout_s=self._connect_timeout,
                request_timeout_s=self._request_timeout,
            )
        try:
            self.metadata = self._client.connect()
        except (OpenPIProtocolError, XPolicyProtocolError) as exc:
            self._client.close()
            self._client = None
            raise PolicyServerError(str(exc)) from exc
