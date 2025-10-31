from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        tokenizer = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            tokenizer: Tokenizer for hierarchical skill generation (required for skill text generation).
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._tokenizer = tokenizer

        # Check if model supports hierarchical skill generation
        self._supports_skill_generation = hasattr(model, "infer_with_memory")

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
            
            # Don't JIT infer_with_memory - it contains autoregressive generation loops
            # that are not compatible with JAX tracing
            if self._supports_skill_generation:
                self._infer_with_memory = model.infer_with_memory

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, memory_text: str | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        
        # Generate skills if model supports hierarchical inference
        skill_json, has_eos = None, False
        
        # Debug logging on first call
        if not hasattr(self, "_first_call_logged"):
            logger.info(f"Policy: supports_skill_generation={self._supports_skill_generation}")
            logger.info(f"Policy: memory_text provided={memory_text is not None}")
            logger.info(f"Policy: is_pytorch_model={self._is_pytorch_model}")
            self._first_call_logged = True
        
        if self._supports_skill_generation:
            # Use hierarchical inference path (generate skills + actions)
            # Provide empty string if no memory_text given
            memory_input = memory_text if memory_text is not None else ""
            
            if not self._is_pytorch_model:
                try:
                    skill_generation_rng, self._rng = jax.random.split(self._rng)
                    actions, skill_json = self._infer_with_memory(
                        skill_generation_rng,
                        observation,
                        memory_text=memory_input,
                        tokenizer=self._tokenizer,
                        generate_skill=True,
                    )
                    # Check if skill ends with EOS
                    has_eos = skill_json is not None and "<EOS_SKILL>" in skill_json
                    # Use actions from infer_with_memory instead of sample_actions
                    outputs = {
                        "state": inputs["state"],
                        "actions": actions,
                        "skill_json": skill_json,
                        "has_eos": has_eos,
                    }
                except Exception as e:
                    logger.error(f"Error in hierarchical inference: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to standard action-only inference
                    outputs = {
                        "state": inputs["state"],
                        "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
                        "skill_json": None,
                        "has_eos": False,
                    }
            else:
                # PyTorch path (if needed later)
                outputs = {
                    "state": inputs["state"],
                    "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
                    "skill_json": None,
                    "has_eos": False,
                }
        else:
            # Standard action-only inference
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
                "skill_json": skill_json,
                "has_eos": has_eos,
            }
        
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            # Handle numpy conversion for JAX arrays
            outputs = {
                k: np.asarray(v[0, ...]) if isinstance(v, (jnp.ndarray, np.ndarray)) and v.ndim > 1 else v
                for k, v in outputs.items()
            }

        # Save skill generation results before transforms
        skill_json_result = outputs.get("skill_json", None)
        has_eos_result = outputs.get("has_eos", False)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        # Re-add skill generation results after transforms
        # (transforms may strip non-action keys)
        outputs["skill_json"] = skill_json_result
        outputs["has_eos"] = has_eos_result

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
