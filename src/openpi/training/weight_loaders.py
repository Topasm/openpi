import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class DualHeadWeightLoader(WeightLoader):
    """Loads weights for Pi0DualHead (TRUE MoE) model from a standard Pi0 checkpoint.

    This loader:
    1. Loads the base Pi0 weights (PaliGemma + LoRA if present)
    2. Skips the old action_out_proj (will be randomly initialized)
    3. Initializes task_router, nav_expert, manip_expert, action_out_proj randomly

    Note: In TRUE MoE architecture, action_out_proj is kept but used differently.
    It now projects the gated expert features instead of LLM outputs directly.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # Skip action_out_proj from checkpoint (incompatible - projects different features)
        # Initialize task_router, nav_expert, manip_expert, action_out_proj randomly
        skip_regex = ".*(action_out_proj|task_router|nav_expert|manip_expert).*"
        missing_regex = ".*(lora|task_router|nav_expert|manip_expert|action_out_proj).*"

        return _merge_params(loaded_params, params, missing_regex=missing_regex, skip_regex=skip_regex)


@dataclasses.dataclass(frozen=True)
class AuxLossWeightLoader(WeightLoader):
    """Loads weights for Pi0AuxLoss model from a standard Pi0 checkpoint.

    This loader:
    1. Loads the base Pi0 weights (PaliGemma + action_out_proj + LoRA if present)
    2. Initializes task_classifier randomly (auxiliary head)

    Note: Unlike MoE, the action_out_proj is compatible and can be loaded from checkpoint.
    Only the task_classifier is new and needs random initialization.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # Only skip task_classifier (new auxiliary head)
        # action_out_proj is compatible and will be loaded
        skip_regex = ".*(task_classifier).*"
        missing_regex = ".*(lora|task_classifier).*"

        return _merge_params(loaded_params, params, missing_regex=missing_regex, skip_regex=skip_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _flatten_dict_with_int_keys(d, parent_key='', sep='/'):
    """Flatten a nested dict, converting integer keys to strings."""
    items = []
    for k, v in d.items():
        # Convert integer keys to strings
        new_key = f"{parent_key}{sep}{str(k)}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict_with_int_keys(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict_with_int_keys(d, sep='/'):
    """Unflatten a dict, converting numeric string keys back to integers where appropriate."""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for i, part in enumerate(parts[:-1]):
            # Try to convert to int if it looks like a number
            try:
                part_key = int(part)
            except ValueError:
                part_key = part

            if part_key not in current:
                current[part_key] = {}
            current = current[part_key]

        # Handle the last part
        last_part = parts[-1]
        try:
            last_key = int(last_part)
        except ValueError:
            last_key = last_part

        current[last_key] = value
    return result


def _merge_params(
    loaded_params: at.Params, params: at.Params, *, missing_regex: str, skip_regex: str | None = None
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.
        skip_regex: Optional regex pattern for keys to skip when loading from checkpoint.

    Returns:
        A new dictionary with the merged parameters.
    """
    # Use custom flatten that handles integer keys (e.g., from nnx.Sequential)
    flat_ref = _flatten_dict_with_int_keys(params, sep="/")
    flat_loaded = _flatten_dict_with_int_keys(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    skip_pattern = re.compile(skip_regex) if skip_regex else None

    for k, v in flat_loaded.items():
        # Skip if matches skip pattern
        if skip_pattern and skip_pattern.fullmatch(k):
            logger.info(f"Skipping checkpoint key (skip_regex match): {k}")
            continue

        if k in flat_ref:
            if v.dtype == flat_ref[k].dtype:
                result[k] = v
            else:
                logger.warning(f"{k} has dtype {v.dtype} but reference has dtype {flat_ref[k].dtype}")
                result[k] = v.astype(flat_ref[k].dtype)
    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            logger.info(f"Initializing randomly (missing from checkpoint): {k}")
            result[k] = flat_ref[k]

    return _unflatten_dict_with_int_keys(result, sep="/")
