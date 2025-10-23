"""Policy wrapper that extracts and returns VLM embeddings along with actions."""

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils


class EmbeddingPolicy(_base_policy.BasePolicy):
    """Policy that extracts and returns VLM embeddings along with actions.
    
    This is useful for visualization and analysis of the VLM embedding space.
    """

    def __init__(
        self,
        model: _pi0.Pi0,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: list[_transforms.DataTransformFn] = None,
        output_transforms: list[_transforms.DataTransformFn] = None,
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._model = model
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._embed_prefix = nnx_utils.module_jit(model.embed_prefix)
        self._input_transform = _transforms.compose(transforms or [])
        self._output_transform = _transforms.compose(output_transforms or [])
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        observation = _model.Observation.from_dict(inputs)
        observation = _model.preprocess_observation(None, observation, train=False)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        
        # Sample actions only (embeddings are extracted separately via extract_embedding())
        actions = self._sample_actions(sample_rng, observation, **self._sample_kwargs)
        
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs
    
    def extract_embedding(self, obs: dict) -> dict:
        """Extract only VLM embeddings without computing actions.
        
        This is useful for collecting embeddings at every step without the overhead
        of action inference.
        
        Args:
            obs: Observation dictionary
            
        Returns:
            Dictionary containing:
                - vlm_embedding: VLM embeddings (seq_len, emb_dim)
                - vlm_mask: Mask for valid tokens (seq_len,)
        """
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        observation = _model.Observation.from_dict(inputs)
        observation = _model.preprocess_observation(None, observation, train=False)

        # Extract VLM embeddings
        prefix_tokens, prefix_mask, prefix_ar_mask = self._embed_prefix(observation)
        
        outputs = {
            "vlm_embedding": prefix_tokens,  # VLM embeddings (B, seq_len, emb_dim)
            "vlm_mask": prefix_mask,  # Mask for valid tokens
        }
        
        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

