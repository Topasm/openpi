"""
Hierarchical Pi0 model for multi-task learning of high-level skills and low-level actions.

This model extends Pi0 to predict both:
1. High-level skill annotations (text generation via language modeling)
2. Low-level action trajectories (flow matching, like original Pi0)

The model is trained with a multi-task loss:
    L_total = λ_skill * L_skill + λ_action * L_action

where:
- L_skill: Cross-entropy loss for next-token prediction of skill text
- L_action: Flow matching loss for action trajectory prediction

Architecture:
- Shared vision encoder (SigLIP) and prefix embedding (Gemma PaliGemma expert)
- Skill prediction head: Uses PaliGemma expert for autoregressive text generation
- Action prediction head: Uses action expert (Gemma) for flow matching (same as Pi0)
"""

import dataclasses
import logging
from typing import Optional

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision and pi0.py"""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0HierarchicalConfig(pi0_config.Pi0Config):
    """
    Configuration for hierarchical Pi0 model.

    Additional parameters:
        skill_loss_weight: Weight for skill prediction loss (default: 10.0)
        action_loss_weight: Weight for action prediction loss (default: 1.0)
        max_skill_tokens: Maximum number of tokens for skill text (default: 64)
        use_hierarchical_tokenizer: Whether to use HierarchicalTokenizer with special tokens (Phase 3)
        vocab_size_override: Override vocabulary size (use when adding special tokens like <EOS_SKILL>)
    """

    skill_loss_weight: float = 10.0
    action_loss_weight: float = 1.0
    max_skill_tokens: int = 64  # Maximum length of skill text
    use_hierarchical_tokenizer: bool = False  # Phase 3: Enable HierarchicalTokenizer
    vocab_size_override: Optional[int] = None  # Override vocab size for special tokens

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Hierarchical":
        return Pi0Hierarchical(self, rngs=nnx.Rngs(rng))


class Pi0Hierarchical(_model.BaseModel):
    """
    Hierarchical Pi0 model that predicts both skills and actions.

    This model extends the standard Pi0 architecture with an additional skill prediction head.
    The vision encoder and prefix embeddings are shared between both tasks.

    Forward pass structure:
        1. Encode observation (images + text prompt) → prefix tokens
        2. For skill prediction: Autoregressively generate skill tokens using PaliGemma
        3. For action prediction: Use action expert with flow matching (standard Pi0)
        4. Compute combined loss
    """

    def __init__(self, config: Pi0HierarchicalConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.skill_loss_weight = config.skill_loss_weight
        self.action_loss_weight = config.action_loss_weight
        self.max_skill_tokens = config.max_skill_tokens
        self.use_hierarchical_tokenizer = config.use_hierarchical_tokenizer

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Shared vision and language model
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Action prediction head (same as Pi0)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Skill prediction head: Use PaliGemma's vocabulary for text generation
        # The PaliGemma expert already has an output projection to vocabulary
        # We'll use the llm's unembedding layer (tied with embedding)
        if config.vocab_size_override is not None:
            self.vocab_size = config.vocab_size_override
            logger.info(f"Using custom vocabulary size: {self.vocab_size}")
            # Note: When using HierarchicalTokenizer with special tokens,
            # the embedding layer will need to be resized. This is handled
            # by calling resize_token_embeddings after model initialization.
        else:
            self.vocab_size = _gemma.PALIGEMMA_VOCAB_SIZE  # 257,152

    def embed_prefix(self, observation: _model.Observation):
        """
        Encode observation (images + text prompt) into prefix tokens.

        This is identical to Pi0's embed_prefix - we reuse the vision and language encoding.

        Args:
            observation: Observation containing images, state, and tokenized prompt

        Returns:
            prefix_tokens: Encoded tokens [B, prefix_len, embed_dim]
            prefix_mask: Attention mask [B, prefix_len]
            prefix_ar_mask: Autoregressive mask [prefix_len]
        """
        tokens = []
        input_mask = []
        ar_mask = []

        # Encode images
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            tokens.append(image_tokens)
            # Repeat mask for each image token
            input_mask.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # Add language tokens (must embed first!)
        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            # Full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    def embed_prefix_with_memory(
        self,
        observation: _model.Observation,
        memory_tokens: Optional[jnp.ndarray] = None,
        memory_mask: Optional[jnp.ndarray] = None
    ):
        """
        Encode observation with ProVideLLM-style memory cache (Phase 1).

        This extends embed_prefix by adding compressed past skill history as text tokens.
        Memory format: ["<L> skill0 </L>", "<L> skill1 </L>", ...]

        Args:
            observation: Current observation (images + text prompt)
            memory_tokens: Past skills encoded as tokens [B, num_memory_tokens]
            memory_mask: Mask for memory tokens [B, num_memory_tokens]

        Returns:
            prefix_tokens: [B, prefix_len + memory_len, embed_dim]
            prefix_mask: [B, prefix_len + memory_len]
            prefix_ar_mask: [prefix_len + memory_len]
        """
        tokens = []
        input_mask = []
        ar_mask = []

        # Add memory tokens FIRST (these are past completed skills)
        if memory_tokens is not None and memory_mask is not None:
            # Embed memory tokens using language model
            memory_embeddings = self.PaliGemma.llm(memory_tokens, method="embed")
            tokens.append(memory_embeddings)
            input_mask.append(memory_mask)
            # Memory tokens use full attention (bidirectional within memory)
            ar_mask += [False] * memory_embeddings.shape[1]

        # Encode current observation images
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            tokens.append(image_tokens)
            # Repeat mask for each image token
            input_mask.append(
                einops.repeat(
                    observation.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens attend to memory + each other
            ar_mask += [False] * image_tokens.shape[1]

        # Add current language tokens (task prompt)
        if observation.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(observation.tokenized_prompt_mask)
            # Full attention across memory + images + language
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    def embed_suffix_action(self, observation: _model.Observation, actions_t, time):
        """
        Embed noisy actions and time for the action expert (flow matching).

        This is identical to Pi0's embed_suffix.

        Args:
            observation: Observation with state
            actions_t: Noisy actions [B, action_horizon, action_dim]
            time: Diffusion time [B]

        Returns:
            action_expert_tokens: [B, action_horizon, embed_dim]
            suffix_mask: [B, action_horizon]
            suffix_ar_mask: [action_horizon]
            adarms_cond: Conditioning for AdaRMS (if pi05)
        """
        # Project actions to embedding space
        action_expert_tokens = self.action_in_proj(actions_t)

        # Add time embedding
        time_emb = posemb_sincos(time, action_expert_tokens.shape[-1], 1.0, 10000.0)

        if self.pi05:
            adarms_cond = nnx.gelu(self.time_mlp_in(time_emb))
            adarms_cond = self.time_mlp_out(adarms_cond)
        else:
            # For non-pi05 models, embed time+state and add directly to tokens
            state_emb = self.state_proj(observation.state)
            time_state_emb = jnp.concatenate([time_emb, state_emb], axis=-1)
            time_cond = nnx.gelu(self.action_time_mlp_in(time_state_emb))
            time_cond = self.action_time_mlp_out(time_cond)
            action_expert_tokens = action_expert_tokens + time_cond[:, None, :]
            # Set adarms_cond to None for non-pi05 models
            adarms_cond = None

        # Add positional embeddings for each action token
        positions = jnp.arange(self.action_horizon)
        pos_emb = posemb_sincos(positions, action_expert_tokens.shape[-1], 1.0, 10000.0)
        action_expert_tokens = action_expert_tokens + pos_emb[None, :, :]

        suffix_mask = jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_)
        # Action tokens use causal attention (first is True, rest are False for prefix-lm style)
        suffix_ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1), dtype=jnp.bool_)

        return action_expert_tokens, suffix_mask, suffix_ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        skill_tokens: Optional[at.Int[at.Array, "b skill_len"]] = None,
        skill_mask: Optional[at.Bool[at.Array, "b skill_len"]] = None,
        memory_tokens: Optional[at.Int[at.Array, "b memory_len"]] = None,
        memory_mask: Optional[at.Bool[at.Array, "b memory_len"]] = None,
        *,
        train: bool = False
    ) -> tuple[at.Float[at.Array, ""], dict]:
        """
        Compute hierarchical multi-task loss with optional memory (Phases 0/1/3).

        Args:
            rng: Random key
            observation: Observation with images, state, tokenized prompt
            actions: Ground truth actions [B, action_horizon, action_dim]
            skill_tokens: Ground truth skill tokens [B, skill_len] (optional)
            skill_mask: Mask for skill tokens [B, skill_len] (optional)
            memory_tokens: Past skills memory tokens [B, memory_len] (optional, Phase 1/3)
            memory_mask: Mask for memory tokens [B, memory_len] (optional, Phase 1/3)
            train: Whether in training mode

        Returns:
            total_loss: Scalar loss (weighted combination)
            loss_dict: Dictionary with individual loss components:
                - "action_loss": Flow matching loss for actions
                - "skill_loss": Cross-entropy loss for skills (if provided)
                - "total_loss": Combined loss
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # ===== Action Loss (Flow Matching) =====
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Forward pass: prefix (with optional memory) + action suffix
        # Phase 0 (no memory) vs Phase 1/3 (with memory)
        if memory_tokens is not None and memory_mask is not None:
            # Phase 1/3: With past skills memory
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_with_memory(
                observation, memory_tokens, memory_mask
            )
        else:
            # Phase 0: No memory
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        action_suffix_tokens, action_suffix_mask, action_suffix_ar_mask, adarms_cond = self.embed_suffix_action(
            observation, x_t, time
        )

        # Concatenate prefix and action suffix
        input_mask = jnp.concatenate([prefix_mask, action_suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, action_suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # Run through LLM
        # Only use AdaRMS conditioning if pi05 is enabled
        # When pi05=False, the conditioning is already added to action_expert_tokens (line 306)
        (prefix_out, action_suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, action_suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond if self.pi05 else None]
        )

        # Compute action loss
        v_t = self.action_out_proj(action_suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t))

        loss_dict = {
            "action_loss": action_loss,
        }

        # ===== Skill Loss (Language Modeling) =====
        if skill_tokens is not None and skill_mask is not None:
            # Forward pass: prefix only (for skill prediction)
            # We need to get the hidden states from PaliGemma expert (index 0)
            # and project them to vocabulary space for next-token prediction

            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

            # Run only prefix through PaliGemma expert
            # LLM returns a list with two elements [paligemma_output, action_expert_output]
            # Since we pass [prefix_tokens, None], action_expert_output will be None
            (prefix_hidden, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
                adarms_cond=[None, None]
            )

            # Project hidden states to vocabulary using Gemma's embedder.decode()
            # This converts [B, seq_len, hidden_dim] -> [B, seq_len, vocab_size]
            # The embedder is wrapped in ToNNX, so we manually perform the decode operation
            # decode(x) = dot(x, embedding_table.T)
            embedding_table = self.PaliGemma.llm.embedder['input_embedding']
            vocab_logits = jnp.dot(prefix_hidden, embedding_table.value.T)
            # Shape: [B, prefix_len, vocab_size=257152]

            # Extract logits for skill tokens
            # The skill tokens should be at the end of the prefix
            # We predict skill_tokens[1:] from skill_tokens[:-1] (teacher forcing)
            skill_len = skill_tokens.shape[1]

            # Get logits for skill prediction positions
            skill_logits = vocab_logits[:, -skill_len:, :]  # [B, skill_len, vocab_size]

            # Shift for next-token prediction
            targets = skill_tokens[:, 1:]  # [B, skill_len-1]
            logits = skill_logits[:, :-1, :]  # [B, skill_len-1, vocab_size]
            mask = skill_mask[:, 1:]  # [B, skill_len-1]

            # Compute cross-entropy loss
            import optax
            token_losses = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits,
                labels=targets
            )  # [B, skill_len-1]

            # Masked average
            masked_loss = token_losses * mask
            skill_loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(mask), 1.0)

            loss_dict["skill_loss"] = skill_loss
        else:
            skill_loss = jnp.array(0.0)
            loss_dict["skill_loss"] = skill_loss

        # ===== Total Loss =====
        total_loss = (
            self.action_loss_weight * action_loss +
            self.skill_loss_weight * skill_loss
        )

        loss_dict["total_loss"] = total_loss

        return total_loss, loss_dict

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        Sample actions using flow matching (identical to Pi0).

        This only samples actions, not skills. Skill generation would require
        a separate sampling method (autoregressive text generation).

        Args:
            rng: Random key
            observation: Observation
            num_steps: Number of ODE integration steps
            noise: Initial noise (optional)

        Returns:
            Sampled actions [B, action_horizon, action_dim]
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Fill KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_action(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def infer_with_memory(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        memory_text: Optional[str] = None,
        tokenizer = None,
        num_steps: int = 10,
        generate_skill: bool = False,
        max_skill_length: int = 32
    ) -> tuple[_model.Actions, Optional[str]]:
        """
        Inference with dynamic memory support for Phase 3.

        This method takes vision+text memory input, generates actions,
        and optionally generates skill predictions for EOS detection.

        Args:
            rng: Random key
            observation: Current observation (images, state, task prompt)
            memory_text: Text memory from past completed skills (optional)
            tokenizer: Tokenizer for memory text and skill generation (optional)
            num_steps: Number of ODE integration steps for action sampling
            generate_skill: Whether to generate skill text (for EOS detection)
            max_skill_length: Maximum length for skill generation

        Returns:
            (actions, skill_text) tuple:
                - actions: Predicted actions [B, action_horizon, action_dim]
                - skill_text: Generated skill text (if generate_skill=True), else None

        Example:
            # Phase 3 inference with memory
            memory_text = "<PAST_SKILL>{...}</PAST_SKILL> <PAST_SKILL>{...}</PAST_SKILL>"
            actions, skill_text = model.infer_with_memory(
                rng, observation, memory_text=memory_text,
                tokenizer=tokenizer, generate_skill=True
            )
            if skill_text.endswith("<EOS_SKILL>"):
                # Skill completed!
                pass
        """
        action_rng, skill_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(None, observation, train=False)

        # Tokenize memory text if provided
        memory_tokens = None
        memory_mask = None
        if memory_text and tokenizer is not None:
            # Tokenize the memory text
            tokens, mask = tokenizer.tokenize(memory_text)
            memory_tokens = jnp.array(tokens)[None, :]  # Add batch dimension
            memory_mask = jnp.array(mask)[None, :]

        # Embed prefix with optional memory
        if memory_tokens is not None and memory_mask is not None:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_with_memory(
                observation, memory_tokens, memory_mask
            )
        else:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # ===== Action Generation (Flow Matching) =====
        # Use the existing sample_actions logic but with memory-aware prefix
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(action_rng, (batch_size, self.action_horizon, self.action_dim))

        # Fill KV cache with prefix (including memory)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_action(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_expanded = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_expanded, suffix_attn_mask], axis=-1)

            positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        # ===== Skill Generation (Optional) =====
        skill_text = None
        if generate_skill and tokenizer is not None:
            # Generate skill text autoregressively using the VQ head
            # Start with BOS token and sample until EOS or max length
            
            # Get embedding table for decoding logits -> tokens
            embedding_table = self.PaliGemma.llm.embedder['input_embedding']
            
            # Initialize with empty skill (will be filled autoregressively)
            batch_size = observation.state.shape[0]
            generated_tokens = []
            
            # Use prefix hidden states as context
            prefix_attn_mask_gen = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions_gen = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_hidden, _), kv_cache_skill = self.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask_gen,
                positions=positions_gen
            )

            # Track the current KV cache size (starts with prefix length)
            current_cache_len = jnp.sum(prefix_mask, axis=-1)[0]  # Scalar: total prefix tokens

            # Autoregressively generate skill tokens
            for step_idx in range(max_skill_length):
                # Project last hidden state to vocabulary
                # vocab_logits shape: [B, vocab_size]
                vocab_logits = jnp.dot(prefix_hidden[:, -1, :], embedding_table.value.T)

                # Sample next token (greedy for now - could use temperature/top-k)
                next_token = jnp.argmax(vocab_logits, axis=-1)  # [B]
                generated_tokens.append(int(next_token[0]))

                # Check if EOS token generated
                EOS_SKILL_TOKEN_ID = 257153
                if int(next_token[0]) == EOS_SKILL_TOKEN_ID:
                    break

                # Embed next token and continue
                next_token_2d = next_token[:, None]  # [B, 1]
                # Manual embedding lookup: next_embedding = embedding_table[next_token_2d]
                next_embedding = embedding_table.value[next_token_2d]  # [B, 1, hidden_dim]

                # Create attention mask following the sample_actions pattern
                # The new token can attend to all previous tokens in KV cache
                next_mask = jnp.ones((batch_size, 1), dtype=jnp.bool_)  # [B, 1]
                next_ar_mask = jnp.array([True], dtype=jnp.bool_)  # [1] - causal for new token
                next_attn_mask = make_attn_mask(next_mask, next_ar_mask)  # [B, 1, 1]

                # Expand the KV cache mask to cover all cached tokens
                # KV cache now has: prefix + all previously generated tokens
                kv_cache_mask = einops.repeat(prefix_mask, "b p -> b 1 p")  # [B, 1, prefix_len]

                # For previously generated tokens, create mask
                if step_idx > 0:
                    # We have generated step_idx tokens already in cache
                    past_generated_mask = jnp.ones((batch_size, 1, step_idx), dtype=jnp.bool_)
                    kv_cache_mask = jnp.concatenate([kv_cache_mask, past_generated_mask], axis=-1)

                # Full mask: [kv_cache_mask, next_attn_mask]
                full_mask = jnp.concatenate([kv_cache_mask, next_attn_mask], axis=-1)
                # Shape: [B, 1, current_cache_len + step_idx + 1]

                # Position for the new token
                next_position = jnp.array([[int(current_cache_len) + step_idx]], dtype=jnp.int32)  # [B, 1]

                # Forward pass to get next hidden state
                # Use PaliGemma expert (index 0) for skill generation
                (next_hidden, _), kv_cache_skill = self.PaliGemma.llm(
                    [next_embedding, None],
                    mask=full_mask,
                    positions=next_position,
                    kv_cache=kv_cache_skill
                )

                # Update prefix_hidden for next iteration
                prefix_hidden = jnp.concatenate([prefix_hidden, next_hidden], axis=1)
            
            # Decode generated tokens to text
            if generated_tokens:
                if tokenizer is not None:
                    skill_text = tokenizer.decode(generated_tokens)
                else:
                    # Fallback: create minimal representation with token IDs
                    skill_text = f"{{\"tokens\": {generated_tokens}}}"
                    # Check if last token was EOS
                    if generated_tokens[-1] == 257153:
                        skill_text += "<EOS_SKILL>"
            else:
                skill_text = None

        return actions, skill_text


# Factory function to create hierarchical config from standard Pi0 config
def hierarchical_config_from_pi0(
    pi0_cfg: pi0_config.Pi0Config,
    skill_loss_weight: float = 1.0,
    action_loss_weight: float = 1.0,
    max_skill_tokens: int = 64
) -> Pi0HierarchicalConfig:
    """
    Create a hierarchical config from a standard Pi0 config.

    Args:
        pi0_cfg: Standard Pi0Config
        skill_loss_weight: Weight for skill prediction loss
        action_loss_weight: Weight for action prediction loss
        max_skill_tokens: Maximum number of skill tokens

    Returns:
        Pi0HierarchicalConfig with same parameters as pi0_cfg plus hierarchical settings
    """
    return Pi0HierarchicalConfig(
        paligemma_variant=pi0_cfg.paligemma_variant,
        action_expert_variant=pi0_cfg.action_expert_variant,
        action_dim=pi0_cfg.action_dim,
        action_horizon=pi0_cfg.action_horizon,
        max_token_len=pi0_cfg.max_token_len,
        dtype=pi0_cfg.dtype,
        pi05=pi0_cfg.pi05,
        skill_loss_weight=skill_loss_weight,
        action_loss_weight=action_loss_weight,
        max_skill_tokens=max_skill_tokens,
    )
