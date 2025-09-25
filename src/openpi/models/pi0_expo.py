import dataclasses
import logging
from typing import Optional, Sequence

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from openpi.models.encoder import MultiEncoder
from openpi.models.critic import MultiStateActionValue
from openpi.models.edit_policy import TanhNormal
from openpi.models.temperature import Temperature

logger = logging.getLogger("openpi")

PALIGEMMA_VOCAB_SIZE = 257_152


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
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
class Pi0ExpoConfig(_model.BaseModelConfig):
    #TODO# jhshin

    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    
    # Edit actor loss parameters
    entropy_scale: float = 1.0
    target_entropy: float = -16.0  # -action_dim/2 for default action_dim=32
    edit_action_scale: float = 1.0
    initial_temperature: float = 1.0
    
    # RL parameters
    discount: float = 0.99
    n_base_samples: int = 2
    n_edit_samples: int = 1
    
    # Encoder parameters
    encoder_sharing: bool = False
    img_latent_dim: int = 512
    txt_latent_dim: int = 512
    state_dim: int = 256
    hidden_dim: int = 256
    out_dim: int = 256
    vocab_size: int = 257152
    image_keys: Optional[Sequence[str]] = None
    d_model: int = 256
    n_layers: int = 2
    kernel_size: int = 3
    dropout_rate: Optional[float] = None
    use_layer_norm: bool = False
    use_film_gate: bool = False

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Expo":
        rngs, rngs_actor, rngs_critic, rngs_target_critic, rngs_edit_actor = jax.random.split(rng, 5)
        return Pi0Expo(
            self,
            rngs_actor=nnx.Rngs(rngs_actor),
            rngs_critic=nnx.Rngs(rngs_critic),
            rngs_target_critic=nnx.Rngs(rngs_target_critic),
            rngs_edit_actor=nnx.Rngs(rngs_edit_actor),
        )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0Expo(_model.BaseModel):
    def __init__(
        self,
        config: Pi0ExpoConfig,
        rngs_actor: nnx.Rngs,
        rngs_critic: nnx.Rngs,
        rngs_target_critic: nnx.Rngs,
        rngs_edit_actor: nnx.Rngs,
    ):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # Store config attributes
        self.discount = config.discount
        self.n_base_samples = config.n_base_samples
        self.n_edit_samples = config.n_edit_samples
        self.entropy_scale = config.entropy_scale
        self.target_entropy = config.target_entropy
        self.edit_action_scale = config.edit_action_scale

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs_actor, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs_actor)

        self.actor = nnx.Dict(
            PaliGemma = nnx.Dict(llm=llm, img=img),
            state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs_actor),
            action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs_actor),
            action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs_actor),
            action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs_actor),
            action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs_actor),
        )


        if config.encoder_sharing:
            self.encoder = MultiEncoder(
                img_dim=config.img_latent_dim,
                txt_dim=config.txt_latent_dim,
                state_dim=config.state_dim,
                state_in_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                out_dim=config.out_dim,
                use_image=True,
                use_text=True,
                use_state=True,
                vocab_size=config.vocab_size,
                image_keys=config.image_keys,
                action_dim=config.action_dim,
                d_model=config.d_model,
                n_layers=config.n_layers,
                kernel_size=config.kernel_size,
                dropout_rate=config.dropout_rate,
                use_layer_norm=config.use_layer_norm,
                use_film_gate=config.use_film_gate,
                rngs=rngs_critic,
            )
            self.critic = MultiStateActionValue(base_cls=self.encoder, rngs=rngs_critic)
            self.edit_actor = TanhNormal(
                base_cls=self.encoder,
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                squash_tanh=True,
                rngs=rngs_edit_actor,
                edit_action_scale=config.edit_action_scale,
            )
        else:
            self.critic = MultiStateActionValue(
                    base_cls=MultiEncoder(
                    img_dim=config.img_latent_dim,
                    txt_dim=config.txt_latent_dim,
                    state_dim=config.state_dim,
                    state_in_dim=config.action_dim,
                    hidden_dim=config.hidden_dim,
                    out_dim=config.out_dim,
                    use_image=True,
                    use_text=True,
                    use_state=True,
                    vocab_size=config.vocab_size,
                    image_keys=config.image_keys,
                    action_dim=config.action_dim,
                    d_model=config.d_model,
                    n_layers=config.n_layers,
                    kernel_size=config.kernel_size,
                    dropout_rate=config.dropout_rate,
                    use_layer_norm=config.use_layer_norm,
                    use_film_gate=config.use_film_gate,
                    rngs=rngs_critic,
                ),
                rngs=rngs_critic,
            )
            self.edit_actor = TanhNormal(
                base_cls=MultiEncoder(
                    img_dim=config.img_latent_dim,
                    txt_dim=config.txt_latent_dim,
                    state_dim=config.state_dim,
                    state_in_dim=config.action_dim,
                    hidden_dim=config.hidden_dim,
                    out_dim=config.out_dim,
                    use_image=True,
                    use_text=True,
                    use_state=True,
                    vocab_size=config.vocab_size,
                    image_keys=config.image_keys,
                    action_dim=config.action_dim,
                    d_model=config.d_model,
                    n_layers=config.n_layers,
                    kernel_size=config.kernel_size,
                    dropout_rate=config.dropout_rate,
                    use_layer_norm=config.use_layer_norm,
                    use_film_gate=config.use_film_gate,
                    rngs=rngs_critic,
                ),
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                rngs=rngs_edit_actor,
            )

        self.target_critic = MultiStateActionValue(
            base_cls=MultiEncoder(
                img_dim=config.img_latent_dim,
                txt_dim=config.txt_latent_dim,
                state_dim=config.state_dim,
                state_in_dim=config.action_dim,
                hidden_dim=config.hidden_dim,
                out_dim=config.out_dim,
                use_image=True,
                use_text=True,
                use_state=True,
                vocab_size=config.vocab_size,
                image_keys=config.image_keys,
                action_dim=config.action_dim,
                d_model=config.d_model,
                n_layers=config.n_layers,
                kernel_size=config.kernel_size,
                dropout_rate=config.dropout_rate,
                use_layer_norm=config.use_layer_norm,
                use_film_gate=config.use_film_gate,
                rngs=rngs_target_critic,
            ),
            rngs=rngs_target_critic,
        )

        self.temp = Temperature(initial_temperature=config.initial_temperature)


    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.actor.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.actor.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.actor.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.actor.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.actor.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.actor.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.actor.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.actor.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.actor.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # 원래 pi0 compute_loss
    def actor_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.actor.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.actor.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @at.typecheck
    def critic_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        rewards: at.Float[at.Array, "b H"],
        next_observation: _model.Observation,
        masks: at.Bool[at.Array, "b"],
        *,
        train: bool = False,
        retrieval_actions: _model.Actions = None,
    ) -> at.Float[at.Array, ""]:

        rng, rng_sample, rng_obs1, rng_obs2 = jax.random.split(rng, 4)
        next_actions = self.sample_OTF_actions(rng_sample, next_observation, retrieval_actions=retrieval_actions)

        gamma_vec = jnp.power(self.discount, jnp.arange(self.action_horizon))[None, :]
        G_n = jnp.sum(gamma_vec * rewards, axis=1)

        next_q = self.target_critic(
            _model.preprocess_observation(rng_obs1, next_observation, train=False),
            next_actions,
            train=False,
        )

        target_q = G_n + (self.discount ** self.action_horizon) * masks.astype(jnp.float32) * next_q
        target_q = jax.lax.stop_gradient(target_q)

        pred_q = self.critic(
            _model.preprocess_observation(rng_obs2, observation, train=True),
            actions,
            train=True,
        )

        return jnp.mean((pred_q - target_q) ** 2)

    @at.typecheck
    def edit_actor_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        retrieval_actions: _model.Actions = None,
    ) -> tuple[at.Float[at.Array, ""], at.Float[at.Array, ""]]:

        rng, rng_obs = jax.random.split(rng)
        observation = _model.preprocess_observation(rng_obs, observation, train=train)

        # Sample actions from edit_actor
        edit_actions, log_probs = self.edit_actor.sample_actions(
            rng, observation, actions, return_logp=True, train=train, retrieval_actions=retrieval_actions
        )
        
        # Scale actions and adjust log probabilities
        # edit_actions = edit_actions * self.edit_action_scale
        # log_probs = log_probs - self.action_horizon * self.action_dim * jnp.log(self.edit_action_scale)

        log_probs = log_probs / (self.action_horizon)
        
        # Add to original actions
        final_actions = edit_actions + actions
        
        # Compute Q-values
        q = self.critic(observation, final_actions, train=False)
        
        # Compute loss
        alpha = jax.lax.stop_gradient(self.temp())
        loss = (self.entropy_scale * log_probs * alpha - q).mean()
        entropy = -log_probs.mean()
        
        return loss, entropy

    @at.typecheck
    def temperature_loss(
        self,
        entropy,
    ) -> at.Float[at.Array, ""]:

        loss = self.temp() * jax.lax.stop_gradient(entropy - self.target_entropy)
        return loss


    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.actor.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.actor.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.actor.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @at.typecheck
    def sample_OTF_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        train: bool = False,
        retrieved_actions : _model.Actions = None,
    ) -> _model.Actions:
        # [consts]
        H = self.action_horizon
        A = self.action_dim
        assert self.n_edit_samples <= self.n_base_samples, "n_edit_samples must be less than or equal to n_base_samples"

        # 1) Base actor N개 샘플
        rngs = jax.random.split(rng, self.n_base_samples + 2)
        rng, base_keys, edit_key = rngs[0], rngs[1:self.n_base_samples+1], rngs[-1]
        base_samples = jax.vmap(lambda k: self.sample_actions(k, observation, num_steps=num_steps))(base_keys)  # [N,B,H,A]
        base_samples = jnp.swapaxes(base_samples, 0, 1)                                                         # [B,N,H,A]

        
        rng, rng_obs = jax.random.split(rng)
        obs = _model.preprocess_observation(rng_obs, observation, train=False)
        
        # 2) Anchor 및 Edit 제안 (벡터화 1-shot)
        anchor_actions = base_samples[:, :self.n_edit_samples]                                                                     # [B,n,H,A]

        if self.n_edit_samples > 0:
            edit_keys_b = jax.random.split(edit_key, anchor_actions.shape[0])

            def sample_edits_for_batch(obs_b, anchors_n, key_b):
                # obs_b: 트리(단일 배치), anchors_n: [n,H,A]
                keys_n = jax.random.split(key_b, anchors_n.shape[0]) # [n]
                # 각 anchor에 대해 edit_actor 한 번씩 호출
                return jax.vmap(
                    lambda k, a: self.edit_actor.sample_actions(k, obs_b, a, n_samples=1, train=False)[0]
                )(keys_n, anchors_n) # [n,H,A]

            edit_props = jax.vmap(sample_edits_for_batch)(obs, anchor_actions, edit_keys_b) # [B,n,H,A]
            edit_samples = anchor_actions + edit_props

            action_candidates = jnp.concatenate([base_samples, edit_samples], axis=1) # [B,M,H,A], M=N+n
        
        else:
            action_candidates = base_samples # [B,N,H,A]

        if retrieved_actions is not None:
            action_candidates = jnp.concatenate([action_candidates, retrieved_actions], axis=1) # [B,M,H,A], M=N+n+K

        def eval_one(obs_b, acts_bm):
            return jax.vmap(
                lambda a: self.critic(obs_b, a, train=False)
            )(acts_bm).squeeze() # [M]

        

        q_values = jax.vmap(eval_one)(obs, action_candidates) # [B,M]
        best_idx = jnp.argmax(q_values, axis=1) # [B]
        best_actions = action_candidates[jnp.arange(action_candidates.shape[0]), best_idx, :, :] # [B,H,A]
        return best_actions

        
        

        # obs_edit_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats=self.n_edit_samples, axis=0), obs)             # [B*n,...]
        # anchors_flat = anchor_actions.transpose(1,0,2,3).reshape(-1, H, A)                                      # [B*n,H,A]

        # # edit_flat = self.edit_actor.sample_actions(edit_key, obs_edit_flat, anchors_flat, n_samples=1, train=train)  # [B*n,1,H,A]
        # # edit_flat = edit_flat[:, 0]                                                                               # [B*n,H,A]


        # edit_keys = jax.random.split(edit_key, anchors_flat.shape[0])
        # edit_flat = jax.vmap(
        #     lambda k, o, a: self.edit_actor.sample_actions(k, o, a, n_samples=1, train=train)[0]
        # )(edit_keys, obs_edit_flat, anchors_flat)
        
        # edit_props = edit_flat.reshape(self.n_edit_samples, -1, H, A).transpose(1,0,2,3)                                             # [B,n,H,A]

        # edit_samples = anchor_actions + self.edit_action_scale * edit_props                                                        # [B,n,H,A]

        # # 3) 후보 합치기
        # action_candidates = jnp.concatenate([base_samples, edit_samples], axis=1)                                 # [B,M,H,A], M=N+n

        # 4) Q 평가





        # actions_flat = action_candidates.reshape(-1, H, A)                                                       # [B*M,H,A]
        # obs_flat = jax.tree_util.tree_map(lambda x: jnp.repeat(x, repeats=action_candidates.shape[1], axis=0), obs)                 # [B*M,...]

        # q_flat = self.critic(obs_flat, actions_flat, train=train)                                                 # [B*M]
        # q_values = q_flat.reshape(action_candidates.shape[:2])                                                                           # [B,M]

        # # 5) argmax 선택
        # best_idx = jnp.argmax(q_values, axis=1)                                                                   # [B]
        # best_actions = jnp.take_along_axis(action_candidates, best_idx[:, None, None, None], axis=1)[:, 0]        # [B,H,A]
        # return best_actions