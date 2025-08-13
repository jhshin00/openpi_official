import dataclasses
import logging

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

from openpi.models.multimodal_critic import PerceiverCritic

PALIGEMMA_VOCAB_SIZE = 257_152

# action_stats = {"mean": [
#                 0.02050294354557991,
#                 0.09887412935495377,
#                 -0.03913182020187378,
#                 0.0005592110683210194,
#                 0.006814933847635984,
#                 -0.00647358363494277,
#                 -0.036161474883556366,
#                 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
#             "std": [
#                 0.2913571000099182,
#                 0.3501919209957123,
#                 0.4437326192855835,
#                 0.03835552558302879,
#                 0.06355606019496918,
#                 0.07581629604101181,
#                 0.9993459582328796,
#                 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
#             "q01": [
#                 -0.7046250000000001,
#                 -0.747375,
#                 -0.9375,
#                 -0.10841571986675261,
#                 -0.16395,
#                 -0.2037045055747032,
#                 -1.0,
#                 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
#             "q99": [
#                 0.8812500000000001,
#                 0.86775,
#                 0.937125,
#                 0.12856071442365646,
#                 0.18419999999999992,
#                 0.3353504996180534,
#                 0.9996,
#                 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#             }

logger = logging.getLogger("openpi")

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
class Pi0FQLConfig(_model.BaseModelConfig):
    discount: float = 0.99
    normalize_q_loss: bool = False
    vocab_size = PALIGEMMA_VOCAB_SIZE
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FQL":
        rngs, rngs_actor, rngs_critic, rngs_critic_target = jax.random.split(rng, 4)
        return Pi0FQL(self, rngs_actor=nnx.Rngs(rngs_actor), rngs_critic=nnx.Rngs(rngs_critic), rngs_critic_target=nnx.Rngs(rngs_critic_target))

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

class Pi0FQL(_model.BaseModel):
    def __init__(self, config: Pi0FQLConfig, rngs_actor: nnx.Rngs, rngs_critic: nnx.Rngs, rngs_critic_target: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

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

        # 2) critic network 초기화
        obs_spec, _ = config.inputs_spec(batch_size=2)
        state_dim = obs_spec.state.shape[-1]

        self.critic = PerceiverCritic(
            embed_dim=512,
            num_heads=8,
            mlp_dim=1024,
            dropout_rate=0.1,
            num_latents=128,
            num_self_attn_layers=6,
            img_patch_size=16,
            img_channels=3,
            state_dim=state_dim,
            action_dim=config.action_dim,
            action_seq_len=config.action_horizon,
            vocab_size=config.vocab_size,
            rngs=rngs_critic,
        )
        self.critic_target = PerceiverCritic(
            embed_dim=512,
            num_heads=8,
            mlp_dim=1024,
            dropout_rate=0.1,
            num_latents=128,
            num_self_attn_layers=6,
            img_patch_size=16,
            img_channels=3,
            state_dim=state_dim,
            action_dim=config.action_dim,
            action_seq_len=config.action_horizon,
            vocab_size=config.vocab_size,
            rngs=rngs_critic_target,
        )

        self.discount = config.discount
        self.normalize_q_loss = config.normalize_q_loss

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
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        rewards: at.Float[at.Array, "b"],
        terminals: at.Bool[at.Array, "b"],
        next_observation: _model.Observation,
        *,
        train: bool = False,
    ) -> tuple[
        at.Float[at.Array, ""],
        tuple[at.Float[at.Array, ""], at.Float[at.Array, ""], at.Float[at.Array, ""], at.Float[at.Array, ""]]
    ]:
        """
        returns: scalar loss = critic_loss + actor_loss

        critic_loss = E[(Q(s,a) - (r + gamma * Q(s',a')))²]
        actor_loss  = flow_loss (||v-u||²) - E[ Q(s, a_sampled) ]
        """

        critic_loss = self.critic_loss(rng, observation, actions, rewards, terminals, next_observation, train=train)
        actor_loss, (flow_loss, q_loss) = self.actor_loss(rng, observation, actions, rewards, terminals, next_observation, train=train)

        # 최종 joint loss
        return critic_loss + actor_loss, (critic_loss, actor_loss, flow_loss, q_loss)
    
    @override
    def flow_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = True
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
        
        # jax.debug.print(
        #     "\n"
        #     "acts  μ {a_mu:.3f} σ {a_std:.3f} max| {a_max:.3f}\n"
        #     "noise μ {n_mu:.3f} σ {n_std:.3f} max| {n_max:.3f}\n"
        #     "x_t   μ {x_mu:.3f} σ {x_std:.3f} max| {x_max:.3f}\n"
        #     "u_t   μ {u_mu:.3f} σ {u_std:.3f} max| {u_max:.3f}\n"
        #     "v_t   μ {v_mu:.3f} σ {v_std:.3f} max| {v_max:.3f}\n"
        #     "‖v-u‖² μ {d_mu:.3f} σ {d_std:.3f} max {d_max:.3f}",
        #     a_mu=jnp.mean(actions),   a_std=jnp.std(actions),   a_max=jnp.max(jnp.abs(actions)),
        #     n_mu=jnp.mean(noise),     n_std=jnp.std(noise),     n_max=jnp.max(jnp.abs(noise)),
        #     x_mu=jnp.mean(x_t),       x_std=jnp.std(x_t),       x_max=jnp.max(jnp.abs(x_t)),
        #     u_mu=jnp.mean(u_t),       u_std=jnp.std(u_t),       u_max=jnp.max(jnp.abs(u_t)),
        #     v_mu=jnp.mean(v_t),       v_std=jnp.std(v_t),       v_max=jnp.max(jnp.abs(v_t)),
        #     d_mu=jnp.mean((v_t-u_t)**2),
        #     d_std=jnp.std((v_t-u_t)**2),
        #     d_max=jnp.max((v_t-u_t)**2),
        # )
        
        # act_mu  = jnp.mean(actions, axis=(0, 1))
        # act_std = jnp.std(actions,  axis=(0, 1))

        # # 8차원씩 슬라이스 출력
        # for i in range(0, self.action_dim, 8):
        #     jax.debug.print(
        #         "acts μ [{:02d}:{:02d}] {}", i, i+8, act_mu[i:i+8]
        #     )
        #     jax.debug.print(
        #         "acts σ [{:02d}:{:02d}] {}", i, i+8, act_std[i:i+8]
        #     )
        
        # for h in range(5):
        #     jax.debug.print("a[0,{h}] = {}", actions[0, h, :32], h=h)
        #     jax.debug.print("u[0,{h}] = {}", u_t[0, h, :32], h=h)
        #     jax.debug.print("v[0,{h}] = {}", v_t[0, h, :32], h=h)

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @at.typecheck
    def critic_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        rewards: at.Float[at.Array, "b"],
        terminals: at.Bool[at.Array, "b"],
        next_observation: _model.Observation,
        *,
        train: bool = True,
    ) -> at.Float[at.Array, ""]:
        
        # jax.debug.print("actions in critic loss : {}", actions[0])
        
        rng, rng_sample, rng_obs = jax.random.split(rng, 3)

        # observation = _model.preprocess_observation(rng_obs, observation, train=train)

        a_next = self.sample_actions(rng_sample, next_observation)
        
        q_next = self.critic_target(_model.preprocess_observation(rng_obs, next_observation, train=train), a_next, train=train)
        # q_next = self.critic_target(next_observation, a_next, train=train)

        mask = 1.0 - terminals.astype(jnp.float32)
        target_q = rewards + self.discount * mask * q_next
        target_q = jax.lax.stop_gradient(target_q)

        q_pred = self.critic(_model.preprocess_observation(rng_obs, observation, train=train), actions, train=train)
        # q_pred = self.critic(observation, actions, train=train)
        critic_loss = jnp.mean((q_pred - target_q) ** 2)

        jax.debug.print(
            "r={r:.3f}, q_next={qn:.3f}, target_q={tq:.3f}, q_pred={qp:.3f}",
            r=rewards[0], qn=q_next[0], tq=target_q[0], qp=q_pred[0]
        )
        jax.debug.print("q_next: mean={:.3f}, max={:.3f}", 
                jnp.mean(q_next), jnp.max(q_next))
        jax.debug.print("q_pred: mean={:.3f}, max={:.3f}", 
                jnp.mean(q_pred), jnp.max(q_pred))
        
        # jax.debug.print("terminal : value {t} / shape {t.shape}\n",t=terminals)
        # jax.debug.print("mask : value {mask} / shape {mask.shape}\n", mask=mask)
        # jax.debug.print("actions : value {t} / shape {t.shape}\n",t=actions)
        # jax.debug.print("a_next : value {t} / shape {t.shape}\n",t=a_next)
        # jax.debug.print("q_next : value {q_next} / shape {q_next.shape}\n", q_next = q_next)
        # jax.debug.print("target_q : value {target_q} / shape {target_q.shape}\n", target_q = target_q)

        return critic_loss

    # #########################################

    @at.typecheck
    def critic_chunk_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        rewards: at.Float[at.Array, "b H"],
        terminals: at.Bool[at.Array, "b"],
        next_observation: _model.Observation,
        reward_is_pad: at.Bool[at.Array, "b H"],
        # actions_is_pad: at.Bool[at.Array, "b H"] | None = None,
        *,
        train: bool = True,
    ) -> at.Float[at.Array, ""]:
        
        H = self.action_horizon
        gamma_vec = jnp.power(self.discount, jnp.arange(H))[None, :] # (1,H)
        
        # Use reward_is_pad to mask out padding rewards
        valid_rewards = rewards * (1 - reward_is_pad.astype(jnp.float32))
        G_n = jnp.sum(gamma_vec * valid_rewards, axis=1)
        
        rng, rng_sample, rng_obs = jax.random.split(rng, 3)

        # critic loss에서는 Q-guided sampling을 사용하지 않음
        # (target Q값을 부풀리는 것을 방지하기 위해)
        a_next = self.sample_actions(rng_sample, next_observation, use_q_guidance=False)
        
        q_next = self.critic_target(_model.preprocess_observation(rng_obs, next_observation, train=train), a_next, train=train)

        # Fix terminal masking: terminals is (B,) and should be applied to each batch
        # If terminal=True, we don't bootstrap (mask=0)
        # If terminal=False, we bootstrap with γ^H * Q(s_{t+H}, a_{t+H})
        mask = (1.0 - terminals.astype(jnp.float32))  # (B,)
        target_q = G_n + (self.discount ** H) * mask * q_next
        target_q = jax.lax.stop_gradient(target_q)

        q_pred = self.critic(_model.preprocess_observation(rng_obs, observation, train=train), actions, train=train)

        critic_loss = jnp.mean((q_pred - target_q) ** 2)

        return critic_loss
    
    @at.typecheck
    def actor_chunk_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        rewards: at.Float[at.Array, "b H"],
        terminals: at.Bool[at.Array, "b"],
        next_observation: _model.Observation,
        # actions_is_pad: at.Bool[at.Array, "b H"] | None = None,
        *,
        train: bool = True,
    ) -> tuple[
        at.Float[at.Array, ""],
        tuple[at.Float[at.Array, ""], at.Float[at.Array, ""]]
    ]:
        rng, rng_phi, rng_obs = jax.random.split(rng, 3)
        chunk_flow_loss = self.flow_loss(rng_phi, observation, actions, train=train)
        flow_loss = jnp.mean(chunk_flow_loss)
        
        # # Q값 제거: BPTT 불안정성 방지를 위해 flow_loss만 사용
        # # Q-guided sampling은 sample_actions에서 처리
        # q_loss = jnp.array(0.0)  # Placeholder for compatibility
        
        # return flow_loss, (flow_loss, q_loss)


        # Q-loss with gradient detaching to prevent BPTT instability
        # Detach critic gradients to avoid backprop through critic
        a_sampled = self.sample_actions(rng_phi, observation, use_q_guidance=False)
        q_val = self.critic(_model.preprocess_observation(rng_obs, observation, train=train), a_sampled, train=train)
        q_loss = -jnp.mean(q_val)
        
        return flow_loss + q_loss, (flow_loss, q_loss)


    # @at.typecheck
    # def critic_loss(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     actions: _model.Actions,
    #     rewards: at.Float[at.Array, "b"],
    #     terminals: at.Bool[at.Array, "b"],
    #     next_observation: _model.Observation,
    #     *,
    #     train: bool = True,
    # ) -> at.Float[at.Array, ""]:
        
    #     rng, rng_sample, rng_obs = jax.random.split(rng, 3)

    #     a_next = self.sample_actions(rng_sample, next_observation)
        
    #     q_next_1 = self.critic(_model.preprocess_observation(rng_obs, next_observation, train=train), a_next, train=train)
    #     q_next_2 = self.critic_target(_model.preprocess_observation(rng_obs, next_observation, train=train), a_next, train=train)
    #     q_next = jnp.minimum(q_next_1, q_next_2)

    #     mask = 1.0 - terminals.astype(jnp.float32)
    #     target_q = rewards + self.discount * mask * q_next
    #     target_q = jax.lax.stop_gradient(target_q)

    #     q_pred = self.critic(_model.preprocess_observation(rng_obs, observation, train=train), actions, train=train)

    #     critic_loss = jnp.mean((q_pred - target_q) ** 2)

    #     return critic_loss
    
    # @at.typecheck
    # def critic_target_loss(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     actions: _model.Actions,
    #     rewards: at.Float[at.Array, "b"],
    #     terminals: at.Bool[at.Array, "b"],
    #     next_observation: _model.Observation,
    #     *,
    #     train: bool = True,
    # ) -> at.Float[at.Array, ""]:
        
    #     rng, rng_sample, rng_obs = jax.random.split(rng, 3)

    #     a_next = self.sample_actions(rng_sample, next_observation)
        
    #     q_next_1 = self.critic(_model.preprocess_observation(rng_obs, next_observation, train=train), a_next, train=train)
    #     q_next_2 = self.critic_target(_model.preprocess_observation(rng_obs, next_observation, train=train), a_next, train=train)
    #     q_next = jnp.minimum(q_next_1, q_next_2)

    #     mask = 1.0 - terminals.astype(jnp.float32)
    #     target_q = rewards + self.discount * mask * q_next
    #     target_q = jax.lax.stop_gradient(target_q)

    #     q_pred = self.critic_target(_model.preprocess_observation(rng_obs, observation, train=train), actions, train=train)

    #     critic_loss = jnp.mean((q_pred - target_q) ** 2)

    #     return critic_loss
    

    # #########################################
    
    @at.typecheck
    def actor_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        rewards: at.Float[at.Array, "b H"],
        terminals: at.Bool[at.Array, "b"],
        next_observation: _model.Observation,
        *,
        train: bool = True,
    ) -> tuple[
        at.Float[at.Array, ""],
        tuple[at.Float[at.Array, ""], at.Float[at.Array, ""]]
    ]:
        
        # jax.debug.print("actions in actor loss : {}", actions[0,0,:])
        rng, rng_phi, rng_sample, rng_obs = jax.random.split(rng, 4)
        chunk_flow_loss = self.flow_loss(rng_phi, observation, actions, train=train)
        flow_loss = jnp.mean(chunk_flow_loss)
        # jax.debug.print("flow_loss : value {t} / shape {t.shape}\n",t=flow_loss)

        a_sampled = self.sample_actions(rng_sample, observation)

        q_val = self.critic(_model.preprocess_observation(rng_obs, observation, train=train), a_sampled, train=train)
        ## 얘는 그냥 observation?으로 해야하나
        q_loss = -jnp.mean(q_val)

        if self.normalize_q_loss:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q_val).mean())
            q_loss = lam * q_loss

        return flow_loss+q_loss, (flow_loss, q_loss)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        num_samples: int = 10,  # Q-guided sampling을 위한 샘플 수
        use_q_guidance: bool = True,  # Q-guided sampling 사용 여부
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        
        if not use_q_guidance:
            # 기존 방식: 단일 샘플 생성
            return self._sample_single_action(rng, observation, num_steps)
        
        # Q-guided sampling: 여러 샘플 생성 후 최고 Q값 선택
        rngs = jax.random.split(rng, num_samples + 1)
        rng, sample_rngs = rngs[0], rngs[1:]
        
        # 1. 여러 action 샘플 생성
        action_samples = jax.vmap(
            lambda r: self._sample_single_action(r, observation, num_steps)
        )(sample_rngs)  # (num_samples, batch_size, action_horizon, action_dim)
        
        # 2. 각 action의 Q값 계산
        q_values = jax.vmap(
            lambda a: self.critic(observation, a, train=False)
        )(action_samples)  # (num_samples, batch_size)
        
        # 3. 가장 높은 Q값의 action 선택
        best_indices = jnp.argmax(q_values, axis=0)  # (batch_size,)
        
        # 4. 선택된 action 반환
        batch_size = observation.state.shape[0]
        selected_actions = action_samples[best_indices, jnp.arange(batch_size)]
        
        return selected_actions
    
    # def _sample_single_action(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    # ) -> _model.Actions:
    #     """기존의 단일 action 샘플링 로직"""
    #     # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    #     # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]
    #     noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    #     # first fill KV cache with a forward pass of the prefix
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.actor.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    #     def step(carry):
    #         x_t, time = carry
    #         suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time, batch_size)
    #         )
    #         # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
    #         # other
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
    #         # prefix tokens
    #         prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
    #         # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    #         assert full_attn_mask.shape == (
    #             batch_size,
    #             suffix_tokens.shape[1],
    #             prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #         )
    #         # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    #         (prefix_out, suffix_out), _ = self.actor.PaliGemma.llm(
    #             [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
    #         )
    #         assert prefix_out is None
    #         v_t = self.actor.action_out_proj(suffix_out[:, -self.action_horizon :])

    #         return x_t + dt * v_t, time + dt

    #     def cond(carry):
    #         x_t, time = carry
    #         # robust to floating-point error
    #         return time >= -dt / 2

    #     x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    #     return x_0

    def _sample_single_action(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        """기존의 단일 action 샘플링 로직"""
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

        def step(i, x_t):
            time = 1 + i * dt
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

            return x_t + dt * v_t

        x_0 = jax.lax.fori_loop(0, num_steps, step, noise)
        return x_0

    # @override
    # def sample_actions(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     *,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    # ) -> _model.Actions:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    #     # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]
    #     noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    #     # first fill KV cache with a forward pass of the prefix
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.actor.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    #     def step(i, x_t):
    #         time = 1 + i * dt
    #         suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time, batch_size)
    #         )
    #         # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
    #         # other
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
    #         # prefix tokens
    #         prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
    #         # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    #         assert full_attn_mask.shape == (
    #             batch_size,
    #             suffix_tokens.shape[1],
    #             prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #         )
    #         # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    #         (prefix_out, suffix_out), _ = self.actor.PaliGemma.llm(
    #             [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
    #         )
    #         assert prefix_out is None
    #         v_t = self.actor.action_out_proj(suffix_out[:, -self.action_horizon :])

    #         return x_t + dt * v_t

    #     x_0 = jax.lax.fori_loop(0, num_steps, step, noise)
    #     return x_0
