import logging
import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import einops

import openpi.models.model as _model
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi.critic")

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

class ResNetBlock(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, strides: tuple[int, int] = (1, 1), rngs=None, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=hidden_features,
            kernel_size=(3, 3),
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=hidden_features,
            out_features=hidden_features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(
            num_features=hidden_features,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(
            num_features=hidden_features,
            rngs=rngs,
        )
        if strides != (1, 1) or in_features != hidden_features:
            self.downsample = nnx.Conv(
                in_features=in_features,
                out_features=hidden_features,
                kernel_size=(1, 1),
                strides=strides,
                padding="SAME",
                rngs=rngs,
            )
        else:
            self.downsample = None

    def __call__(self, x: at.Float[at.Array, "b h w c"], train: bool = True) -> at.Float[at.Array, "b h' w' c"]:
        residual = x
        # conv1
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nnx.silu(x)
        # conv2
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return nnx.silu(residual + x)


class ResNetEncoder(nnx.Module):
    def __init__(
        self,
        in_channels: int = 3,
        block_channels: tuple[int, ...] = (64, 128, 256, 512),
        out_dim: int = 256,
        rngs=None,
        **kwargs,
    ):
        super().__init__()

        hidden_ch = 64
        self.conv_init = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_ch,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.bn_init = nnx.BatchNorm(
            num_features=hidden_ch,
            rngs=rngs,
        )

        self.num_blocks = len(block_channels)

        in_ch = hidden_ch
        for i, ch in enumerate(block_channels):
            stride = (1, 1) if i == 0 else (2, 2)

            block = ResNetBlock(
                in_features=in_ch,
                hidden_features=ch,
                strides=stride,
                rngs=rngs,
            )
            setattr(self, f"block_{i}", block)
            in_ch = ch

        self.proj = nnx.Linear(
            in_features=block_channels[-1],
            out_features=out_dim,
            rngs=rngs,
        )

        self.out_dim = out_dim

    def __call__(
        self,
        img: at.Float[at.Array, "b H W C"],
        train: bool = True,
    ) -> at.Float[at.Array, "b d"]:
        x = self.conv_init(img)  # (32, 128, 128, 64)
        x = self.bn_init(x, use_running_average=not train)
        x = nnx.silu(x)
        x = nnx.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding="SAME",
        )  # (32, 64, 64, 64)
        for i in range(self.num_blocks):
            x = getattr(self, f"block_{i}")(x, train=train)
        # (32,64,64,64) - (32,32,32,128) - (32,16,16,256) - (b,8,8,512)
        x = x.mean(axis=(1, 2))  # (32, 512)

        return self.proj(x)  # (32, 256)

PALIGEMMA_VOCAB_SIZE = 257_152

class Critic(nnx.Module):
    def __init__(
        self,
        prompt_embed_dim: int = 64,
        img_proj_dim: int = 256,
        state_dim: int = 32,
        act_dim: int = 32,
        act_horizon: int = 50,
        mlp_hidden: int = 256,
        dropout_rate: float = 0.1,
        vocab_size: int = PALIGEMMA_VOCAB_SIZE,
        num_heads: int = 4,
        attn_dim: int = 128,
        rngs=None,
    ):
        super().__init__()
        self.prompt_embed_dim = prompt_embed_dim
        self.img_proj_dim = img_proj_dim
        self.act_horizon = act_horizon
        self.mlp_hidden = mlp_hidden
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.img_enc = ResNetEncoder(
            in_channels=3,
            block_channels=(64, 128, 256, 512),
            out_dim=256,
            rngs=rngs,
        )
        self.prompt_embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=prompt_embed_dim,
            rngs=rngs,
        )
        self.prompt_proj = nnx.Linear(
            in_features=prompt_embed_dim,
            out_features=attn_dim,
            rngs=rngs,
        )
        self.img_proj = nnx.Linear(
            in_features=self.img_enc.out_dim,
            out_features=attn_dim,
            rngs=rngs,
        )
        self.state_proj = nnx.Linear(
            in_features=state_dim,
            out_features=attn_dim,
            rngs=rngs,
        )
        self.action_proj1 = nnx.Linear(
            in_features=act_horizon * act_dim,
            out_features=self.mlp_hidden,
            rngs=rngs,
        )
        self.action_proj2 = nnx.Linear(
            in_features=self.mlp_hidden,
            out_features=self.attn_dim,
            rngs=rngs,
        )
        self.action_proj3 = nnx.Linear(
            in_features=self.attn_dim,
            out_features=attn_dim,
            rngs=rngs,
        )
        
        # self.action_in_proj = nnx.Linear(
        #     in_features=act_dim,
        #     out_features=attn_dim,
        #     rngs=rngs,
        # )
        # self.action_time_mlp_in = nnx.Linear(
        #     in_features=2 * attn_dim,
        #     out_features=attn_dim,
        #     rngs=rngs,
        # )
        # self.action_time_mlp_out = nnx.Linear(
        #     in_features=attn_dim,
        #     out_features=attn_dim,
        #     rngs=rngs,
        # )

        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=attn_dim,
            qkv_features=attn_dim,
            out_features=attn_dim,
            dropout_rate=dropout_rate,
            decode=False,
            normalize_qk=True,
            rngs=rngs,
        )
        self.MLP_head1 = nnx.Linear(
            in_features=attn_dim,
            out_features=mlp_hidden,
            rngs=rngs,
        )
        self.MLP_head2 = nnx.Linear(
            in_features=mlp_hidden,
            out_features=1,
            rngs=rngs,
        )

    def __call__(
        self,
        rng: at.KeyArrayLike,
        observations: _model.Observation,
        actions: at.Float[at.Array, "b H A"],
        train: bool = True,
    ) -> at.Float[at.Array, "b"]:
        b = observations.state.shape[0]
        D = self.attn_dim

        # 1) Prompt embedding
        observations = _model.preprocess_observation(rng, observations, train=train)
        tokens = observations.tokenized_prompt  # [b, L] or None
        if tokens is None:
            prompt_proj = jnp.zeros((b, 1, D), dtype=jnp.float32)
        else:
            prompt_emb = self.prompt_embed(tokens)
            prompt_proj = self.prompt_proj(prompt_emb)  # [b, L, D]

        # 2) Image embeddings
        feats = []
        for key in IMAGE_KEYS:
            enc = self.img_enc(observations.images[key], train=train)  # [b, 256]
            feats.append(enc)
        img_seq = jnp.stack(feats, axis=1)  # [b, N, 256]
        img_proj = nnx.silu(self.img_proj(img_seq))  # [b, N, D]
        
        # 3) State embedding
        st = observations.state  # [b, A]
        st_proj = nnx.silu(self.state_proj(st))  # [b, D]
        st_proj = st_proj[:, None, :]  # [b, 1, D]

        # 4) Action embedding
        # time_emb = 1
        # time_tokens = einops.repeat(time_emb, "b emb -> b H emb", H=self.act_horizon)
        # action_tokens = self.action_in_proj(actions)
        # action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        # action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        # action_time_tokens = nnx.silu(action_time_tokens)
        # action_time_tokens = self.action_time_mlp_out(action_time_tokens) # [b, H, d]
        


        flat = actions.reshape((b, -1))  # [b, H*A]
        act_proj = nnx.silu(self.action_proj1(flat))
        act_proj = nnx.silu(self.action_proj2(act_proj))
        act_proj = nnx.silu(self.action_proj3(act_proj))  # [b, D]
        act_proj = act_proj[:, None, :]  # [b, 1, D]

        # 5) Fuse all modalities into one sequence
        fusion_seq = jnp.concatenate([prompt_proj, img_proj, st_proj, act_proj], axis=1)  # [b, L+N+2, D]

        # nan_mask = jnp.isnan(fusion_seq)
        # has_nan = nan_mask.any()
        # jax.debug.print("nan 존재 여부5: {}", has_nan)
        # jax.debug.print("nan 위치: {}", jnp.argmax(nan_mask))

        # # 6) Multimodal self-attention
        attn_out = self.self_attn(
            inputs_q=fusion_seq,
            deterministic=True,
        )  # [b, T, D]

        attn_out = jnp.mean(attn_out, axis=1)

        # 7) Head MLP
        q = nnx.silu(self.MLP_head1(attn_out))
        q = self.MLP_head2(q)  # [b, 1]
        return jnp.squeeze(q, axis=1)  # [b]


# import flax.nnx as nnx
# from openpi.shared import array_typing as at
# import jax.numpy as jnp

# class FeatureCritic(nnx.Module):
#     """Simple critic that takes features as input (for Pi0FQL)."""
    
#     def __init__(
#         self,
#         feature_dim: int,
#         hidden_dims: tuple[int, ...] = (512, 256, 128),
#         rngs=None,
#     ):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.hidden_dims = hidden_dims
        
#         # Build MLP layers
#         layers = []
#         in_dim = feature_dim
#         for hidden_dim in hidden_dims:
#             layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
#             in_dim = hidden_dim
#         layers.append(nnx.Linear(in_dim, 1, rngs=rngs))
        
#         self.layers = layers
    
#     def __call__(self, features: at.Float[at.Array, "b h d"]) -> at.Float[at.Array, "b"]:
#         """Forward pass through the critic network.
        
#         Args:
#             features: Input features of shape [batch_size, horizon, feature_dim]
            
#         Returns:
#             Q-values of shape [batch_size]
#         """
#         batch_size, horizon, feature_dim = features.shape
        
#         # Process each timestep individually
#         q_values = []
#         for t in range(horizon):
#             x = features[:, t, :]  # [batch, feature_dim]
#             for i, layer in enumerate(self.layers[:-1]):
#                 x = layer(x)
#                 x = nnx.swish(x)  # Use swish activation
            
#             # Final layer (no activation)
#             q_t = self.layers[-1](x)  # [batch, 1]
#             q_values.append(q_t)
        
#         # Stack and average across horizon
#         q_values = jnp.stack(q_values, axis=1)  # [batch, horizon, 1]
#         q_values = jnp.mean(q_values, axis=1)  # [batch, 1]
#         return jnp.squeeze(q_values, axis=-1)  # [batch]