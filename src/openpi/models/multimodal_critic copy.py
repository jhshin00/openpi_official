import jax
import jax.numpy as jnp
from flax import nnx

import openpi.models.model as _model
from openpi.shared import array_typing as at

PALIGEMMA_VOCAB_SIZE = 257_152

def make_cross_mask(obs: _model.Observation, img_patch_size):
    patch_masks = []

    L_img = (_model.IMAGE_RESOLUTION[0] // img_patch_size) ** 2

    # for key in _model.IMAGE_KEYS:
    #     m = obs.image_masks[key][:, None] # (B,1)
    #     m = jnp.repeat(m, L_img, axis=1)

    patch_masks = [jnp.repeat(obs.image_masks[key][:, None], L_img, axis=1) for key in _model.IMAGE_KEYS] # (B,L_img) X 3
    patch_masks = jnp.concatenate(patch_masks, axis=1) # (B,L_img*3)

    state_mask = jnp.ones((obs.state.shape[0],1), dtype=bool) # (B,1)

    lang_mask = obs.tokenized_prompt_mask # (B,L_txt)

    key_mask = jnp.concatenate([patch_masks, state_mask, lang_mask], axis=1) # (B,T)
    attn_mask = key_mask[:, None, None, :] # (B,1,1,T)

    return attn_mask

def patch_posemb(patch_h: int, patch_w: int, embed_dim: int, min_period: float = 1e-4, max_period: float = 10.0):
    """
    2-D sin-cos positional embedding for image patches.

    Args
    ----
    img_patch_size        : patch grid height/width  (e.g. 14, 14 for 224×224 with 16×16 patch)
    embed_dim   : must be 짝수. 절반(row)·절반(col)에 나눔
    min_period / max_period : 주파수 범위

    Returns
    -------
    pe : (1, img_patch_size**2, embed_dim)  —  batch 차원은 1이라 브로드캐스트용
    """
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")
    
    def _sincos(pos, out_dim_half):
        half = out_dim_half // 2
        fraction = jnp.linspace(0.0, 1.0, half)
        period = min_period * (max_period / min_period) ** fraction
        angles = pos * (1.0 / period) * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

    row_emb = _sincos(jnp.arange(patch_h)[:, None], embed_dim // 2) # (h, dim/2)
    col_emb = _sincos(jnp.arange(patch_w)[:, None], embed_dim // 2) # (w, dim/2)

    row_emb = row_emb[:, None, :].repeat(patch_w, axis=1) # (h, w, dim/2)
    col_emb = col_emb[None, :, :].repeat(patch_h, axis=0) # (h, w, dim/2)

    pe = jnp.concatenate([row_emb, col_emb], axis=-1) # (h, w, dim)
    return pe.reshape(1, patch_h*patch_w, embed_dim) # (1, L_img, dim)

def act_posemb(act_horizon: int, embed_dim: int,  min_period: float = 1e-4, max_period: float = 10.0):
    """
    1-D sin-cos positional embedding for action sequence.

    Args
    ----
    act_horizon     : action_horizon (예: 50)
    embed_dim   : must be 짝수
    Returns
    -------
    pe : (1, act_horizon, embed_dim)
    """
    if embed_dim % 2:
        raise ValueError("embed_dim must be even")

    fraction = jnp.linspace(0.0, 1.0, embed_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    angles = jnp.arange(act_horizon)[:, None] * (1.0 / period) * 2 * jnp.pi
    pe = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (L, D)
    return pe[None, :, :]                                              # (1, L, D)


class PatchEmbed(nnx.Module):
    """Convert an image to a sequence of patch embeddings."""
    def __init__(self, embed_dim, img_patch_size, img_channels, rngs):
        super().__init__()
        ps = img_patch_size
        self.proj = nnx.Conv(
            in_features=img_channels,
            out_features=embed_dim,
            kernel_size=(ps, ps),
            strides=(ps, ps),
            padding="VALID",
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, img):                        # (B,H,W,C)
        x = self.proj(img)                          # (B, H/ps, W/ps, D)
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # → (B, L, D)
        return x

class MLP(nnx.Module):
    def __init__(self, embed_dim, mlp_dim, dropout_rate, rngs):
        super().__init__()
        self.fc1 = nnx.Linear(
            in_features=embed_dim,
            out_features=mlp_dim,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            in_features=mlp_dim,
            out_features=embed_dim,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, deterministic: bool = True):
        x = self.fc1(x)
        x = nnx.silu(x)
        if not deterministic:
            x = self.dropout(x)
        x = self.fc2(x)
        if not deterministic:
            x = self.dropout(x)
        return x

class CrossAttentionBlock(nnx.Module):
    """Cross-attention: queries come from `latent`, keys/values from `tokens`."""
    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_dim,
            dropout_rate,
            rngs,
    ):
        super().__init__()
        self.ln_q = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.ln_kv = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dropout_rate=dropout_rate,
            decode=False,
            normalize_qk=False,
            rngs=rngs,
        )
        self.mlp = MLP(embed_dim, mlp_dim, dropout_rate, rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.ln_out = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)

    def __call__(self, q: jax.Array, kv: jax.Array, mask = None, deterministic: bool = True):
        q = self.ln_q(q)
        k = self.ln_kv(kv)
        v = k  # same as keys
        attended = self.cross_attn(inputs_q=q, inputs_k=k, inputs_v=v, mask=mask, deterministic=deterministic)
        if not deterministic:
            attended = self.dropout(attended)
        latent = q + attended
        # Feed‑forward on latents
        y = self.ln_out(latent)
        y = self.mlp(y, deterministic=deterministic)
        latent = latent + y
        return latent # latent와 동일 shape
    
class PreLNBlock(nnx.Module):
    """Standard Transformer block: Pre-LN + MH-Attention + MLP."""
    def __init__(
            self,
            embed_dim,
            num_heads,
            mlp_dim,
            dropout_rate,
            rngs,
    ):
        super().__init__()
        self.ln1 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dropout_rate=dropout_rate,
            decode=False,
            normalize_qk=False,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.mlp = MLP(embed_dim, mlp_dim, dropout_rate, rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x, mask: jax.Array | None = None, deterministic: bool = True):
        y = self.ln1(x)
        y = self.attn(inputs_q=y, mask=mask, deterministic=deterministic)
        if not deterministic:
            y = self.dropout(y)
        x = x + y
        z = self.ln2(x)
        z = self.mlp(z, deterministic=deterministic)
        x = x + z
        return x

class PerceiverCritic(nnx.Module):
    def __init__(
            self,
            embed_dim: int=512,
            num_heads: int = 8,
            mlp_dim: int = 2048,
            dropout_rate: float = 0.1,
            num_latents: int = 128,
            num_self_attn_layers: int = 6,
            img_patch_size: int = 16,
            img_channels: int = 3,
            state_dim: int = 7,
            action_dim: int = 7,
            action_seq_len: int = 50,
            vocab_size: int = PALIGEMMA_VOCAB_SIZE,
            rngs = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.img_patch_size = img_patch_size
        self.action_seq_len = action_seq_len

        self.img_embed = PatchEmbed(embed_dim, img_patch_size, img_channels, rngs)

        self.state_embed = nnx.Linear(
            in_features=state_dim,
            out_features=embed_dim,
            rngs=rngs,
        )
        
        self.action_embed = nnx.Linear(
            in_features=action_dim,
            out_features=embed_dim,
            rngs=rngs,
        )

        self.language_embed = nnx.Embed(
            num_embeddings=vocab_size,
            features=embed_dim,
            rngs=rngs,
        )

        lat_init = jax.random.normal(rngs(), (1, num_latents, embed_dim)) * 0.02
        self.latents = nnx.Param(lat_init)
        # self.latents = nnx.Param(jnp.zeros((1, num_latents, embed_dim)))

        # Cross Attention in
        self.cross_attn_in = CrossAttentionBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs)

        # Self Attention
        self.self_attn1 = PreLNBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs)
        self.self_attn2 = PreLNBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs)
        self.self_attn3 = PreLNBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs)

        # Cross Attention out
        self.cross_attn_out = CrossAttentionBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs)

        self.q_head_ln = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.q_head_mlp = nnx.Linear(
            in_features=embed_dim,
            out_features=embed_dim // 2,
            rngs=rngs,
        )
        self.q_head_out = nnx.Linear(
            in_features=embed_dim // 2,
            out_features=1,
            rngs=rngs,
        )

    def __call__(
        self,
        observations: _model.Observation,
        actions: at.Float[at.Array, "b H A"],
        # actions_is_pad: at.Bool[at.Array, "b H"] | None = None,
        train: bool = True,
    ) -> at.Float[at.Array, "b"]:
        
        # Use full action sequence instead of just first action
        # actions = actions[:, 0, :] # (b,A)  # REMOVED
        
        state_tokens = self.state_embed(observations.state)[:, None, :]  # (B,1,D)
        B = state_tokens.shape[0]

        img_toks = [self.img_embed(observations.images[key]) for key in _model.IMAGE_KEYS]  # (32,196,512)
        img_tokens = jnp.concatenate(img_toks, axis=1) # (32,196*3,512)

        pe_img = patch_posemb(14, 14, self.embed_dim)
        pe_img    = jnp.tile(pe_img, (1, len(_model.IMAGE_KEYS), 1))
        img_tokens = img_tokens + pe_img
        
        lang_tokens = self.language_embed(observations.tokenized_prompt)      # (B,L,D)

        all_tokens = jnp.concatenate(
            [img_tokens, state_tokens, lang_tokens], axis=1
        )  # (B, T, D)

        latents = jnp.tile(self.latents, (B, 1, 1))

        mask = make_cross_mask(observations, self.img_patch_size)
        # mask = None

        cross1_out = self.cross_attn_in(latents, all_tokens, mask=mask, deterministic=not train)

        self1_out = self.self_attn1(cross1_out, deterministic=not train)
        self2_out = self.self_attn2(self1_out, deterministic=not train)
        self3_out = self.self_attn3(self2_out, deterministic=not train)

        # Process full action sequence with positional embeddings
        # Remove actions_is_pad handling for consistency with sampled actions
        # Sampled actions are always full sequences, so no padding needed
        act_tokens = self.action_embed(actions.reshape(-1, self.action_dim))  # (B*H, D)
        act_tokens = act_tokens.reshape(B, -1, self.embed_dim)  # (B, H, D)
        pe_act = act_posemb(self.action_seq_len, self.embed_dim)  # (1, H, D)
        act_tokens = act_tokens + pe_act  # Add positional embeddings

        # No attention mask needed since all actions are valid
        cross2_out = self.cross_attn_out(act_tokens, self3_out, deterministic=not train)

        # Use last token for value estimation (simplified pooling)
        pooled = cross2_out[:, -1]  # (B, D)
            
        x = self.q_head_ln(pooled)
        x = self.q_head_mlp(x)
        q_logits = self.q_head_out(x).squeeze(-1)
        # Use softplus + clip for better gradients than sigmoid
        # softplus(x) = log(1 + exp(x)) gives smooth gradients
        # Clip to [0, 1] for Q-value range
        q = jnp.clip(nnx.softplus(q_logits), 0.0, 1.0)
        return q