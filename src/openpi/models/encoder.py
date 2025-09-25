from typing import Optional, Sequence, Dict
import jax
import jax.numpy as jnp
from flax import nnx

from openpi.models import model as _model

PALIGEMMA_VOCAB_SIZE = 257_152

# =====================
# Utils
# =====================
def masked_mean(x: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
    """
    x:    [B, T, D]
    mask: [B, T] or None  (True/1=keep)
    return: [B, D]
    """
    if mask is None:
        return x.mean(axis=1)
    w = mask[..., None]
    denom = jnp.clip(w.sum(axis=1), min=1e-6)
    return (x * w).sum(axis=1) / denom

def sinusoidal_positional_encoding(T: int, D: int) -> jnp.ndarray:
    """
    return: [T, D]  (sin-cos PE)
    """
    position = jnp.arange(T)[:, None]
    div_term = jnp.exp(jnp.arange(0, D, 2) * (-jnp.log(10000.0) / D))
    pe = jnp.zeros((T, D), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

# =====================
# Encoders (nnx)
# =====================
class ImageEncoder(nnx.Module):
    """
    입력:
      x: [B, H=224, W=224, C=3]
    출력:
      z: [B, out_dim]
    """
    def __init__(self, in_channels: int = 3, out_dim: int = 512, rngs: nnx.Rngs = None):
        super().__init__()
        # Conv: [B,H,W,C] -> [B,H/2,W/2,32] -> ... -> [B, H/16, W/16, 256]
        self.conv1 = nnx.Conv(in_features=in_channels, out_features=32,
                              kernel_size=(5,5), strides=(2,2), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64,
                              kernel_size=(3,3), strides=(2,2), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(in_features=64, out_features=128,
                              kernel_size=(3,3), strides=(2,2), padding="SAME", rngs=rngs)
        self.conv4 = nnx.Conv(in_features=128, out_features=256,
                              kernel_size=(3,3), strides=(2,2), padding="SAME", rngs=rngs)
        self.fc    = nnx.Linear(in_features=256, out_features=out_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B,224,224,3] or [224,224,3] (when called from vmap)
        x = x.astype(jnp.float32)
        
        # Add batch dimension if missing (when called from vmap)
        if x.ndim == 3:
            x = x[None, ...]  # [1, 224, 224, 3]
            
        x = nnx.gelu(self.conv1(x))  # [B,112,112,32]
        x = nnx.gelu(self.conv2(x))  # [B, 56, 56,64]
        x = nnx.gelu(self.conv3(x))  # [B, 28, 28,128]
        x = nnx.gelu(self.conv4(x))  # [B, 14, 14,256]
        x = x.mean(axis=(1,2))       # [B,256]   (global average pool)
        x = self.fc(x)               # [B,out_dim]
            
        return x

class ViewPooler(nnx.Module):
    """
    feats: [B, V, D]  (예: V=2 for rgb_base/left_wrist)
    return: [B, D]
    """
    def __init__(self): super().__init__()
    def __call__(self, feats: jnp.ndarray) -> jnp.ndarray:
        return feats.mean(axis=1)

class LanguageEncoder(nnx.Module):
    """
    token_ids: [B, T]
    mask:      [B, T] or None
    return:    [B, embed_dim] (or proj_dim)
    """
    def __init__(self,
                 vocab_size: int = PALIGEMMA_VOCAB_SIZE,
                 embed_dim: int = 512,
                 proj_dim: Optional[int] = None,
                 use_positional_encoding: bool = True,
                 rngs: nnx.Rngs = None):
        super().__init__()
        self.use_pe = use_positional_encoding
        self.embed = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.ln    = nnx.LayerNorm(num_features=embed_dim, rngs=rngs)
        self.proj  = nnx.Linear(embed_dim, proj_dim, rngs=rngs) if (proj_dim is not None and proj_dim != embed_dim) else None

    def __call__(self, token_ids: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # token_ids: [B,T] or [T] (when called from vmap)
        if token_ids.ndim == 1:
            # Add batch dimension if missing (when called from vmap)
            token_ids = token_ids[None, ...]  # [1, T]
            mask = mask[None, ...] if mask is not None else None  # [1, T] or None
            
        x = self.embed(token_ids)                     # [B,T,D]
        if self.use_pe:
            _, T, D = x.shape
            x = x + sinusoidal_positional_encoding(T, D)[None, ...]  # [B,T,D]
        x = self.ln(x)                                # [B,T,D]
        x = masked_mean(x, mask)                      # [B,D]
        if self.proj is not None:
            x = self.proj(x)                          # [B,proj_dim]
            
        return x

class StateEncoder(nnx.Module):
    """
    x: [B, state_in_dim]   (여기서 state_in_dim == action_dim == 32)
    return: [B, embed_dim] (기본 256)
    """
    def __init__(self, state_in_dim: int, embed_dim: int = 256, hidden: int = 256, rngs: nnx.Rngs = None):
        super().__init__()
        self.fc1 = nnx.Linear(state_in_dim, hidden, rngs=rngs)  # [B,hidden]
        self.ln  = nnx.LayerNorm(num_features=hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, embed_dim, rngs=rngs)     # [B,embed_dim]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, state_in_dim] or [state_in_dim] (when called from vmap)
        if x.ndim == 1:
            # Add batch dimension if missing (when called from vmap)
            x = x[None, ...]  # [1, state_in_dim]
            
        x = nnx.gelu(self.fc1(x))  # [B,hidden]
        x = self.ln(x)             # [B,hidden]
        x = self.fc2(x)            # [B,embed_dim]
            
        return x

class ActionEncoder1D(nnx.Module):
    def __init__(self, action_dim: int, d_model: int = 256, n_layers: int = 2, kernel_size: int = 3, rngs: nnx.Rngs = None):
        super().__init__()
        self.action_dim = action_dim            # ← 추가: 확실한 판별을 위해 보관
        self.d_model = d_model
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.fc_in = nnx.Linear(action_dim, d_model, rngs=rngs)
        self.convs = [nnx.Conv(d_model, d_model, (kernel_size,), strides=(1,), padding="SAME", rngs=rngs)
                      for _ in range(n_layers)]
        self.lns   = [nnx.LayerNorm(num_features=d_model, rngs=rngs) for _ in range(n_layers)]
        self.att_w = nnx.Linear(d_model, 1, rngs=rngs)

    def __call__(self, actions: jnp.ndarray, *, train: bool = False) -> jnp.ndarray:
        if actions.ndim == 2:
            actions = actions[None, ...]

        B, H, A = actions.shape
        x = self.fc_in(actions)                         # [B,H,D]
        x = x + sinusoidal_positional_encoding(H, self.d_model)[None, ...]
        for conv, ln in zip(self.convs, self.lns):
            res = x
            x = conv(x); x = nnx.gelu(x); x = ln(x)
            x = x + res
        w = self.att_w(x)                               # [B,H,1]
        w = nnx.softmax(w, axis=1)
        x = (x * w).sum(axis=1)                         # [B,D]

        return x



# =====================
# Trunk & Fusion (nnx)
# =====================
class ObsTrunk(nnx.Module):
    """
    입력(obs):
      obs.images: dict with keys {"rgb_base","left_wrist"}, 각 뷰는 [B,224,224,3]
                  또는 단일 배열 [B,224,224,3] (dict가 아닌 경우)
      obs.tokenized_prompt: [B,T] (int ids)
      obs.tokenized_prompt_mask: [B,T] (optional)
      obs.state: [B, state_in_dim] (여기선 32)

    출력:
      trunk feature: [B, hidden_dims[-1]]
    """
    def __init__(self,
                 img_dim: int = 512,          # 각 이미지 인코더 출력 D_img
                 txt_dim: int = 512,          # 텍스트 인코더 출력 D_txt
                 state_dim: int = 256,        # 상태 인코더 출력 D_state
                 state_in_dim: int = 0,       # 상태 입력 차원(=32)
                 hidden_dim: int = 256,
                 out_dim: int = 256,
                 use_image: bool = True,
                 use_text: bool = True,
                 use_state: bool = True,
                 vocab_size: int = PALIGEMMA_VOCAB_SIZE,
                 image_keys: Optional[Sequence[str]] = ("rgb_base", "left_wrist"),
                 dropout_rate: float = 0.0,
                 use_layer_norm: bool = True,
                 rngs: nnx.Rngs = None):
        super().__init__()
        self.use_image = use_image
        self.use_text  = use_text
        self.use_state = use_state
        self.image_keys = image_keys
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_layer_norm = use_layer_norm
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs) if dropout_rate and dropout_rate > 0 else None

        if use_image:
            # 각 뷰마다 같은 인코더 공유(가중치 공유) — 필요 시 분리 가능
            self.img_enc = ImageEncoder(in_channels=3, out_dim=img_dim, rngs=rngs)
            self.view_pool = ViewPooler()  # [B,V,D_img] -> [B,D_img]
        if use_text:
            self.txt_enc = LanguageEncoder(vocab_size=vocab_size, embed_dim=txt_dim, proj_dim=txt_dim, rngs=rngs)
        if use_state:
            assert state_in_dim > 0, "ObsTrunkNNX: use_state=True이면 state_in_dim이 필요합니다."
            self.st_enc = StateEncoder(state_in_dim=state_in_dim, embed_dim=state_dim, rngs=rngs)

        # small MLP head: concat([img?, txt?, state?]) -> MLP
        # in_dim = D_img*1 + D_txt*1 + D_state*1 (뷰는 view_pool로 이미 집계됨)
        in_dim = img_dim * int(use_image) + txt_dim * int(use_text) + state_dim * int(use_state)
        self.linear = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.out_linear = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, obs: _model.Observation, *, train: bool = False) -> jnp.ndarray:
        feats = []

        # ----- Images -----
        if self.use_image and obs.images is not None:
            if isinstance(obs.images, dict):
                # 각 키: "rgb_base", "left_wrist"  (각각 [B,224,224,3])
                keys = self.image_keys if self.image_keys is not None else list(obs.images.keys())
                per_view = [self.img_enc(obs.images[k]) for k in keys]  # 각 [B,img_dim]
                views = jnp.stack(per_view, axis=1)                     # [B,V,img_dim], V=len(keys)=2
                img_feat = self.view_pool(views)                        # [B,img_dim]
            else:
                # 단일 이미지 텐서 [B,224,224,3]
                img_feat = self.img_enc(obs.images)                     # [B,img_dim]
            feats.append(img_feat)

        # ----- Text -----
        if self.use_text:
            if obs.tokenized_prompt is None:
                raise ValueError("ObsTrunkNNX: tokenized_prompt가 필요합니다.")
            # tokenized_prompt: [B,T], tokenized_prompt_mask: [B,T] (optional)
            txt_feat = self.txt_enc(obs.tokenized_prompt, getattr(obs, "tokenized_prompt_mask", None))  # [B,txt_dim]
            feats.append(txt_feat)

        # ----- State -----
        if self.use_state and obs.state is not None:
            # obs.state: [B, state_in_dim]  (여기선 32)
            st_feat = self.st_enc(obs.state)                            # [B,state_dim]
            feats.append(st_feat)

        if len(feats) == 0:
            raise ValueError("ObsTrunkNNX: 사용할 모달리티가 없습니다.")

        x = jnp.concatenate(feats, axis=-1)                              # [B, in_dim]
        # small MLP
        x = self.linear(x)
        x = nnx.gelu(x)
        x = self.layer_norm(x)
        x = self.out_linear(x)
        return x                                                         # [B, out_dim]

class FiLMGate(nnx.Module):
    """
    s_feat: [B, D_s]
    a_feat: [B, D_a]
    return: [B, D_s]
    """
    def __init__(self, s_dim: int, a_dim: int, rngs: nnx.Rngs = None):
        super().__init__()
        self.fc = nnx.Linear(a_dim, s_dim, rngs=rngs)  # [B,D_a]->[B,D_s]
    def __call__(self, s_feat: jnp.ndarray, a_feat: jnp.ndarray) -> jnp.ndarray:
        g = jax.nn.sigmoid(self.fc(a_feat))            # [B,D_s]
        return s_feat * g                              # [B,D_s]

class MultiEncoder(nnx.Module):
    """
    입력:
      observations:
        images: {"rgb_base":[B,224,224,3], "left_wrist":[B,224,224,3]} (dict) 또는 [B,224,224,3]
        tokenized_prompt: [B,T], tokenized_prompt_mask: [B,T] (optional)
        state: [B,32]  (state_in_dim=32)
      actions: [B,H,32] 또는 [B,32]

    출력:
      z: [B, hidden_dims[-1]]
    """
    def __init__(self,
                 img_dim: int = 512,
                 txt_dim: int = 512,
                 state_dim: int = 256,
                 state_in_dim: int = 32,          # ← state 입력 차원(=action_dim)
                 hidden_dim: int = 256,
                 out_dim: int = 256,
                 use_image: bool = True,
                 use_text: bool = True,
                 use_state: bool = True,
                 vocab_size: int = PALIGEMMA_VOCAB_SIZE,
                 image_keys: Optional[Sequence[str]] = ("rgb_base", "left_wrist"),
                 # action encoder
                 action_dim: int = 32,            # ← 액션 차원
                 d_model: int = 256,
                 n_layers: int = 2,
                 kernel_size: int = 3,
                 # fusion/head
                 dropout_rate: float = 0.0,
                 use_layer_norm: bool = False,
                 use_film_gate: bool = True,
                 rngs: nnx.Rngs = None):
        super().__init__()
        self.use_film_gate = use_film_gate

        # Obs trunk: [B, out_dim]
        self.trunk = ObsTrunk(img_dim=img_dim,
                                 txt_dim=txt_dim,
                                 state_dim=state_dim,
                                 state_in_dim=state_in_dim,
                                 hidden_dim=hidden_dim,
                                 out_dim=out_dim,
                                 use_image=use_image,
                                 use_text=use_text,
                                 use_state=use_state,
                                 vocab_size=vocab_size,
                                 image_keys=image_keys,
                                 dropout_rate=dropout_rate,
                                 use_layer_norm=True,
                                 rngs=rngs)

        # Action encoder: [B, D_a] (D_a=d_model)
        self.act_enc = ActionEncoder1D(action_dim=action_dim, d_model=d_model, n_layers=n_layers, kernel_size=kernel_size, rngs=rngs)

        # Fusion head
        self.s_dim_last = out_dim         # = out_dim
        self.a_dim_last = d_model
        if use_film_gate:
            self.film = FiLMGate(s_dim=self.s_dim_last, a_dim=self.a_dim_last, rngs=rngs)

        in_dim = self.s_dim_last + self.a_dim_last     # concat 차원
        self.head1 = nnx.Linear(in_dim, self.s_dim_last, rngs=rngs)   # [B,in_dim]->[B,D_s]
        self.head_ln = nnx.LayerNorm(num_features=self.s_dim_last, rngs=rngs)
        self.out_dim = self.s_dim_last                 # base 출력 차원

    def __call__(self, observations: _model.Observation, actions: _model.Actions, *, train: bool = False) -> jnp.ndarray:
        s_feat = self.trunk(observations, train=train)   # [B, D_s]
        a_feat = self.act_enc(actions, train=train)      # [B, D_a]

        s_mod = self.film(s_feat, a_feat) if self.use_film_gate else s_feat  # [B,D_s]
        h = jnp.concatenate([s_mod, a_feat], axis=-1)          # [B, D_s + D_a]
        z = self.head1(h)                                      # [B, D_s]
        z = nnx.gelu(self.head_ln(z))                          # [B, D_s]
        return z                                               # [B, out_dim]