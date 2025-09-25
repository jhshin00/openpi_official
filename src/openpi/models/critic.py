from flax import nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.encoder import MultiEncoder


# =====================
# Critic (nnx)
# =====================
class MultiStateActionValue(nnx.Module):
    """
    Q(s, a_{1:H}) = Linear( MultiEncoder(s, a_{1:H}) )
    입력:
      observations: 위와 동일
      actions: [B,H,32] 또는 [B,32]
    출력:
      q: [B]
    """
    def __init__(self, base_cls: MultiEncoder, rngs: nnx.Rngs = None):
        super().__init__()
        self.base_cls = base_cls
        self.q_head = nnx.Linear(self.base_cls.out_dim, 1, rngs=rngs)  # [B,out_dim]->[B,1]

    def __call__(self, observations: _model.Observation, actions: _model.Actions, *, train: bool = False) -> jnp.ndarray:
        feat = self.base_cls(observations, actions, train=train)  # [B, out_dim]
        q = self.q_head(feat).squeeze(-1)                           # [B]
        return q


# rngs = nnx.Rngs(0)
# base = MultiEncoder(
#     img_dim=512, txt_dim=512, state_dim=256,
#     state_in_dim=32,          # state 길이
#     hidden_dims=(256,256),
#     image_keys=("rgb_base","left_wrist"),
#     action_dim=32, d_model=256, n_layers=2, kernel_size=3,
#     use_film_gate=True, rngs=rngs,
# )
# critic = MultiStateActionValue(base, rngs=rngs)

# obs.images: {"rgb_base": [B,224,224,3], "left_wrist": [B,224,224,3]}
# obs.tokenized_prompt: [B,T] (int32)
# obs.tokenized_prompt_mask: [B,T] (bool or {0,1})  # optional
# obs.state: [B,32]
# actions: [B,H,32] 또는 [B,32]

# q = critic(obs, actions, training=True)  # [B]