import functools
from typing import Optional

# TFP JAX substrate
try:
    import tensorflow_probability.substrates.jax as tfp
except ImportError as e:
    raise ImportError("tensorflow_probability[substrates-jax]가 필요합니다.") from e

tfd = tfp.distributions
tfb = tfp.bijectors

from flax import nnx
import jax.numpy as jnp
from openpi.shared import array_typing as at
from openpi.models import model as _model  # 타입 힌트용(옵션)


# # --------------------------------------------------
# # Tanh wrapper (기존 동일)
# # --------------------------------------------------
# class TanhTransformedDistribution(tfd.TransformedDistribution):
#     def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
#         super().__init__(distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args)

#     def mode(self) -> jnp.ndarray:
#         return self.bijector.forward(self.distribution.mode())


# --------------------------------------------------
# Normal (nnx) — MultiEncoder 기반 edit policy head
# --------------------------------------------------
class Normal(nnx.Module):
    """
    base_cls: nnx 모듈 인스턴스 (예: MultiEncoder)
              __call__(observations, actions_ctx, training=False) -> [B, D_feat]
              그리고 .out_dim(int) 속성을 가져야 함.
    반환: tfd.Distribution (MVNDiag) 또는 TanhTransformedDistribution
    """
    def __init__(
        self,
        base_cls: nnx.Module,           # MultiEncoder(nnx) 인스턴스
        action_dim: int = 32,                # 예: 32
        action_horizon: int = 50,
        log_std_min: float = -20.0,
        log_std_max: float =  2.0,
        state_dependent_std: bool = True,
        squash_tanh: bool = False,
        *,
        rngs: nnx.Rngs | None = None,   # 서브레이어 init용 RNG
        edit_action_scale: float = 0.05,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.base_cls = base_cls
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        self.squash_tanh = squash_tanh

        self.edit_action_scale = edit_action_scale

        # 헤드 in_features는 base의 출력 차원과 같아야 함
        in_features = getattr(self.base_cls, "out_dim", None)
        if in_features is None:
            raise ValueError("base_cls.out_dim이 필요합니다 (MultiEncoder에서 설정됨).")

        self.means_head = nnx.Linear(in_features, action_dim*action_horizon, rngs=rngs)

        if self.state_dependent_std:
            self.log_stds_head = nnx.Linear(in_features, action_dim*action_horizon, rngs=rngs)
            self.log_stds_param = None
        else:
            # 학습 가능한 [A] 파라미터 (배치에 브로드캐스트)
            self.log_stds_head = None
            self.log_stds_param = nnx.Param(jnp.zeros((action_dim*action_horizon,), dtype=jnp.float32))

    def __call__(self, *base_args, **base_kwargs) -> tfd.Distribution:
        """
        보통: dist = policy(observations, actions_ctx, training=True/False)
        base_args/base_kwargs는 그대로 base_cls로 전달됨.
        """
        # 1) base features
        x = self.base_cls(*base_args, **base_kwargs)            # [B, D_feat]

        # 2) heads
        means = self.means_head(x)                              # [B, H*A]
        if self.state_dependent_std:
            log_stds = self.log_stds_head(x)                    # [B, H*A]
        else:
            # [A] → [B, A] 브로드캐스트
            log_stds = jnp.broadcast_to(self.log_stds_param.value, means.shape)

        # 3) 안정 범위로 클립 후 분포 생성
        log_stds = jnp.clip(log_stds, min=self.log_std_min, max=self.log_std_max)   # [B, H*A]
        dist = tfd.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        if self.squash_tanh:
            bijector = tfb.Chain([tfb.Scale(scale=self.edit_action_scale), tfb.Tanh()])
            return tfd.TransformedDistribution(distribution=dist, bijector=bijector)
        else:
            return dist

        # 4) tanh squash (필요 시)
        # return (tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh()) if self.squash_tanh else dist)
        # #return TanhTransformedDistribution(dist) if self.squash_tanh else dist

    
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        anchor_action: _model.Actions,
        *,
        n_samples: int = 1,
        train: bool = True,
        return_logp: bool = False,
    ) -> _model.Actions:

        dist = self(observation, anchor_action, train=train)


        if n_samples == 1:
            a = dist.sample(seed=rng)                          # [B, H*A]
            logp = dist.log_prob(a)                            # [B]
            a = a.reshape(a.shape[0], self.action_horizon, self.action_dim)  # [B,H,A]
            if return_logp:
                return a, logp
            return a
        else:
            a = dist.sample(seed=rng, sample_shape=(n_samples,))   # [K, B, H*A]
            a = jnp.swapaxes(a, 0, 1)                              # [B, K, H*A]
            a = a.reshape(a.shape[0], a.shape[1], self.action_horizon, self.action_dim)  # [B,K,H,A]

            logp = dist.log_prob(a.reshape(a.shape[0], a.shape[1], -1))  # [B,K]
            if return_logp:
                return a, logp
            return a

# 편의용 partial (이름은 유지)
TanhNormal = functools.partial(Normal, squash_tanh=True)