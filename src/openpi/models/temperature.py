from flax import nnx
import jax.numpy as jnp


class Temperature(nnx.Module):
    def __init__(self, initial_temperature: float = 1.0):
        # log_temp를 학습 가능한 파라미터로 등록
        self.log_temp = nnx.Param(jnp.log(initial_temperature))

    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.log_temp.value)