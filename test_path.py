import jax
import flax.nnx as nnx
import optax

class model(nnx.Module):
    def __init__(self, rng):
        super().__init__()
        self.linear = nnx.Linear(5,3, rngs=rng)
        self.drop = nnx.Dropout(0.9, rngs=rng)
        
    def __call__(self, x):
        return self.drop(self.linear(x))
    
class critic(nnx.Module):
    def __init__(self, rng):
        super().__init__()
        self.model = model(rng)
        
    def __call__(self, x):
        return self.model(x)
    
class critic_target(nnx.Module):
    def __init__(self, rng):
        super().__init__()
        self.model = model(rng)
        
    def __call__(self, x):
        return self.model_target(x)

critic = critic(nnx.Rngs(jax.random.PRNGKey(0)))
critic_target = critic_target(nnx.Rngs(jax.random.PRNGKey(1)))

# tx = optax.adam(0.005)
# tx_state = tx.init(nnx.state(critic))

# nnx.state(critic)['model']
# nnx.state(critic_target)['model']

# new_tree = optax.incremental_update(
#     nnx.state(critic)['model'],
#     nnx.state(critic_target)['model_target'],
#     0.5,
# )

print(nnx.state(critic))
print("111\n")

print(nnx.state(critic_target).filter(nnx.Param))
print("222\n")


# tx.update(nnx.state(critic_target), tx_state, nnx.state(critic))

# print(nnx.state(critic))
# print("333\n")

# new_critic_params = optax.apply_updates(nnx.state(critic), nnx.state(critic_target).filter(nnx.Param))
# print(new_critic_params)


nnx.update(critic, nnx.state(critic_target).filter(nnx.Param))
print(nnx.state(critic))
print("333")

# print(new_tree)
# print("333\n")

# print(new_tree.to_pure_dict())
# print("444\n")

# print({'model_target': new_tree.to_pure_dict()})
# print("555\n")

# print(nnx.state(critic_target).to_pure_dict())
# print("666\n")

# print(nnx.statelib.State({'model_target': new_tree.to_pure_dict()}))
# print("777\n")

# nnx.state(critic_target).replace_by_pure_dict({'model_target': new_tree.to_pure_dict()})

# print(nnx.state(critic_target))
# print("\n")

  
# State({
#   'action_embed': {
#     'bias': VariableState(
#       type=Param,
#       value=Traced<ShapedArray(float32[512])>with<DynamicJaxprTrace>
#     ),
#     'kernel': VariableState(
#       type=Param,
#       value=Traced<ShapedArray(float32[32,512])>with<DynamicJaxprTrace>
#     )
#   }
  