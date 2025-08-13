import jax
dt = -1.0/10

def step(carry):
    x, t = carry
    jax.debug.print("x: {} and t: {}",x, t)
    return x+1, t+dt

def cond(carry):
    x, t = carry
    return t >= -dt / 2

def step_for(i, x):
    t = 1 + i * dt
    jax.debug.print("x: {} and t: {}",x, t)
    return x+1

x, t = jax.lax.while_loop(cond, step, (0, 1.0))
print(x)
print("while end")

x_0 = jax.lax.fori_loop(0, 10, step_for, 0)
print(x_0)
print("for end")