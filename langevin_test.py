import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import langevin
import cProfile as profile
import pstats

pr = profile.Profile()
pr.disable()

V = lambda x: jnp.linalg.norm(x) ** 2 / 2
# gradV = lambda x: x
h = 0.01
T = 10
d = 1
gradV = jax.grad(V)

num_data = 1
key = jax.random.PRNGKey(4)
key_split = jax.random.split(key, num_data)
pr.enable()
data = [langevin.run_mala(V, gradV, h, T, d, key=key_split[i]) for i in range(num_data)]
pr.disable()

pstats.Stats(pr).sort_stats(pstats.SortKey.CALLS).print_stats()
