import jax
import jax.numpy as jnp


def const_init(d, val=0.01):
    return lambda: val * jnp.ones(shape=(d,))


def vmap_run_mala_handle(gibbs_measure, h, T, d, init_val=lambda: None):
    """Run multiple MALA simulations using vectorization with vmap."""
    if init_val() is None:
        init_val = const_init(d)

    V = gibbs_measure.potential
    gradV = jax.grad(V)

    return lambda key: run_mala(V, gradV, h, T, d, init_val, key)


def run_mala(V, gradV, h, T, d, init_val, key):
    """Simplified version for use with vmap."""
    sim_length = int(T / h) + 1

    # Pre-generate all random numbers
    key1, key2 = jax.random.split(key, 2)
    rand_norms = jax.random.normal(key=key1, shape=(sim_length, d))
    rand_unifs = jax.random.uniform(key=key2, shape=(sim_length,))

    # Initialize the state
    x = jnp.array(init_val(), dtype=jnp.float32).reshape(d)

    # Simple loop
    for t in range(sim_length):
        # Propose new state
        grad_x = gradV(x)
        mean = x - h * grad_x
        y = mean + jnp.sqrt(2 * h) * rand_norms[t]

        # Compute acceptance ratio directly for simplicity
        grad_y = gradV(y)
        mean_yx = y - h * grad_y

        diff_xy = y - mean
        log_q_xy = -jnp.sum(diff_xy ** 2) / (4 * h)

        diff_yx = x - mean_yx
        log_q_yx = -jnp.sum(diff_yx ** 2) / (4 * h)

        log_ratio = -V(y) + V(x) + log_q_yx - log_q_xy
        accept = rand_unifs[t] < jnp.minimum(1.0, jnp.exp(log_ratio))

        # Update state
        x = jnp.where(accept, y, x)

    return x
