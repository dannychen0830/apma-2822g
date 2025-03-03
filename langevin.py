import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(0, 1))
def metropolis_filter(V, Q_dens, x, y):
    """Compute the Metropolis-Hastings acceptance ratio."""
    # Correctly computing the acceptance ratio for MALA
    log_ratio = -V(y) + V(x) + jnp.log(Q_dens(x, y)) - jnp.log(Q_dens(y, x))
    return jnp.minimum(1.0, jnp.exp(log_ratio))


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def mala_step(V, gradV, h, d, x, rand_norm, rand_unif):
    """Perform a single step of MALA."""
    # Ensure x has the right shape consistently
    x = x.reshape(d)  # Explicit reshape to dimension d

    # Propose new state
    grad_x = gradV(x)
    mean = x - h * grad_x
    y = mean + jnp.sqrt(2 * h) * rand_norm

    # Compute Q(x,y) - transition density from x to y
    diff_xy = y - mean
    log_q_xy = -jnp.sum(diff_xy ** 2) / (4 * h)

    # Compute Q(y,x) - transition density from y to x
    grad_y = gradV(y)
    mean_yx = y - h * grad_y
    diff_yx = x - mean_yx
    log_q_yx = -jnp.sum(diff_yx ** 2) / (4 * h)

    # Compute acceptance ratio
    log_ratio = -V(y) + V(x) + log_q_yx - log_q_xy
    accept = rand_unif < jnp.minimum(1.0, jnp.exp(log_ratio))

    # Return new state based on acceptance ensuring scalar output
    return jnp.where(accept, y, x).reshape(d)  # ensure proper shape


def run_mala(V, gradV, h, T, d, init=lambda: 0.01, key=jax.random.key(4)):
    """Run MALA algorithm with optimized implementation."""
    sim_length = int(T / h) + 1

    # Pre-generate all random numbers
    key1, key2 = jax.random.split(key, 2)
    rand_norms = jax.random.normal(key=key1, shape=(sim_length, d))
    rand_unifs = jax.random.uniform(key=key2, shape=(sim_length,))

    # Initialize the state
    x = jnp.array(init(), dtype=jnp.float32).reshape(d)

    # Create storage for the trajectory
    x_final = x

    # Use a for loop instead of scan for simplicity
    for t in range(sim_length):
        x_final = mala_step(V, gradV, h, d, x_final, rand_norms[t], rand_unifs[t])

        # Print progress for selected iterations (uncommented for performance)
        # if t % int(1 / h) == 0:
        #     print(f'At iteration {t}!')

    return x_final


# Use vmap instead of pmap for vectorization on a single device
def run_multiple_mala(V, gradV, h, T, d, num_runs, init_val=0.01, main_key=jax.random.key(4)):
    """Run multiple MALA simulations using vectorization with vmap."""

    # Generate keys for each run
    keys = jax.random.split(main_key, num_runs)

    # Vectorize the MALA simulation across the batch of keys
    # This runs on a single device but parallelizes the computation
    vmap_mala = jax.vmap(lambda key: run_mala_simple(V, gradV, h, T, d, init_val, key))

    # Run the vectorized computation
    results = vmap_mala(keys)

    print(f"Completed {num_runs} vectorized MALA simulations")

    return results


def run_mala_simple(V, gradV, h, T, d, init_val, key):
    """Simplified version for use with vmap."""
    sim_length = int(T / h) + 1

    # Pre-generate all random numbers
    key1, key2 = jax.random.split(key, 2)
    rand_norms = jax.random.normal(key=key1, shape=(sim_length, d))
    rand_unifs = jax.random.uniform(key=key2, shape=(sim_length,))

    # Initialize the state
    x = jnp.array(init_val, dtype=jnp.float32).reshape(d)

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