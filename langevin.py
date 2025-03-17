import jax
import jax.numpy as jnp


def const_init(d, val=0.01):
    return lambda key: val * jnp.ones(shape=(d,))


def sphere_init(d):
    """Initialize a point on the unit sphere in d dimensions."""
    return lambda key: normalize(jax.random.normal(key=key, shape=(d,)))


def normalize(v):
    """Normalize a vector to have unit length."""
    return v / jnp.sqrt(jnp.sum(v ** 2))


def project_to_tangent(x, v):
    """Project vector v onto the tangent space of the sphere at point x."""
    return v - jnp.sum(x * v) * x


def vmap_run_mala_handle(gibbs_measure, h, T, d, init_val=None):
    """Run multiple MALA simulations using vectorization with vmap."""
    if init_val is None:
        init_val = const_init(d)

    V = gibbs_measure.potential
    gradV = jax.grad(V)

    return lambda key: run_mala(V, gradV, h, T, d, init_val, key)


def vmap_run_sphere_mala_handle(gibbs_measure, h, T, d, init_val=None):
    """Run multiple sphere MALA simulations using vectorization with vmap."""
    if init_val is None:
        init_val = sphere_init(d)

    V = gibbs_measure.potential
    # For sphere, we need to compute the Riemannian gradient
    euclidean_gradV = jax.grad(V)

    def riemannian_gradV(x):
        """Compute the Riemannian gradient on the sphere."""
        euclidean_grad = euclidean_gradV(x)
        return project_to_tangent(x, euclidean_grad)

    return lambda key: run_sphere_mala(V, riemannian_gradV, h, T, d, init_val, key)


def run_mala(V, gradV, h, T, d, init_val, key):
    """Simplified version for use with vmap."""
    sim_length = int(T / h) + 1

    # Pre-generate all random numbers
    key_init, key1, key2 = jax.random.split(key, 3)
    rand_norms = jax.random.normal(key=key1, shape=(sim_length, d))
    rand_unifs = jax.random.uniform(key=key2, shape=(sim_length,))

    # Initialize the state
    x = jnp.array(init_val(key_init), dtype=jnp.float32).reshape(d)

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


def run_sphere_mala(V, riemannian_gradV, h, T, d, init_val, key):
    """Optimized MALA on the sphere with pre-generated random numbers."""
    sim_length = int(T / h) + 1

    # Pre-generate all random numbers (same as in run_mala)
    key_init, key1, key2 = jax.random.split(key, 3)
    rand_norms = jax.random.normal(key=key1, shape=(sim_length, d))
    rand_unifs = jax.random.uniform(key=key2, shape=(sim_length,))

    # Initialize the state on the sphere

    x = init_val(key_init)

    # Simple loop
    for t in range(sim_length):
        # Compute Riemannian gradient
        grad_x = riemannian_gradV(x)

        # Propose new state: first make Euclidean step
        mean = x - h * grad_x

        # Project pre-generated noise to tangent space
        base_noise = rand_norms[t]
        tangent_noise = project_to_tangent(x, base_noise)

        # Add scaled noise in tangent space
        y_pre = mean + jnp.sqrt(2 * h) * tangent_noise

        # Project back to sphere
        y = normalize(y_pre)

        # Compute acceptance ratio
        grad_y = riemannian_gradV(y)
        mean_yx = y - h * grad_y

        # For sphere MALA, account for the manifold geometry
        tangent_y_to_x = project_to_tangent(y, x - y)
        tangent_x_to_y = project_to_tangent(x, y - x)

        log_q_xy = -jnp.sum(tangent_x_to_y ** 2) / (4 * h)
        log_q_yx = -jnp.sum(tangent_y_to_x ** 2) / (4 * h)

        log_ratio = -V(y) + V(x) + log_q_yx - log_q_xy
        accept = rand_unifs[t] < jnp.minimum(1.0, jnp.exp(log_ratio))

        # Update state
        x = jnp.where(accept, y, x)

    return x