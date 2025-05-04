import jax
import jax.numpy as jnp
from langevin import run_mala, normal_init


def l2n(x, n):
    """Calculate L2 norm scaled by n."""
    return jnp.square(jnp.sum(jnp.power(x, 2))) / n


def z_function(s, beta):
    """Z function."""
    return (beta * s) ** 2


def d1z_function(s, beta):
    """Derivative of Z function."""
    return 2 * (beta ** 2) * s


def h_function(x, y, g_matrix, n):
    """H function."""
    return (x.T @ g_matrix @ x) / jnp.sqrt(n) + x.T @ y


def d1h_function(x, y, g_matrix, n):
    """Gradient of H function with respect to x."""
    return jax.grad(lambda x: h_function(x, y, g_matrix, n))(x)


def r_function(s):
    """R function for density."""
    c = jnp.maximum(s, 1e-8)
    return jnp.where(c > 0.1, 1.5 * (c - 0.01 / c - 0.2 * jnp.log(c / 0.1)), 0)


def generate_symmetric_matrix(n, beta, key):
    """Generate a symmetric random matrix with JAX."""
    matrix = beta * jax.random.normal(key, (n, n))
    return (matrix + matrix.T) / 2


def orthogonal_basis(y, key, n):
    """Generate orthogonal basis using QR decomposition."""
    random_matrix = jax.random.normal(key, (n, n - 1))
    combined = jnp.hstack([y.reshape(-1, 1), random_matrix])
    q, _ = jnp.linalg.qr(combined)
    return q[:, 1:]


def projection(z, y, ortho_key, n):
    """Project z onto the space orthogonal to y."""
    ortho_basis_mat = orthogonal_basis(y, ortho_key, n)
    return (y / jnp.sqrt(l2n(y, n)) + ortho_basis_mat @ z) / jnp.sqrt(1 + l2n(z, n))


def hp_function(z, y, ortho_key, g_matrix, n):
    """HP function."""
    proj = projection(z, y, ortho_key, n)
    return h_function(proj, y, g_matrix, n) - (n / 2) * jnp.log(1 + l2n(z, n))


def vp_function(z, y, t_param, ortho_key, g_matrix, n):
    """VP function (potential)."""
    return hp_function(z, y, ortho_key, g_matrix, n) - (t_param * n / 2) * r_function(l2n(z, n))


def approximate_mean_computation(y, t_param, k_param, g_matrix, beta, n):
    """Calculate approximate mean using AMP algorithm."""
    # Initialize with zeros
    q = 0.0
    m = jnp.zeros(n)

    # Iterate k_param times
    for k in range(k_param):
        # Calculate new q
        q_new = 1 - 1 / (1 + d1z_function(q, beta) + t_param)

        # Calculate new w
        grad_h = d1h_function(m, y, g_matrix, n)
        w_new = grad_h + y - 2 * (beta ** 2) * (1 - q) * m

        # Calculate new m
        m_new = (1 - q_new) * w_new

        # Update state for next iteration
        q = q_new
        m = m_new

    return m


def gradT(z, n):
    return -2 * (1 - l2n(z, n)) * z


def gradL(z, n):
    return z / (1 - l2n(z, n))


def gradF(z, y, g_matrix, n):
    return gradH(g_matrix, z, n) + y + gradT(z, n) + gradL(z, n)


def gradH(g_matrix, z, n):
    return 2 * g_matrix @ z / jnp.sqrt(n)


def grad_tap(z, y, n, g_matrix):
    return gradH(g_matrix, z, n) + y + gradT(z, n) + gradL(z, n)


def dg_tap_energy(init_state, y, n, g_matrix, learning_rate, k_param):
    for k in range(k_param):
        init_state = init_state - learning_rate * grad_tap(init_state, y, n, g_matrix)
    return init_state


def subroutine(key, k_param, s_param, t_param, g_matrix, beta, eta, n, verbose=True):
    """Run the subroutine to compute the final y.

    Pre-generates all random noise for efficiency.
    """
    # Initialize state
    e = jnp.zeros(n)
    b = jnp.zeros(n)

    # Pre-generate all random noise at once
    noise = jax.random.normal(key, (int(t_param), n))

    # Loop over iterations
    for i in range(int(t_param)):
        if verbose and i % 100 == 0:
            print(f"Subroutine Iteration: {i}")

        # Compute m with current parameters
        m = approximate_mean_computation(e, s_param * jnp.array(i, dtype=jnp.float32),
                                         k_param, g_matrix, beta, n)

        m = dg_tap_energy(m, e, n, g_matrix, eta, k_param)

        # Get the pre-generated noise for this step
        w = noise[i]

        # Update b and e
        b = b + jnp.sqrt(s_param) * w
        e = e + s_param * m + jnp.sqrt(s_param) * w

    return e


def run_stoch_loc(g_matrix, n, beta, eta, k=35, s=None, t=None, t_mala=20, h=0.01,
                  verbose=True, key=jax.random.PRNGKey(9)):
    """
    Main function to run stochastic localization algorithm.

    Args:
        n: Dimension parameter
        beta: Scaling parameter
        seed: Random seed
        k: Iteration parameter for approximate_mean_computation
        s: Step size for subroutine (defaults to 1/n^2 if None)
        t: Number of iterations for subroutine (defaults to 20*n if None)
        t_mala: Number of MALA iterations
        h: Step size for MALA
        run_subroutine: Whether to run the subroutine or use zeros
        verbose: Whether to print progress

    Returns:
        MALA samples
    """
    # Set defaults based on n
    if s is None:
        s = 1 / jnp.power(n, 2)
    if t is None:
        t = 20 * n

    # Initialize keys for all random operations
    keys = jax.random.split(key, 2)
    subroutine_key, mala_key = keys

    # Run the subroutine
    if verbose:
        print("Running subroutine...")
    final_y = subroutine(subroutine_key, k, s, t, g_matrix, beta, eta, n, verbose)

    if verbose:
        print("Running MALA...")

    # Split the key again for orthogonal basis generation
    ortho_key = jax.random.fold_in(key, 103)  # Using a different fold_in value

    # Create the potential function for Gibbs measure
    potential_fn = lambda z: vp_function(z, final_y, t_mala, ortho_key, g_matrix, n)

    # Create the gradient function using JAX
    grad_potential_fn = jax.grad(potential_fn)

    # Run MALA
    sample = run_mala(
        potential_fn,
        grad_potential_fn,
        h,
        t_mala,
        n - 1,
        normal_init(n - 1),
        key=mala_key,
        verbose=verbose
    )

    if verbose:
        print("MALA completed.")

    return projection(sample, final_y, ortho_key, n)


def vmap_run_stoch_loc_handle(g_matrix, n, beta, eta=0.001, k=35, s=None, t=None, t_mala=20, h=0.01):
    """Run multiple MALA simulations using vectorization with vmap."""
    return lambda key: run_stoch_loc(g_matrix, n, beta, eta, k, s, t, t_mala, h, verbose=False, key=key)


def mp_run_stoch_loc_handle(key, g_matrix, n, beta, eta, k, s, t, t_mala, h):
    return run_stoch_loc(g_matrix, n, beta, eta, k, s, t, t_mala, h, verbose=False, key=key)


def vmap_run_stoch_loc_handle_jit(g_matrix, n, beta, eta=0.001, k=35, s=None, t=None, t_mala=20, h=0.01):
    """Run multiple MALA simulations using vectorization with vmap."""
    # Apply jit for better performance
    jitted_run = jax.jit(lambda key: run_stoch_loc(
        g_matrix, n, beta, eta, k, s, t, t_mala, h, verbose=False, key=key
    ))
    return jitted_run