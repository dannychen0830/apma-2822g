import jax
import jax.numpy as jnp
    from functools import partial
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


def subroutine(key, k_param, s_param, t_param, g_matrix, beta, n, verbose=True):
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

        # Get the pre-generated noise for this step
        w = noise[i]

        # Update b and e
        b = b + jnp.sqrt(s_param) * w
        e = e + s_param * m + jnp.sqrt(s_param) * w

    return e


def run_stoch_loc(n, beta, seed=0, k=35, s=None, t=None, t_mala=20, h=0.01,
                  run_subroutine=True, verbose=True):
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
    master_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(master_key, 3)
    g_key, subroutine_key, mala_key = keys

    # Generate G matrix
    g_matrix = generate_symmetric_matrix(n, beta, g_key)

    # Run the subroutine if required
    if run_subroutine:
        if verbose:
            print("Running subroutine...")
        final_y = subroutine(subroutine_key, k, s, t, g_matrix, beta, n, verbose)
    else:
        final_y = jnp.zeros(n)

    if verbose:
        print("Running MALA...")

    # Split the key again for orthogonal basis generation
    ortho_key = jax.random.fold_in(master_key, 103)  # Using a different fold_in value

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

    return sample


def test_alg():
    """Test function that uses the original global variables."""
    n = 20
    beta = 0.25
    seed = 50

    # Run with default parameters
    result = run_stoch_loc(n=n, beta=beta, seed=seed, verbose=True)
    print(result)
    return result


if __name__ == "__main__":
    test_alg()