import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from stochastic_localization import *
from langevin import *
from mcmc import *
import gibbs_measure as gm
import time
from misc import *


def test_uniform(num_samples, num_processes=4, n=20, beta=0.25, eta=0.001, k=35, s=None, t=None, t_mala=2, h=0.01):
    """ test stochatic localization with disorder g_matrix be the identity matrix

    Args:
        num_samples: Number of simulations to run
        n: Dimension parameter
        beta: Scaling parameter
        eta: Learning rate for gradient descent
        batch_size: Size of batches for processing (None means process all at once)
        use_pmap: Whether to use pmap for multi-device parallelism
    """

    key = jax.random.PRNGKey(int(time.time()) % 100000)

    g_key, alg_key = jax.random.split(key, 2)
    g_matrix = jnp.eye(n)  # Identity matrix

    # start timing
    start_time = time.time()
    print(f"Running {num_samples} Stochastic Localization simulations with uniform disorder (identity matrix)...")

    # Use enhanced vmap_multiple_run function with batch_size and use_pmap options
    if num_processes > 1:
        samples = mp_multiple_run(
            mp_run_stoch_loc_handle,
            num_samples,
            (g_matrix, n, beta, eta, k, s, t, t_mala, h),
            main_key=alg_key,
            num_processes=num_processes,
        )
    else:
        samples = vmap_multiple_run(
            vmap_run_stoch_loc_handle(g_matrix, n, beta, eta),
            num_samples,
            main_key=alg_key
        )

    # end timing
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")

    samples_np = np.array(samples)
    # Save samples to file
    output_file = f"data/stoch_loc_uniform_samples_n_{n}_num_samples_{num_samples}.npy"
    np.save(output_file, samples_np)
    print(f"Samples saved to {os.path.abspath(output_file)}")

    return samples_np


def localization_test(num_samples, g_matrix, n, beta, eta, k=35, s=None, t=None, t_mala=20, h=0.01,
                  verbose=True, key=jax.random.PRNGKey(9)):
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
    np.save(f"./data/localization_test_final_y.npy", final_y)

    if verbose:
        print("Running MALA...")

    # Split the key again for orthogonal basis generation
    ortho_key = jax.random.fold_in(key, 103)  # Using a different fold_in value

    # Create the potential function for Gibbs measure
    potential_fn = lambda z: vp_function(z, final_y, t_mala, ortho_key, g_matrix, n)

    # Create the gradient function using JAX
    grad_potential_fn = jax.grad(potential_fn)

    # Run MALA
    mala_handle = lambda key: run_mala(
        potential_fn,
        grad_potential_fn,
        h,
        t_mala,
        n - 1,
        normal_init(n - 1),
        key=key,
        verbose=True
    )

    samples = vmap_multiple_run(mala_handle, num_samples, main_key=mala_key)

    samples_np = np.array(samples)
    # Save samples to file
    output_file = f"./data/localization_test_sample.npy"
    np.save(output_file, samples_np)
    print(f"Samples saved to {os.path.abspath(output_file)}")
    #
    # num_directions = 4
    # proj_key = jax.random.PRNGKey(0)
    # key_list = jax.random.split(proj_key, num_directions)
    # direction_list = [jax.random.normal(shape=(n-1,), key=key)
    #                   / jnp.linalg.norm(jax.random.normal(shape=(n-1,), key=key))
    #                   for key in key_list]
    #
    # plot_projection(samples, direction_list, gm.GibbsMeasure('localization_test',None, None, None), show_plot=True)

    return samples_np


def sphere_projection_density(x, dim):
    return jax.scipy.special.gamma(dim / 2) / (jnp.sqrt(jnp.pi) * jax.scipy.special.gamma((dim - 1) / 2)) * (1 - jnp.sum(x ** 2)) ** ((dim - 3) / 2)


if __name__ == "__main__":
    n = 20
    num_samples = 1000
    gibbs_measure = gm.GibbsMeasure('gaussian',
                                    lambda x: jnp.sum(x ** 2) / 2,
                                    n,
                                    'sphere',
                                    )

    test_samples = vmap_multiple_run(vmap_run_sphere_mala_handle(gibbs_measure, 0.01, 10, n), 5000)

    # samples = test_uniform(num_samples=num_samples, n=n, num_processes=6, beta=0.25, eta=0.001)
    # print('start visualization')
    # visualize_results(samples, gibbs_measure, show_plot=True)

    # localization_test(1000, jnp.eye(n), n, beta=0.25, eta=0.001, k=35, s=None, t=None, t_mala=20, h=0.01)

    samples = np.load(f'data/stoch_loc_uniform_samples_n_{n}_num_samples_{num_samples}.npy')
    print(np.count_nonzero(np.isnan(samples)))  # Check for NaN values
    samples_no_nan = samples[~np.isnan(samples).any(axis=1)]  # Remove rows with NaN values
    identity_matrix = jnp.eye(n)
    direction_list = [identity_matrix[i,:] for i in range(min(n, 10))]
    direction_names = [i for i in range(min(n, 10))]
    plot_projection(samples_no_nan,
                    direction_list,
                    gm.GibbsMeasure('localization_test_coordinates', None, None, None),
                    direction_name=direction_names,
                    show_plot=True,
                    test_sample=test_samples,
                    save_fig=f"./data/stoch_loc_unif_test_n_{n}_num_sample_{num_samples}.png",
                    density_functions=[lambda x: sphere_projection_density(x, dim=n) for i in range(n)])