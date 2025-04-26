import os
from langevin import *
from mcmc import *
import gibbs_measure as gm
import time
from misc import *


def run_test(gibbs_measure, h=0.01, T=10, d=2, num_samples=10000, init_val=None, spherical=False):
    """Run a test with the given potential function."""
    print(f"\n=== Running test for {gibbs_measure.name} potential ===")

    # Start timing
    start_time = time.time()

    # Create a key for random number generation
    key = jax.random.PRNGKey(int(time.time()) % 100000)

    print(f"Running {num_samples} MALA simulations...")

    # Run the parallelized implementation
    if spherical:
        samples = vmap_multiple_run(vmap_run_sphere_mala_handle(gibbs_measure, h, T, d, init_val), num_samples, key)
    else:
        samples = vmap_multiple_run(vmap_run_mala_handle(gibbs_measure, h, T, d, init_val), num_samples, key)

    # Convert to numpy array
    samples_np = np.array(samples)

    # End timing
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")

    # Save samples to file
    output_file = f"./data/spherical_{gibbs_measure.name}_samples.npy" if spherical \
        else f"./data/{gibbs_measure.name}_samples.npy"
    with open(output_file, 'wb') as f:
        np.save(f, samples_np)
    print(f"Samples saved to {os.path.abspath(output_file)}")

    return samples_np


if __name__ == '__main__':
    # Set run parameters
    generate_new_data = False
    show_plot = True
    spherical = False
    # gibbs_measure = gm.GibbsMeasure('gaussian',
    #                                 lambda x: jnp.sum(x ** 2) / 2, 2,
    #                                 'sphere' if spherical else 'reals',
    #                                 )
    gibbs_measure = gm.GibbsMeasure('double_well',
                                    lambda x: jnp.sum(x ** 4) - jnp.sum(x ** 2),
                                    2,
                                    'sphere' if spherical else 'reals')

    if generate_new_data:
        samples = run_test(gibbs_measure, h=0.01, T=20, d=2, num_samples=10000, spherical=spherical)

    else:
        # Load previously generated data
        samples = np.load(f"./data/spherical_{gibbs_measure.name}_samples.npy") if spherical \
            else np.load(f"./data/{gibbs_measure.name}_samples.npy")

    # Visualize loaded results
    visualize_results(samples, gibbs_measure, show_plot=show_plot)

    num_directions = 4
    proj_key = jax.random.PRNGKey(0)
    key_list = jax.random.split(proj_key, num_directions)
    direction_list = [jax.random.normal(shape=(gibbs_measure.dim,), key=key) / jnp.linalg.norm(jax.random.normal(shape=(gibbs_measure.dim,), key=key))
                      for key in key_list]

    plot_projection(samples, direction_list, gibbs_measure, show_plot=show_plot)
    order_list = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2]]
    test_moments(samples, gibbs_measure, order_list)