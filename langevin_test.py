import matplotlib.pyplot as plt
import numpy as np
import os
from langevin import *
from mcmc import *
import gibbs_measure as gm
import time
import pytest
import itertools as it


def run_test(gibbs_measure, h=0.01, T=10, d=2, num_samples=10000, init_val=lambda: None):
    """Run a test with the given potential function."""
    print(f"\n=== Running test for {gibbs_measure.name} potential ===")

    # Start timing
    start_time = time.time()

    # Define gradient of potential
    gradV = jax.grad(gibbs_measure.potential)

    # Create a key for random number generation
    key = jax.random.PRNGKey(int(time.time()) % 100000)

    print(f"Running {num_samples} MALA simulations...")

    # Run the parallelized implementation
    samples = vmap_multiple_run(vmap_run_mala_handle(gibbs_measure, h, T, d, init_val), num_samples, key)

    # Convert to numpy array
    samples_np = np.array(samples)

    # End timing
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")

    # Save samples to file
    output_file = f"./data/{gibbs_measure.name}_samples.npy"
    with open(output_file, 'wb') as f:
        np.save(f, samples_np)
    print(f"Samples saved to {os.path.abspath(output_file)}")

    return samples_np


def visualize_results(samples, gibbs_measure, show_plot=True):
    """Create visualizations comparing empirical and theoretical distributions."""
    # Extract x and y components
    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    # Set up the figure
    fig = plt.figure(figsize=(6, 4))

    # Now create the 3D visualization with empirical histogram and theoretical surface
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    # Create histogram data
    hist, xedges, yedges = np.histogram2d(x_samples, y_samples, bins=20, density=True)

    # Compute centers of bins
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X_hist, Y_hist = np.meshgrid(x_centers, y_centers)

    # Plot the 3D histogram bars
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    # Transpose histogram to match meshgrid dimensions
    hist = hist.T

    # Plot histogram as bars
    ax1.bar3d(
        X_hist.flatten(),
        Y_hist.flatten(),
        np.zeros_like(hist.flatten()),
        dx,
        dy,
        hist.flatten(),
        color='skyblue',
        alpha=0.1,
        shade=True
    )

    # Create a finer grid for the theoretical surface
    x_surf = np.linspace(min(x_samples), max(x_samples), 50)
    y_surf = np.linspace(min(y_samples), max(y_samples), 50)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

    # Compute theoretical density
    gibbs_measure.compute_normalization()
    density = gibbs_measure.density()
    density_unfold = lambda *x: float(density(jnp.array(x)))
    Z_surf = np.zeros(X_surf.shape)
    for i in range(X_surf.shape[0]):
        for j in range(X_surf.shape[1]):
            Z_surf[i, j] = density_unfold(X_surf[i, j], Y_surf[i, j])

    # Plot theoretical surface
    ax1.plot_surface(
        X_surf, Y_surf, Z_surf,
        cmap='viridis',
        alpha=0.7,
        linewidth=0,
        antialiased=True
    )

    # Set labels and title
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Density')
    ax1.set_title(f'3D Comparison: {gibbs_measure.name} (Blue: Empirical, Surface: Theoretical)')

    # Adjust viewing angle
    ax1.view_init(elev=30, azim=30)

    plt.tight_layout()
    plt.savefig(f"./data/{gibbs_measure.name}_comparison_3d.png", dpi=300)
    if show_plot:
        plt.show()


if __name__ == '__main__':
    # Set run parameters
    generate_new_data = True
    show_plot = False
    # gibbs_measure = gm.GibbsMeasure('gaussian', lambda x: jnp.sum(x ** 2) / 2, 2, 'reals')
    gibbs_measure = gm.GibbsMeasure('double_well', lambda x: jnp.sum(x ** 4) - jnp.sum(x ** 2), 2, 'reals')

    if generate_new_data:
        samples = run_test(gibbs_measure, h=0.01, T=20, d=2, num_samples=10000)

    else:
        # Load previously generated data
        samples = np.load(f"./data/{gibbs_measure.name}_samples.npy")

    # Visualize loaded results
    visualize_results(samples, gibbs_measure, show_plot=show_plot)
    order_list = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2]]
    for order in order_list:
        emp_moment = np.mean(np.prod(np.power(samples, np.array(order)), axis=1))
        theory_moment, _ = gibbs_measure.compute_moments(np.array(order))
        assert np.isclose(emp_moment, theory_moment, atol=0.03)