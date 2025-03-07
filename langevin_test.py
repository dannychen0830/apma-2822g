import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import langevin
import numpy as np
import os
from scipy import integrate
from matplotlib.colors import LogNorm
import time


def gaussian_potential(x):
    """2D quadratic potential: V(x) = |x|^2/2"""
    return jnp.sum(x ** 2) / 2


def double_well_potential(x):
    """2D double well potential: V(x) = (x₁-1)²(x₁+1)² + (x₂-1)²(x₂+1)²"""
    return (x[0] - 1) ** 2 * (x[0] + 1) ** 2 + (x[1] - 1) ** 2 * (x[1] + 1) ** 2


def gaussian_density(x, y):
    """Theoretical density for 2D Gaussian: π(x) ∝ exp(-|x|²/2)"""
    return np.exp(-(x ** 2 + y ** 2) / 2) / (2 * np.pi)


def double_well_density(x, y):
    """Theoretical density for 2D double well: π(x) ∝ exp(-(x₁-1)²(x₁+1)² - (x₂-1)²(x₂+1)²)"""
    potential = (x - 1) ** 2 * (x + 1) ** 2 + (y - 1) ** 2 * (y + 1) ** 2
    unnormalized = np.exp(-potential)

    # Compute normalization constant via numerical integration
    def integrand(x, y):
        potential = (x - 1) ** 2 * (x + 1) ** 2 + (y - 1) ** 2 * (y + 1) ** 2
        return np.exp(-potential)

    # Use a sufficiently large but finite integration domain
    normalization, _ = integrate.dblquad(integrand, -5, 5, lambda x: -5, lambda x: 5)

    return unnormalized / normalization


def potential_types(type_name):
    if type_name == 'guassian':
        return gaussian_potential, gaussian_density
    elif type_name == 'double-well':
        return double_well_potential, double_well_density
    else:
        raise ValueError('do not recognize potential type!')


def compute_theoretical_moments(density):
    """Compute theoretical moments for the given potential type."""


    # Function to integrate for normalization
    def integrand_norm(x, y):
        return density(x, y)

    # Functions to integrate for moments
    def integrand_mean_x(x, y):
        return x * density(x, y)

    def integrand_mean_y(x, y):
        return y * density(x, y)

    def integrand_second_x(x, y):
        return x ** 2 * density(x, y)

    def integrand_second_y(x, y):
        return y ** 2 * density(x, y)

    def integrand_third_x(x, y):
        return x ** 3 * density(x, y)

    def integrand_third_y(x, y):
        return y ** 3 * density(x, y)

    # Integrate over a sufficiently large domain
    bounds = (-5, 5)
    Z, _ = integrate.dblquad(integrand_norm, bounds[0], bounds[1], lambda x: bounds[0], lambda x: bounds[1])

    mean_x, _ = integrate.dblquad(integrand_mean_x, bounds[0], bounds[1], lambda x: bounds[0], lambda x: bounds[1])
    mean_y, _ = integrate.dblquad(integrand_mean_y, bounds[0], bounds[1], lambda x: bounds[0], lambda x: bounds[1])

    second_x, _ = integrate.dblquad(integrand_second_x, bounds[0], bounds[1], lambda x: bounds[0],
                                    lambda x: bounds[1])
    second_y, _ = integrate.dblquad(integrand_second_y, bounds[0], bounds[1], lambda x: bounds[0],
                                    lambda x: bounds[1])

    third_x, _ = integrate.dblquad(integrand_third_x, bounds[0], bounds[1], lambda x: bounds[0],
                                   lambda x: bounds[1])
    third_y, _ = integrate.dblquad(integrand_third_y, bounds[0], bounds[1], lambda x: bounds[0],
                                   lambda x: bounds[1])

    # Normalize by Z
    return {
        "1st_moment": np.array([mean_x / Z, mean_y / Z]),
        "2nd_moment": np.array([second_x / Z, second_y / Z]),
        "3rd_moment": np.array([third_x / Z, third_y / Z])
    }


def run_test(potential_name, h=0.01, T=10, d=2, num_samples=10000, init_val=lambda: None):
    """Run a test with the given potential function."""
    print(f"\n=== Running test for {potential_name} potential ===")
    potential_func, __ = potential_types(potential_name)

    # Start timing
    start_time = time.time()

    # Define gradient of potential
    gradV = jax.grad(potential_func)

    # Create a key for random number generation
    key = jax.random.PRNGKey(int(time.time()) % 100000)

    print(f"Running {num_samples} MALA simulations...")

    # Run the parallelized implementation
    samples = langevin.run_multiple_mala(potential_func, gradV, h, T, d, num_samples,
                                         init_val=init_val, main_key=key)

    # Convert to numpy array
    samples_np = np.array(samples)

    # End timing
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")

    # Save samples to file
    output_file = f"{potential_name}_samples.npy"
    with open(output_file, 'wb') as f:
        np.save(f, samples_np)
    print(f"Samples saved to {os.path.abspath(output_file)}")

    # Calculate empirical moments
    empirical_moments = {
        "1st_moment": np.mean(samples_np, axis=0),
        "2nd_moment": np.mean(samples_np ** 2, axis=0),
        "3rd_moment": np.mean(samples_np ** 3, axis=0)
    }

    # Get theoretical moments
    theoretical_moments = compute_theoretical_moments(potential_name)

    # Print moment comparison
    print("\nMoment comparison:")
    print("Moment | Empirical (x, y) | Theoretical (x, y)")
    print("-" * 50)
    print(
        f"1st    | ({empirical_moments['1st_moment'][0]:.4f}, {empirical_moments['1st_moment'][1]:.4f}) | ({theoretical_moments['1st_moment'][0]:.4f}, {theoretical_moments['1st_moment'][1]:.4f})")
    print(
        f"2nd    | ({empirical_moments['2nd_moment'][0]:.4f}, {empirical_moments['2nd_moment'][1]:.4f}) | ({theoretical_moments['2nd_moment'][0]:.4f}, {theoretical_moments['2nd_moment'][1]:.4f})")
    print(
        f"3rd    | ({empirical_moments['3rd_moment'][0]:.4f}, {empirical_moments['3rd_moment'][1]:.4f}) | ({theoretical_moments['3rd_moment'][0]:.4f}, {theoretical_moments['3rd_moment'][1]:.4f})")

    # Create visualization
    visualize_results(samples_np, potential_name)

    return samples_np


def visualize_results(samples, potential_name):
    """Create visualizations comparing empirical and theoretical distributions."""
    # Extract x and y components
    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    # Set up the figure
    fig = plt.figure(figsize=(6, 4))

    # Now create the 3D visualization with empirical histogram and theoretical surface
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')

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
        dx * 0.9,
        dy * 0.9,
        hist.flatten(),
        color='skyblue',
        alpha=0.6,
        shade=True
    )

    # Create a finer grid for the theoretical surface
    x_surf = np.linspace(min(x_samples), max(x_samples), 50)
    y_surf = np.linspace(min(y_samples), max(y_samples), 50)
    X_surf, Y_surf = np.meshgrid(x_surf, y_surf)

    # Compute theoretical density
    if potential_name == "gaussian":
        Z_surf = gaussian_theoretical_density(X_surf, Y_surf)
    else:  # double_well
        Z_surf = double_well_theoretical_density(X_surf, Y_surf)

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
    ax1.set_title(f'3D Comparison: {potential_name} (Blue: Empirical, Surface: Theoretical)')

    # Adjust viewing angle
    ax1.view_init(elev=30, azim=30)

    plt.tight_layout()
    plt.savefig(f"{potential_name}_comparison_3d.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # Set run parameters
    generate_new_data = True
    potential_type = 'gaussian'

    if generate_new_data:
        gaussian_samples = run_test(potential_type, h=0.01, T=20, d=2, num_samples=100)

    else:
        # Load previously generated data
        gaussian_samples = np.load("gaussian_samples.npy")
        double_well_samples = np.load("double_well_samples.npy")

        # Visualize loaded results
        visualize_results(gaussian_samples, "gaussian")
        visualize_results(double_well_samples, "double_well")