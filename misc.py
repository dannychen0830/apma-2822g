import numpy as np
import jax.numpy as jnp
import scipy.integrate as spi
import matplotlib.pyplot as plt


# TODO: unit test integrators

def _spherical_to_cartesian(phis, dim):
    """Convert spherical coordinates to Cartesian coordinates on the unit sphere.

    Args:
        phis: Array of angular coordinates (phi_1, phi_2, ..., phi_{d-1})
              where phi_i is in [0, pi] for i < d-1 and phi_{d-1} is in [0, 2pi]

    Returns:
        Cartesian coordinates on the unit sphere
    """
    x = jnp.zeros(dim)

    # First coordinate is special
    x = x.at[0].set(jnp.cos(phis[0]))

    # Middle coordinates
    for i in range(1, dim - 1):
        term = jnp.sin(phis[0])
        for j in range(1, i):
            term = term * jnp.sin(phis[j])
        term = term * jnp.cos(phis[i])
        x = x.at[i].set(term)

    # Last coordinate is special too
    if dim > 1:
        term = jnp.sin(phis[0])
        for j in range(1, dim - 1):
            term = term * jnp.sin(phis[j])
        x = x.at[dim - 1].set(term)

    return x


def _spherical_jacobian(phis, dim):
    """Compute the Jacobian determinant for spherical coordinates.

    Args:
        phis: Array of angular coordinates

    Returns:
        Jacobian determinant value
    """
    jacobian = 1.0
    for i in range(dim - 2):
        jacobian *= jnp.sin(phis[i]) ** (dim - i - 2)
    return jacobian


def _get_sphere_integration_limits(dim):
    """Get the integration limits for spherical coordinates.

    Returns:
        List of (lower, upper) tuples for each angular coordinate
    """
    # phi_1 to phi_{d-2} range from 0 to pi, phi_{d-1} ranges from 0 to 2pi
    return [(0, jnp.pi)] * (dim - 2) + [(0, 2 * jnp.pi)]


def _create_sphere_integrand(func, dim):
    """Create an integrand function for integration over the sphere.

    Args:
        func: Function to integrate (taking Cartesian coordinates as input)

    Returns:
        Function taking spherical coordinates suitable for nquad integration
    """

    def integrand(*phis):
        x = _spherical_to_cartesian(phis, dim)
        return func(x) * _spherical_jacobian(phis, dim)

    return integrand


def _integrate_over_sphere(func, dim):
    """Integrate a function over the unit sphere.

    Args:
        func: Function to integrate (taking Cartesian coordinates as input)

    Returns:
        Tuple of (result, error estimate)
    """
    if dim == 1:
        # Special case for 1D: just two points {-1, 1}
        left_point = jnp.array([-1.0])
        right_point = jnp.array([1.0])
        result = func(left_point) + func(right_point)
        return result, 0.0
    else:
        # Use scipy's nquad for multi-dimensional integration
        limits = _get_sphere_integration_limits(dim)
        integrand = _create_sphere_integrand(func, dim)
        return spi.nquad(integrand, limits)


def plot_projection(samples, direction_list, gibbs_measure, direction_name=None, show_plot=True):
    """Plot the projection of the samples onto the specified direction and compare with theretical density"""
    projection = [samples @ direction for direction in direction_list]

    # Set up the figure
    num_directions = len(direction_list)
    nrow = int(jnp.sqrt(num_directions))
    ncol = num_directions // nrow + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(6, 4))

    # Create histogram data
    for i, direction in enumerate(direction_list):
        ridx = i // ncol
        cidx = i % ncol

        hist, edges = np.histogram(projection[i], bins=20, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        dx = edges[1] - edges[0]
        axs[ridx, cidx].bar(centers, hist, width=dx, color='skyblue')
        # TODO: rounding stil has decimals????
        axs[ridx, cidx].set_title(f"Projection on {np.round(direction, decimals=1) if direction_name is None else direction_name[i]}")

    plt.tight_layout()
    plt.savefig(f"./data/{gibbs_measure.name}_projection.png", dpi=300)
    if show_plot:
        plt.show()


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
    if gibbs_measure.normalization is None:
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
    ax1.set_title(f'Histogram against theoretical density for {gibbs_measure.name} potential on {gibbs_measure.state_space}')

    # Adjust viewing angle
    ax1.view_init(elev=30, azim=30)

    plt.tight_layout()
    plt.savefig(f"./data/{gibbs_measure.name}_comparison_{gibbs_measure.state_space}.png", dpi=300)
    if show_plot:
        plt.show()


def test_moments(samples, gibbs_measure, order_list, atol=0.03):
    for order in order_list:
        emp_moment = jnp.mean(np.prod(np.power(samples, np.array(order)), axis=1))
        theory_moment, _ = gibbs_measure.compute_moments(np.array(order))
        print(f"Empirical moment: {emp_moment}, theoretical moment: {theory_moment}")
        assert jnp.isclose(emp_moment, theory_moment, atol=atol)

