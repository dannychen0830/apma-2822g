import numpy as np
import jax
import jax.numpy as jnp
import scipy.integrate as spi
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def plot_projection(samples, direction_list, gibbs_measure, direction_name=None,
                    show_plot=True, test_sample=None, save_fig=None, density_functions=None):
    """Plot the projection of the samples onto the specified direction and compare with theoretical density

    Parameters:
    -----------
    samples : array-like
        High dimensional samples
    direction_list : list of array-like
        List of directions to project onto
    gibbs_measure : object
        Gibbs measure object with name attribute
    direction_name : list of str, optional
        Names for the directions
    show_plot : bool, optional
        Whether to show the plot
    test_sample : array-like, optional
        Additional samples to compare against
    save_fig : str, optional
        Path to save the figure
    density_functions : list of callable, optional
        List of functions representing the theoretical density for each projection
    """

    projection = [samples @ direction for direction in direction_list]
    # If test_sample is provided, add it to the projection
    if test_sample is not None:
        test_projection = [test_sample @ direction for direction in direction_list]

    # Set up the figure
    num_directions = len(direction_list)
    nrow = int(jnp.sqrt(num_directions))
    ncol = num_directions // nrow + 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(6, 4))

    # Handle the case where axs is not a 2D array (when there's only one subplot)
    if num_directions == 1:
        axs = np.array([[axs]])

    # Track what elements are present for the legend
    has_empirical = False
    has_test_sample = False
    has_theoretical = False

    # Create histogram data
    for i, direction in enumerate(direction_list):
        ridx = i // ncol
        cidx = i % ncol

        # Get current axis
        if nrow == 1 and ncol == 1:
            ax = axs[0, 0]
        elif nrow == 1:
            ax = axs[cidx]
        elif ncol == 1:
            ax = axs[ridx]
        else:
            ax = axs[ridx, cidx]

        hist, edges = np.histogram(projection[i], bins=20, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        dx = edges[1] - edges[0]
        ax.bar(centers, hist, width=dx, color='skyblue', alpha=0.7)
        has_empirical = True

        if test_sample is not None:
            test_hist, test_edge = np.histogram(test_projection[i], bins=edges, density=True)
            test_centers = (test_edge[:-1] + test_edge[1:]) / 2
            test_dx = test_edge[1] - test_edge[0]
            ax.bar(centers, test_hist, width=dx, color='red', alpha=0.3, zorder=2)
            has_test_sample = True

        # Plot theoretical density if provided
        if density_functions is not None and i < len(density_functions) and density_functions[i] is not None:
            # Create a smooth x-axis for the density function
            x_min, x_max = edges[0], edges[-1]
            x = np.linspace(x_min, x_max, 200)
            # Compute theoretical density
            try:
                density = jnp.vectorize(density_functions[i])(x)
                ax.plot(x, density, 'r-', linewidth=2, zorder=3)
                has_theoretical = True
            except Exception as e:
                print(f"Error plotting theoretical density for direction {i}: {e}")

        # Add title
        if direction_name is None:
            title = f"Projection on {np.round(direction, decimals=1)}"
        else:
            title = f"Projection on {direction_name[i]}"
        ax.set_title(title)

    # Hide empty subplots
    for i in range(num_directions, nrow * ncol):
        ridx = i // ncol
        cidx = i % ncol
        if nrow == 1:
            if cidx < len(axs):
                axs[cidx].set_visible(False)
        elif ncol == 1:
            if ridx < len(axs):
                axs[ridx].set_visible(False)
        else:
            if ridx < len(axs) and cidx < len(axs[0]):
                axs[ridx, cidx].set_visible(False)

    # Create a single legend for the entire figure
    legend_elements = []
    if has_empirical:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='skyblue', alpha=0.7, label='SL'))
    if has_test_sample:
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3, label='MALA'))
    if has_theoretical:
        legend_elements.append(Line2D([0], [0], color='r', linewidth=2, label='true density'))

    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0),
                   ncol=len(legend_elements), bbox_transform=fig.transFigure, fontsize='small')
        # Add some extra space at the bottom for the legend
        plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    # Adjust after tight_layout to ensure legend is visible
    if legend_elements:
        plt.subplots_adjust(bottom=0.15)

    if save_fig is None:
        plt.savefig(f"./data/{gibbs_measure.name}_projection.png", dpi=300)
    else:
        plt.savefig(save_fig, dpi=300)
    if show_plot:
        plt.show()

    return fig, axs


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

