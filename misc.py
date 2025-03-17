import jax.numpy as jnp
import scipy.integrate as spi


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
