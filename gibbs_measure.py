import jax
import jax.numpy as jnp
import scipy.integrate as spi


class GibbsMeasure:

    def __init__(self, name, potential, dim, state_space, normalization=None):
        self.name = name
        self.potential = potential
        self.dim = dim
        self.state_space = state_space
        self.normalization = None

    def unnormalized_density(self):
        return lambda x: jnp.exp(-self.potential(x))

    def compute_normalization(self, domain=None):
        # if domain is the reals, numerically integrate the unnormalized density
        if self.state_space == 'reals':
            # set default domain to be [-5, 5]^dim
            if domain is None:
                domain = self.dim * ((-5, 5),)
            self.normalization = spi.nquad(lambda *x: self.unnormalized_density()(jnp.array(x)), domain)
        else:
            raise ValueError('do not recognize state space!')

    def density(self):
        if self.normalization is None:
            raise ValueError('normalization not computed yet!')
        return lambda x: self.unnormalized_density(x) / self.normalization

    def compute_moments(self, order, domain=None):
        if self.state_space == 'reals':
            if domain is None:
                domain = self.dim * ((-5, 5),)
            def integrand(x):
                return jnp.prod(jnp.power(x, order)) * self.density()(x)
            print(integrand(jnp.array([1, 2])))
            return spi.nquad(integrand, domain)
