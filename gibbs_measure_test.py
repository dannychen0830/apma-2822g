import pytest
import gibbs_measure as gm
import jax.numpy as jnp
import itertools as it

@pytest.mark.parametrize(
    'dim_list', [(1,), (2,)]
)
def test_gibbs_measure(dim_list):
    for d in dim_list:
        gaussian = gm.GibbsMeasure('gaussian', lambda x: jnp.sum(x ** 2), d, 'reals')
        assert jnp.isclose(gaussian.compute_normalization(), (2 * jnp.pi) ** (d / 2), atol=1e-3)
        # test all moments of degree 2
        for order in it.product(range(d), range(d)):
            assert jnp.isclose(gaussian.compute_moments(order), int(order[0] == order[1]), atol=1e-3)
