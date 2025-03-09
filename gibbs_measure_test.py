import pytest
import gibbs_measure as gm
import jax.numpy as jnp
import itertools as it


@pytest.mark.parametrize(
    'dim_list', [(1,), (2,)]
)
def test_gibbs_measure(dim_list):
    for d in dim_list:
        gaussian = gm.GibbsMeasure('gaussian', lambda x: jnp.sum(x ** 2) / 2, d, 'reals')
        normalization, _ = gaussian.compute_normalization()
        assert jnp.isclose(normalization, (2 * jnp.pi) ** (d / 2), atol=1e-3)
        # test all moments of degree 2
        for idx in it.product(range(d), range(d)):
            order = jnp.array([idx.count(i) for i in range(d)])
            moments, __ = gaussian.compute_moments(order)
            eq_fun = lambda i, j: 1 if i == j else 0
            assert jnp.isclose(moments, eq_fun(idx[0], idx[1]), atol=1e-3)

