import jax


def vmap_multiple_run(mcmc_func, num_runs, main_key=jax.random.key(4)):
    """Run multiple MCMC simulations using vectorization with vmap."""
    keys = jax.random.split(main_key, num_runs)
    vmap_func = jax.vmap(mcmc_func)
    return vmap_func(keys)