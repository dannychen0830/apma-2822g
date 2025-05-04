import jax
import multiprocessing as mp


def vmap_multiple_run(mcmc_func, num_runs, main_key=jax.random.key(4)):
    """Run multiple MCMC simulations using vectorization with vmap."""
    keys = jax.random.split(main_key, num_runs)
    vmap_func = jax.vmap(mcmc_func)
    return vmap_func(keys)


def mp_multiple_run(mcmc_func, num_runs, arg, main_key=jax.random.key(4), num_processes=4):
    """Run multiple MCMC simulations using multiprocessing."""
    keys = jax.random.split(main_key, num_runs)
    with mp.Pool(num_processes) as pool:
        results = pool.starmap(mcmc_func, [(k,) + arg for k in keys])
    # results = [mcmc_func(*((k,) + arg)) for k in keys]
    return results