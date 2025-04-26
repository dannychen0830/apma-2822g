import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
from functools import partial


def reconfigure_jax(core_count=8):
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={core_count}"
    # Need to re-initialize JAX with new configuration
    import importlib
    importlib.reload(jax)

    print(f"JAX devices available: {jax.device_count()}")
    print(f"Device types: {[d.device_kind for d in jax.devices()]}")


def vmap_multiple_run(mcmc_func, num_runs, main_key=jax.random.PRNGKey(4), batch_size=None, use_pmap=False):
    """Run multiple MCMC simulations using vectorization with vmap and optional pmap.

    Optimized for multicore CPUs by configuring virtual devices when needed.

    Args:
        mcmc_func: Function to vectorize
        num_runs: Number of total simulation runs
        main_key: Main PRNG key for generating random seeds
        batch_size: Size of batches for processing (None means process all at once)
        use_pmap: Whether to use pmap for multi-device parallelism

    Returns:
        Array of results from all simulation runs
    """
    # Generate all keys upfront
    keys = jax.random.split(main_key, num_runs)

    # When batch_size is None and use_pmap is True, use all virtual devices
    if batch_size is None and use_pmap and jax.device_count() > 1:
        device_count = jax.device_count()
        # Divide workload among available devices
        runs_per_device = (num_runs + device_count - 1) // device_count

        # Create batches of runs for each device
        # We need to pad to ensure each device gets same number of runs
        total_padded_size = runs_per_device * device_count
        padded_keys = jnp.pad(
            keys,
            ((0, total_padded_size - keys.shape[0]), (0, 0)),
            mode='constant'
        )

        # Reshape for pmap (devices, runs_per_device, key_size)
        reshaped_keys = padded_keys.reshape(device_count, runs_per_device, -1)

        # Use pmap+vmap
        pmapped_func = jax.pmap(jax.vmap(mcmc_func))
        results = pmapped_func(reshaped_keys)

        # Reshape and trim padding
        results = results.reshape(-1, *results.shape[2:])
        results = results[:num_runs]

        return results

    elif batch_size is None:
        # Original behavior: run all at once with vmap
        vmap_func = jax.vmap(mcmc_func)
        return vmap_func(keys)

    else:
        # Process in batches
        results = []
        num_batches = (num_runs + batch_size - 1) // batch_size  # Ceiling division

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_runs)
            batch_keys = keys[start_idx:end_idx]

            if use_pmap and jax.device_count() > 1:
                # Split batch across devices
                devices_to_use = min(jax.device_count(), batch_keys.shape[0])
                keys_per_device = (end_idx - start_idx + devices_to_use - 1) // devices_to_use

                # Pad to be divisible by device count
                padded_size = keys_per_device * devices_to_use
                padded_keys = jnp.pad(
                    batch_keys,
                    ((0, padded_size - batch_keys.shape[0]), (0, 0)),
                    mode='constant'
                )

                # Reshape for pmap
                reshaped_keys = padded_keys.reshape(devices_to_use, keys_per_device, -1)

                # Run with pmap+vmap
                pmapped_func = jax.pmap(jax.vmap(mcmc_func))
                batch_results = pmapped_func(reshaped_keys)

                # Reshape and trim padding
                batch_results = batch_results.reshape(-1, *batch_results.shape[2:])
                batch_results = batch_results[:end_idx - start_idx]
            else:
                # Just use vmap
                vmap_func = jax.vmap(mcmc_func)
                batch_results = vmap_func(batch_keys)

            results.append(batch_results)

        # Combine all batches
        return jnp.concatenate(results)