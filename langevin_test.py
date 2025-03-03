import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import langevin
import numpy as np
import os


if __name__ == '__main__':
    read_data = True

    if read_data:
        output_file = "mala_simulation_results.npy"
        data_np = np.load(output_file)

        plt.hist(data_np)
        plt.show()

    else:
        # Define potential function and parameters
        V = lambda x: jnp.linalg.norm(x) ** 2 / 2
        h = 0.01
        T = 10
        d = 1
        gradV = jax.grad(V)

        # Number of simulations to run in parallel
        num_data = 1000

        # Create a single key
        key = jax.random.PRNGKey(4)

        print(f"Running {num_data} MALA simulations...")

        # Run the parallelized implementation
        data = langevin.run_multiple_mala(V, gradV, h, T, d, num_data, init_val=0.01, main_key=key)

        # Convert to numpy array for saving
        data_np = np.array(data)

        # Save the data to a file
        output_file = "mala_simulation_results.npy"
        with open(output_file, 'wb') as f:
            np.save(f, data_np)

        print(f"Data saved to {os.path.abspath(output_file)}")

        # Print some summary statistics
        print(f"Mean of results: {jnp.mean(data)}")
        print(f"Std dev of results: {jnp.std(data)}")
        print(f"Min value: {jnp.min(data)}")
        print(f"Max value: {jnp.max(data)}")

