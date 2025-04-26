import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Noise intensity (can be adjusted)
noise_strength = 10.0

# Time settings
dt = 0.01
total_time = 20
steps = int(total_time / dt)
time = np.linspace(0, total_time, steps)

# Number of ensemble members
num_ensembles = 50


def lorenz_drift(x, y, z):
    """Calculate drift term (Lorenz system)"""
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def simulate_sde():
    """Simulate an ensemble of Lorenz SDEs"""
    # Initialize with slightly different starting positions
    base_init = np.array([1.0, 1.0, 1.0])
    ensemble_paths = np.zeros((num_ensembles, steps, 3))

    for e in range(num_ensembles):
        # Add small perturbation to initial conditions for each ensemble member
        perturbation = np.random.normal(0, 0.1, 3)
        x = base_init + perturbation
        ensemble_paths[e, 0] = x

        # Euler-Maruyama method for SDE integration
        for i in range(1, steps):
            drift = lorenz_drift(x[0], x[1], x[2])
            # Brownian motion term (stochastic part)
            dW = np.random.normal(0, np.sqrt(dt), 3)
            # Update using Euler-Maruyama
            x = x + drift * dt + noise_strength * dW
            ensemble_paths[e, i] = x

    return ensemble_paths


# Run simulation
ensemble_paths = simulate_sde()

# Set up 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initial plot setup
lines = []
for e in range(num_ensembles):
    line, = ax.plot([], [], [], lw=1, alpha=0.7)
    lines.append(line)

# Plot title with parameters
ax.set_title(f"Stochastic Lorenz Attractor\nσ={sigma}, ρ={rho}, β={beta}, noise={noise_strength}")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set axis limits based on data
max_val = np.max(np.abs(ensemble_paths)) * 1.1
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([0, max_val * 2])


# Animation function
def update(frame):
    display_points = 100  # Number of points to display in tail
    start = max(0, frame - display_points)

    for e, line in enumerate(lines):
        x = ensemble_paths[e, start:frame + 1, 0]
        y = ensemble_paths[e, start:frame + 1, 1]
        z = ensemble_paths[e, start:frame + 1, 2]
        line.set_data(x, y)
        line.set_3d_properties(z)

    # Rotate view for 3D effect
    ax.view_init(elev=30, azim=frame / 2)
    return lines


# Create animation
ani = FuncAnimation(fig, update, frames=range(1, steps, 5),
                    interval=20, blit=False)

plt.tight_layout()
plt.show()


# If you want to save the animation
# ani.save('stochastic_lorenz.mp4', writer='ffmpeg', fps=30)

# Alternative: Plot static trajectories if animation is too heavy
def plot_static_trajectories():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each ensemble trajectory with different colors
    for e in range(num_ensembles):
        ax.plot(ensemble_paths[e, :, 0],
                ensemble_paths[e, :, 1],
                ensemble_paths[e, :, 2],
                lw=0.8, alpha=0.7)

    ax.set_title(f"Stochastic Lorenz Attractor Ensemble\nσ={sigma}, ρ={rho}, β={beta}, noise={noise_strength}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()

# Uncomment this to run the static plot instead of animation
# plot_static_trajectories()