from amp import *
import jax


def compute_approx_mean(gradH, gradF, y, t, num_amp_steps, num_gd_steps, gd_step_size):
    m = amp(gradH, num_amp_steps, t, y)
    u = ngd(gradF, m[-1], num_gd_steps, gd_step_size, y)
    return u[-1]


key = jax.random.PRNGKey(4)
