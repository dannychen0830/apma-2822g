import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gibbs_measure import GibbsMeasure
from langevin import run_mala, normal_init
from mcmc import vmap_multiple_run

rng = np.random.default_rng(50)

def symmetrize(A):
    return (A + A.T) / 2

N = 20
b = 0.25

G = b * symmetrize(rng.standard_normal((N, N)))

def L2N(x):
    return jnp.square(jnp.sum(jnp.power(x, 2))) / N

def Z(s):
    return (b * s) ** 2

def D1Z(s):
    return 2 * (b ** 2) * s

def H(x, y):
    return (x.T @ G @ x) / jnp.sqrt(N) + x.T @ y

def D1H(x):
    L = np.zeros(N)

    for i in range(N):
        for j in range(N):
            L[i] += (2 / np.sqrt(N)) * G[i, j] * x[j]

    return L

rng = np.random.default_rng()

# Add TAP Free Energy + Correction.
def approximate_mean_computation(y, t, K):
    W = [np.zeros(N)]
    M = [np.zeros(N) for _ in range(2)]

    Q = [0.0]

    for k in range(K):
        Q.append(1 - 1 / (1 + D1Z(Q[k]) + t))
        W.append(D1H(M[k + 1]) + y - 2 * (b ** 2) * (1 - Q[k]) * M[k])
        M.append((1 - Q[k + 1]) * W[k + 1])

    return M[-1]

# Determine Value of Constants.
def subroutine(K = 35, S = 1 / np.power(N, 2), T = 20 * np.power(N, 1)):
    E = [np.zeros(N)]
    B = [np.zeros(N)]

    for i in range(T):
        if i % 100 == 0:
            print("Subroutine Iteration: " + str(i))

        M = approximate_mean_computation(E[i], S * i, K)
        W = rng.standard_normal(N)

        B.append(B[i] + np.sqrt(S) * W)
        E.append(E[i] + S * M + np.sqrt(S) * W)

    return E[-1]

def R(s):
    c = jnp.maximum(s, 1e-8)

    return jnp.where(c > 0.1, 1.5 * (c - 0.01 / c - 0.2 * jnp.log(c / 0.1)), 0)

def orthogonal(y):
    Q, _ = jnp.linalg.qr(jnp.hstack([y.reshape(-1, 1), rng.standard_normal((N, N - 1))]))

    return Q[:, 1:]

def projection(z, y):
    return (y / jnp.sqrt(L2N(y)) + orthogonal(y) @ z) / jnp.sqrt(1 + L2N(z))

def HP(z, y):
    return H(projection(z, y), y) - (N / 2) * jnp.log(1 + L2N(z))

def VP(z, y, T = 20):
    return HP(z, y) - (T * N / 2) * R(L2N(z))

def density(z, y, T = 20):
    return np.exp(VP(z, y, T))

# Use MALA.
def metropolis_hastings(y, sample_count = 1e4, proposal_std = 1.0, burn = 5e2):
    current = np.zeros(N - 1)
    samples = []
    accepted = 0

    total = int(sample_count + burn)

    for _ in range(total):
        if _ % 1e1 == 0:
            print("Iteration: " + str(_))

        proposed = current + np.random.normal(scale = proposal_std, size = N - 1)

        P1 = density(current, y)
        P2 = density(proposed, y)

        if P2 == 0:
            continue

        accept_ratio = min(1, P1 / P2)

        if np.random.rand() < accept_ratio:
            current = proposed
            accepted += 1

            if _ >= burn:
                samples.append(current)

    print("Acceptance Rate: " + str(accepted / total))
    return np.array(samples)

# testM = metropolis_hastings(subroutine())
T = 20
h = 0.01
T_mala = 20
final_y = subroutine()
nu_proj = GibbsMeasure('nu_proj', lambda x: VP(x, final_y, T=20), N - 1, 'real')
mala_key = jax.random.PRNGKey(0)
sample = run_mala(nu_proj.potential,
                  jax.grad(nu_proj.potential),
                  h,
                  T_mala,
                  N - 1,
                  normal_init(N-1),
                  key=mala_key,
                  verbose=True)

print(sample)
