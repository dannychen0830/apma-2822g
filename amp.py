import math as mt
import numpy as np
import matplotlib as mlt

# Using the SK Hamiltonian on the N-Sphere

N = 100

# Initialize Gaussian Matrix
A = np.random.randn(N, N)

# Symmetrizes the Matrix A
G = (A + A.T)/2

# Gradient of the Hamiltonian
def gradH(s):
    """
    Computes the gradient of H.

    Parameters:
    s (Array of Floats): The value of the spin, takes values in R^N.

    Returns:
    An array which represents the evaluation of the gradient of H at s.
    """

    L = np.ones(N)

    for i in range(N):
        # Temporary Variable
        v = 0

        for j in range(N):
            v += (2/mt.sqrt(N)) * G[i, j] * s[j]

        L[i] = v

    return L

def gradL(s):
    """
    Computes the derivative of Log (cf. Eq. 2.5).

    Parameters:
    s (Array of Floats): The value of the spin, takes values in R^N.

    Returns:
    An array which represents the evaluation of the gradient of Log at u.
    """

    # Temporary Function
    f = lambda v: -1/(1 - v)

    I = np.linalg.norm(s)

    return f(I/N) * (2/N) * s

def gradT(s):
    """
    Computes the derivative of Theta (cf. Eq. 2.4).

    Parameters:
    s (Array of Floats): The value of the spin, takes values in R^N.

    Returns:
    An array which represents the evaluation of the gradient of Theta at s.
    """

    I = np.linalg.norm(s)

    return 2 * (I/N - 1) * (2/N) * s

# Gradient of the TAP Free Energy
def gradF(s, y):
    """
    Computes the gradient of F.

    Parameters:
    s (Array of Floats): The value of the spin, takes values in R^N.
    y (Array of Floats): A spatial variable, takes values in R^N.

    Returns:
    An array which represents the evaluation of the gradient of N at (s; y).
    """

    return gradH(s) + y + (N/2) * (gradT(s) + gradL(s))

# AMP Iteration
def amp(K, t, y):
    """
    Computes an AMP iteration.

    Parameters:
    K (Int): The number of AMP steps - 1.
    t (Float): A time parameter, is positive.
    y (Array of Floats): A spatial variable, takes values in R^N.

    Returns:
    A list whose last element is an approximation of the mean function at time t, before NGD.
    """

    # q: Array of Floats
    q = np.zeros(K)
    c = np.zeros(K)

    # w: Array of Vectors
    w = [np.zeros(N) for _ in range(K)]

    # Messages
    M = [np.zeros(N) for _ in range(2)]

    for k in range(1, K):
        # Temporary Variable
        v = 2 * q[k - 1] + t + 1

        # q-Update
        q[k] = 1 - 1/v
        c[k] = 1 - q[k]

        # w-Update
        w[k] = gradH(M[k]) + y - 2 * c[k - 1] * M[k - 1]

        # Message Update
        M.append(c[k] * w[k])

    return M

def ngd(X, K, z, y):
    """
    Computes a NGD iteration.

    Parameters:
    X (Array of Floats): The starting point of the NGD.
    K (Int): The number of NGD steps.
    z (Float): The step-size parameter.
    y (Array of Floats): A spatial variable, takes values in R^N.

    Returns:
    A list whose last element is an approximation of the mean function at time t, after NGD.
    """

    L = [X]

    for i in range(K):
        L.append(L[i] - z * gradF(L[i], y))

    return L