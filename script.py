
import math
import numpy as np
import matplotlib.pyplot as plt
from kernel import Kernel

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times"
plt.rc('text', usetex=True)

def generate_random_sphere(n_train, m):
    u = np.random.randn(n_train, m)
    u /= np.sqrt(np.sum(u**2, axis=1))[:, np.newaxis]
    return u

def get_stepsize(T, gamma_0=1):
    gamma = np.arange(T, dtype=float)
    gamma += 1
    np.sqrt(gamma, out=gamma)
    np.divide(1, gamma, out=gamma)
    gamma *= gamma_0
    return gamma

def target_func(x, phases=0, omega=2*np.pi):
    out = x + phases
    out *= omega
    np.sin(out, out=out)
    return out

def median_error(y_pred, y_test, inplace=True):
    if inplace:
        err = y_pred
        err -= y_test
    else:
        err = y_pred - y_test
    err **= 2
    err = np.sum(err, axis=1)
    np.sqrt(err, out=err)
    np.abs(err, out=err)
    return err.mean()

# Hyperparameters
random_init = False # Random weight initialization
n_train = 10000     # Training samples
n_repr = 50         # Low rank representation
noise = 0           # Noise level
n_test = 100        # Testing samples

# Experiment parameters (dimensinon, experiments, stepsize)
ms = [1, 2, 3, 5, 7, 10, 13, 17, 21, 26, 31, 37, 43, 50]
num_exp = 10
gammas = [1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 2.5e0, 5.e0]
gammas = [1e-1]
ms = [1]

# Placeholders
error_w = np.zeros((num_exp, len(ms), len(gammas)))
error_f = np.zeros((num_exp, len(ms), len(gammas)))
full_error_w = np.zeros((n_train, num_exp, len(ms), len(gammas)))
full_error_f = np.zeros((n_train, num_exp, len(ms), len(gammas)))
K_train = np.zeros((1, n_repr))

# Testing set
x_test = np.linspace(0, 1, n_test, dtype=np.float32)[:, np.newaxis]

# Kernel
sigma = .1
kernel = Kernel('gaussian', sigma=sigma)

# Randomness control
np.random.seed(0)

# Function representation
x_repr = np.random.rand(n_repr, 1)
kernel.set_support(x_repr)
K_test = kernel(x_test).T

for i_e in range(num_exp):
    # Training samples
    x_train = np.random.rand(n_train, 1)

    for i_m, m in enumerate(ms):
        # Labels
        phases = np.random.rand(m)
        y_test = target_func(x_test, phases=phases)
        y_train = target_func(x_train, phases=phases)
        # y_train += noise * np.random.randn(*y_train.shape)
        # min_error = noise * np.sqrt(2) * math.gamma((m+1) / 2) / math.gamma(m/2)

        # Random questions
        u = generate_random_sphere(*y_train.shape)

        # Placeholders
        theta_w = np.empty((n_repr, m))
        theta_f = np.empty((n_repr, m))
        theta_ave_w = np.empty((n_repr, m))
        theta_ave_f = np.empty((n_repr, m))
        grad = np.empty(theta_w.shape)

        for i_g, gamma_0 in enumerate(gammas):
            gamma = get_stepsize(n_train, gamma_0)

            # Descent initialization
            if random_init:
                theta_w[:] = .3 * np.random.randn(*grad.shape)
                theta_f[:] = .3 * np.random.randn(*grad.shape)
            else:
                theta_w[:] = 0
                theta_f[:] = 0
            theta_ave_w[:] = 0
            theta_ave_f[:] = 0

            for i in range(n_train):
                # New point
                K_train[:] = kernel(x_train[i:i+1]).T

                # Weakly supervised update
                epsilon = np.sign((K_train @ theta_w - y_train[i]) @ u[i])
                grad[:] = 1
                grad *= u[i]
                grad *= K_train.T
                grad *= epsilon * gamma[i]
                theta_w -= grad

                # Fully supervised update
                grad[:] = 1
                grad *= K_train @ theta_f - y_train[i]
                grad /= np.sqrt(np.sum(grad[0]**2))
                grad *= K_train.T
                grad *= gamma[i]
                theta_f -= grad

                # Averaging iterates
                theta_ave_w *= i
                theta_ave_w += theta_w
                theta_ave_w /= (i+1)
                theta_ave_f *= i
                theta_ave_f += theta_f
                theta_ave_f /= (i+1)

                full_error_w[i, i_e, i_m, i_g] = median_error(K_test @ theta_ave_w, y_test)
                full_error_f[i, i_e, i_m, i_g] = median_error(K_test @ theta_ave_f, y_test)

            # Computing error
            error_w[i_e, i_m, i_g] = median_error(K_test @ theta_ave_w, y_test)
            error_f[i_e, i_m, i_g] = median_error(K_test @ theta_ave_f, y_test)
    print(i_e, end=', ')