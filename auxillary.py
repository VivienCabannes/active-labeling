
import numpy as np


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