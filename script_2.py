
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

useful_col = [49, 50, 52, 54, 55, 56, 57]
df = pd.read_csv('bottle.csv', usecols=useful_col, dtype=float)
print(df.count())
df.dropna(inplace=True)
df.set_index(np.arange(len(df)), inplace=True)

input_col = ['R_Depth', 'R_TEMP', 'R_SALINITY', 'R_SVA', 'R_DYNHT']
output_col = ['R_O2', 'R_O2Sat']

n_train = 500000
n_repr = 1000
causal = False

if causal:
    # Training set
    x_train = df.iloc[:n_train][input_col].values
    y_train = df.iloc[:n_train][output_col].values

    # Testing set
    x_test = df.iloc[n_train:][input_col].values
    y_test = df.iloc[n_train:][output_col].values
else:
    # Training set
    index_train = np.random.choice(len(df), n_train, replace=False)
    x_train = df.iloc[index_train][input_col].values
    y_train = df.iloc[index_train][output_col].values

    # Testing set
    index_test = np.ones(len(df), dtype=np.bool_)
    index_test[index_train] = False
    x_test = df.iloc[index_test][input_col].values
    y_test = df.iloc[index_test][output_col].values

    # Normalization
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)

    x_train -= x_mean
    x_train /= x_std
    x_test -= x_mean
    x_test /= x_std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std

# Function parameterization
x_repr = x_train[np.random.choice(len(x_train), n_repr, replace=False)]
y_test = df.iloc[index_test][output_col].values

# Small testset
n_small_test = 1000
ind = np.random.choice(len(x_test), n_small_test, replace=True)
x_small_test = x_test[ind]
y_small_test = y_test[ind]

# Kernel
sigma = .1
kernel = Kernel('gaussian', sigma=sigma)
kernel.set_support(x_repr)

ns_test = 1000
K_test = kernel(x_small_test)
K_train = np.zeros((1, n_repr))

# Randomness control
np.random.seed(0)

# Random questions
u = generate_random_sphere(*y_train.shape)

# Placeholders
theta_w = np.empty((n_repr, 2))
theta_ave_w = np.empty((n_repr, 2))
theta_f = np.empty((n_repr, 2))
theta_ave_f = np.empty((n_repr, 2))
grad = np.empty(theta_w.shape)

gammas = np.logspace(1, 3, num=5)

# Placeholders
n_small_train = 20000
full_error_w = np.zeros((n_small_train, len(gammas)))
full_error_f = np.zeros((n_small_train, len(gammas)))
for i_g, gamma_0 in enumerate(gammas):
    gamma = get_stepsize(n_train, gamma_0)

    # Descent initialization
    random_init = True
    if random_init:
        theta_w[:] = .3 * np.random.randn(*grad.shape)
        theta_f[:] = .3 * np.random.randn(*grad.shape)
    else:
        theta_w[:] = 0
        theta_f[:] = 0
    theta_ave_f[:] = 0

    for i in range(n_small_train):
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

        y_pred_w = K_test @ theta_ave_w
        y_pred_w *= y_std
        y_pred_w += y_mean
        full_error_w[i, i_g] = median_error(y_pred_w, y_small_test)

        y_pred_f = K_test @ theta_ave_f
        y_pred_f *= y_std
        y_pred_f += y_mean
        full_error_f[i, i_g] = median_error(y_pred_f, y_small_test)
        if not i % 1000:
            print(i//1000, end=',')

        if np.isnan(theta_ave_w).any():
            break


exit()