
from statistics import median
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from activelabeling import (
    Kernel,
    data_dir,
    generate_random_sphere,
    get_stepsize,
    mean_error,
    median_error
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times"
plt.rc('text', usetex=True)

cols = [3, 4, 5, 6, 7, 8, 10]
df = pd.read_csv(data_dir / 'weatherHistory.csv', usecols=cols, dtype=float)

input_col = ['Temperature (C)', 'Humidity',
       'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
       'Pressure (millibars)']
output_col = ['Apparent Temperature (C)']

n_train = 70000
causal = True

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

# Small trainset and testset
n_small_train = 10000
ind = np.random.choice(len(x_train), n_small_train, replace=True)
x_small_train = x_train[ind]
y_small_train = y_train[ind]

n_small_test = 1000
ind = np.random.choice(len(x_test), n_small_test, replace=True)
x_small_test = x_test[ind]
y_small_test = y_test[ind]

# Normalized output (the easy part to learn)
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std
y_small_test -= y_mean
y_small_test /= y_std

# Real temperature baseline
def easiest_baseline():
    baseline = median_error(x_test[:, :1], y_test, inplace=False)
    print(baseline)

easiest_baseline()

# Descent parameters
gammas = [1e-2]
num_it = n_train
resampling = False

# Randomness control
np.random.seed(0)

# Random questions
n_train, m = y_train.shape
u = generate_random_sphere(num_it, m)
v = .3 * np.random.randn(num_it)

# Kernel
sigma = 1e1
kernel = Kernel('gaussian', sigma=sigma)
lambd = 1e-6

# Function parameterization
n_repr = 100
ind = np.random.choice(len(x_train), n_repr, replace=True)
x_repr = x_train[ind]
kernel.set_support(x_repr)
K_repr = kernel.get_k()
K_test = kernel(x_test).T

# Assume that parameters where initialized around a good model
def good_init():
    K = kernel(x_repr)
    lambd = 1e-6 * n_repr
    K += lambd * np.eye(len(K))
    return np.linalg.solve(K, y_train[ind])

theta_init = good_init()
theta_init += 1e-1 * np.random.randn(*theta_init.shape)

# Placeholders
full_error_w = np.zeros((num_it, len(gammas)))
full_error_f = np.zeros((num_it, len(gammas)))
theta_w = np.empty((n_repr, 1))
theta_ave_w = np.empty((n_repr, 1))
theta_f = np.empty((n_repr, 1))
theta_ave_f = np.empty((n_repr, 1))
grad = np.empty(theta_w.shape)
K_train = kernel(x_repr[:1]).T

if resampling:
    I = np.random.choice(n_train, num_it, replace=True)
else:
    I = np.arange(num_it)

for i_g, gamma_0 in enumerate(gammas):
    gamma = get_stepsize(num_it, gamma_0)

    # Descent initialization
    theta_w[:] = theta_init
    theta_f[:] = theta_init
    theta_ave_f[:] = 0
    theta_ave_w[:] = 0

    for i in range(n_repr, num_it):
        # New point
        ind = I[i]
        K_train[:] = kernel(x_train[ind:ind+1]).T

        # Active update
        epsilon = np.sign((K_train @ theta_w - y_train[ind]) @ u[i])
        grad[:] = 1
        grad *= u[i]
        grad *= K_train.T
        grad *= epsilon
        grad += lambd * K_repr @ theta_w
        grad *= gamma[i]
        theta_w -= grad

        # Passive update
        eps_sig = np.sign((y_train[ind] @ u[i]) - v[i])
        eps_fun = np.sign(((K_train @ theta_f) @ u[i]) - v[i])
        if eps_sig != eps_fun:
            grad[:] = 1
            grad *= u[i]
            grad *= K_train.T
            grad *= eps_fun
            grad += lambd * K_repr @ theta_f
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
        full_error_w[i, i_g] = median_error(y_pred_w, y_test)

        y_pred_f = K_test @ theta_ave_f
        full_error_f[i, i_g] = median_error(y_pred_f, y_test)
        if not i % int(1e4):
            print(i//int(1e4), end=',')

# fig, ax = plt.subplots(figsize=(2.5, 1.75))
fig, ax = plt.subplots(figsize=(10, 7))
for i in range(len(gammas)):
    a, = ax.plot(full_error_w[n_repr:, i])
    b, = ax.plot(full_error_f[n_repr:, i])

c, = ax.plot(np.arange(n_train-n_repr), np.full(n_train-n_repr, median_error(x_test[:, :1], y_test, inplace=False)), c='k', linestyle='--')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend([a, b, c], ['active', 'passive', 'baseline'], prop={'size': 6})

# plt.yticks(fontsize=6)
# plt.xticks(fontsize=6)
ax.set_title(r"Weather dataset", size=10)
ax.set_xlabel(r"Iteration $T$", size=8)
ax.set_ylabel(r"Test risk of $\bar\theta_T$", size=8)
fig.tight_layout()
fig.savefig("real_temp.pdf")