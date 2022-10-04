
import numpy as np
import matplotlib.pyplot as plt

from activelabeling.auxillary import target_func
from activelabeling.kernel import Kernel


plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times"

np.random.seed(100)


def solution(x_train, kernel, gamma=.3, verbose=False):

    n_train = x_train.shape[0]
    kernel.set_support(x_train)
    K = kernel(x_train)
    alpha = np.zeros(n_train, float)

    if isinstance(gamma, float):
        gamma = np.full(n_train, gamma, float)

    if verbose:
        history = np.zeros((n_train, 2), float)

    for i in range(n_train):
        z = K[i] @ alpha
        epsilon = np.sign(z - target_func(x_train[i,0]))
        alpha[i] -= epsilon * gamma[i]
        if verbose:
            history[i, 0] = epsilon
            history[i, 1] = z

    return alpha, history


n_train = 30
x_train = np.random.rand(n_train)[:, np.newaxis]

sigma = .2
kernel = Kernel('gaussian', sigma=sigma)

x_test = np.linspace(0, 1, 50)
y_test = target_func(x_test)

alpha, history = solution(x_train, kernel, verbose=True)
k_test = kernel(x_test[:, np.newaxis]).T

fig, ax = plt.subplots(1,1, figsize=(2.5, 1.75))
ind = history[:, 0] == 1
a = ax.scatter(x_train[ind, 0], history[ind, 1], s=10)
ind = history[:, 0] == -1
b = ax.scatter(x_train[ind, 0], history[ind, 1], s=10)
c, = ax.plot(x_test, y_test, linestyle='--')
y_pred = k_test @ alpha
d, = ax.plot(x_test, y_pred, alpha=.7)
ax.legend([c,d], ['Signal', 'Reconstruction'], prop={"size": 8})
ax.set_ylim(-2.5, 2.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Active strategy', size=10)
ax.set_xlabel(r'Inputs',size=8)
ax.set_ylabel(r'Outputs', size=8)
fig.tight_layout()
fig.savefig('one_d_active.pdf')


epsilon_bis = np.zeros((n_train, 2), float)
epsilon_bis[:, 1] = np.random.randn(n_train)

alpha_bis = np.zeros(n_train, float)
gamma_0 = .7
gamma = np.full(n_train, gamma_0, float)
K = kernel(x_train)
for i in range(n_train):
    epsilon_bis[i, 0] = np.sign(epsilon_bis[i, 1] - target_func(x_train[i,0]))
    err = np.sign(epsilon_bis[i, 1] - K[i] @ alpha_bis)
    if err != epsilon_bis[i, 0]:
        alpha_bis[i] -= epsilon_bis[i, 0] * gamma[i]

fig, ax = plt.subplots(1,1, figsize=(2.5, 1.75))
ind = epsilon_bis[:, 0] == 1
a = ax.scatter(x_train[ind, 0], epsilon_bis[ind, 1], s=10)
ind = epsilon_bis[:, 0] == -1
b = ax.scatter(x_train[ind, 0], epsilon_bis[ind, 1], s=10)
c, = ax.plot(x_test, y_test, linestyle='--')
y_pred_bis = k_test @ alpha_bis
d, = ax.plot(x_test, y_pred_bis, alpha=.7)
ax.legend([a,b], ['Above', 'Below'], prop={"size": 8})
ax.set_ylim(-2.5, 2.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Passive strategy', size=10)
ax.set_xlabel(r'Inputs',size=8)
ax.set_ylabel(r'Outputs', size=8)
fig.tight_layout()
fig.savefig('one_d_passive.pdf')