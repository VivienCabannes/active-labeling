import numpy as np
import matplotlib.pyplot as plt

from activelabeling.auxillary import target_func
from activelabeling.kernel import Kernel


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times"

np.random.seed(1000)

n_train = 64
x_train = np.random.rand(n_train)[:, np.newaxis]

sigma = 0.2
kernel = Kernel("gaussian", sigma=sigma)
kernel.set_support(x_train)
K = kernel(x_train)

x_test = np.linspace(0, 1, 50)
y_test = target_func(x_test)
k_test = kernel(x_test[:, np.newaxis]).T


def myplot(epsilon, alpha, alpha_update, ax):
    ind = epsilon[:, 0] == 1
    ax.scatter(x_train[ind, 0], epsilon[ind, 1], s=10)
    ind = epsilon[:, 0] == -1
    ax.scatter(x_train[ind, 0], epsilon[ind, 1], s=10)
    ax.plot(x_test, y_test, alpha=0.5, linestyle="--", c="C0")
    y_pred = k_test @ alpha
    ax.plot(x_test, y_pred, alpha=0.5, c="C1", linestyle="--")
    y_pred_update = k_test @ alpha_update
    ax.plot(x_test, y_pred_update, alpha=0.7, c="C1")
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks([])
    ax.set_yticks([])


alpha = np.zeros(n_train, float)
epsilon = np.zeros((n_train, 2), float)
gamma_0 = 2
gamma = np.arange(n_train, dtype=float)
gamma += 1
np.sqrt(gamma, out=gamma)
np.divide(1, gamma, out=gamma)
gamma *= gamma_0

j = 0
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i in range(n_train):
    z = K[i] @ alpha
    epsilon[i, 0] = np.sign(z - target_func(x_train[i, 0]))
    epsilon[i, 1] = z
    update = epsilon[i, 0] * gamma[i]
    if i + 1 in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
        ax = axes[j // 4, j % 4]
        alpha_update = alpha.copy()
        alpha_update[i] -= update
        myplot(epsilon, alpha, alpha_update, ax)
        ax.set_title(r"$T=" + str(i + 1) + "$", size=18)
        j += 1
    alpha[i] -= update

fig.tight_layout()
fig.savefig("one_d_frame.pdf")
