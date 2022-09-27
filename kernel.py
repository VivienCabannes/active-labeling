
import numpy as np


class Kernel:
    """
    Computation of classical kernels

    Parameters
    ----------
    kernel : {'gaussian'}
        Name of the kernel to use.
    sigma : int, optional
        Parameter for various kernel: standard deviation for Gaussian kernel.

    Examples
    --------
    >>> import numpy as np
    >>> x_support = np.random.randn(50, 10)
    >>> kernel_computer = Kernel('Gaussian', sigma=3)
    >>> kernel_computer.set_support(x_support)
    >>> x = np.random.randn(30, 10)
    >>> k = kernel_computer(x)
    """

    def __init__(self, kernel, **kwargs):
        self.kernel = kernel.lower()
        if self.kernel == "gaussian":
            self.sigma2 = 2 * (kwargs['sigma'] ** 2)
        if self.kernel == "laplacian":
            self.sigma = kwargs['sigma']
        self._call_method = getattr(self, self.kernel + '_kernel')

    def set_support(self, x):
        """Set train support for kernel method.

        Parameters
        ----------
        x : ndarray
            Training set given as a design matrix, of shape (nb_points, input_dim).
        """
        self.reset()
        self.x = x

    def __call__(self, x):
        """Kernel computation.

        Parameters
        ----------
        x : ndarray
            Points to compute kernel, of shape (nb_points, input_dim).

        Returns
        -------
        out : ndarray
            kernel matrix k(x, x_support).
        """
        return self._call_method(x)

    def get_k(self):
        """Kernel computations.

        Get kernel matrix on support points.
        """
        return self(self.x)

    def gaussian_kernel(self, x):
        """Gaussian kernel.

        Implement k(x, y) = exp(-norm{x - y}^2 / (2 * sigma2)).
        """
        K = self.x @ x.T
        K *= 2
        if not hasattr(self, "_attr_1"):
            self._attr1 = np.sum(self.x ** 2, axis=1)[:, np.newaxis]
        K -= self._attr1
        K -= np.sum(x ** 2, axis=1)
        K /= self.sigma2
        np.exp(K, out=K)
        return K

    def laplacian_kernel(self, x):
        """Laplacian kernel
        return exp(-norm{x - y} / (sigma))
        sigma = kernel_parameter
        """
        K = self.x @ x.T
        K *= -2
        if not hasattr(self, "_attr_1"):
            self._attri_1 = np.sum(self.x ** 2, axis=1)[:, np.newaxis]
        K += self._attri_1
        K += np.sum(x ** 2, axis=1)
        K[K < 0] = 0
        np.sqrt(K, out=K)
        K /= - self.sigma
        np.exp(K, out=K)
        return K

    def linear_kernel(self, x):
        """Linear kernel.

        Implement k(x, y) = x^T y.
        """
        return self.x @ x.T

    def reset(self):
        """Resetting attributes."""
        if hasattr(self, "_attr_1"):
            delattr(self, "_attr_1")


class RidgeRegressor:
    """
    Regression weights of kernel Ridge regression

    Parameters
    ----------
    kernel : Kernel Object
        Backend kernel for regression.

    Examples
    --------
    >>> import numpy as np
    >>> kernel = Kernel('Gaussian', sigma=3)
    >>> krr = RidgeRegressor(kernel)
    >>> x_support = np.random.randn(50, 10)
    >>> kernel.set_support(x_support)
    >>> krr.set_xtrain(x_support)
    >>> krr.update_lambda(1)
    >>> x = np.random.randn(30, 10)
    >>> alpha = krr(x)
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def set_xtrain(self, x_train):
        self.K = self.kernel(x_train)
        A = self.K @ self.K.T
        self.w_0, self.v = np.linalg.eigh(A)
        self.b = self.v.T @ self.K

    def update_lambda(self, lambd):
        """Setting Tikhonov regularization parameter

        Useful to try several regularization parameter.
        """
        w = self.w_0 + lambd
        w **= -1
        self.beta = (self.v * w) @ self.b

    def __call__(self, x_test):
        """Weighting scheme computation.

        Parameters
        ----------
        x_test : ndarray
            Points to compute kernel ridge regression weights, of shape (nb_points, input_dim).

        Returns
        -------
        out : ndarray
            Similarity matrix of size (nb_points, n_train) given by kernel ridge regression.
        """
        K_x = self.kernel(x_test)
        return K_x.T @ self.beta