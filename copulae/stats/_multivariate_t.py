from collections import abc
from typing import Optional

import numpy as np
import numpy.linalg as la
from scipy.special import beta, gamma
from scipy.stats import multivariate_normal as mvn

from copulae.types import Numeric, OptNumeric


def _is_psd(M: Numeric):
    if isinstance(M, (float, int)):
        M = np.asarray([[M]], float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        return False

    ev = la.eigvalsh(M)
    if ev.min() < -np.finfo('f').eps or not np.allclose(M, M.T):
        return False

    return True


class multivariate_t:
    _T_LIMIT = 10000  # beyond this limit, we assume normal distribution

    @staticmethod
    def _process_parameters(dim: OptNumeric, mean: OptNumeric, cov: OptNumeric, df: Optional[float]):
        if dim is None:
            if cov is None:
                dim = 1
                cov = np.array([[1.]], dtype=float)
            else:
                if isinstance(cov, (float, int)):
                    cov = np.array([[cov]], dtype=float)

                dim = len(cov)
                cov = np.asarray(cov, dtype=float).reshape(dim, dim)
        else:
            if not np.isscalar(dim):
                raise ValueError("Dimension of random variable must be a scalar.")

        if mean is None:
            mean = np.zeros(dim)
        elif isinstance(mean, (float, int)):
            mean = np.repeat(mean, dim)
        else:
            mean = np.asarray(mean, dtype=float)

        # checks done here
        if mean.shape[0] != dim or mean.ndim != 1:
            raise ValueError("Array 'mean' must be a vector of length %d." % dim)

        if not _is_psd(cov):
            raise ValueError("Matrix 'cov' must be positive semi-definite")

        if df <= 0:
            raise ValueError("Degrees of freedom 'df' must be greater than 0")
        elif df is None:
            df = 4.6692  # In honour of Feigenbaum
        return dim, mean, cov, df

    @staticmethod
    def _process_input(x: Numeric, dim: int):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return x[np.newaxis]
        elif x.ndim == 1:
            return x[:, np.newaxis] if dim == 1 else x[np.newaxis, :]
        return x

    @classmethod
    def pdf(cls, x: Numeric, mean: Numeric = None, cov: Numeric = 1, df: float = None):
        dim, mean, cov, df = cls._process_parameters(None, mean, cov, df)
        x = cls._process_input(x, dim)

        if df > cls._T_LIMIT:  # if greater than 500, just assume to be normal
            return mvn.pdf(x, mean=mean, cov=cov)

        # multivariate t distribution from https://en.wikipedia.org/wiki/Multivariate_t-distribution
        # break into 3 portions, we use beta function to avoid some amount of numeric overflow
        a = gamma(dim / 2)
        b = beta(df / 2, dim / 2) * ((df * np.pi) ** (dim / 2)) * (la.det(cov) ** 0.5)

        x_us = x - mean  # difference between x and mu (mean)
        c = np.diag(1 + (x_us @ la.inv(cov) @ x_us.T) / df) ** ((dim + df) / 2)

        return a / (b * c)

    @classmethod
    def logpdf(cls, x: Numeric, mean: Numeric = None, cov: Numeric = 1, df: float = None):
        return np.log(cls.pdf(x, mean, cov, df))

    @classmethod
    def cdf(cls, x: Numeric, mean: Numeric = None, cov: Numeric = 1, df: float = None):
        raise NotImplementedError
        # dim, mean, cov, df = cls._process_parameters(None, mean, cov, df)
        # x = cls._process_input(x, dim)
        #
        # C = la.cholesky(cov)
        # err, eps, = np.inf, 1e-6
        #
        # _g = 3  # standard deviations for monte carlo method
        # N = len(x)
        # int_vals, var_sums = np.zeros(N), np.zeros(N)
        #
        # for xi in range(N):
        #     int_val, var_sum, n = 0, 0, 0
        #     while err > eps:
        #         ws = np.random.uniform(size=dim)
        #
        #         ys = np.zeros(dim)
        #         n += 1
        #         for i in range(dim):
        #             dfi = df + i
        #             ys[i] = np.sqrt((df + (ys ** 2).sum()) / dfi) / t.pdf(ws[i], dfi)
        #             if (C[i, :] * ys).sum() > x[xi, i]:
        #                 break
        #         else:
        #             var_sum += (n - 1) / n * (1 - int_val) ** 2
        #             int_val += (1 - int_val) / n
        #             err = _g * np.sqrt(var_sum / (n * (n - 1)))
        #             print(err)
        #     int_vals[xi] = int_val
        #     var_sums[xi] = var_sum
        #
        # return int_vals

    @classmethod
    def logcdf(cls, x: Numeric, mean: Numeric = None, cov: Numeric = 1, df: float = None):
        return np.log(cls.cdf(x, mean, cov, df))

    @classmethod
    def rvs(cls, mean: OptNumeric = None, cov: Numeric = 1, df: float = None, size: Numeric = 1, type_='shifted',
            random_state: Optional[int] = None):
        """
        Draw random samples from a multivariate student T distribution

        :param mean: float, array
            Mean of the distribution (default zero)
        :param cov: float, matrix
            Covariance matrix of the distribution (default one)
        :param df: float
            Degrees of freedom
        :param size: int, array of int
            Number of samples to draw (default 1).
        :param type_: str, default 'shifted'
            Type of non-central multivariate t distribution. 'Kshirsagar' is the non-central t-distribution needed for
            calculating the power of multiple contrast tests under a normality assumption. 'Shifted' is a location
             location shifted version of the central t-distribution. This non-central multivariate t distribution
             appears for example as the Bayesian posterior distribution for the regression coefficients in a linear
             regression. In the central case both types coincide
        :param random_state: int, optional
            use it for drawing the random variates. If None (or np.random), the global np.random state is used. Default
            is None.
        :return: numpy array
            simulated random variates
        """
        if not (type(size) is int or all(type(s) is int for s in size)):
            raise ValueError('size argument must be integer or iterable of integers')

        if df is None:
            raise ValueError("Degrees of freedom 'df' must be a scalar float and greater than 0")
        elif df <= 0:
            raise ValueError("Degrees of freedom 'df' must be greater than 0")
        elif df > cls._T_LIMIT:
            return np.asarray(mvn.rvs(mean, cov, size, random_state), dtype=float)

        if random_state is not None:
            np.random.seed(random_state)
        d = np.sqrt(np.random.chisquare(df, size) / df)

        if isinstance(size, int):
            if size > 1:
                d = d.reshape(size, -1)
            size_is_iterable = False
        else:
            d = d.reshape(*size, -1)
            size_is_iterable = True

        # size and dim used to reshape generated tensor correctly before dividing by chi-square rvs
        dim = 1 if not isinstance(cov, abc.Sized) else len(cov)

        if type_.casefold() == 'kshirsagar':
            r = mvn.rvs(mean, cov, size, random_state)
            if size_is_iterable:
                r = r.reshape(*size, dim)
            r /= d

        elif type_.casefold() == 'shifted':
            r = mvn.rvs(np.zeros(len(cov)), cov, size, random_state)
            if size_is_iterable:
                r = r.reshape(*size, dim)
            r /= d

            if mean is not None:
                r += mean  # location shifting
        else:
            raise ValueError(f"Unknown centrality type {type_}. Use one of 'Kshirsagar', 'shifted'")

        return np.asarray(r, dtype=float)
