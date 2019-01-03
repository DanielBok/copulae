from typing import Union

import numpy as np
from scipy.stats import multivariate_normal as mvn, norm

from .abstract import AbstractEllipticalCopula
from copulae.types import Array


class GaussianCopula(AbstractEllipticalCopula):
    """
    The Gaussian (Normal) Copula. It is elliptical and symmetric which gives it nice analytical properties. The
    Gaussian Copula is fully determined by its correlation matrix.

    Gaussian Copulas do not model tail dependencies very well, it's tail is flat. Take not that by symmetry,
    it gives equal weight to tail scenarios. In English, this means upside scenarios happen as often as downside
    scenarios.
    """

    def __init__(self, dim=2):
        super().__init__(dim, "Gaussian")
        self._rhos = np.zeros(sum(range(dim)))

    @property
    def params(self):
        return self._rhos

    @params.setter
    def params(self, params: Union[float, np.ndarray, list]):
        """
        Sets the covariance parameters for the copulae

        :param params: float, numpy array
            Covariance parameters. If it is a single float number, function will broadcast it to an array of similar
            length as the rhos
        """
        if type(params) in {float, int}:
            params = np.repeat(params, len(self._rhos))
        self._rhos = np.array(params)

    def tau(self):
        return np.arcsin(self.params) * 2 / np.pi

    def rho(self):
        return np.arcsin(self.params / 2) * 6 / np.pi

    def lambda_(self):
        i01 = (self.params == 1).astype(float)
        return self._lambda_(i01, i01)

    def itau(self, tau: Array):
        return np.sin(tau * np.pi / 2)

    def irho(self, rho: Array):
        return np.sin(np.array(rho) * np.pi / 6) * 2

    def cdf(self, x: np.ndarray, log=False):
        # self._check_dimension(x)
        sigma = self.sigma
        return mvn.logcdf(x, cov=sigma) if log else mvn.cdf(x, cov=sigma)

    def ppf(self, x: np.ndarray):
        # self._check_dimension(x)
        return norm.ppf(x)

    def pdf(self, x: np.ndarray, log=False):
        # self._check_dimension(x)
        sigma = self.sigma

        q = norm.ppf(x)
        r = mvn.logpdf(q, cov=sigma) - norm.logpdf(q).sum(1)

        return r if log else np.exp(r)

    def __random__(self, n: int, seed: int = None):
        r = mvn.rvs(cov=self.sigma, size=n, random_state=seed)
        return norm.cdf(r)
