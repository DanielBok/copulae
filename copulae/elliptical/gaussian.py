from typing import Union

import numpy as np

from copulae.stats import multivariate_normal as mvn, norm
from copulae.types import Array
from .abstract import AbstractEllipticalCopula
from .decorators import quantile


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
        n = sum(range(dim))
        self._rhos = np.zeros(n)
        self.params_bounds = np.repeat(-1., n), np.repeat(1., n)

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
        self._rhos = np.asarray(params)

    @property
    def __lambda__(self):
        res = (self._rhos == 1).astype(float)
        return res, res

    def irho(self, rho: Array):
        return np.sin(np.array(rho) * np.pi / 6) * 2

    @quantile('normal')
    def cdf(self, x: np.ndarray, log=False):
        sigma = self.sigma
        return mvn.logcdf(x, cov=sigma) if log else mvn.cdf(x, cov=sigma)

    @quantile('normal')
    def pdf(self, x: np.ndarray, log=False):
        sigma = self.sigma
        d = mvn.logpdf(x, cov=sigma) - norm.logpdf(x).sum(1)
        return d if log else np.exp(d)

    def __random__(self, n: int, seed: int = None):
        r = mvn.rvs(cov=self.sigma, size=n, random_state=seed)
        return norm.cdf(r)

    def summary(self):
        print(self)

    def __str__(self):
        msg = f"""
Gaussian Copula with {self.dim} dimensions

Correlation Matrix (P):
{self.sigma}
        """.strip()

        if self.fit_stats is not None:
            msg += f'\n\n{self.fit_stats}'

        return msg
