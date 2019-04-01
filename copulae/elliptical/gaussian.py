from typing import Union

import numpy as np

from copulae.copula import Summary, TailDep
from copulae.elliptical.abstract import AbstractEllipticalCopula
from copulae.stats import multivariate_normal as mvn, norm
from copulae.types import Array
from copulae.utility import array_io


class GaussianCopula(AbstractEllipticalCopula):
    r"""
    The Gaussian (Normal) copula. It is elliptical and symmetric which gives it nice analytical properties. The
    Gaussian copula is determined entirely by its correlation matrix.

    Gaussian copulas do not model tail dependencies very well, it's tail is flat. Take not that by symmetry,
    it gives equal weight to tail scenarios. In English, this means upside scenarios happen as often as downside
    scenarios.

    A Gaussian copula is fined as

    .. math::

        C_\Sigma (u_1, \dots, u_d) = \Phi_\Sigma (N^{-1} (u_1), \dots, N^{-1} (u_d))

    where :math:`\Sigma` is the covariance matrix which is the parameter of the Gaussian copula and
    :math:`N^{-1}` is the quantile (inverse cdf) function
    """

    def __init__(self, dim=2):
        """
        Creates a Gaussian copula instance

        Parameters
        ----------
        dim: int, optional
            Dimension of the copula
        """

        super().__init__(dim, "Gaussian")
        n = sum(range(dim))
        self._rhos = np.zeros(n)
        self._bounds = np.repeat(-1., n), np.repeat(1., n)

    @array_io(dim=2)
    def cdf(self, x: np.ndarray, log=False):
        q = norm.ppf(x)
        sigma = self.sigma
        return mvn.logcdf(q, cov=sigma) if log else mvn.cdf(q, cov=sigma)

    @array_io
    def irho(self, rho: Array):
        return np.sin(np.array(rho) * np.pi / 6) * 2

    def lambda_(self):
        res = (self._rhos == 1).astype(float)
        return TailDep(res, res)

    @property
    def params(self):
        """
        The covariance parameters for the Gaussian copula

        Returns
        -------
        ndarray
            Correlation matrix of the Gaussian copula
        """
        return self._rhos

    @params.setter
    def params(self, params: Union[float, np.ndarray, list]):
        if isinstance(params, (float, int)):
            params = np.repeat(params, len(self._rhos))
        self._rhos = np.asarray(params)

    @array_io(dim=2)
    def pdf(self, u: np.ndarray, log=False):
        sigma = self.sigma
        q = norm.ppf(u)
        d = mvn.logpdf(q, cov=sigma) - norm.logpdf(q).sum(1)
        return d if log else np.exp(d)

    def random(self, n: int, seed: int = None):
        r = mvn.rvs(cov=self.sigma, size=n, random_state=seed)
        return norm.cdf(r)

    def summary(self):
        return Summary(self, {
            'Correlation Matrix': self.sigma
        })
