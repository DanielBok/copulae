from typing import NamedTuple, Union

import numpy as np

from copulae.copula import Summary, TailDep
from copulae.elliptical.abstract import AbstractEllipticalCopula
from copulae.stats import multivariate_t as mvt, t
from copulae.types import Array
from copulae.utility import array_io


class StudentParams(NamedTuple):
    df: float
    rho: np.ndarray


class StudentCopula(AbstractEllipticalCopula):
    r"""
    The Student (T) Copula. It is elliptical and symmetric which gives it nice analytical properties. The
    Student copula is determined by its correlation matrix and the degrees of freedom. Student copulas have
    fatter tails as compared to Gaussian copulas.

    A Student copula is fined as

    .. math::

        C_{\Sigma, \nu} (u_1, \dots, u_d) = \mathbf{t}_{\Sigma, \nu} (t_\nu^{-1} (u_1), \dots, t_\nu^{-1} (u_d))

    where :math:`\Sigma` and :math:`\nu` are the covariance matrix and degrees of freedom which describes the
    Student copula and :math:`t_\nu^{-1}` is the quantile (inverse cdf) function
    """

    def __init__(self, dim=2, df=1):
        """
        Creates a Student copula instance

        Parameters
        ----------
        dim: int, optional
            Dimension of the copula

        df: float, optional
            Degrees of freedom of the copula
        """
        super().__init__(dim, 'Student')

        n = sum(range(dim))
        self._df = df
        self._rhos = np.zeros(n)

        lower, upper = np.repeat(-1., n + 1), np.repeat(1., n + 1)
        lower[0], upper[0] = 0, np.inf  # bounds for df, the rest are correlation
        self._bounds = (lower, upper)

    @array_io(dim=2)
    def cdf(self, x: np.ndarray, log=False):
        sigma = self.sigma
        df = self._df
        q = t.ppf(x, df)
        return mvt.logcdf(q, cov=sigma, df=df) if log else mvt.cdf(q, cov=sigma, df=df)

    def fit(self, data: np.ndarray, x0: np.ndarray = None, method='mpl', fix_df=False, est_var=False, verbose=1,
            optim_options: dict = None):
        if fix_df:
            optim_options['constraints'].append({'type': 'eq', 'fun': lambda x: x[0] - self._df})  # df doesn't change

        return super().fit(data, x0, method, est_var, verbose, optim_options)

    def irho(self, rho: Array):
        """
        irho is not implemented for t copula
        """
        return NotImplemented()

    @property
    def lambda_(self):
        df = self._df
        rho = self._rhos
        if np.isinf(df):
            res = (self._rhos == 1).astype(float)
        else:
            res = 2 * t.cdf(- np.sqrt((df + 1) * (1 - rho)), df=df + 1)

        return TailDep(res, res)

    @property
    def params(self) -> StudentParams:
        """
        The parameters of the Student copula. A tuple where the first value is the degrees of freedom and the
        subsequent values are the correlation matrix parameters

        Returns
        -------
        df: float
            Degrees of freedom
        corr: ndarray
            Correlation parameters

        """
        return StudentParams(self._df, self._rhos)

    @params.setter
    def params(self, params: Union[np.ndarray, StudentParams]):
        if isinstance(params, StudentParams):
            self._df = params.df
            self._rhos = params.rho
        else:
            params = np.asarray(params)
            if len(params) != 1 + sum(range(self.dim)):
                raise ValueError('Incompatible parameters for student copula')

            df = params[0]
            if df <= 0:
                raise ValueError('Degrees of freedom must be greater than 0')

            self._df = df
            self._rhos = params[1:]

    @array_io(dim=2)
    def pdf(self, u: np.ndarray, log=False):
        sigma = self.sigma
        df = self._df
        q = t.ppf(u, df)
        d = mvt.logpdf(q, cov=sigma, df=df) - t.logpdf(q, df=df).sum(1)
        return d if log else np.exp(d)

    def random(self, n: int, seed: int = None):
        r = mvt.rvs(cov=self.sigma, df=self._df, size=n, random_state=seed)
        return t.cdf(r, self._df)

    def summary(self):
        return Summary(self, {
            'Degree of Freedom': self.params.df,
            'Correlation Matrix': self.sigma
        })
