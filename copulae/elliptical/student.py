from typing import NamedTuple, Union

import numpy as np

from copulae.stats import multivariate_t as mvt, t
from copulae.types import Array
from .abstract import AbstractEllipticalCopula
from .decorators import quantile


class StudentParams(NamedTuple):
    df: float
    rho: np.ndarray


class StudentCopula(AbstractEllipticalCopula):

    def __init__(self, dim=2, df=1):
        super().__init__(dim, 'Student')

        n = sum(range(dim))
        self._df = df
        self._rhos = np.zeros(n)

        lower, upper = np.repeat(-1., n + 1), np.repeat(1., n + 1)
        lower[0], upper[0] = 0, np.inf  # bounds for df, the rest are correlation
        self.params_bounds = lower, upper

    @quantile('student')
    def cdf(self, x: np.ndarray, log=False):
        sigma = self.sigma
        df = self._df
        return mvt.logcdf(x, cov=sigma, df=df) if log else mvt.cdf(x, cov=sigma, df=df)

    @quantile('student')
    def pdf(self, x: np.ndarray, log=False):
        sigma = self.sigma
        df = self._df
        d = mvt.logpdf(x, cov=sigma, df=df) - t.logpdf(x, df=df).sum(1)
        return d if log else np.exp(d)

    @property
    def params(self) -> StudentParams:
        return StudentParams(self._df, self._rhos)

    @params.setter
    def params(self, params: Union[np.ndarray, StudentParams]):
        if type(params) is StudentParams:
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

    def fit(self, data: np.ndarray, x0: np.ndarray = None, method='mpl', fix_df=False, est_var=False, verbose=1,
            optim_options: dict = None):
        if fix_df:
            optim_options['constraints'].append({'type': 'eq', 'fun': lambda x: x[0] - self._df})  # df doesn't change

        return super().fit(data, x0, method, est_var, verbose, optim_options)

    def irho(self, rho: Array):
        """
        irho is not implemented for t copula

        :param rho: numpy array
        :return: Not Implemented
        """
        return NotImplemented()

    @property
    def __lambda__(self):
        df = self._df
        rho = self._rhos
        if np.isinf(df):
            res = (self._rhos == 1).astype(float)
        else:
            res = 2 * t.cdf(- np.sqrt((df + 1) * (1 - rho)), df=df + 1)

        return res, res

    def __random__(self, n: int, seed: int = None):
        r = mvt.rvs(cov=self.sigma, df=self._df, size=n, random_state=seed)
        return t.cdf(r, self._df)

    def __str__(self):
        msg = f"""
Student T Copula with {self.dim} dimensions

Degrees of Freedom: {self._df}

Correlation Matrix (P):
    {self.sigma}
""".strip()

        if self.fit_stats is not None:
            msg += f'\n\n{self.fit_stats}'

        return msg

    def summary(self):
        print(self)
