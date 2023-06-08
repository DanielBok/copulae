from typing import Collection, Literal, NamedTuple, Union

import numpy as np

from copulae.copula import Summary, TailDep
from copulae.copula.base import EstimationMethod
from copulae.elliptical.abstract import AbstractEllipticalCopula
from copulae.stats import multivariate_t as mvt, t
from copulae.types import Array, Ties
from copulae.utility.annotations import *


class StudentParams(NamedTuple):
    df: float
    rho: np.ndarray


class StudentCopula(AbstractEllipticalCopula[StudentParams]):
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

        eps = 1e-6
        lower, upper = np.repeat(-1., n + 1) - eps, np.repeat(1., n + 1) + eps
        lower[0], upper[0] = 0, np.inf  # bounds for df, the rest are correlation
        self._bounds = (lower, upper)

    @validate_data_dim({"x": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def cdf(self, x: np.ndarray, log=False):
        sigma = self.sigma
        df = self._df
        q = t.ppf(x, df)
        return mvt.logcdf(q, cov=sigma, df=df) if log else mvt.cdf(q, cov=sigma, df=df)

    def fit(self, data: np.ndarray, x0: Union[Collection[float], np.ndarray] = None, method: EstimationMethod = 'ml',
            optim_options: dict = None, ties: Ties = 'average', verbose=1, to_pobs=True, scale=1.0, fix_df=False):
        """
        Fit the copula with specified data

        Parameters
        ----------
        data: ndarray
            Array of data used to fit copula. Usually, data should be the pseudo observations

        x0: ndarray
            Initial starting point. If value is None, best starting point will be estimated

        method: { 'ml', 'irho', 'itau' }, optional
            Method of fitting. Supported methods are: 'ml' - Maximum Likelihood, 'irho' - Inverse Spearman Rho,
            'itau' - Inverse Kendall Tau

        optim_options: dict, optional
            Keyword arguments to pass into :func:`scipy.optimize.minimize`

        ties: { 'average', 'min', 'max', 'dense', 'ordinal' }, optional
            Specifies how ranks should be computed if there are ties in any of the coordinate samples. This is
            effective only if the data has not been converted to its pseudo observations form

        verbose: int, optional
            Log level for the estimator. The higher the number, the more verbose it is. 0 prints nothing.

        to_pobs: bool
            If True, casts the input data along the column axis to uniform marginals (i.e. convert variables to
            values between [0, 1]). Set this to False if the input data are already uniform marginals.

        scale: float
            Amount to scale the objective function value of the numerical optimizer. This is helpful in
            achieving higher accuracy as it increases the sensitivity of the optimizer. The downside is
            that the optimizer could likely run longer as a result. Defaults to 1.

        fix_df: bool, optional
            If True, the degree of freedom specified by the user (param) is fixed and will not change. Otherwise,
            the degree of freedom is subject to changes during the fitting phase.

        See Also
        --------
        :code:`scipy.optimize.minimize`: the `scipy minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_ function use for optimization
        """
        if fix_df:
            if optim_options is None:
                optim_options = {'constraints': []}
            optim_options['constraints'].append({'type': 'eq', 'fun': lambda x: x[0] - self._df})  # df doesn't change

        return super().fit(data, x0, method, optim_options, ties, verbose, to_pobs, scale)

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
    def params(self):
        """
        The parameters of the Student copula. A tuple where the first value is the degrees of freedom and the
        subsequent values are the correlation matrix parameters

        Returns
        -------
        StudentParams:
            A dataclass with 2 properties

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

    @validate_data_dim({"u": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def pdf(self, u: np.ndarray, log=False):
        sigma = self.sigma
        df = self._df
        q = t.ppf(u, df)
        d = mvt.logpdf(q, cov=sigma, df=df) - t.logpdf(q, df=df).sum(1)
        return d if log else np.exp(d)

    @cast_output
    def random(self, n: int, seed: int = None):
        r = mvt.rvs(cov=self.sigma, df=self._df, size=n, random_state=seed)
        return t.cdf(r, self._df)

    @select_summary
    def summary(self, category: Literal['copula', 'fit'] = 'copula'):
        return Summary(self, {
            'Degree of Freedom': self.params.df,
            'Correlation Matrix': self.sigma
        })
