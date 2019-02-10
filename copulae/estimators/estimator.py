from typing import Optional

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from statsmodels.stats.correlation_tools import corr_nearest

from copulae.copula.abstract import AbstractCopula as Copula, FitStats
from copulae.core import tri_indices
from copulae.stats import kendall_tau, pearson_rho, spearman_rho
from copulae.utils import format_docstring, merge_dict
from .est_max_likelihood import MaxLikelihoodEstimator

__estimator_params_docs__ = """
        :param copula: copula
            Copula that will be fitted
        :param data: numpy array
            Array of data used to fit copula. Usually, data should be the pseudo observations
        :param x0: numpy array
            Initial starting point. If value is None, best starting point will be estimated
        :param method:
            Method of fitting. Supported methods are:
                ml - maximize likelihood,
        :param est_var: bool
            Whether to estimate variance of the fitted copula
        :param verbose: int
            Log level for the estimator. The higher the number, the more verbose. 0 prints nothing
        :param optim_options: dict
            keyword arguments to pass into scipy.optimize.minimize
""".strip()


class CopulaEstimator:

    @format_docstring(params_doc=__estimator_params_docs__)
    def __init__(self, copula: Copula, data: np.ndarray, x0: np.ndarray = None, method='ml', est_var=False,
                 verbose=1, optim_options: Optional[dict] = None):
        """
        Estimator for any copula

        By passing the copula into class object, the copula will be automatically fitted

        {params_doc}
        """
        from copulae import AbstractEllipticalCopula

        self.copula = copula
        self._is_elliptical = isinstance(copula, AbstractEllipticalCopula)
        self.data = data
        self._est_var = est_var

        self._method = method.lower()

        if np.any(data) < 0 or np.any(data) > 1:
            raise ValueError("data must be in [0, 1] -- you probably forgot to convert data to pseudo-observations")
        elif len(data) < self.copula.dim:
            raise ValueError("number of data (rows) must be greater than its dimension")

        # default optim options is the first dictionary. We have set the default options for Nelder-Mead
        self.__optim_options = optim_options or {}

        self._x0 = x0
        self._verbose = verbose

        self.fit()  # fit the copula

    def fit(self):
        m = self._method
        if m in {'ml', 'mpl'}:
            MaxLikelihoodEstimator(self.copula, self.data, self.initial_params, self.optim_options, self._est_var,
                                   self._verbose).fit(m)
        elif m == 'itau':
            raise NotImplementedError
        elif m == 'irho':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _fit_icor(self, method: str):
        """
        Inversion of Spearman's rho or Kendall's tau Estimator

        :param method: str
            Indicates whether Spearman's rho or Kendall's tau shall be used
        :return: numpy array
            estimates for the copula
        """

        if self._is_elliptical and hasattr(self.copula, 'df'):
            # T copula
            pass

        estimate = self._est_copula_cor(method)
        self.copula.params = estimate

        d = self.copula.dim
        var_est = np.full((d, d), np.nan)
        if self._est_var:
            # TODO calculate estimate variance [icor]
            pass

        method = 'Inversion of ' + "Kendall's Tau" if method == 'tau' else "Spearman's Rho"
        self.copula.fit_stats = FitStats(estimate, var_est, method, np.nan, len(self.data))

        return estimate

    def _est_copula_cor(self, method: str):
        """
        Estimates Parameter Matrix from Matrix of Kendall's Taus / Spearman's Rhos

        :param method: str
            the rank correlation used, one of 'tau' or 'rho' representing Kendall and Spearman respectively
        :return:
        """
        cop = self.copula
        data = self.data

        ii = tri_indices(cop.dim, 1, 'lower')
        if method == 'tau':
            tau = kendall_tau(data)[ii]
            theta = cop.itau(tau)
        elif method == 'rho':
            rho = spearman_rho(data)[ii]
            theta = cop.irho(rho)
        else:
            raise ValueError("method should be one of 'tau', 'rho'")

        if not self._is_elliptical:
            return theta

        M = np.identity(cop.dim)
        M[tri_indices(cop.dim, 1)] = np.tile(theta, 2)
        M = corr_nearest(M)
        return M[ii]

    def _optimize(self) -> OptimizeResult:
        return minimize(self._est_copula_log_lik, self.initial_params, **self.optim_options)

    @property
    def initial_params(self):
        if self._x0 is not None:
            return self._x0

        if self._is_elliptical:
            corr = pearson_rho(self.data)
            rhos = corr[tri_indices(self.copula.dim, 1, 'lower')]
            if hasattr(self.copula, '_df'):
                # T-distribution
                return np.array([4.669, *rhos])  # set df as Feigenbaum's constant
            else:
                # Gaussian
                return rhos

        try:
            start = self._fit_icor('tau')
            ll = self._est_copula_log_lik(start)

            if np.isfinite(ll):
                return start
            else:
                # TODO implement custom start for Clayton Copula

                if not np.isnan(ll) and np.isinf(ll):
                    # for perfectly correlated data
                    return start
                return self.copula.params
        except NotImplementedError:
            return self.copula.params

    @property
    def optim_options(self):
        data = self.data
        verbose = self._verbose
        options = self.__optim_options

        max_iter = min(len(data) * 250, 20000)
        disp = verbose >= 2

        options.setdefault('method', 'SLSQP')
        method_is = _method_is(options['method'])
        bounds = [(l, u) for l, u in zip(*self.copula.params_bounds)]
        if method_is('Nelder-Mead'):
            return merge_dict({
                'options': {
                    'maxiter': max_iter,
                    'disp': disp,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                },
            }, options)
        elif method_is('BFGS'):
            return merge_dict({
                'options': {
                    'maxiter': max_iter,
                    'disp': disp,
                    'gtol': 1e-4,
                }
            }, options)
        elif method_is('SLSQP'):
            return merge_dict({
                'bounds': bounds,
                'options': {
                    'maxiter': max_iter,
                    'ftol': 1e-06,
                    'iprint': 1,
                    'disp': disp,
                    'eps': 1.5e-8
                }
            }, options)
        elif method_is('COBYLA'):
            return merge_dict({
                'bounds': bounds,
                'options': {
                    'maxiter': max_iter,
                    'rhobeg': 1.0,
                    'disp': disp,
                    'catol': 0.0002
                }
            }, options)
        elif method_is('trust-constr'):
            return merge_dict({
                'maxiter': max_iter,
                'disp': disp,
                'bounds': bounds
            }, options)
        else:
            return options


def _method_is(method: str):
    def compare(b: str):
        return method.casefold() == b.casefold()

    return compare
