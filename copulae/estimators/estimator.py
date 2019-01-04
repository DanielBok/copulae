from typing import Optional

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from statsmodels.stats.correlation_tools import corr_nearest

from copulae.copula.abstract import AbstractCopula as Copula, FitStats
from copulae.math_tools import kendall_tau, pearson_rho, spearman_rho, tri_indices
from copulae.utils import format_docstring, merge_dict

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
        self.copula = copula
        self.data = data
        self.est_var = est_var

        self._method = method.lower()

        if np.any(data) < 0 or np.any(data) > 1:
            raise ValueError("data must be in [0, 1] -- you probably forgot to convert data to pseudo-observations")

        # default optim options is the first dictionary. We have set the default options for Nelder-Mead
        self.__optim_options = optim_options or {}

        self.copula.params = 0 if x0 is None else x0
        self._verbose = verbose

        self.fit()  # fit the copula

    def fit(self):
        m = self._method
        if m in {'ml', 'mpl'}:
            self._fit_ml(m)
        else:
            raise NotImplementedError

    def _fit_ml(self, method):
        """
        Maximum Likelihood Estimator for Copulas

        :param method: str
            'ml' or 'mpl'. This will determine the variance estimate
        :return: numpy array
            estimates for the copula
        """

        res = self._optimize()

        if not res['success']:
            if self._verbose >= 1:
                _warn_no_convergence()
            return

        estimate = res['x']
        self.copula.params = estimate

        d = self.copula.dim

        var_est = np.full((d, d), np.nan)
        if self.est_var:
            # TODO calculate variance estimate [ml]
            if method == 'ml':
                pass
            else:  # method == 'mpl'
                pass

        method = f"Maximum {'pseudo-' if method == 'mpl' else ''}likelihood"
        self.copula.fit_stats = FitStats(estimate, var_est, method, res['fun'], len(self.data), self.optim_options,
                                         res)

        return estimate

    def _fit_icor(self, method: str):
        """
        Inversion of Spearman's rho or Kendall's tau Estimator

        :param method: str
            Indicates whether Spearman's rho or Kendall's tau shall be used
        :return: numpy array
            estimates for the copula
        """

        if self.copula.is_elliptical and hasattr(self.copula, 'df'):
            # T copula
            pass

        estimate = self._est_copula_cor(method)
        self.copula.params = estimate

        d = self.copula.dim
        var_est = np.full((d, d), np.nan)
        if self.est_var:
            # TODO calculate estimate variance [icor]
            pass

        method = 'Inversion of ' + "Kendall's Tau" if method == 'tau' else "Spearman's Rho"
        self.copula.fit_stats = FitStats(estimate, var_est, method, np.nan, len(self.data))

        return estimate

    def _est_copula_log_lik(self, param: np.ndarray) -> float:
        """
        Calculates the log likelihood after setting the new parameters (inserted from the optimizer) of the copula

        :param param: numpy array
            parameters of the copula
        :return: float
            negative log likelihood
        """
        old_params = self.copula.params
        try:
            self.copula.params = param
            return -self.copula.log_lik(self.data)
        except ValueError:
            # Sometimes, we need to constrain the parameters so that they do not become invalid. Example: elliptical
            # copula parameters need to have PSD covariance. However, since it is not explicitly stated in the
            # constraints, we would hit a ValueError when determining the log likelihood. In this case, we revert to
            # the previous parameter and stop the optimization
            self.copula.params = old_params
            raise RuntimeError

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

        if not cop.is_elliptical:
            return theta

        M = np.identity(cop.dim)
        M[tri_indices(cop.dim, 1)] = np.tile(theta, 2)
        M = corr_nearest(M)
        return M[ii]

    def _optimize(self) -> OptimizeResult:
        # TODO convert to constrained optimization for Elliptical Copulas where cov matrix is constrained to be PSD
        try:
            return minimize(self._est_copula_log_lik, self.initial_params, **self.optim_options)
        except RuntimeError:
            return OptimizeResult({
                'x': self.copula.params,
                'fun': self.copula.log_lik(self.data),
                'success': True,
                'message': 'termination due to runtime errors. If elliptical copulas, it is likely because optimizer '
                           'descended to a non-positive semi-definite region.'
            })

    @property
    def initial_params(self):
        if self.copula.is_elliptical and not hasattr(self.copula, 'df'):
            # Gaussian
            corr = pearson_rho(self.data)
            return corr[tri_indices(self.copula.dim, 1, 'lower')]

        params = self.copula.params

        if not np.allclose(params, 0):
            return params

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

        method_is = _method_is(options.get('method', 'Nelder-Mead'))
        if method_is('Nelder-Mead'):
            return merge_dict({
                'method': 'Nelder-Mead',
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
        else:
            # TODO set other defaults for other optimizers like BFGS (supposedly faster) and SLSQP (constrained)
            return options


def _method_is(method: str):
    def compare(b: str):
        return method.casefold() == b.casefold()

    return compare


def _warn_no_convergence():
    print("Warning: Possible convergence problem with copula fitting")
