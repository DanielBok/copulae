from typing import Optional

import numpy as np

from copulae.copula.estimator.misc import is_elliptical
from copulae.core import tri_indices
from copulae.stats import pearson_rho
from copulae.utility import merge_dict
from .corr_inversion import CorrInversionEstimator
from .max_likelihood import MaxLikelihoodEstimator

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

EstimationMethod = Literal['ml', 'mpl', 'irho', 'itau']


class CopulaEstimator:
    def __init__(self, copula, data: np.ndarray, x0: np.ndarray = None, method: EstimationMethod = 'ml', verbose=1,
                 optim_options: Optional[dict] = None):
        """
        Estimator for any copula

        By passing the copula into class object, the copula will be automatically fitted

        Parameters
        ----------
        data: ndarray
            Array of data used to fit copula. Usually, data should be the pseudo observations

        x0: ndarray
            Initial starting point. If value is None, best starting point will be estimated

        method: { 'ml', 'mpl', 'irho', 'itau' }
            Method of fitting. Supported methods are: 'ml' - Maximum Likelihood, 'mpl' - Maximum Pseudo-likelihood,
            'irho' - Inverse Spearman Rho, 'itau' - Inverse Kendall Tau

        verbose: int
            Log level for the estimator. The higher the number, the more verbose it is. 0 prints nothing.

        optim_options: dict
            Keyword arguments to pass into scipy.optimize.minimize

        See Also
        --------
        :code:`scipy.optimize.minimize`: the optimization function
        """

        self.copula = copula
        self.data = data
        self.x0 = x0
        self.verbose = verbose
        self.method = method.lower()

        # default optim options is the first dictionary. We have set the default options for Nelder-Mead
        self.options = form_options(optim_options or {}, verbose, data, copula.bounds)

        if np.any(data) < 0 or np.any(data) > 1:
            raise ValueError("data must be in [0, 1] -- you probably forgot to convert data to pseudo-observations")
        elif len(data) < self.copula.dim:
            raise ValueError("number of data (rows) must be greater than its dimension")

        self.fit()  # fit the copula

    def fit(self):
        m = self.method
        if m in {'ml', 'mpl'}:
            MaxLikelihoodEstimator(self.copula, self.data, self.initial_params, self.options, self.verbose).fit(m)
        elif m in ('itau', 'irho'):
            CorrInversionEstimator(self.copula, self.data, self.verbose).fit(m)
        else:
            raise NotImplementedError(f"'{m}' is not implemented")

    @property
    def initial_params(self) -> np.ndarray:
        if self.x0 is not None:
            return self.x0

        if is_elliptical(self.copula):
            corr = pearson_rho(self.data)
            rhos = corr[tri_indices(self.copula.dim, 1, 'lower')]
            if hasattr(self.copula, '_df'):
                # T-distribution
                return np.array([4.669, *rhos])  # set df as Feigenbaum's constant
            else:
                # Gaussian
                return rhos

        try:
            start = CorrInversionEstimator(self.copula, self.data, self.verbose).fit('itau')

            ll = self.copula.log_lik(self.data)

            if np.isfinite(ll):
                return start
            else:
                if self.copula.name.lower() == 'clayton' and self.copula.dim == 2:
                    # The support of bivariate claytonCopula with negative parameter is not
                    # the full unit square; the further from 0, the more restricted.
                    while start < 0:
                        start += .2
                        self.copula.params = start
                        if np.isfinite(self.copula.log_lik(self.data)):
                            break

                if not np.isnan(ll) and np.isinf(ll):
                    # for perfectly correlated data
                    return start
                return self.copula.params
        except NotImplementedError:
            return self.copula.params


def form_options(options: dict, verbose: int, data: np.ndarray, bounds):
    def method_is(method: str):
        return options['method'].casefold() == method.casefold()

    max_iter = min(len(data) * 250, 20000)
    disp = verbose >= 2

    options.setdefault('method', 'SLSQP')

    lb, ub = bounds
    if isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
        bounds = [bounds]
    else:
        # lower and upper bounds are arrays
        bounds = [(l, u) for l, u in zip(lb, ub)]

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
