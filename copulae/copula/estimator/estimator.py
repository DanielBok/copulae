from typing import Any, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

from copulae.copula.estimator.misc import is_archimedean, is_elliptical
from copulae.core import tri_indices
from copulae.stats import pearson_rho
from copulae.types import Array, Ties
from copulae.utility.dict import merge_dict
from .corr_inversion import estimate_corr_inverse_params
from .max_likelihood import estimate_max_likelihood_params

__all__ = ['fit_copula', 'EstimationMethod']


class Copula(Protocol):
    def pdf(self, u: Array, log=False) -> Union[np.ndarray, float]:
        raise NotImplementedError

    def log_lik(self, data: np.ndarray, *, to_pobs=True, ties: Ties = 'average') -> float:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return NotImplemented

    @property
    def dim(self) -> int:
        return NotImplemented

    @property
    def params(self) -> Any:
        return NotImplemented

    @params.setter
    def params(self, x: Any):
        raise NotImplementedError

    @property
    def bounds(self) -> Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]:
        return NotImplemented


EstimationMethod = Literal['ml', 'irho', 'itau']


def fit_copula(copula: Copula, data: Union[pd.DataFrame, np.ndarray],
               x0: Optional[Union[np.ndarray, float]], method: EstimationMethod,
               verbose: int, optim_options: Optional[dict], scale: float):
    """
    Estimator for any copula

    By passing the copula into class object, the copula will be automatically fitted

    Parameters
    ----------
    copula
        The copula instance

    data: ndarray
        Array of data used to fit copula. Usually, data should be the pseudo observations

    x0: ndarray
        Initial starting point. If value is None, best starting point will be estimated

    method: { 'ml', 'irho', 'itau' }
        Method of fitting. Supported methods are: 'ml' - Maximum Likelihood, 'irho' - Inverse Spearman Rho,
        'itau' - Inverse Kendall Tau

    verbose: int
        Log level for the estimator. The higher the number, the more verbose it is. 0 prints nothing.

    optim_options: dict
        Keyword arguments to pass into scipy.optimize.minimize

    scale: float
        Amount to scale the objective function value. This is helpful in achieving higher accuracy
        as it increases the sensitivity of the optimizer. The downside is that the optimizer could
        likely run longer as a result

    See Also
    --------
    :code:`scipy.optimize.minimize`: the optimization function
    """
    options = form_options(optim_options or {}, verbose, data, copula)
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if np.any(data) < 0 or np.any(data) > 1:
        raise ValueError("data must be in [0, 1] -- you probably forgot to convert data to pseudo-observations")
    elif len(data) < copula.dim:
        raise ValueError("number of data (rows) must be greater than its dimension")

    m = method.lower()
    if m in {'ml'}:
        x0 = initial_params(copula, data, x0)
        return estimate_max_likelihood_params(copula, data, x0, options, verbose, scale)
    elif m in ('itau', 'irho'):
        return estimate_corr_inverse_params(copula, data, m)
    else:
        raise NotImplementedError(f"'{m}' is not implemented")


def initial_params(copula: Copula, data: np.ndarray, x0: np.ndarray):
    # ensure that initial is defined. If it is defined, checks that all x0 values are finite
    # neither infinite nor nan
    if x0 is not None and np.all(np.isfinite(x0)):
        return x0

    if is_elliptical(copula):
        corr = pearson_rho(data)
        rhos = corr[tri_indices(copula.dim, 1, 'lower')]
        if copula.name.lower() == 'student':
            # T-distribution
            return np.array([4.669, *rhos])  # set df as Feigenbaum's constant
        else:
            # Gaussian
            return rhos

    try:
        start = estimate_corr_inverse_params(copula, data, 'itau').params
        ll = copula.log_lik(data)

        if np.isfinite(ll):
            return start
        else:
            if copula.name.lower() == 'clayton' and copula.dim == 2:
                # The support of bivariate claytonCopula with negative parameter is not
                # the full unit square; the further from 0, the more restricted.
                while start < 0:
                    start += .2
                    copula.params = start
                    if np.isfinite(copula.log_lik(data)):
                        break

            if not np.isnan(ll) and np.isinf(ll):
                # for perfectly correlated data
                return start
            return copula.params
    except NotImplementedError:
        return copula.params


def form_options(options: dict, verbose: int, data: np.ndarray, copula: Copula):
    def method_is(method: str):
        return options['method'].casefold() == method.casefold()

    max_iter = min(len(data) * 250, 20000)
    disp = verbose >= 2

    if is_archimedean(copula):
        options.setdefault('method', 'Nelder-Mead')
    else:
        options.setdefault('method', 'SLSQP')

    lb, ub = copula.bounds
    if np.isscalar(lb) and np.isscalar(ub):
        bounds = [[lb, ub]]
    else:
        assert not np.isscalar(lb) and not np.isscalar(ub), "bounds must be both scalars or both vectors"
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
