from typing import Collection, Literal, Protocol, Union

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from copulae.copula.estimator.misc import warn_no_convergence
from copulae.copula.estimator.summary import FitSummary

__all__ = ['estimate_max_likelihood_params']


def estimate_max_likelihood_params(copula, data: np.ndarray, x0: Union[np.ndarray, float],
                                   optim_options: dict, verbose: int, scale: float):
    """
    Fits the copula with the Maximum Likelihood Estimator

    Parameters
    ----------
    copula
        Copula whose parameters are to be estimated
    data
        Data to fit the copula with
    x0
        Initial parameters for optimization
    optim_options
        optimizer options
    verbose
        Verbosity level for the optimizer
    scale
        Amount to scale the objective function value. This is helpful in achieving higher accuracy
        as it increases the sensitivity of the optimizer. The downside is that the optimizer could
        likely run longer as a result
    """

    def calc_log_lik(param: Union[float, Collection[float]]) -> float:
        """
        Calculates the log likelihood after setting the new parameters (inserted from the optimizer) of the copula

        Parameters
        ----------
        param: ndarray
            Parameters of the copula

        Returns
        -------
        float
            Negative log likelihood of the copula

        """

        if any(np.isnan(np.ravel(param))):
            return np.inf
        try:
            copula.params = param
            return -copula.log_lik(data, to_pobs=False) * scale
        except ValueError:  # error encountered when setting invalid parameters
            return np.inf

    res: OptimizeResult = minimize(calc_log_lik, x0, **optim_options)
    if not res['success']:
        if verbose >= 1:
            warn_no_convergence()
        estimate = np.nan if np.isscalar(x0) else np.repeat(np.nan, len(x0))
    else:
        estimate = res['x']
        copula.params = estimate

    return FitSummary(estimate, "Maximum likelihood", -res['fun'], len(data), optim_options, res)
