from typing import Collection, Union

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from copulae.copula.estimator.misc import warn_no_convergence
from copulae.copula.summary import FitSummary


class MaxLikelihoodEstimator:
    def __init__(self, copula, data: np.ndarray, initial_params: np.ndarray, optim_options: dict, verbose: int):
        """
        Maximum likelihood estimator

        Parameters
        ----------
        copula:
            Copula whose parameters are to be estimated

        data: ndarray
            Data to fit the copula with

        initial_params: float or ndarray
            Initial parameters for optimization

        optim_options: dict
            optimizer options

        verbose: int
            Verbosity level for the optimizer

        """
        self.copula = copula
        self.data = data
        self.initial_params = initial_params
        self.optim_options = optim_options
        self.verbose = verbose

    def fit(self, method):
        """
        Fits the copula with the Maximum Likelihood Estimator

        Parameters
        ----------
        method: {'ml', 'mpl'}
            This will determine the variance estimate

        Returns
        -------
        ndarray
            Estimated parameters for the copula

        """

        res: OptimizeResult = minimize(self.copula_log_lik, self.initial_params, **self.optim_options)

        if not res['success']:
            if self.verbose >= 1:
                warn_no_convergence()
            return

        estimate = res['x']
        self.copula.params = estimate

        method = f"Maximum {'pseudo-' if method == 'mpl' else ''}likelihood"
        self.copula.fit_smry = FitSummary(estimate, method, res['fun'], len(self.data), self.optim_options, res)

        return estimate

    def copula_log_lik(self, param: Union[float, Collection[float]]) -> float:
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
        try:
            self.copula.params = param
            return -self.copula.log_lik(self.data, to_pobs=False)
        except ValueError:  # error encountered when setting invalid parameters
            return np.inf
