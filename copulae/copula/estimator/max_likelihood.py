from typing import Collection, Union

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from copulae.copula.estimator.misc import warn_no_convergence
from copulae.copula.summary import FitSummary


class MaxLikelihoodEstimator:
    def __init__(self, copula, data: np.ndarray, initial_params: Union[float, Collection[float]],
                 optim_options: dict, est_var: bool, verbose: int):
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

        est_var: bool
            If True, calculates the variance estimates

        verbose: int
            Verbosity level for the optimizer

        """
        self.copula = copula
        self.data = data
        self.initial_params = initial_params
        self.optim_options = optim_options
        self._verbose = verbose
        self._est_var = est_var

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
            if self._verbose >= 1:
                warn_no_convergence()
            return

        estimate = res['x']
        self.copula.params = estimate

        d = self.copula.dim

        var_est = np.full((d, d), np.nan)
        if self._est_var:
            # TODO calculate variance estimate [ml]
            if method == 'ml':
                pass
            else:  # method == 'mpl'
                pass

        method = f"Maximum {'pseudo-' if method == 'mpl' else ''}likelihood"
        self.copula.fit_smry = FitSummary(estimate, var_est, method, res['fun'], len(self.data), self.optim_options,
                                          res)

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
