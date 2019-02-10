import numpy as np
from scipy.optimize import OptimizeResult, minimize

from copulae.copula.abstract import AbstractCopula as Copula, FitStats
from .utils import warn_no_convergence


class MaxLikelihoodEstimator:
    def __init__(self, copula: Copula, data: np.ndarray, initial_params: np.ndarray, optim_options, est_var: bool,
                 verbose: int):
        """
        Maximum Likelihood Estimator for Copulas

        :param copula: BaseCopula
            copula to be inverted
        :param data: ndarray
            data to fit copula with
        :param initial_params: ndarray
            initial parameters for copula
        :param optim_options: dict
            optimizer options
        :param est_var: bool
            If true, calculates variance estimates
        :param verbose: int
            verbosity level for optimizer
        """

        self.copula = copula
        self.data = data
        self.initial_params = initial_params
        self.optim_options = optim_options
        self._verbose = verbose
        self._est_var = est_var

    def fit(self, method):
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
        self.copula.fit_stats = FitStats(estimate, var_est, method, res['fun'], len(self.data), self.optim_options,
                                         res)

        return estimate

    def copula_log_lik(self, param: np.ndarray) -> float:
        """
        Calculates the log likelihood after setting the new parameters (inserted from the optimizer) of the copula

        :param param: numpy array
            parameters of the copula
        :return: float
            negative log likelihood
        """
        try:
            self.copula.params = param
            return -self.copula.log_lik(self.data)
        except ValueError:  # error encountered when setting invalid parameters
            return np.inf

    def _optimize(self) -> OptimizeResult:
        return minimize(self.copula_log_lik, self.initial_params, **self.optim_options)
