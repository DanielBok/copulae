import numpy as np

from copulae.copula.abstract import AbstractCopula as Copula
from copulae.copula.summary import FitSummary
from copulae.copula.utils import fit_cor, is_archimedean, is_elliptical
from copulae.core import rank_data


class CorrInversionEstimator:
    def __init__(self, copula: Copula, data: np.ndarray, est_var: bool, verbose: int):
        """
        Inversion of Spearman's rho or Kendall's tau Estimator

        Parameters
        ----------
        copula: AbstractCopula
            Copula whose parameters are to be estimated

        data: ndarray
            Data to fit the copula with

        est_var: bool
            If True, calculates the variance estimates

        verbose: int
            Verbosity level for the optimizer
        """

        self.copula = copula
        self.data = data
        self._verbose = verbose
        self.est_var = est_var

    def fit(self, typ: str) -> np.ndarray:
        """
        Fits the copula with the inversion of Spearman's rho or Kendall's tau Estimator

        Parameters
        ----------
        typ: {'irho', 'itau'}
            The type of rank correlation measure to use. 'itau' uses Kendall's tau while 'irho' uses Spearman's rho

        Returns
        -------
        ndarray
            Estimates for the copula
        """
        typ = typ.lower()
        if typ not in ('itau', 'irho'):
            raise ValueError("Correlation Inversion must be either 'itau' or 'irho'")

        icor = fit_cor(self.copula, self.data, typ)

        if is_elliptical(self.copula):
            estimate = icor
            self.copula.params[:] = estimate
        elif is_archimedean(self.copula):
            estimate = np.mean(icor)
            self.copula.params = estimate
        else:
            raise NotImplementedError(f"Have not developed for '{self.copula.name} copula'")

        var_est = np.nan
        if self.est_var:
            # TODO calculate variance estimate. Need to implement variance
            pass
        method = f"Inversion of {'Spearman Rho' if typ == 'irho' else 'Kendall Tau'} Correlation"

        self.copula.fit_smry = FitSummary(estimate, var_est, method, self.copula.log_lik(self.data), len(self.data))

        return estimate

    def variance(self, typ: str):
        """
        Variance of the Inversion of a Rank Correlation Measure Estimator

        Parameters
        ----------
        typ: {'itau', 'irho'}
            The type of rank correlation measure to use

        Returns
        -------
        float:
            Variance of the inversion of rank correlation measure estimator
        """

        u = self.data
        dim = self.copula.dim
        nrow = len(self.data)
        ncol = dim * (dim - 1) // 2
        v = np.zeros((nrow, ncol))

        if typ == 'itau':
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    ec = 2 * np.sum([(u[:, i] <= u[k, i]) & (u[:, j] <= u[k, j]) for k in range(nrow)]) / nrow
                    v[:, i * (dim - 1) + j] = ec - u[:, i] - u[:, j]
        else:
            ord = np.argsort(-u, 1, ) + 1
            ordb = rank_data(u, 1)

            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    a = np.array([0, *np.cumsum(u[ord[:, i], j][nrow - ordb[:, i]]) / nrow])
                    b = np.array([0, *np.cumsum(u[ord[:, j], i][nrow - ordb[:, j]]) / nrow])
                    v[:, i * (dim - 1) + j] = u[:, i] * u[:, j] + a + b
        # TODO complete rest of function GETL

        raise NotImplementedError
