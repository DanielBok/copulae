from typing import Literal

import numpy as np

from copulae.copula.estimator.misc import is_archimedean, is_elliptical
from copulae.copula.estimator.summary import FitSummary
from copulae.core import create_cov_matrix, near_psd, tri_indices
from copulae.stats import kendall_tau, spearman_rho

__all__ = ['estimate_corr_inverse_params']


def estimate_corr_inverse_params(copula, data: np.ndarray, type_: Literal['itau', 'irho']):
    """
    Fits the copula with the inversion of Spearman's rho or Kendall's tau Estimator

    Parameters
    ----------
    copula
        Copula whose parameters are to be estimated
    data
        Data to fit the copula with
    type_ : {'irho', 'itau'}
        The type of rank correlation measure to use. 'itau' uses Kendall's tau while 'irho' uses Spearman's rho
    """

    type_ = type_.lower()
    if type_ not in ('itau', 'irho'):
        raise ValueError("Correlation Inversion must be either 'itau' or 'irho'")

    icor = fit_cor(copula, data, type_)

    if is_elliptical(copula):
        from copulae.elliptical.student import StudentCopula, StudentParams
        estimate = icor
        if isinstance(copula, StudentCopula):
            # itau and irho must fix degree of freedom
            copula.params = StudentParams(copula.params.df, estimate)
        else:
            copula.params[:] = estimate

    elif is_archimedean(copula):
        estimate = copula.params = np.mean(icor)
    else:
        raise NotImplementedError(f"Have not developed for '{copula.name} copula'")

    method = f"Inversion of {'Spearman Rho' if type_ == 'irho' else 'Kendall Tau'} Correlation"
    return FitSummary(estimate, method, copula.log_lik(data), len(data))


def fit_cor(copula, data: np.ndarray, typ: str) -> np.ndarray:
    """
    Constructs parameter matrix from matrix of Kendall's Taus or Spearman's Rho

    Parameters
    ----------
    copula: BaseCopula
        Copula instance

    data: ndarray
        Data to fit copula with
    typ: {'irho', 'itau'}
        The type of rank correlation measure to use. 'itau' uses Kendall's tau while 'irho' uses Spearman's rho

    Returns
    -------
    ndarray
        Parameter matrix is copula is elliptical. Otherwise, a vector
    """

    indices = tri_indices(copula.dim, 1, 'lower')
    if typ == 'itau':
        tau = np.asarray(kendall_tau(data))[indices]
        theta = copula.itau(tau)
    elif typ == 'irho':
        rho = np.asarray(spearman_rho(data))[indices]
        theta = copula.irho(rho)
    else:
        raise ValueError("Correlation Inversion must be either 'itau' or 'irho'")

    if is_elliptical(copula):
        theta = near_psd(create_cov_matrix(theta))[indices]

    return theta
