from typing import Union

import numpy as np

from copulae.copula.abstract import AbstractCopula as Copula
from copulae.core import create_cov_matrix, near_psd, tri_indices
from copulae.stats import kendall_tau, spearman_rho

InitialParam = Union[float, np.ndarray]


def is_elliptical(copula: Copula):
    return copula.name.lower() in ('gaussian', 'student')


def is_archimedean(copula: Copula):
    return copula.name.lower() in ('clayton', 'gumbel', 'frank', 'joe', 'amh')


def warn_no_convergence():
    print("Warning: Possible convergence problem with copula fitting")


def fit_cor(copula: Copula, data: np.ndarray, typ: str) -> np.ndarray:
    """
    Constructs parameter matrix from matrix of Kendall's Taus or Spearman's Rho

    Parameters
    ----------
    copula: AbstractCopula
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
        tau = kendall_tau(data)[indices]
        theta = copula.itau(tau)
    elif typ == 'irho':
        rho = spearman_rho(data)[indices]
        theta = copula.irho(rho)
    else:
        raise ValueError("Correlation Inversion must be either 'itau' or 'irho'")

    if is_elliptical(copula):
        theta = near_psd(create_cov_matrix(theta))[indices]

    return theta
