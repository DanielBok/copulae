from typing import Union

import numpy as np
from scipy.stats import multivariate_normal as mvn

from .marginals import gmm_marginal_ppf
from .parameter import GMCParam

__all__ = ['gmm_log_likelihood', 'gmm_log_likelihood_marginals', 'gmcm_log_likelihood']


def gmm_log_likelihood(x: np.ndarray, param: GMCParam) -> float:
    """
    Log likelihood of the quantile observations given a Gaussian Mixture Model

    Parameters
    ----------
    x : np.ndarray
        Quantiles
    param : GMCParam
        The Gaussian Mixture Copula parameters

    Returns
    -------
    float
        The log likelihood
    """
    marginals = np.zeros(len(x))

    for i, prob in enumerate(param.prob):
        marginals += prob * mvn.pdf(x, param.means[i], param.covs[i], allow_singular=True)

    return np.log(marginals).sum()


def gmm_log_likelihood_marginals(x: np.ndarray, param: GMCParam) -> Union[np.ndarray, float]:
    """
    Log likelihood of the quantile observations given a Gaussian Mixture Model along each axis (dimension)
    of the individual Gaussian component weighted by the component's probability of occurrence.

    Parameters
    ----------
    x : np.ndarray
        Quantiles
    param : GMCParam
        The Gaussian Mixture Copula parameters

    Returns
    -------
    float
        The marginal log likelihood
    """
    sigmas = np.array([np.diag(c) for c in param.covs])
    marginals = np.zeros_like(x)

    for p, mus, sds in zip(param.prob, param.means, sigmas):
        for j, (m, s) in enumerate(zip(mus, sds)):
            marginals[:, j] += p * mvn.pdf(x[:, j], m, s)

    return np.log(marginals).sum(0)  # aggregate across each dimension


def gmcm_log_likelihood(x: np.ndarray, param: GMCParam) -> float:
    """
    Log likelihood of the quantile observations given a Gaussian Mixture Copula Model

    Parameters
    ----------
    x : np.ndarray
        Quantiles
    param : GMCParam
        The Gaussian Mixture Copula parameters

    Returns
    -------
    float
        The log likelihood
    """
    q = gmm_marginal_ppf(x, param)
    gmm_ll = gmm_log_likelihood_marginals(q, param)
    return gmm_log_likelihood(q, param) - sum(gmm_ll)
