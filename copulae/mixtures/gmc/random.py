import numpy as np
from numpy.random import choice
from scipy.stats import multivariate_normal as mvn

from .marginals import gmm_marginal_cdf
from .parameter import GMCParam

__all__ = ['random_gmcm']


def random_gmcm(n: int, param: GMCParam):
    """
    Generates random variables from a Gaussian Mixture Copula Model

    Parameters
    ----------
    n : int
        The number of instances to generate
    param : GMCParam
        The Gaussian Mixture Copula parameter

    Returns
    -------
    np.ndarray
        An array of random variables
    """
    z = random_gmm(n, param)  # latent realizations from Gaussian mixture model
    return gmm_marginal_cdf(z, param)


def random_gmm(n: int, param: GMCParam):
    """Generates random variables from a Gaussian Mixture Model"""

    output = np.empty((n, param.n_dim))
    order = choice(range(param.n_clusters), n, p=param.prob)
    for i in range(param.n_clusters):
        k = sum(order == i)
        output[order == i] = mvn.rvs(param.means[i], cov=param.covs[i], size=k)

    return output
