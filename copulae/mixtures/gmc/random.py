import numpy as np
from numpy.random import choice
from scipy.stats import multivariate_normal as mvn

from .marginals import p_gmm_marginal
from .param import GMCParam

__all__ = ['random_gmcm']


def random_gmcm(n: int, p: GMCParam):
    """
    Generates random variables from a Gaussian Mixture Copula Model

    Parameters
    ----------
    n : int
        The number of instances to generate
    p : GMCParam
        The Gaussian Mixture Copula paramter

    Returns
    -------
    np.ndarray
        An array of random variables
    """
    z = random_gmm(n, p)  # latent realizations from Gaussian mixture model
    return p_gmm_marginal(z, p)


def random_gmm(n: int, p: GMCParam):
    """Generates random variables from a Gaussian Mixture Model"""

    output = np.empty((n, p.n_dim))
    order = choice(range(p.n_clusters), n, p=p.prob)
    for i in range(p.n_clusters):
        k = sum(order == i)
        output[order == i] = mvn.rvs(p.means[i], cov=p.covs[i], size=k)

    return output
