"""
Functions here calculate the probability distribution, density and quantile of
the marginals
"""
import numpy as np

from .param import GMCParam


def p_gmm_marginal(z: np.ndarray, p: GMCParam):
    dist = np.zeros_like(z)
    for mus, sigmas, prob in zip(p.means, p.covs, p.prob):  # type: np.ndarray, np.ndarray, float
        for j in range(p.n_dim):
            m = mus[j]
            s = sigmas[j, j] ** 0.5
            dist[:, j] = dist[:, j] + prob * approximate_pnorm(z[:, j], m, s)

    return dist


def approximate_pnorm(vec: np.ndarray, mu: float, sd: float):
    """
    Approximate univariate Gaussian CDF, applied marginally
    Abramowitz, Stegun p. 299 (7.1.25) (using error function) improved.
    """
    a1 = 0.3480242
    a2 = -0.0958798
    a3 = 0.7478556
    p = 0.47047
    sqrt2 = 1.4142136

    z = (vec - mu) / (sd * sqrt2)
    zi = np.abs(z)
    t = 1 / (1 + p * zi)
    chunk = 0.5 * (a1 * t + a2 * t ** 2 + a3 * t ** 3) * np.exp(-(zi ** 2))

    return np.where(z < 0, chunk, 1 - chunk)
