"""
Functions here calculate the probability distribution, density and quantile of
the marginals
"""
import numpy as np
from scipy.interpolate import interp1d

from .parameter import GMCParam


def gmm_marginal_ppf(q: np.ndarray, param: GMCParam, resolution=2000, spread=5, validate=False):
    """
    Approximates the inverse cdf of the input given the GMC parameters

    Parameters
    ----------
    q : np.ndarray
        Marginal probability values. Must be between [0, 1]
    param : GMCParam
        The Gaussian Mixture Copula parameters
    resolution : int
        The number of values used for approximation. The higher the resolution, the finer the interpolation.
        However, it also comes at higher computation cost
    spread : int
        The number of standard deviation to approximate the range of marginal probability
    validate : bool
        If True, validates that all input marginal probability vector are between [0, 1] and raises an
        ValueError if condition isn't met.

    Returns
    -------
    np.ndarray
        Quantile corresponding to the lower tail probability q.
    """
    if validate and ((q < 0).any() or (q > 1).any()):
        raise ValueError("Invalid probability marginal values detected. Ensure that are values are between [0, 1]")

    # number of samples for each cluster with a minimum of 2
    n_samples = np.maximum(np.round(param.prob * resolution), 2).astype(int)

    # create evaluation grid
    grid = np.empty((n_samples.sum(), param.n_dim))
    i = 0
    for n, mu, sigma2 in zip(n_samples, param.means, param.covs):
        sigma = np.sqrt(np.diag(sigma2))
        grid[i:(i + n)] = np.linspace(mu - spread * sigma, mu + spread * sigma, n)
        i += n

    dist = gmm_marginal_cdf(grid, param)

    ppf = np.empty_like(q)
    for i in range(param.n_dim):
        # does an inverse cdf, ppf, to get the marginals
        ppf[:, i] = interp1d(dist[:, i], grid[:, i], fill_value='interpolate')(q[:, i])

    is_nan = np.isnan(ppf)
    if is_nan.any():
        ppf[is_nan & (q >= 1)] = np.inf  # infinity because marginal is greater or equal to 1
        ppf[is_nan & (q <= 0)] = -np.inf  # infinity because marginal is less than or equal to 0

    return ppf


def gmm_marginal_cdf(x: np.ndarray, param: GMCParam):
    """
    Applies the approximation of the inverse cdf marginally for the input given the GMC parameters

    Notes
    -----
    The approximation is taken from `Abramowitz and Stegun's Handbook of Mathematical
    Functions <http://people.math.sfu.ca/~cbm/aands/toc.htm>`_ formula 7.1.25.

    Parameters
    ----------
    x : np.ndarray
        Vector of quantiles
    param : GMCParam
        The Gaussian Mixture Copula parameters

    Returns
    -------
    np.ndarray
        Cumulative distribution function evaluated at `x`
    """

    sigmas = np.repeat(np.sqrt([np.diag(c) for c in param.covs]).T[np.newaxis, ...], len(x), axis=0)
    means = np.repeat(param.means.T[np.newaxis, ...], len(x), axis=0)

    a1 = 0.3480242
    a2 = -0.0958798
    a3 = 0.7478556
    rho = 0.47047
    sqrt2 = 1.4142136

    zi = (np.repeat(x[..., np.newaxis], param.n_clusters, axis=2) - means) / (sigmas * sqrt2)
    za = np.abs(zi)
    t = 1 / (1 + rho * za)
    erf = 0.5 * (a1 * t + a2 * t ** 2 + a3 * t ** 3) * np.exp(-(za ** 2))
    return np.where(zi < 0, erf, 1 - erf) @ param.prob
