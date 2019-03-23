import numpy as np
from scipy.special import gammaln, gamma

__all__ = ['comb', 'perm']


def comb(n, r, log=False):
    """
    Generalized combination function. Unlike standard combination functions which uses factorials, function can take
    floats as it uses the gamma function.

    Parameters
    ----------
    n: {array_like, scalar}
        Numeric scalar or vector

    r: {array_like, scalar}
        Numeric scalar or vector

    log: bool, optional
        If true, returns the log of the combination function

    Returns
    -------
    comb: ndarray or scalar
        Number of combinations
    """
    n, r = np.asarray(n), np.asarray(r)

    if log:
        return gammaln(n + 1) - gammaln(n + 1 - r) - gammaln(r + 1)
    return gamma(n + 1) / (gamma(n + 1 - r) * gamma(r + 1))


def perm(n, r, log=False):
    """
    Generalized permutation function. Unlike standard permutation functions which uses factorials, function can take
    floats as it uses the gamma function.

    Parameters
    ----------
    n: {array_like, scalar}
        Numeric scalar or vector

    r: {array_like, scalar}
        Numeric scalar or vector

    log: bool, optional
        If true, returns the log of the permutation function

    Returns
    -------
    perm: ndarray or scalar
        Number of permutations
    """
    n, r = np.asarray(n), np.asarray(r)

    if log:
        return gammaln(n + 1) - gammaln(n + 1 - r)
    return gamma(n + 1) / gamma(n + 1 - r)
