import numpy as np
from scipy.special import gammaln, gamma

__all__ = ['comb', 'perm']


def comb(n, r, log=False, nan_to_0=False):
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

    nan_to_0: bool, optional
        If true, changes nan values to 0

    Returns
    -------
    comb: ndarray or scalar
        Number of combinations
    """
    n, r = np.asarray(n), np.asarray(r)

    if log:
        res = gammaln(n + 1) - gammaln(n + 1 - r) - gammaln(r + 1)
    else:
        res = gamma(n + 1) / (gamma(n + 1 - r) * gamma(r + 1))

    if nan_to_0:
        res[np.isnan(res)] = 0
    return res


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
