import numpy as np
from scipy.special import gammaln

from copulae.types import Numeric

__all__ = ['comb', 'perm']


def comb(n: Numeric, r: Numeric, log=False):
    """
    Generalized combination function. Unlike standard combination functions which uses factorials, function can take
    floats as it uses the gamma function.

    :param n: iterable[float, int], float, int
    :param r: iterable[float, int], float, int
    :param log: boolean, default False
        If true, returns the log of the combination function
    :return: iterable[float, int], float
        nCr value
    """
    n, r = np.asarray(n), np.asarray(r)

    ncr = gammaln(n + 1) - gammaln(n + 1 - r) - gammaln(r + 1)
    return ncr if log else np.exp(ncr)


def perm(n: Numeric, r: Numeric, log=False):
    """
    Generalized combination function. Unlike standard combination functions which uses factorials, function can take
    floats as it uses the gamma function.

    :param n: iterable[float, int], float, int
    :param r: iterable[float, int], float, int
    :param log: boolean, default False
        If true, returns the log of the combination function
    :return: iterable[float, int], float
        nCr value
    """
    n, r = np.asarray(n), np.asarray(r)

    npr = gammaln(n + 1) - gammaln(n + 1 - r)
    return npr if log else np.exp(npr)
