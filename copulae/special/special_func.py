from typing import Iterable, Union

import numpy as np
from scipy.stats import logistic as logis

from copulae.special.combinatorics import comb
from copulae.special.dilog import dilog, dilog_complex

__all__ = ['eulerian', 'eulerian_all', 'log1mexp', 'log1pexp', 'poly_log', 'polyn_eval', 'sign_ff',
           'stirling_first', 'stirling_first_all', 'stirling_second', 'stirling_second_all']


def eulerian(n, k):
    r"""
    Computes the Eulerian numbers.

    The Eulerian number :math:`\binom{n}{k}` gives the number of permutations of
    {1,2,...,n} having :math:`k` permutation ascents. A description of a permutation
    ascent is as given. Let :math:`p = {a_1, a_2, \dots, a_n}` be a permutation.
    Then :math:`i` is a permutation ascent if :math:`a_i < a_{i+1}`. For example,
    the permutation {1,2,3,4} is composed of three ascents, namely {1,2}, {2,3}, and {3,4}.

    Parameters
    ----------
    n: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    k: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    Returns
    -------
    int
        Eulerian number
    """
    try:
        k, n = int(k), int(n)
    except (ValueError, TypeError):
        raise TypeError("`k` and `n` must both be integers")

    assert 0 <= k <= n, "`k` must be in the range of [0, `n`]"

    if k == n:
        return 0
    if k == 0:
        return 1
    if k >= (n + 1) // 2:
        k = n - k - 1
    k1 = k + 1
    sig = np.ones(k1)
    sig[1::2] *= -1

    return int(round(sum(sig * comb(n + 1, np.arange(k + 1)) * np.arange(k1, 0, -1, np.float64) ** n)))


def eulerian_all(n):
    """
    Computes the full vector of Eulerian numbers

    Parameters
    ----------
    n: int
        Positive integer

    Returns
    -------
    list of int
        All Eulerian numbers at :math:`n`
    """
    assert n >= 0 and isinstance(n, int), "`n` must be an integer >= 0"
    if n == 0:
        return 1

    return [eulerian(n, i) for i in range(n)]


def log1mexp(x):
    r"""
    Calculates log(1 - exp(-x))

    Parameters
    ----------
    x: array_like
        data

    Returns
    -------
    array_like
        Results of :math:`\log(1 - e^{-x})`
    """
    return np.log(1 - np.exp(-x))


def log1pexp(x):
    r"""
    Calculates log(1 + exp(x))

    Parameters
    ----------
    x: array_like
        data

    Returns
    -------
    array_like
        Results of :math:`\log(1 - e^{-x})`
    """
    return np.log(1 + np.exp(x))


def poly_log(z, s, method='default', log=False) -> Union[float, np.ndarray]:
    r"""
    Computes the polylogarithm function. Current implementation only takes care of :math:`s \leq 0`

    .. math::

        L_s(z) = \sum_{k=1}^\infty \frac{z^k}{k^s}

    Parameters
    ----------
    z: {array_like, scalar}
        Numeric or complex vector

    s: {array_like, scalar}
        Complex number

    method: {'default', 'neg-stirling', 'neg-eulerian'}
        Algorithm used for calculating poly logarithms

    log: bool
        If True, returns the log of poly logarithms

    Returns
    -------
    {array_like, scalar}
        Poly logarithms
    """
    if isinstance(z, (complex, int, float)):
        z = np.ravel(z)

    method = method.lower()
    assert method in ('default', 'neg-stirling', 'neg-eulerian')
    if s == 2 and method == 'default':
        return dilog_complex(z) if np.any(np.iscomplex(z)) else dilog(z)

    elif method == 'default':
        method = 'neg-stirling'

    assert float(s).is_integer() and s <= 1

    if s == 1:
        r = -np.log1p(-z)
        return np.log(r) if log else r

    iz = 1 - z
    n = abs(int(s))

    if method == 'neg-stirling':
        r = z / iz
        f = np.cumprod([1, *range(1, n + 1)])
        s = stirling_second_all(n + 1)
        p = polyn_eval(f * s, r)
        if log:
            res = np.log(p) + logis.ppf(z)
        else:
            res = r * p

    else:
        #  method == 'neg-eulerian'
        p = polyn_eval(eulerian_all(n), z)
        if log:
            res = np.log(p) + np.log(z) - (n + 1) * np.log1p(-z)
        else:
            res = z * p / iz ** (n + 1)

    return res.item(0) if res.size == 1 else res


def polyn_eval(coef, x) -> Union[float, np.ndarray]:
    r"""
    Polynomial evaluation via Horner scheme

    Evaluate a univariate polynomial at `x` (typically a vector). For a given vector of coefficients <coef>,
    the polynomial :math:`c_1 + c_2x + \dots + c_{p+1}x^p`

    Parameters
    ----------
    coef: array_like
        Coefficients

    x: array_like
        Evaluation points

    Returns
    -------
    {float, ndarray}
        numeric vector or scalar, with the same dimensions as x, containing the polynomial values
    """
    if isinstance(coef, (float, int)):
        coef = [coef]
    if isinstance(x, (float, int)):
        x = [x]

    m = len(coef)

    res = np.zeros(len(x), np.float64)
    for i, xi in enumerate(x):
        if m == 1:
            r = coef[0]
        else:
            r = coef[-1]
            for j in coef[:-1][::-1]:
                r = j + r * xi

        res[i] = r

    return res.item(0) if res.size == 1 else res


def sign_ff(alpha: float, j: Union[Iterable[int], int], d: Union[Iterable[int], int]):
    r"""
    The sign of :math:`\binom{\alpha * j}{d} \cdot (-1)^{d-j}`

    Parameters
    ----------
    alpha: float
        Parameter in (0, 1]

    j: int
        integer in [0, d]

    d: int
        integer >= 0

    Returns
    -------
    ndarray
        signs
    """
    assert 0 < alpha <= 1, "`alpha` must be between (0, 1]"

    d, j = np.asarray(d, int), np.asarray(j, int)

    assert np.all(d >= 0), "`d` must be >= 0"
    assert np.all(j >= 0), "all elements in `j` must be >= 0"

    if d.ndim == 0 and j.ndim == 0:
        d, j = int(d), int(j)
    elif d.ndim == 0 and j.ndim > 0:
        d = np.repeat(d, len(j))
    elif j.ndim == 0 and d.ndim > 0:
        j = np.repeat(j, len(d))
    else:
        if len(d) > len(j):
            j = np.asarray([j[i % len(j)] for i in range(len(d))])
        elif len(j) > len(d):
            d = np.asarray([d[i % len(d)] for i in range(len(j))])

    if alpha == 1:
        res = (j > d) * (-1) ** (d - j) + (j == d)
    else:
        res = np.where(j > d, np.nan, 0)
        x = alpha * j
        ind = x != np.floor(x)
        res[ind] = (-1) ** (j[ind] - np.ceil(x[ind]))

    return res.item(0) if res.size == 1 else res


def stirling_first(n: int, k: int):
    """
    Computes Stirling number of the first kind

    Parameters
    ----------
    n: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    k: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    Returns
    -------
    int
        Stirling number of the first kind
    """
    try:
        k, n = int(k), int(n)
    except (ValueError, TypeError):
        raise TypeError("`k` and `n` must both be integers")

    assert 0 <= k <= n, "`k` must be in the range of [0, `n`]"

    if n == 0 or n == k:
        return 1
    if k == 0:
        return 0

    s = [1, *[0] * (k - 1)]
    for i in range(1, n):
        last_row = [*s]
        s[0] = - i * last_row[0]
        for j in range(1, k):
            s[j] = last_row[j - 1] - i * last_row[j]

    return s[-1]


def stirling_first_all(n: int):
    """
    Computes all the Stirling number of the first kind for a given `n`

    Parameters
    ----------
    n: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    Returns
    -------
    list of int
        A vector of all the Stirling number of the first kind

    """
    return [stirling_first(n, k + 1) for k in range(n)]


def stirling_second(n: int, k: int):
    """
    Computes Stirling number of the second kind

    Parameters
    ----------
    n: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    k: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    Returns
    -------
    int
        Stirling number of the second kind
    """
    try:
        k, n = int(k), int(n)
    except (ValueError, TypeError):
        raise TypeError("`k` and `n` must both be integers")

    assert 0 <= k <= n, "`k` must be in the range of [0, `n`]"

    if n == 0 or n == k:
        return 1
    if k == 0:
        return 0

    s = [1, *[0] * (k - 1)]
    for _ in range(1, n):
        last_row = [*s]
        for i in range(1, k):
            s[i] = (i + 1) * last_row[i] + last_row[i - 1]

    return s[-1]


def stirling_second_all(n: int):
    """
    Computes all the Stirling number of the second kind for a given `n`

    Parameters
    ----------
    n: int
        Numeric. If non-integer passed in, will attempt to cast as integer

    Returns
    -------
    list of int
        A vector of all the Stirling number of the second kind
    """
    return [stirling_second(n, k + 1) for k in range(n)]
