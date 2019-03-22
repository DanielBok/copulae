from typing import Iterable, Union

import numpy as np

__all__ = ['polyn_eval', 'sign_ff', 'stirling_first', 'stirling_first_all', 'stirling_second', 'stirling_second_all']


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

    res = np.zeros(len(x))
    for i, xi in enumerate(x):
        if m == 1:
            r = coef[0]
        else:
            r = coef[-1]
            for j in coef[:-1][::-1]:
                r = j + r * xi

        res[i] = r

    return float(res) if res.size == 1 else res


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
    if not (0 < alpha <= 1):
        raise ValueError("<alpha> must be between (0, 1]")

    d, j = np.asarray(d, int), np.asarray(j, int)
    if not np.all(d >= 0):
        raise ValueError("<d> must be >= 0")
    if not np.all(j >= 0):
        raise ValueError("all elements in <j> must be >= 0")

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

    return int(res) if np.size == 1 else res


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
        raise TypeError("<k> and <n> must both be integers")

    if k < 0 or k > n:
        raise ValueError("<k> must be in the range of [0, <n>]")

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
    List[int]
        a list of all the Stirling number of the first kind

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
        raise TypeError("<k> and <n> must both be integers")

    if k < 0 or k > n:
        raise ValueError("<k> must be in the range of [0, <n>]")

    # s = np.zeros((n, k)).astype(np.uint64)
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
    List[int]
        a list of all the Stirling number of the second kind

    """
    return [stirling_second(n, k + 1) for k in range(n)]
