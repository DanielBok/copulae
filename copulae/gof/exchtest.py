import numpy as np

from copulae.core import pseudo_obs
from .common import TestStatistic

__all__ = ['exch_test']


def exch_test(x, y, N=1000, m=0, ties='average'):
    """
    Test for assessing the exchangeability of the underlying bivariate copula based on the empirical copula.
    The test statistics are defined in the first two references. Approximate p-values for the test statistics are
    obtained by means of a multiplier technique if there are no ties in the component series of the bivariate
    data, or by means of an appropriate bootstrap otherwise.

    Parameters
    ----------
    x: array_like
        first vector for the exchangeability test

    y: array_like
        second vector for the exchangeability test

    N: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.

    m: int
        If m = 0, integration in the Cramér–von Mises statistic is carried out with respect to the empirical copula.
        If m > 0, integration is carried out with respect to the Lebesgue measure and m specifies the size of the
        integration grid.

    ties: str, optional
        String specifying how ranks should be computed if there are ties in any of the coordinate samples. Options
        include 'average', 'min', 'max', 'dense', 'ordinal'.

    Returns
    -------

    """
    x = pseudo_obs(x, ties)
    y = pseudo_obs(y, ties)
    u = np.vstack([x, y]).T

    assert isinstance(m, int) and m >= 0, "size of the integration grid must be an integer >= 0"
    assert x.ndim == 1 and y.ndim == 1, "x and y must be vectors. Exchangeability tests is bivariate"
    assert isinstance(N, int) and N >= 1, "number of replications for exchangeability test must be a positive integer"

    n = len(u)
    if m > 0:
        xis = np.linspace(1 / m, 1 - 1 / m, m)
        g = np.stack([np.tile(xis, m), np.repeat(xis, m)]).T
        ng = m * m
    else:
        g = u
        ng = n

    s = _exch_test_stat(u, g, n, ng)

    if False:
        pass
    else:
        s0 = _exch_test_cn(u, g, n, ng, N)

    return TestStatistic(
        s,
        (np.sum(s0 >= s) + 0.5) / (N + 1),
        "Test of exchangeability for bivariate copulas"
    )


def _cn(u, v, n):
    """
    Utility function for the exchangeability test based on Cn

    Parameters
    ----------
    u: ndarray
        a matrix

    v: ndarray or list of floats
        a vector with 2 elements

    n: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.
    """
    res = np.sum((u[:, 0] <= v[0]) * (u[:, 1] <= v[1]))
    return res / n


def _der_cn_1(u, v, n, x, y):
    """
    Utility function for the exchangeability test based on Cn

    Parameters
    ----------
    u: ndarray
        first vector for the exchangeability test

    v: ndarray
        second vector for the exchangeability test

    n: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.

    x: float
        exchangeability bounds

    y: float
        exchangeability bounds

    Returns
    -------
    float
    """
    inv_sqrt_n = 1 / np.sqrt(n)
    if x < inv_sqrt_n:
        x = inv_sqrt_n
    elif x > 1 - inv_sqrt_n:
        x = 1 - inv_sqrt_n

    u = np.vstack([u, v]).T
    return (_cn(u, [x + inv_sqrt_n, y], n) - _cn(u, [x - inv_sqrt_n, y], n)) / (2 * inv_sqrt_n)


def _der_cn_2(u, v, n, x, y):
    """
    Utility function for the exchangeability test based on Cn

    Parameters
    ----------
    u: ndarray
        first vector for the exchangeability test

    v: ndarray
        second vector for the exchangeability test

    n: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.

    x: float
        exchangeability bounds

    y: float
        exchangeability bounds

    Returns
    -------
    float
    """
    inv_sqrt_n = 1.0 / np.sqrt(n)

    if y < inv_sqrt_n:
        y = inv_sqrt_n
    elif y > 1 - inv_sqrt_n:
        y = 1 - inv_sqrt_n

    u = np.vstack([u, v]).T
    return (_cn(u, [x, y + inv_sqrt_n], n) - _cn(u, [x, y - inv_sqrt_n], n)) / (2 * inv_sqrt_n)


def _exch_test_cn(u, g, n, m, N):
    """
    Exchangeability test based on the empirical copula

    Parameters
    ----------
    u: ndarray
        Data to test for exchangeability

    g: ndarray
        Integrand grid

    n: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.
    """
    influ = np.zeros(m * n)

    for i in range(m):
        x, y = g[i, 0], g[i, 1]
        d1xy = _der_cn_1(u[:, 0], u[:, 1], n, x, y)
        d2xy = _der_cn_2(u[:, 0], u[:, 1], n, x, y)
        d1yx = _der_cn_1(u[:, 0], u[:, 1], n, y, x)
        d2yx = _der_cn_2(u[:, 0], u[:, 1], n, y, x)

        influ[i * n: (1 + i) * n] = (u[:, 0] <= x) * (u[:, 1] <= y) \
                                    - d1xy * (u[:, 0] <= x) - d2xy * (u[:, 1] <= y) \
                                    - (u[:, 0] <= y) * (u[:, 1] <= x) \
                                    + d1yx * (u[:, 0] <= y) + d2yx * (u[:, 1] <= x)
    influ /= np.sqrt(n)

    s0 = np.zeros(N)
    np.random.seed(8888)
    for i in range(N):
        random = np.random.normal(size=n)
        mean = random.mean()
        for j in range(m):
            process = (random - mean) * influ[j * n: (j + 1) * n]
            s0[i] += np.sum(process * process)
    return s0 / m


def _exch_test_stat(u, g, n, m):
    """
    Statistic for the exchangeability test based on the empirical copula

    Parameters
    ----------
    u: ndarray
        Data to test for exchangeability

    g: ndarray
        Integrand grid

    n: int
        number of rows in u

    m: int
        number of rows in g

    Returns
    -------
    float:
        Value of the test statistic for exchangeability test
    """

    s = 0.0

    for i in range(m):
        diff = _cn(u, g[i], n) - _cn(u, g[i, ::-1], n)
        s += diff * diff

    return s * n / m
