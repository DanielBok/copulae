from libc.math cimport sqrt
from libc.stdint cimport int64_t
from libc.stdlib cimport rand, RAND_MAX

import numpy as np

from copulae.core import pseudo_obs


cdef double cn(double[:] u, double[:] v, double x, double y, int n) nogil:
    """
    Utility function for the exchangeability test based on Cn

    Parameters
    ----------
    u: ndarray
        a vector

    v: ndarray
        a vector
    
    x: float
        a double
    
    y: float
        a double

    n: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.
    """
    cdef:
        double res = 0.0
        int i

    for i in range(n):
        res += (u[i] <= x) * (v[i] <= y)
    return res / n


cdef double der_cn_1(double[:] u, double[:] v, double x, double y, int n) nogil:
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
    cdef:
        double inv_sqrt_n = 1 / sqrt(n)
    if x < inv_sqrt_n:
        x = inv_sqrt_n
    elif x > 1 - inv_sqrt_n:
        x = 1 - inv_sqrt_n

    return (cn(u, v, x + inv_sqrt_n, y, n) - cn(u, v, x - inv_sqrt_n, y, n)) / (2 * inv_sqrt_n)


cdef double der_cn_2(double[:] u, double[:] v, double x, double y, int n) nogil:
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
    inv_sqrt_n = 1.0 / sqrt(n)

    if y < inv_sqrt_n:
        y = inv_sqrt_n
    elif y > 1 - inv_sqrt_n:
        y = 1 - inv_sqrt_n

    return (cn(u, v, x, y + inv_sqrt_n, n) - cn(u, v, x, y - inv_sqrt_n, n)) / (2 * inv_sqrt_n)


def exch_test_cn(double[:, :] u, double[:, :] g, int n, int m, int N, int random_state=8888):
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

    m: int
        size of the integration grid

    N: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.

    random_state: int
        Random state used to standardize calculated p-value
    """
    cdef:
        double[:] influ = np.zeros(m * n), s0 = np.zeros(N), random
        int i, j, k
        double x, y, d1xy, d2xy, d1yx, d2yx, mean, process
        double[:] U = u[:, 0], V = u[:, 1]

    for i in range(m):
        x, y = g[i, 0], g[i, 1]
        d1xy = der_cn_1(U, V, x, y, n)
        d2xy = der_cn_2(U, V, x, y, n)
        d1yx = der_cn_1(U, V, y, x, n)
        d2yx = der_cn_2(U, V, y, x, n)

        for j in range(n):
            influ[i * n + j] = ((U[j] <= x) * (V[j] <= y)
                               - d1xy * (U[j] <= x) - d2xy * (V[j] <= y)
                               - (U[j] <= y) * (V[j] <= x)
                               + d1yx * (U[j] <= y) + d2yx * (V[j] <= x)) / sqrt(n)

    np.random.seed(random_state)

    for i in range(N):
        random = np.random.normal(size=n)
        mean = 0.0
        for j in range(n):
            mean += random[j] / n

        for j in range(m):
            process = 0.0
            for k in range(n):
                process += (random[k] - mean) * influ[k + j * n]
            s0[i] += process * process
        s0[i] /= m

    return np.asarray(s0)


def exch_test_stat(double[:, :] u, double[:, :] g, int n, int m):
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
        size of the integration grid

    Returns
    -------
    float
        Value of the test statistic for exchangeability test
    """
    cdef:
        double s = 0.0, diff, x, y
        int i
        double[:] a = u[:, 0], b = u[:, 1]

    for i in range(m):
        x, y = g[i, 0], g[i, 1]
        diff = cn(a, b, x, y, n) - cn(a, b, y, x, n)
        s += diff * diff

    return s * n / m


def exch_replication(long[:, :] ir, double[:, :] u, double[:, :] g, int n, int m, int ng):
    """
    One instance of the bootstrap replication

    Parameters
    ----------
    ir: ndarray

    u: ndarray
        Data to test for exchangeability

    g: ndarray
        Integrand grid

    n: int
        number of rows in u

    m: int
        size of the integration grid

    ng: int
        number of rows in g

    Returns
    -------
    float
        Value of the test statistic for exchangeability test
    """
    cdef:
        int i, j, s1, s2
        int64_t[:] order
        double[:, :] ub = np.copy(u), tub

    for i in range(n):
        s1, s2 = (0, 1) if (<double>rand() / RAND_MAX) > 0.5 else (1, 0)
        ub[i, 0] = u[i, s1]
        ub[i, 1] = u[i, s2]

    for i in range(2):
        order = np.argsort(ub[:, i])

        tub = np.copy(ub)
        for j in range(n):
            ub[j] = tub[<int>order[j]]

        tub = np.copy(ub)
        for j in range(n):
            ub[j, i] = tub[ir[j, i], i]

    ub = pseudo_obs(ub)

    if m == 0:
        g = ub

    return exch_test_stat(ub, g, n, ng)
