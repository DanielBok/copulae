from libc.stdint cimport int64_t
from libc.stdlib cimport rand, RAND_MAX

import numpy as np

from copulae.core import pseudo_obs


def rad_sym_test_stat(const double[:] u, const int n, const int p):
    """
    Statistic for the radial symmetry test based on the empirical copula

    Parameters
    ----------
    u: ndarray
        Pseudo-observations

    n: int
        Number of observations

    p: int
        Dimension of data

    Returns
    -------
    float
        Test statistics
    """
    cdef:
        double s = 0, diff
        int i

    for i in range(n):
        diff = diff_cn(u, n, p, i)
        s += diff * diff

    return s


def rad_sym_replicate(double[:, :] u, long[:, :] ir, const int n, const int p, bint has_ties):
    """One instance of bootstrap replication for radial symmetry test"""
    cdef:
        double[:, :] ub = np.copy(u), tub
        int64_t[:] order
        int i, j

    for i in range(n):
        if (<double>rand() / RAND_MAX) < 0.5:
            for j in range(p):
                ub[i, j] = 1 - u[i, j]

    if has_ties:
        for i in range(p):
            order = np.argsort(ub[:, i])

            tub = np.copy(ub)
            for j in range(n):
                ub[j] = tub[<int>order[j]]

            tub = np.copy(ub)
            for j in range(n):
                ub[j, i] = tub[ir[j, i], i]

    ub = pseudo_obs(ub)
    return rad_sym_test_stat(np.ravel(ub, 'F'), n, p)


cdef double diff_cn(const double[:] u, const int n, const int p, const int k) nogil:
    """
    Difference between the multivariate empirical copula and the multivariate survival empirical copula
    for the radial symmetry test

    Parameters
    ----------
    u: ndarray
        Pseudo-observations

    n: int
        Number of observations

    p: int
        Dimension of data

    k: int
        "line" of `u` at which to compute the difference

    Returns
    -------
    float
        The value of the difference at u[k + n * j], j=1...p`
    """
    cdef:
        double sumind = 0.0
        int i, j, ind1, ind2

    for i in range(n):
        ind1 = 1
        ind2 = 1

        for j in range(p):
            ind1 *= (u[i + n * j] <= u[k + n * j])
            ind2 *= (1.0 - u[i + n * j] <= u[k + n * j])

        sumind += <double>ind1 - <double>ind2

    return sumind / n
