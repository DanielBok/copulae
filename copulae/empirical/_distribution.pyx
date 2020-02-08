cimport numpy as cnp
cimport scipy.special.cython_special as csc
from libc.math cimport fmin, fmax

import numpy as np


ctypedef double (*func_type)(double, double, int) nogil


def emp_copula_dist(cnp.ndarray[cnp.npy_float64, ndim=2] X,
                    cnp.ndarray[cnp.npy_float64, ndim=2] Y,
                    double offset,
                    int typ):  # Cn_C
    cdef:
        int i
        double v
        int nrow_X = len(X), nrow_Y = len(Y), ncol = X.shape[1]
        cnp.ndarray[cnp.npy_float64, ndim=1] U = np.ravel(X, order='F'), V = np.ravel(Y, order='F')
        double[::1] res = np.repeat(np.nan, nrow_X)

    for i in range(nrow_X):
        if typ == 0:  # default empirical copula
            res[i] = multivariate_emp_cop_dist_func(U, V, nrow_X, nrow_Y, ncol, i, offset, emp_cop)

        if typ == 1:  # empirical beta copula
            res[i] = multivariate_emp_cop_dist_func(U, V, nrow_X, nrow_Y, ncol, i, offset, emp_beta_cop)

        if typ == 2:  # empirical checkerboard copula
            res[i] = multivariate_emp_cop_dist_func(U, V, nrow_X, nrow_Y, ncol, i, offset, emp_multi_linear_copula)

    return np.asarray(res)


cdef double multivariate_emp_cop_dist_func(double[::1] X,
                                           double[::1] Y,
                                           int nrow_X,
                                           int nrow_Y,
                                           int ncol,
                                           int k,
                                           double offset,
                                           func_type f) nogil:  # Cn_f
    cdef:
        double sum_prod = 0.0, prod
        int i, j

    for i in range(nrow_Y):
        prod = 1.0
        for j in range(ncol):
            prod *= f(Y[i + nrow_Y * j], X[k + nrow_X * j], nrow_Y)
        sum_prod += prod

    return sum_prod / (nrow_Y + offset)


cdef inline double emp_cop(double u, double v, int n) nogil:
    return 1.0 if u <= v else 0.0

cdef inline double emp_beta_cop(double u, double v, int n) nogil:
    return csc.btdtr(u * (n + 1), (n + 1) * (1.0 - u), v)


cdef inline double emp_multi_linear_copula(double u, double v, int n) nogil:
    return fmin(fmax(n * v - (n + 1) * u + 1, 0), 1.0)
