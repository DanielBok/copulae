cimport numpy as cnp
import scipy.special.cython_special as csc
from libc.math cimport fmin, fmax
import numpy as np



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
        res[i] = multivariate_emp_cop_dist_func(U, V, nrow_X, nrow_Y, ncol, i, offset, typ)
    return np.asarray(res)


cdef double multivariate_emp_cop_dist_func(double[::1] X,
                                           double[::1] Y,
                                           int nrow_X,
                                           int nrow_Y,
                                           int ncol,
                                           int k,
                                           double offset,
                                           int option):  # Cn_f
    cdef:
        double sum_prod = 0.0, prod
        int i, j

    for i in range(nrow_Y):
        prod = 1.0
        for j in range(ncol):
            u = Y[i + nrow_Y * j]
            v = X[k + nrow_X * j]
            n = nrow_Y

            if option == 0:
                prod *= (1.0 if u <= v else 0.0)
            elif option == 1:
                prod *= csc.betainc(u * (n + 1), (n + 1) * (1.0 - u), v)
            elif option == 2:
                prod *= fmin(fmax(n * v - (n + 1) * u + 1, 0), 1.0)
            else:
                raise ValueError(f"Invalid option: {option}. Expected [0, 1, 2].")
        sum_prod += prod

    return sum_prod / (nrow_Y + offset)
