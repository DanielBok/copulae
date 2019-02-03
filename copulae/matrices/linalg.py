from typing import Tuple, Union

import numpy as np
import numpy.linalg as la
from statsmodels.stats.correlation_tools import cov_nearest

__all__ = ["corr2cov", "cov2corr", "is_psd", "is_symmetric", "near_psd"]


def corr2cov(corr, std) -> np.ndarray:
    """
    Convert correlation matrix to covariance matrix given standard deviation

    This function does not convert subclasses of ndarrays. This requires
    that multiplication is defined elementwise. np.ma.array are allowed, but
    not matrices.

    :param corr: array_like, 2d
        correlation matrix
    :param std: array_like, 1d
        standard deviation
    :return:
        cov: ndarray (subclass)
            covariance matrix
    """

    corr = np.asanyarray(corr)
    std_ = np.asanyarray(std)
    cov = corr * np.outer(std_, std_)
    return cov


def cov2corr(cov, return_std=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert covariance matrix to correlation matrix

    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    :param cov: array_like, 2d
        covariance matrix, see Notes
    :param return_std: bool, default False
        If this is true then the standard deviation is also returned. By default only the correlation matrix is returned.

    :return:
        corr: ndarray (subclass)
            correlation matrix
        std: ndarray [optional]
            standard deviation
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr


def is_psd(M: np.ndarray, tol=1e-8) -> bool:
    """
    Tests if matrix is positive semi-definite

    :param M: ndarray
        (n x n) matrix
    :param tol: float
        numeric tolerance to check for equality
    :return: bool
        True if matrix is positive semi-definite, else False
    """

    M = np.asarray(M)
    if not is_symmetric(M, tol):
        return False

    return (la.eigvalsh(M) >= 0).all()


def is_symmetric(M: np.ndarray, tol=1e-8) -> bool:
    """
    Tests if a matrix is symmetric

    :param M: ndarray
        (n x n) matrix
    :param tol: float
        numeric tolerance to check for equality
    :return: bool
        True if matrix is positive symmetric, else False
    """
    if M.ndim != 2:
        return False

    r, c = M.shape
    if r != c:
        return False

    return np.allclose(M, M.T, atol=tol)


def near_psd(cov, method='clipped', threshold=1e-15, n_fact=100) -> np.ndarray:
    """
    Find the nearest covariance matrix that is positive (semi-) definite

    This converts the covariance matrix to a correlation matrix. Then, finds the nearest correlation matrix that is
    positive semi-definite and converts it back to a covariance matrix using the initial standard deviation.

    The smallest eigenvalue of the intermediate correlation matrix is approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix might be negative, but zero within a
    numerical error, for example in the range of -1e-16.

    Assumes input covariance matrix is symmetric.

    :param cov: ndarray, (k,k)
        initial covariance matrix
    :param method: string
        If "clipped", this function clips the eigenvalues, replacing eigenvalues smaller than the threshold by the
        threshold. The new matrix is normalized, so that the diagonal elements are one. Compared to "nearest", the
        distance between the original correlation matrix and the positive definite correlation matrix is larger.
        However, it is much faster since it only computes eigenvalues once.
        If "nearest", then the function iteratively adjust the correlation matrix by clipping the
        eigenvalues of a difference matrix. The diagonal elements are set to one.
    :param threshold: float
        clipping threshold for smallest eigenvalue
    :param n_fact: int
        factor to determine the maximum number of iterations if method is set to "nearest"
    :return: ndarray
        corrected covariance matrix
    """
    cov = np.asarray(cov)

    if not is_symmetric(cov):
        raise ValueError('covariance matrix must be symmetric')

    if is_psd(cov):
        return cov

    return cov_nearest(cov, method, threshold, n_fact, False)
