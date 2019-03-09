from typing import Tuple, Union

import numpy as np
import numpy.linalg as la
from statsmodels.stats.correlation_tools import cov_nearest

__all__ = ["corr2cov", "cov2corr", "is_psd", "is_symmetric", "near_psd"]


def corr2cov(corr, std) -> np.ndarray:
    """
    Convert correlation matrix to covariance matrix given standard deviation

    Parameters
    ----------
    corr: (N, N) array like
        Correlation matrix

    std: (1, N) array like
        Vector of standard deviation

    Returns
    -------
    ndarray
        (N, N) covariance matrix
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

    Parameters
    ----------
    cov: (N, N) array like
        Covariance matrix

    return_std: bool, optional
         If True then the standard deviation is also returned. By default only the correlation matrix is returned.

    Returns
    -------
    corr: (N, N) ndarray
        Correlation matrix

    std: (1, N) ndarray
        Standard deviation
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr


def is_psd(M: np.ndarray, strict=False, tol=1e-8) -> bool:
    """
    Tests if matrix is positive semi-definite

    Parameters
    ----------
    M: (N, N) ndarray
        Matrix to be tested for positive semi-definiteness

    strict: bool
        If True, tests for posotive definiteness

    tol: float
        Numeric tolerance to check for equality

    Returns
    -------
    bool
        True if matrix is positive (semi-)definite, else False
    """
    if isinstance(M, (int, float)):
        return M >= 0

    M = np.asarray(M)
    if not is_symmetric(M, tol):
        return False

    if strict:
        return (la.eigvalsh(M) > 0).all()
    else:
        return (la.eigvalsh(M) >= 0).all()


def is_symmetric(M: np.ndarray, tol=1e-8) -> bool:
    """
    Tests if matrix is symmetric

    Parameters
    ----------
    M: (N, N) ndarray
        Matrix to be tested for symmetry

    tol: float
        Numeric tolerance to check for equality

    Returns
    -------
    bool
        True if matrix is symmetric
    """
    if M.ndim != 2:
        return False

    r, c = M.shape
    if r != c:
        return False

    return np.allclose(M, M.T, atol=tol)


def near_psd(cov, method='clipped', threshold=1e-15, n_fact=100) -> np.ndarray:
    """
    Finds the nearest covariance matrix that is positive (semi-) definite

    This converts the covariance matrix to a correlation matrix. Then, finds the nearest correlation matrix that is
    positive semi-definite and converts it back to a covariance matrix using the initial standard deviation.

    The smallest eigenvalue of the intermediate correlation matrix is approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix might be negative, but zero within a
    numerical error, for example in the range of -1e-16.

    Input covariance matrix must be symmetric.

    Parameters
    ----------
    cov: (N, N) array like
        Initial covariance matrix

    method: { 'clipped', 'nearest' }, optional
         If "clipped", this function clips the eigenvalues, replacing eigenvalues smaller than the threshold by the
        threshold. The new matrix is normalized, so that the diagonal elements are one. Compared to "nearest", the
        distance between the original correlation matrix and the positive definite correlation matrix is larger.
        However, it is much faster since it only computes eigenvalues once.

        If "nearest", then the function iteratively adjust the correlation matrix by clipping the
        eigenvalues of a difference matrix. The diagonal elements are set to one.

    threshold: float
        Clipping threshold for smallest eigenvalue

    n_fact: int
        Factor to determine the maximum number of iterations if method is set to "nearest"

    Returns
    -------
    ndarray
        positive semi-definite matrix
    """
    cov = np.asarray(cov)

    if not is_symmetric(cov):
        raise ValueError('covariance matrix must be symmetric')

    if is_psd(cov):
        return cov

    return cov_nearest(cov, method, threshold, n_fact, False)
