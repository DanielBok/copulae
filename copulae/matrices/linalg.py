import numpy as np
import numpy.linalg as la
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.stats.moment_helpers import corr2cov, cov2corr

__all__ = ["corr2cov", "cov2corr", "is_psd", "is_symmetric", "near_psd"]


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
