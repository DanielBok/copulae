import numpy as np
import numpy.linalg as la

__all__ = ["is_PSD"]


def is_PSD(M: np.ndarray) -> bool:
    """
    Tests if matrix is positive semi-definite

    :param M: numpy array
        (n x n) matrix
    :return: bool
        True if matrix is positive semi-definite, else False
    """

    M = np.asarray(M)
    if not np.allclose(M, M.T):
        return False

    return (la.eigvalsh(M) >= 0).all()
