import numpy as np
from copulae.matrices import tri_indices


def create_cov_matrix(params: np.ndarray):
    """
    Creates a matrix from a given vector of parameters.

    Useful for elliptical copulas where we translate the rhos to the covariance matrix

    :param params: numpy array, 1d
        Vector of parameters
    :return: numpy array, 2d
        Square matrix where the upper and lower triangles are the parameters and the diagonal is a vector of 1
    """

    c = len(params)
    d = int(1 + (1 + 4 * 2 * c) ** 0.5) // 2

    sigma = np.identity(d)
    sigma[tri_indices(d, 1)] = np.tile(params, 2)
    return sigma
