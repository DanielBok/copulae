import warnings

import numpy as np


def valid_rows_in_u(U: np.ndarray) -> np.ndarray:
    """
    Checks that the matrix U supplied has elements between 0 and 1 inclusive.

    :param U: ndarray, matrix
        matrix where rows is the number of data points and columns is the dimension
    :return: ndarray
        a boolean vector that indicates which rows are okay
    """

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return (~np.isnan(U) & (0 <= U) & (U <= 1)).all(1)
