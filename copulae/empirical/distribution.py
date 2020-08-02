from typing import Optional

import numpy as np

from ._distribution import emp_copula_dist

__all__ = ["emp_dist_func"]


def emp_dist_func(x: np.ndarray, y: np.ndarray, smoothing: Optional[str] = "none", offset: float = 0.0):
    """
    Empirical (cumulative) distribution function

    Parameters
    ----------
    x
        Matrix of evaluation points. These are the points that are "new" and need to be evaluated against an
        empirical distribution formed by :code:`y`

    y
        Matrix of data points forming the empirical distribution

    smoothing
        If not specified (default), the empirical distribution function or copula is computed. If "beta", the
        empirical beta copula is computed. If "checkerboard", the empirical checkerboard copula is computed.

    offset
        Used in scaling the result for the density and distribution functions. Defaults to 0.

    Returns
    -------
    ndarray
        Computes the CDF of the empirical copula
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    smoothing = _map_smoothing(smoothing)
    offset = _validate_offset(offset)

    assert np.ndim(x) == 2 and np.ndim(y) == 2, "input data must be matrices"
    assert x.shape[1] == y.shape[1], "input data must have the same dimensions"

    return emp_copula_dist(x, y, offset, smoothing)


def _map_smoothing(smoothing: Optional[str]):
    if smoothing is None:
        smoothing = "none"

    res = {
        "none": 0,
        "beta": 1,
        "checkerboard": 2,
    }.get(smoothing.lower(), None)
    assert res is not None, "smoothing must be 'beta', 'checkerboard' or None"

    return res


def _validate_offset(offset: float):
    assert isinstance(offset, (int, float)), "offset must be numeric"
    return float(offset)
