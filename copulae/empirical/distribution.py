from typing import Optional

import numpy as np

from copulae.core import pseudo_obs
from ._distribution import emp_copula_dist

__all__ = ["emp_dist_func"]


def emp_dist_func(x: np.ndarray, y: np.ndarray, smoothing: Optional[str] = "none", ties="average", offset: float = 0.0):
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

    ties
        The method used to assign ranks to tied elements. The options are 'average', 'min', 'max', 'dense'
        and 'ordinal'.
        'average': The average of the ranks that would have been assigned to all the tied values is assigned to each
            value.
        'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to each
            value. (This is also referred to as "competition" ranking.)
        'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
        'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after those
            assigned to the tied elements. 'ordinal': All values are given a distinct rank, corresponding to
            the order that the values occur in `a`.

    offset
        Used in scaling the result for the density and distribution functions. Defaults to 0.

    Returns
    -------
    ndarray
        Computes the CDF of the empirical copula
    """
    smoothing = _map_smoothing(smoothing)
    x, y, ncol = _validate_inputs(x, y, ties)
    offset = _validate_offset(offset)

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


def _validate_inputs(x: np.ndarray, y: np.ndarray, ties="average"):
    assert np.ndim(x) == 2 and np.ndim(y) == 2, "input data must be matrices"

    x = pseudo_obs(x, ties)
    y = pseudo_obs(y, ties)

    assert x.shape[1] == y.shape[1], "input data must have the same dimensions"
    return x, y, x.shape[1]
