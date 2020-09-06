from functools import wraps

import numpy as np

from copulae.copula import BaseCopula
from copulae.types import Array

__all__ = ['array_io', 'array_io_mcd']


def array_io(func=None, optional=False):
    """
    Decorator that ensures that the first input of function gets converted to a
    numpy array of the dimension specified. The return value will also be converted
    to a scalar if the size is 1
    """
    assert func is None or callable(func), "`func` can either be None or a callable"

    def decorator(f):
        @wraps(f)
        def internal(cls: BaseCopula, x: Array = None, *args, **kwargs):
            if not optional and x is None:
                raise ValueError("Input array cannot be None")

            if x is not None:
                x = np.asarray(x)

            res = np.asarray(f(cls, x, *args, **kwargs))
            if res.size == 1:
                return float(res)

            res = np.asarray(res, np.float_)
            return res.squeeze() if 1 in res.shape else res

        return internal

    return decorator(func) if func else decorator


def array_io_mcd(func):
    """
    Decorator that ensures that the first input of function gets converted to a
    numpy array of the dimension specified. The return value will also be converted
    to a scalar if the size is 1. Unlike :func:`array_io`, this decorator ensures that
    the input data must match the copula's dimension (mcd = match copula dimension)
    and that it will auto-cast the input array to a 2-D array
    """

    @wraps(func)
    def internal(cls: BaseCopula, x: Array = None, *args, **kwargs):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[None, :]

        if x.ndim != 2:
            raise ValueError("Input array must have 1 or 2 dimensions")
        if x.shape[1] != cls.dim:
            raise ValueError("Input array must have same dimension as copula")

        res = np.asarray(func(cls, x, *args, **kwargs))
        if res.size == 1:
            return float(res)

        res = np.asarray(res, np.float_)
        return res.squeeze() if 1 in res.shape else res

    return internal
