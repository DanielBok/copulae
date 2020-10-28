from functools import wraps

import numpy as np

from copulae.copula import BaseCopula
from copulae.types import Array

__all__ = ['array_io']


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
