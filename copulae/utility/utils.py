from functools import wraps

import numpy as np

__all__ = ['array_io', 'as_array', 'merge_dict', 'merge_dicts', ]


def array_io(func=None, dim=0, optional=False):
    """
    Decorator that ensures that the first input of function gets converted to a
    numpy array of the dimension specified. The return value will also be converted
    to a scalar if the size is 1
    """
    assert func is None or callable(func), "`func` can either be None or a callable"

    def decorator(f):
        @wraps(f)
        def internal(cls, x=None, *args, **kwargs):
            assert not (x is None and not optional), "Input array is None when it is not optional"

            if dim == 1:
                x = np.asarray(x)
                if x.ndim == 0:
                    x = x.ravel()
                assert x.ndim == 1, 'Dimension of x must be 1'

            elif dim == 2:
                x = np.asarray(x)
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                assert x.ndim == 2 and x.shape[1] == cls.dim, 'Input data shape does not match copula dimension'

            elif x is not None:
                x = np.asarray(x)

            res = np.asarray(f(cls, x, *args, **kwargs))
            return res.item(0) if res.size == 1 else res

        return internal

    return decorator(func) if func else decorator


def as_array(x, dtype=None, copy=False) -> np.ndarray:
    """
    Converts a scalar or array into a numpy array

    Parameters
    ----------
    x: {array_like, scalar}
        Numeric input

    dtype
        Array type

    copy: bool
        If true, returns a copy of the array

    Returns
    -------
    ndarray
        Numpy array
    """
    x = np.asarray(x, dtype)
    x = x.copy() if copy else x
    return x.ravel() if x.ndim == 0 else x


def merge_dict(a: dict, b: dict) -> dict:
    """
    Merge 2 dictionaries.

    If the parent and child shares a similar key and the value of that key is a dictionary, the key will be recursively
    merged. Otherwise, the child value will override the parent value.

    Parameters
    ----------
    a dict:
        Parent dictionary

    b dict:
        Child dictionary

    Returns
    -------
    dict
        Merged dictionary
    """
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                a[key] = merge_dict(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def merge_dicts(*dicts: dict) -> dict:
    """
    Merge multiple dictionaries recursively

    Internally, it calls :code:`merge_dict` recursively.

    Parameters
    ----------
    dicts
        a list of dictionaries

    Returns
    -------
    dict
        Merged dictionary

    See Also
    --------
    :code:`merge_dict`: merge 2 dictionaries
    """
    """
    
    :param dicts: List[Dict]
        a list of dictionaries
    :return: dict
        merged dictionary
    """

    a = dicts[0]
    if len(dicts) == 1:
        return dicts[0]

    for b in dicts:
        a = merge_dict(a, b)
    return a
