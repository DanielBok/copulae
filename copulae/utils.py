from functools import wraps
import numpy as np


def merge_dict(a: dict, b: dict) -> dict:
    """
    Merge 2 dictionaries.

    If the parent and child shares a similar key and the value of that key is a dictionary, the key will be recursively
    merged. Otherwise, the child value will override the parent value.

    :param a: dict
        parent dictionary
    :param b: dict
        child dictionary
    :return: dict
        merged dictionary
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


def format_docstring(*args, **kwargs):
    def decorator(func):
        func.__doc__ = func.__doc__.format(*args, **kwargs)
        return func

    return decorator


def reshape_data(func):
    """
    Helper that ensures that inputs of pdf and cdf function gets converted to a 2D array and output if a single
    value gets converted to a scalar
    """

    @wraps(func)
    def decorator(cls, x, *args, **kwargs):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError("input array must be a vector or matrix")

        if x.shape[1] != cls.dim:
            raise ValueError('number of columns in input data does not match copula dimension')

        res = np.asarray(func(cls, x, *args, **kwargs))

        if res.ndim == 1 and len(res) == 1:
            res = res[0]

        return res

    return decorator
