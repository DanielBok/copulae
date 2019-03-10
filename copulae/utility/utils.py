from functools import wraps

import numpy as np

__all__ = ['merge_dict', 'merge_dicts', 'reshape_data', 'reshape_output']


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

        if res.size == 1:
            res = float(res)

        return res

    return decorator


def reshape_output(func):
    """
    Helpers function that converts the output to a float if the size of the output is 1
    """

    @wraps(func)
    def decorator(cls, x=None, *args, **kwargs):
        x = np.asarray(x) if x is not None else x
        res = np.asarray(func(cls, x, *args, **kwargs))

        if res.size == 1:
            res = float(res)
        return res

    return decorator
