from functools import wraps

import numpy as np


def _array_io(f):
    @wraps(f)
    def decorator(x):
        res = np.asarray(f(np.asarray(x)))
        return res.item(0) if res.size == 1 else res

    return decorator


@_array_io
def cospi(x):
    """
    Cosine function where every element is multiplied by pi.

    .. math::

        cospi(x) = cos(x * pi)

    Parameters
    ----------
    x: {array_like, scalar}
        Input array in degrees

    Returns
    -------
    ndarray
        The corresponding cosine values. This is a scalar if x is a scalar.
    """
    return np.cos(x * np.pi)


@_array_io
def cospi2(x):
    """
    Cosine function where every element is multiplied by pi / 2.

    .. math::

        cospi(x) = cos(x * pi / 2)

    Parameters
    ----------
    x: {array_like, scalar}
        Input array in degrees

    Returns
    -------
    ndarray
        The corresponding cosine values. This is a scalar if x is a scalar.
    """
    return np.cos(x * np.pi / 2)


@_array_io
def sinpi(x):
    """
    Sine function where every element is multiplied by pi.

    .. math::

        sinpi(x) = sin(x * pi)

    Parameters
    ----------
    x: {array_like, scalar}
        Input array in degrees

    Returns
    -------
    ndarray
        The corresponding sine values. This is a scalar if x is a scalar.
    """
    return np.sin(x * np.pi)


@_array_io
def sinpi2(x):
    """
    Sine function where every element is multiplied by pi / 2.

    .. math::

        sinpi(x) = sin(x * pi / 2)

    Parameters
    ----------
    x: {array_like, scalar}
        Input array in degrees

    Returns
    -------
    ndarray
        The corresponding sine values. This is a scalar if x is a scalar.
    """
    return np.sin(x * np.pi / 2)


@_array_io
def tanpi(x):
    """
    Tangent function where every element is multiplied by pi.

    .. math::

        tanpi(x) = tan(x * pi)

    Parameters
    ----------
    x: {array_like, scalar}
        Input array in degrees

    Returns
    -------
    ndarray
        The corresponding tangent values. This is a scalar if x is a scalar.

    """
    return np.tan(x * np.pi)


@_array_io
def tanpi2(x):
    """
    Tangent function where every element is multiplied by pi / 2.

    .. math::

        tanpi(x) = tan(x * pi / 2)

    Parameters
    ----------
    x: {array_like, scalar}
        Input array in degrees

    Returns
    -------
    {array_like, scalar}
        The corresponding Tangent values. This is a scalar if x is a scalar.
    """
    return np.tan(x * np.pi / 2)
