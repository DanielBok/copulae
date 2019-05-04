from ._specfunc import debye_1 as _d1, debye_2 as _d2

__all__ = ['debye_1', 'debye_2']


def debye_1(x, threaded=True):
    r"""
    Computes the first-order Debye function

    .. math::

        D_1(x) = \frac{1}{x} \int^x_0 \frac{t}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    threaded: bool
        If true, use parallelism for calculations

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return _d1(x, threaded)


def debye_2(x, threaded=True):
    r"""
    Computes the second-order Debye function

    .. math::
        D_2(x) = \frac{2}{x^2} \int^x_0 \frac{t^2}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    threaded: bool
        If true, use parallelism for calculations

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return _d2(x, threaded)
