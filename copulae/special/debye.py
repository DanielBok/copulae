import numba as nb
import numpy as np

from copulae.special import _machine as M
from copulae.special._cheb import cheb_eval

__all__ = ['debye_n', 'debye_1', 'debye_2', 'debye_3', 'debye_4', 'debye_5', 'debye_6']


def debye_n(x, order=1):
    r"""
    Computes the nth order Debye function

    .. math::

        D_n(x) = n/x^n \int^x_0 (t^n/(e^t - 1)) dt

    where :math:`n` represents the order

    Parameters
    ----------
    x: array_like
        Real values

    order: {1, 2, 3, 4, 5, 6}
        Order of the Debye function. Limited to 6.

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    assert 1 <= order <= 6, "Debye order must be between [1, 6]"
    if order == 1:
        return debye_1(x)
    elif order == 2:
        return debye_2(x)
    elif order == 3:
        return debye_3(x)
    elif order == 4:
        return debye_4(x)
    elif order == 5:
        return debye_5(x)
    elif order == 6:
        return debye_6(x)


@nb.vectorize([nb.float64(nb.float64)], cache=True)
def debye_1(x):  # pragma: no cover
    r"""
    Computes the first-order Debye function

    .. math::

        D_1(x) = \frac{1}{x} \int^x_0 \frac{t}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    constants = [
        2.4006597190381410194,
        0.1937213042189360089,
        -0.62329124554895770e-02,
        0.3511174770206480e-03,
        -0.228222466701231e-04,
        0.15805467875030e-05,
        -0.1135378197072e-06,
        0.83583361188e-08,
        -0.6264424787e-09,
        0.476033489e-10,
        -0.36574154e-11,
        0.2835431e-12,
        -0.221473e-13,
        0.17409e-14,
        -0.1376e-15,
        0.109e-16,
        -0.9e-18
    ]

    val_infinity = 1.64493406684822644
    xcut = -M.LOG_DBL_MIN

    if x < 0:
        return np.nan
    elif x < 2 * M.SQRT_DBL_EPSILON:
        return 1 - 0.25 * x + x ** 2 / 36
    elif x <= 4:
        return cheb_eval(constants, x * x / 8 - 1) - 0.25 * x
    elif x <= -(M.M_LN2 + M.LOG_DBL_EPSILON):
        nexp = int(np.floor(xcut / x))
        ex = np.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            total *= ex
            total += (1 + 1 / (x * i)) / i

        return val_infinity / x - total * ex
    elif x < xcut:
        return (val_infinity - np.exp(-x) * (x + 1)) / x
    else:
        return val_infinity / x


@nb.vectorize([nb.float64(nb.float64)], cache=True)
def debye_2(x):  # pragma: no cover
    r"""
    Computes the second-order Debye function

    .. math::
        D_2(x) = \frac{2}{x^2} \int^x_0 \frac{t^2}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    constants = [
        2.5943810232570770282,
        0.2863357204530719834,
        -0.102062656158046713e-01,
        0.6049109775346844e-03,
        -0.405257658950210e-04,
        0.28633826328811e-05,
        -0.2086394303065e-06,
        0.155237875826e-07,
        -0.11731280087e-08,
        0.897358589e-10,
        -0.69317614e-11,
        0.5398057e-12,
        -0.423241e-13,
        0.33378e-14,
        -0.2645e-15,
        0.211e-16,
        -0.17e-17,
        0.1e-18
    ]

    val_infinity = 4.80822761263837714
    xcut = -M.LOG_DBL_MIN

    if x < 0:
        return np.nan
    elif x < 2 * M.M_SQRT2 * M.SQRT_DBL_EPSILON:
        return 1 - x / 3 + x ** 2 / 24
    elif x <= 4:
        return cheb_eval(constants, x * x / 8 - 1) - x / 3
    elif x < - (M.M_LN2 - M.LOG_DBL_EPSILON):
        nexp = int(np.floor(xcut / x))
        ex = np.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            total *= ex
            xi = x * i
            total += (1 + 2 / xi + 2 / (xi ** 2)) / i

        return val_infinity / x ** 2 - 2 * total * ex
    elif x < xcut:
        x2 = x ** 2
        total = 2 + 2 * x + x2
        return (val_infinity - 2 * total * np.exp(-x)) / x2
    else:
        return val_infinity / x ** 2


@nb.vectorize([nb.float64(nb.float64)], cache=True)
def debye_3(x):  # pragma: no cover
    r"""
    Computes the third-order Debye function

    .. math::

        D_3(x) = \frac{3}{x^3} \int^x_0 \frac{t^3}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    constants = [
        2.707737068327440945,
        0.340068135211091751,
        -0.12945150184440869e-01,
        0.7963755380173816e-03,
        -0.546360009590824e-04,
        0.39243019598805e-05,
        -0.2894032823539e-06,
        0.217317613962e-07,
        -0.16542099950e-08,
        0.1272796189e-09,
        -0.987963460e-11,
        0.7725074e-12,
        -0.607797e-13,
        0.48076e-14,
        -0.3820e-15,
        0.305e-16,
        -0.24e-17
    ]

    val_infinity = 19.4818182068004875
    xcut = -M.LOG_DBL_MIN

    if x < 0:
        return np.nan
    elif x < 2.0 * M.M_SQRT2 * M.SQRT_DBL_EPSILON:
        return 1 - 3 * x / 8 + x ** 2 / 20
    elif x <= 4:
        return cheb_eval(constants, x * x / 8 - 1) - 0.375 * x
    elif x < - (M.M_LN2 - M.LOG_DBL_EPSILON):
        nexp = int(np.floor(xcut / x))
        ex = np.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += (((6 * xinv + 6) * xinv + 3) * xinv + 1) / i

        return val_infinity / x ** 3 - 3 * total * ex
    elif x < xcut:
        total = 6 + 6 * x + 3 * x ** 2 + x ** 3
        return (val_infinity - 3 * total * np.exp(-x)) / x ** 3
    else:
        return val_infinity / x ** 3


@nb.vectorize([nb.float64(nb.float64)], cache=True)
def debye_4(x):  # pragma: no cover
    r"""
    Computes the fourth-order Debye function

    .. math::

        D_4(x) = \frac{4}{x^4} \int^x_0 \frac{t^4}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    constants = [
        2.781869415020523460,
        0.374976783526892863,
        -0.14940907399031583e-01,
        0.945679811437042e-03,
        -0.66132916138933e-04,
        0.4815632982144e-05,
        -0.3588083958759e-06,
        0.271601187416e-07,
        -0.20807099122e-08,
        0.1609383869e-09,
        -0.125470979e-10,
        0.9847265e-12,
        -0.777237e-13,
        0.61648e-14,
        -0.4911e-15,
        0.393e-16,
        -0.32e-17
    ]

    val_infinity = 99.5450644937635129
    xcut = -M.LOG_DBL_MIN

    if x < 0:
        return np.nan
    elif x < 2.0 * M.M_SQRT2 * M.SQRT_DBL_EPSILON:
        return 1 - 2 * x / 5 + x ** 2 / 18
    elif x <= 4:
        return cheb_eval(constants, x * x / 8 - 1) - 0.4 * x
    elif x < - (M.M_LN2 - M.LOG_DBL_EPSILON):
        nexp = int(np.floor(xcut / x))
        ex = np.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += ((((24 * xinv + 24) * xinv + 12) * xinv + 4) * xinv + 1) / i

        return val_infinity / x ** 4 - 4 * total * ex
    elif x < xcut:
        total = 24 + 24 * x + 12 * x ** 2 + 4 * x ** 3 + x ** 4
        return (val_infinity - 4 * total * np.exp(-x)) / x ** 4
    else:
        return val_infinity / x ** 4


@nb.vectorize([nb.float64(nb.float64)], cache=True)
def debye_5(x):  # pragma: no cover
    r"""
    Computes the fifth-order Debye function

    .. math::

        D_5(x) = \frac{5}{x^5} \int^x_0 \frac{t^5}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    constants = [
        2.8340269546834530149,
        0.3994098857106266445,
        -0.164566764773099646e-1,
        0.10652138340664541e-2,
        -0.756730374875418e-4,
        0.55745985240273e-5,
        -0.4190692330918e-6,
        0.319456143678e-7,
        -0.24613318171e-8,
        0.1912801633e-9,
        -0.149720049e-10,
        0.11790312e-11,
        -0.933329e-13,
        0.74218e-14,
        -0.5925e-15,
        0.475e-16,
        -0.39e-17
    ]

    val_infinity = 610.405837190669483828710757875
    xcut = -M.LOG_DBL_MIN

    if x < 0:
        return np.nan
    elif x < 2.0 * M.M_SQRT2 * M.SQRT_DBL_EPSILON:
        return 1 - 5 * x / 12 + 5 * x ** 2 / 84
    elif x <= 4:
        return cheb_eval(constants, x * x / 8 - 1) - 5 * x / 12
    elif x < - (M.M_LN2 - M.LOG_DBL_EPSILON):
        nexp = int(np.floor(xcut / x))
        ex = np.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += (((((120 * xinv + 120) * xinv + 60) * xinv + 20) * xinv + 5) * xinv + 1) / i
        return val_infinity / x ** 5 - 5 * total * ex
    elif x < xcut:
        total = 120 + 120 * x + 60 * x ** 2 + 20 * x ** 3 + 5 * x ** 4 + x ** 5
        return (val_infinity - 5 * total * np.exp(-x)) / x ** 5
    else:
        return val_infinity / x ** 5


@nb.vectorize([nb.float64(nb.float64)], cache=True)
def debye_6(x):  # pragma: no cover
    r"""
    Computes the sixth-order Debye function

    .. math::

        D_6(x) = \frac{6}{x^6} \int^x_0 \frac{t^6}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    constants = [
        2.8726727134130122113,
        0.4174375352339027746,
        -0.176453849354067873e-1,
        0.11629852733494556e-2,
        -0.837118027357117e-4,
        0.62283611596189e-5,
        -0.4718644465636e-6,
        0.361950397806e-7,
        -0.28030368010e-8,
        0.2187681983e-9,
        -0.171857387e-10,
        0.13575809e-11,
        -0.1077580e-12,
        0.85893e-14,
        -0.6872e-15,
        0.552e-16,
        -0.44e-17
    ]

    val_infinity = 4356.06887828990661194792541535
    xcut = -M.LOG_DBL_MIN

    if x < 0:
        return np.nan
    elif x < 2.0 * M.M_SQRT2 * M.SQRT_DBL_EPSILON:
        return 1 - 3 * x / 7 + x ** 2 / 16
    elif x <= 4:
        return cheb_eval(constants, x * x / 8 - 1) - 3 * x / 7
    elif x < - (M.M_LN2 - M.LOG_DBL_EPSILON):
        nexp = int(np.floor(xcut / x))
        ex = np.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += ((((((720 * xinv + 720) * xinv + 360) * xinv + 120) * xinv + 30) * xinv + 6) * xinv + 1) / i
        return val_infinity / x ** 6 - 6 * total * ex
    elif x < xcut:
        total = 720 + 720 * x + 360 * x ** 2 + 120 * x ** 3 + 30 * x ** 4 + 6 * x ** 5 + x ** 6
        return (val_infinity - 6 * total * np.exp(-x)) / x ** 6
    else:
        return val_infinity / x ** 6
