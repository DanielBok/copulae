import numpy as np

import copulae.special._machine as M
from copulae.utility import as_array


def dilog(x):
    r"""
    Computes the dilogarithm for a real argument. In Lewin’s notation this is  :math:`Li_2(x)`,
    the real part of the dilogarithm of a real :math:`x`. It is defined by the integral
    representation :math:`Li_2(x) = -\Re \int_0^x \frac{\log(1-s)}{s} ds`.

    Note that :math:`\Im(Li_2(x)) = 0 \forall x \leq 1` and :math:`\Im(Li_2(x)) = -\pi\log(x) \forall x > 1`.

    Parameters
    ----------
    x: {array_like, scalar}
        Numeric vector input

    Returns
    -------
    {array_like, scalar}
        Real Dilog output
    """
    x = as_array(x)
    x[x >= 0] = _dilog_xge0(x[x >= 0])

    xm = x[x < 0]
    x1 = _dilog_xge0(-xm)
    x2 = _dilog_xge0(xm * xm)
    x[x < 0] = -x1 + 0.5 * x2

    return x


def _dilog_xge0(x):
    r"""Calculates dilog for real :math:`x \geq 0`"""

    res = np.zeros_like(x)

    # first mask
    m1 = x > 2
    if np.any(m1):
        xx = x[m1]
        t1 = M.M_PI ** 2 / 3
        t2 = _dilog_series_2(1 / xx)
        t3 = 0.5 * np.log(xx) * np.log(xx)
        res[m1] = t1 - t2 - t3

    # second mask
    m2 = (x > 1.01) & (x <= 2)
    if np.any(m2):
        xx = x[m2]
        t1 = M.M_PI ** 2 / 6
        t2 = _dilog_series_2(1 - 1 / xx)
        t3 = np.log(xx) * (np.log(1 - 1 / xx) + 0.5 * np.log(xx))
        res[m2] = t1 + t2 - t3

    # third mask
    m3 = (x > 1) & (x <= 1.01)
    if np.any(m3):
        xx = x[m3]
        e = xx - 1
        lne = np.log(e)
        c0 = M.M_PI ** 2 / 6
        c1 = 1 - lne
        c2 = -(1 - 2 * lne) / 4
        c3 = (1 - 3 * lne) / 9
        c4 = -(1 - 4 * lne) / 16
        c5 = (1 - 5 * lne) / 25
        c6 = -(1 - 6 * lne) / 36
        c7 = (1 - 7 * lne) / 49
        c8 = -(1 - 8 * lne) / 64
        res[m3] = c0 + e * (c1 + e * (c2 + e * (c3 + e * (c4 + e * (c5 + e * (c6 + e * (c7 + e * c8)))))))

    # fourth mask
    m4 = x == 1
    if np.any(m4):
        res[m4] = M.M_PI ** 2 / 6

    # fifth mask
    m5 = (x > 0.5) & (x < 1)
    if np.any(m5):
        xx = x[m5]
        t1 = M.M_PI ** 2 / 6
        t2 = _dilog_series_2(1 - xx)
        t3 = np.log(xx) * np.log(1 - xx)
        res[m5] = t1 - t2 - t3

    # sixth mask
    m6 = (x > 0.25) & (x <= 0.5)
    if np.any(m6):
        res[m6] = _dilog_series_2(x[m6])

    # seventh mask
    m7 = (x > 0) & (x <= 0.25)
    if np.any(m7):
        res[m7] = _dilog_series_1(x[m7])

    return float(res) if res.size == 1 else res


def _dilog_series_1(x: np.ndarray):
    total = x.copy()
    term = x.copy()

    mask = np.ones_like(x)

    for k in range(2, 1000):
        rk2 = ((k - 1) / k) ** 2
        term *= (x * rk2 * mask)
        total += term

        mask *= ((term / total) >= M.DBL_EPSILON)
        if np.alltrue(mask == 0):
            break
    else:
        raise RuntimeError('Max iterations hit for dilog series 1')

    return total


def _dilog_series_2(x: np.ndarray):
    xx = x.copy()
    total = 0.5 * x

    mask = np.ones_like(x)
    for k in range(2, 100):
        xx *= x
        ds = xx / (k * k * (k + 1)) * mask
        total += ds

        mask *= ~((k >= 10) & (np.abs(ds / total) < 0.5 * M.DBL_EPSILON))
        if np.all(np.isclose(mask, 0)):
            break

    t = np.full_like(x, total)

    m1 = x > 0.01
    t[m1] += (1 - x[m1]) * np.log(1 - x[m1]) / x[m1]

    xx = x[~m1]
    tt = 1 / 3 + xx * (0.25 + xx * (0.2 + xx * (1 / 6 + xx * (1 / 7 + xx / 8))))
    t[~m1] += (xx - 1) * (1 + xx * (0.5 + xx * tt))

    return t + 1
