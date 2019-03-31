import numpy as np

from copulae.special import _machine as M
from copulae.special._log import complex_log_e
from copulae.special.clausen import clausen
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
    x = as_array(x, copy=True)
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

    return res.item(0) if res.size == 1 else res


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
    else:  # pragma: no cover
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


# noinspection PyPep8Naming
def dilog_complex(z) -> np.ndarray:
    r"""
    This function computes the full complex-valued dilogarithm for the complex argument
    :math=:`z = r \exp(i \theta)`.

    Parameters
    ----------
    z: {array_like, complex}
        Numeric complex vector input

    Returns
    -------
    {array_like, scalar}
        Complex Dilog output
    """
    z = as_array(z, complex, True)
    Z = np.ravel(z)

    R, Theta = np.abs(Z), np.angle(Z)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    ZETA2 = M.M_PI ** 2 / 6
    R2 = X * X + Y * Y

    log_x = np.log(X)
    atans = np.arctan2(Y, X)

    res = np.zeros_like(Z, complex)
    for i, x, y, r2 in zip(range(len(Z)), X, Y, R2):
        if y == 0:
            c = (-M.M_PI * log_x[i]) if x >= 1.0 else 0.0
            res[i] = complex(dilog(x), c)
        elif abs(r2 - 1) <= M.DBL_EPSILON:
            t = atans[i]
            t1 = t * t / 4
            t2 = M.M_PI * abs(t) / 2
            r = ZETA2 + t1 - t2
            res[i] = complex(r, clausen(t))
        elif r2 < 1:
            res[i] = complex(*_dilogc_unit_disk(x, y))
        else:
            re, im = _dilogc_unit_disk(x / r2, -y / r2)
            t = atans[i]
            re1 = np.log(r2 ** 0.5)
            im1 = (-1 if t < 0 else 1) * (abs(t) - M.M_PI)
            re2 = re1 ** 2 - im1 ** 2
            im2 = 2 * re1 * im1

            res[i] = complex(-re - 0.5 * re2 - ZETA2, -im - 0.5 * im2)

    return res.item(0) if res.size == 1 else res.reshape(z.shape)


def _dilogc_fundamental(r, x, y):
    if r > 0.98:
        return _dilogc_series_3(r, x, y)
    elif r > 0.25:
        return _dilogc_series_2(r, x, y)
    else:
        return _dilogc_series_1(r, x, y)


def _dilogc_unit_disk(x: float, y: float):
    magic_split_value = 0.732

    r = np.hypot(x, y)

    if x > magic_split_value:
        x_tmp = 1.0 - x
        y_tmp = -y
        r_tmp = np.hypot(x_tmp, y_tmp)

        re, im = _dilogc_fundamental(r_tmp, x_tmp, y_tmp)
        a = np.log(r)
        b = np.log(r_tmp)
        c = np.arctan2(y, x)
        d = np.arctan2(y_tmp, x_tmp)

        re = -re + M.M_PI ** 2 / 6 - a * b + c * d
        im = -im - b * c - a * d
        return re, im
    else:
        return _dilogc_fundamental(r, x, y)


def _dilogc_series_1(r, x, y):
    cos_theta = x / r
    sin_theta = y / r

    alpha = 1 - cos_theta
    beta = sin_theta

    ck, sk, rk = cos_theta, sin_theta, r

    re, im = r * ck, r * sk

    for k in range(2, 50 + int(-22 / np.log(r))):
        ck_tmp = ck
        ck = ck - (alpha * ck + beta * sk)
        sk = sk - (alpha * sk - beta * ck_tmp)
        rk *= r
        dr = rk / (k * k) * ck
        di = rk / (k * k) * sk
        re += dr
        im += di
        if abs((dr * dr + di * di) / (re * re + im * im)) < M.DBL_EPSILON ** 2:
            break

    return re, im


def _dilogc_series_2(r, x, y):
    if r == 0:  # pragma: no cover
        return .0, .0
    re, im = _series_2_c(r, x, y)

    om_r, om_i = complex_log_e(1 - x, -y)
    tx = (om_r * x + om_i * y) / (r ** 2)
    ty = (-om_r * y + om_i * x) / (r ** 2)
    rx = (1.0 - x) * tx + y * ty
    ry = (1.0 - x) * ty - y * tx

    return re + rx + 1, im + ry


def _dilogc_series_3(r, x, y):
    theta = np.arctan2(y, x)
    cos_theta = x / r
    sin_theta = y / r
    omc = 1.0 - cos_theta

    re = [
        M.M_PI ** 2 / 6 + 0.25 * (theta ** 2 - 2 * M.M_PI * abs(theta)),
        -0.5 * np.log(2 * omc),
        -0.5,
        -0.5 / omc,
        0,
        0.5 * (2.0 + cos_theta) / (omc ** 2),
        0
    ]

    im = [
        clausen(theta),
        -np.arctan2(-sin_theta, omc),
        0.5 * sin_theta / omc,
        0,
        -0.5 * sin_theta / (omc ** 2),
        0,
        0.5 * sin_theta / (omc ** 5) * (8.0 * omc - sin_theta * sin_theta * (3.0 + cos_theta))
    ]

    sum_re, sum_im = re[0], im[0]
    a, an, nfact = np.log(r), 1, 1
    for n in range(1, 7):
        an *= a
        nfact *= n
        t = an / nfact
        sum_re += t * re[n]
        sum_im += t * im[n]

    return sum_re, sum_im


def _series_2_c(r, x, y):
    cos_theta = x / r
    sin_theta = y / r

    alpha = 1 - cos_theta
    beta = sin_theta

    ck, sk, rk = cos_theta, sin_theta, r

    re, im = 0.5 * r * ck, 0.5 * r * sk

    for k in range(2, 30 + int(18.0 / (-np.log(r)))):
        ck_tmp = ck
        ck = ck - (alpha * ck + beta * sk)
        sk = sk - (alpha * sk - beta * ck_tmp)
        rk *= r
        dr = rk / (k * k * (k + 1.0)) * ck
        di = rk / (k * k * (k + 1.0)) * sk
        re += dr
        im += di
        if abs((dr ** 2 + di ** 2) / (re ** 2 + im ** 2)) < M.DBL_EPSILON ** 2:
            break
    return re, im
