from libc cimport math as cm
cimport numpy as cnp

from cython.parallel import prange
import numpy as np

ctypedef double (*Fn1R) (double) nogil
ctypedef ComplexResult (*Fn1C) (double, double) nogil
ctypedef struct ComplexResult:
    double real
    double imag

cdef:
    double PI = 3.14159265358979323846264338328
    double DBL_EPSILON = 2.220446049250313e-16
    double SQRT_DBL_EPSILON = 1.4901161193847656e-08
    double LOG_DBL_EPSILON = -36.04365338911715
    double LOG_DBL_MIN = -708.3964185322641
    double M_LN2 = 0.69314718055994530941723212146
    double M_SQRT2 = 1.41421356237309504880168872421

    double[::1] clausen_constants = np.array([
        2.142694363766688447e+00,
        0.723324281221257925e-01,
        0.101642475021151164e-02,
        0.3245250328531645e-04,
        0.133315187571472e-05,
        0.6213240591653e-07,
        0.313004135337e-08,
        0.16635723056e-09,
        0.919659293e-11,
        0.52400462e-12,
        0.3058040e-13,
        0.18197e-14,
        0.1100e-15,
        0.68e-17,
        0.4e-18
    ])

    double[::1] debye1_constant = np.array([
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
    ])

    double[::1] debye2_constant = np.array([
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
    ])


def clausen(x, bint threaded=True):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _clausen(arr[0])
    if threaded and n > 1:
        map_dbl_p(_clausen, arr, n)
    else:
        map_dbl_s(_clausen, arr, n)

    return arr.reshape(x.shape)


def debye_1(x, bint threaded=True):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_1(arr[0])
    elif threaded:
        map_dbl_p(_debye_1, arr, n)
    else:
        map_dbl_s(_debye_1, arr, n)

    return arr.reshape(x.shape)


def debye_2(x, bint threaded=True):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_2(arr[0])
    elif threaded:
        map_dbl_p(_debye_2, arr, n)
    else:
        map_dbl_s(_debye_2, arr, n)

    return arr.reshape(x.shape)


def dilog(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _dilog(arr[0])
    if threaded and n > 1:
        map_dbl_p(_dilog, arr, n)
    else:
        map_dbl_s(_dilog, arr, n)

    return arr.reshape(x.shape)


def dilog_complex(r, theta, bint threaded):
    r = np.asarray(r, float)
    theta = np.asarray(theta, float)

    cdef:
        ComplexResult c
        cnp.ndarray[cnp.npy_float64, ndim=1] r_vec = r.ravel()
        cnp.ndarray[cnp.npy_float64, ndim=1] i_vec = theta.ravel()
        int n = r_vec.size

    assert r.shape == theta.shape, "Radius of complex vector must have same shape as the angled part"

    if n == 1:
        c = _dilog_complex(r_vec[0], i_vec[0])
        return c.real + 1j * c.imag
    if threaded:
        mapc_dbl_p(_dilog_complex, r_vec, i_vec, n)
    else:
        mapc_dbl_s(_dilog_complex, r_vec, i_vec, n)

    return (r_vec + 1j * i_vec).reshape(r.shape)


cdef double angle_restrict_pos_err(double theta) nogil:
    cdef:
        double two_pi = 2 * PI
        double y = 2 * cm.floor(theta / two_pi)
        double r = theta - y * 2 * two_pi

    if cm.fabs(theta) > 0.0625 / DBL_EPSILON:
        return cm.NAN

    if r > two_pi:
        r -= two_pi
    elif r < 0:
        r += two_pi

    return r


cdef double cheb_eval(double[::1] constants, double x, int a, int b) nogil:
    cdef:
        double d = 0, dd = 0
        double y = (2. * x - a - b) / (b - a)
        double y2 = 2 * y
        size_t i, n = len(constants)

    for i in range(n - 1, 0, -1):
        d, dd = y2 * d - dd + constants[i], d

    return y * d - dd + 0.5 * constants[0]


cdef void map_dbl_p(Fn1R f, double[::1] x, int size) nogil:
    # Parallel
    cdef int i
    for i in prange(size, nogil=True):
        x[i] = f(x[i])


cdef void map_dbl_s(Fn1R f, double[::1] x, int size) nogil:
    # single
    cdef int i
    for i in range(size):
        x[i] = f(x[i])


cdef void mapc_dbl_p(Fn1C f, double[::1] r, double[::1] t, int size) nogil:
    # Parallel
    cdef:
        ComplexResult c
        int i

    for i in prange(size, nogil=True):
        c = f(r[i], t[i])
        r[i] = c.real
        t[i] = c.imag


cdef void mapc_dbl_s(Fn1C f, double[::1] r, double[::1] t, int size) nogil:
    # Single
    cdef:
        ComplexResult c
        int i

    for i in range(size):
        c = f(r[i], t[i])
        r[i] = c.real
        t[i] = c.imag


cdef double _clausen(double x) nogil:
    cdef:
        double res, sr
        double x_cut = PI * DBL_EPSILON
        int sgn = 1

    if x < 0:
        x = -x
        sgn = -1

    x = sr = angle_restrict_pos_err(x)

    if x > PI:
        x = (6.28125 - x) + 1.9353071795864769253e-03
        sgn = -sgn

    if x == 0.0:
        return 0
    elif x < x_cut:
        res = x * (1 - cm.log(x))
    else:
        sr = cheb_eval(clausen_constants, 2 * (x * x / (PI ** 2) - 0.5), -1, 1)
        res = x * (sr - cm.log(x))

    return res * sgn


cdef double _debye_1(double x) nogil:
    cdef:
        double res = 0
        double val_infinity = 1.64493406684822644
        int i, nexp
        double total, ex
        double X_CUT = -LOG_DBL_MIN

    if x < 0:
        return cm.NAN

    elif x < 2 * SQRT_DBL_EPSILON:
        res = 1 - 0.25 * x + x ** 2 / 36.

    elif x <= 4:
        c = cheb_eval(debye1_constant, x * x / 8. - 1, -1, 1)
        res = c  - 0.25 * x

    elif x <= -(M_LN2 + LOG_DBL_EPSILON):
        nexp = <int> cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            total *= ex
            total += (1 + 1 / (x * i)) / i

        res = val_infinity / x - total * ex

    elif x < X_CUT:
        res = (val_infinity - cm.exp(-x) * (x + 1)) / x

    else:
        res = val_infinity / x

    return res


cdef double _debye_2(double x) nogil:
    cdef:
        double res = 0
        double val_infinity = 4.80822761263837714, x2 = x ** 2
        int i, nexp
        double total, ex, xi
        double X_CUT = -LOG_DBL_MIN

    if x < 0:
        return cm.NAN

    elif x < 2 * M_SQRT2 * SQRT_DBL_EPSILON:
        res = 1 - x / 3 + x2 / 24

    elif x <= 4:
        c = cheb_eval(debye2_constant, x * x / 8. - 1, -1, 1)
        res = c - x / 3.

    elif x < - (M_LN2 + LOG_DBL_EPSILON):
        nexp = <int> cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            total *= ex
            xi = x * i
            total += (1 + 2 / xi + 2 / (xi ** 2)) / i

        res = val_infinity / x ** 2 - 2 * total * ex

    elif x < X_CUT:
        total = 2 + 2 * x + x2
        res = (val_infinity - 2 * total * cm.exp(-x)) / x2

    else:
        res = val_infinity / x2

    return res

cdef double _dilog(double x) nogil:
    cdef:
        double res = 0, d1, d2

    if x >= 0:
        return dilog_xge0(x)
    d1 = dilog_xge0(-x)
    d2 = dilog_xge0(x * x)
    res = 0.5 * d2 - d1
    return res

cdef double dilog_xge0(double x) nogil:
    """Calculates dilog for real :math:`x \geq 0"""
    cdef:
        double res = 0
        double ser
        double log_x = cm.log(x)
        double t1, t2, t3
        int i

    if x > 2:
        ser = dilog_series_2(1. / x)
        t1 = PI * PI / 3
        t2 = ser
        t3 = 0.5 * log_x ** 2
        res = t1 - t2 - t3

    elif x > 1.01:
        ser = dilog_series_2(1 - 1. / x)
        t1 = PI ** 2 / 6
        t2 = ser
        t3 = log_x * (cm.log(1 - 1. / x) + 0.5 * log_x)
        res = t1 + t2 - t3

    elif x > 1:
        t2 = cm.log(x - 1)
        t3 = 0

        for i in range(8, 0, -1):
            t1 = (-1) ** (i + 1) * (1 - i * t2) / (i * i)
            t3 = (x - 1) * (t3 + t1)

        res = t3 + PI * PI / 6

    elif cm.fabs(x - 1) <= DBL_EPSILON * 10:
        res = PI ** 2 / 6

    elif x > 0.5:
        ser = dilog_series_2(1 - x)
        t1 = PI ** 2 / 6
        t2 = ser
        t3 = log_x * cm.log(1 - x)
        res = t1 - t2 - t3

    elif x > 0.25:
        return dilog_series_2(x)

    elif x > 0:
        return dilog_series_1(x)

    return res

cdef double dilog_series_1(double x) nogil:
    cdef:
        double rk2, term = x, total = x
        int k

    for k in range(2, 1000):
        rk2 = ((k - 1.0) / k) ** 2
        term *= x * rk2
        total += term

        if cm.fabs(term / total) < DBL_EPSILON:
            return total

    # Max iteration hit. dilog_series_1 could not converge
    return cm.NAN


cdef double dilog_series_2(double x) nogil:
    cdef:
        double res = 0
        double total = 0.5 * x, y = x, z = 0
        double ds
        int k

    for k in range(2, 100):
        y *= x
        ds = y / (k * k * (k + 1))
        total += ds
        if k >= 10 and cm.fabs(ds / total) < 0.5 * DBL_EPSILON:
            break

    res = total

    if x > 0.01:
        z = (1 - x) * cm.log(1 - x) / x
    else:
        for k in range(8, 1, -1):
            z = x * (1.0 / k + z)
        z = (x - 1) * (1 + z)

    return res + z + 1


cdef ComplexResult _dilog_complex(double r, double theta) nogil:
    cdef:
        ComplexResult c = make_c_0()
        double x = r * cm.cos(theta)
        double y = r * cm.sin(theta)
        double zeta2 = PI ** 2 / 6
        double r2 = x * x + y * y
        double t1, t2

        double ln_minusz_re, ln_minusz_im, lmz2_re, lmz2_im

    if cm.fabs(y) < 10 * DBL_EPSILON:
        c.real = _dilog(x)
        if x >= 1:
            c.imag = -PI * cm.log(x)

    elif cm.fabs(r2 - 1) <= DBL_EPSILON:
        t1 = theta * theta / 4
        t2 = PI * cm.fabs(theta) / 2
        c.real = zeta2 + t1 - t2

        c.imag = _clausen(theta)

    elif r2 < 1:
        return dilogc_unit_disk(x, y)

    else:
        c = dilogc_unit_disk(x / r2, - y / r2)
        ln_minusz_re = cm.log(r)
        ln_minusz_im = (-1.0 if theta < 0.0 else 1.0) * (cm.fabs(theta) - PI)
        lmz2_re = ln_minusz_re ** 2 - ln_minusz_im ** 2
        lmz2_im = 2.0 * ln_minusz_re * ln_minusz_im

        c.real = -c.real - 0.5 * lmz2_re - zeta2
        c.imag = -c.imag - 0.5 * lmz2_im

    return c


cdef ComplexResult complex_log(double zr, double zi) nogil:
    cdef:
        ComplexResult res = make_c_0()
        double ax, ay, min_, max_

    if zr == 0 and zi == 0:
        res.real = cm.NAN
        res.imag = cm.NAN
        return res

    ax = cm.fabs(zr)
    ay = cm.fabs(zi)
    min_ = cm.fmin(ax, ay)
    max_ = cm.fmax(ax, ay)

    res.real = cm.log(max_) + 0.5 * cm.log(1 + (min_ / max_) ** 2)
    res.imag = cm.atan2(zi, zr)

    return res

cdef inline ComplexResult dilogc_fundamental(double r, double x, double y) nogil:
    if r > 0.98:
        return dilogc_series_3(r, x, y)
    elif r > 0.25:
        return dilogc_series_2(r, x, y)
    else:
        return dilogc_series_1(r, x, y)


cdef ComplexResult dilogc_unit_disk(double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        ComplexResult tmp_c
        double r = cm.hypot(x, y)
        double zeta2 = PI ** 2 / 6
        double x_tmp, y_tmp, r_tmp, lnz, lnomz, argz, argomz

    if x > 0.732:  # magic split value
        x_tmp = 1.0 - x
        y_tmp = -y
        r_tmp = cm.hypot(x_tmp, y_tmp)
        tmp_c = dilogc_fundamental(r_tmp, x_tmp, y_tmp)

        lnz = cm.log(r)  # log(|z|)
        lnomz = cm.log(r_tmp)  # log(|1 - z|)
        argz = cm.atan2(y, x)  # arg(z)
        argomz = cm.atan2(y_tmp, x_tmp)  # arg(1 - z)

        c.real = -tmp_c.real + zeta2 - lnz * lnomz + argz * argomz
        c.imag = -tmp_c.imag - argz * lnomz - argomz * lnz

        return c
    else:
        return dilogc_fundamental(r, x, y)


cdef ComplexResult dilogc_series_1(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        double cos_theta = x / r
        double sin_theta = y / r
        double alpha = 1 - cos_theta
        double beta = sin_theta
        double ck = cos_theta
        double sk = sin_theta
        double rk = r
        double real = r * ck
        double imag = r * sk
        int k, kmax = 50 + <int> (-22 / cm.log(r))

    for k in range(2, kmax):
        ck_tmp = ck
        ck = ck - (alpha * ck + beta * sk)
        sk = sk - (alpha * sk - beta * ck_tmp)
        rk *= r
        dr = rk / (k * k) * ck
        di = rk / (k * k) * sk
        real += dr
        imag += di
        if cm.fabs((dr * dr + di * di) / (real ** 2 + imag ** 2)) < DBL_EPSILON ** 2:
            break

    c.real = real
    c.imag = imag
    return c


cdef ComplexResult dilogc_series_2(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        ComplexResult ln_omz, sum_c
        double r2 = r ** 2
        double tx, ty, rx, ry

    if cm.fabs(r) <= DBL_EPSILON * 10:
        return c

    sum_c = series_2_c(r, x, y)
    ln_omz = complex_log(1 - x, -y)

    tx = (ln_omz.real * x + ln_omz.imag * y) / r2
    ty = (-ln_omz.real * y + ln_omz.imag * x) / r2
    rx = (1 - x) * tx + y * ty
    ry = (1 - x) * ty - y * tx

    c.real = sum_c.real + rx + 1
    c.imag = sum_c.imag + ry

    return c

cdef ComplexResult dilogc_series_3(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        double theta = cm.atan2(y, x)
        double cos_theta = x / r
        double sin_theta = y / r
        double omc = 1.0 - cos_theta

        double claus = _clausen(theta)

        double*re = [
            PI ** 2 / 6 + 0.25 * (theta ** 2 - 2 * PI * cm.fabs(theta)),
            -0.5 * cm.log(2 * omc),
            -0.5,
            -0.5 / omc,
            0,
            0.5 * (2.0 + cos_theta) / (omc ** 2),
            0
        ]
        double*im = [
            claus,
            -cm.atan2(-sin_theta, omc),
            0.5 * sin_theta / omc,
            0,
            -0.5 * sin_theta / (omc ** 2),
            0,
            0.5 * sin_theta / (omc ** 5) * (8.0 * omc - sin_theta * sin_theta * (3.0 + cos_theta))
        ]
        double sum_re = re[0], sum_im = im[0]
        double a = cm.log(r), an = 1.0, nfact = 1.0
        int n

    for n in range(1, 7):
        an *= a
        nfact *= n
        sum_re += an / nfact * re[n]
        sum_im += an / nfact * im[n]

    c.real = sum_re
    c.imag = sum_im
    return c

cdef ComplexResult series_2_c(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        double cos_theta = x / r
        double sin_theta = y / r
        double alpha = 1 - cos_theta
        double beta = sin_theta
        double ck = cos_theta
        double sk = sin_theta
        double rk = r
        double real = 0.5 * r * ck
        double imag = 0.5 * r * sk
        double ck_tmp, di, dr
        int k, kmax = 30 + <int> (18.0 / (-cm.log(r)))
        double limit = DBL_EPSILON ** 2

    for k in range(2, kmax):
        ck_tmp = ck
        ck = ck - (alpha * ck + beta * sk)
        sk = sk - (alpha * sk - beta * ck_tmp)
        rk *= r
        dr = rk / (k * k * (k + 1.0)) * ck
        di = rk / (k * k * (k + 1.0)) * sk
        real += dr
        imag += di
        if cm.fabs((dr ** 2 + di ** 2) / (real ** 2 + imag ** 2)) < limit:
            break

    c.real = real
    c.imag = imag
    return c


cdef ComplexResult make_c_0() nogil:
    cdef ComplexResult c
    c.real, c.imag = 0, 0
    return c
