import numpy as np
from scipy.special import gamma, gammaln

from copulae.special.optimize import find_root
from copulae.special.trig import tanpi2, cospi2
from .common import *

__all__ = ['aux_f1', 'aux_f2', 'pareto']


def aux_f1(x, alpha, beta, log=False):
    r"""
    Auxiliary for pdf calculation when :math:`\alpha \neq 1`

    Parameters
    ----------
    x: scalar
        Input vector

    alpha: float
        Value of the index parameter alpha in the interval = (0, 2]

    beta: float
        Skew parameter

    log: bool
        If True, returns the log of the auxiliary function

    Returns
    -------
    scalar
        Output of auxiliary function
    """
    bt = beta * tanpi2(alpha)
    zeta = -bt
    theta0 = min(max(-PI2, np.arctan(bt) / alpha), PI2)

    if bt == 0:
        zeta_tol = 4e-16
    elif 1 - abs(beta) < 0.01 or alpha < 0.01:
        zeta_tol = 2e-15
    else:
        zeta_tol = 5e-5

    x_m_zeta = abs(x - zeta)

    if log:
        f_zeta = gammaln(1 + 1 / alpha) + np.log(np.cos(theta0)) - (np.log(PI) + np.log1p(zeta ** 2) / (2 * alpha))
    else:
        f_zeta = gamma(1 + 1 / alpha) * np.cos(theta0) / (PI * (1 + zeta ** 2) ** (1 / (2 * alpha)))

    if np.isfinite(x) and x_m_zeta <= zeta_tol * (zeta_tol + np.abs([x, zeta]).max()):
        return f_zeta

    if x < zeta:
        theta0 = -theta0
        if alpha < 1e-17:
            beta = -beta
            x = -x

    if alpha < 1e-17:
        r = np.log(alpha) + np.log1p(beta) - (1 + np.log(2 * x + PI * alpha * beta))
        return r if log else np.exp(r)

    def g1(th):
        a_1 = alpha - 1
        at = alpha * theta0

        if np.isclose(abs(PI2 - np.sign(a_1) * th), 0, atol=1.4e-14):
            return 0

        att = (th + theta0) * alpha
        return np.cos(att - th) * (np.cos(at) * np.cos(th) * (x_m_zeta / np.sin(att)) ** alpha) ** (1 / a_1)

    def g2(th):
        return xexp(g1(th))

    g_pi = g1(PI2)
    g_t0 = g1(-theta0)

    if (alpha >= 1 and ((~np.isnan(g_pi) and g_pi > LARGE_EXP_POW) or np.isclose(g_t0, 0))) or \
            (alpha < 1 and ((~np.isnan(g_t0) and g_t0 > LARGE_EXP_POW) or np.isclose(g_pi, 0))):
        return -np.inf if log else 0

    g_ = g1(eplus(-theta0)) if alpha >= 1 else g1(PI2E)
    if np.isnan(g_) and max(x_m_zeta, x_m_zeta / abs(x)) < 0.01:
        return f_zeta

    if (alpha >= 1 and g_pi > 1) or (alpha < 1 and g_pi < 1):
        theta2, mid = PI2E, g2(PI2E)
    elif (alpha >= 1 > g_t0) or (alpha < 1 < g_t0):
        theta2, mid = eplus(-theta0), g2(eplus(-theta0))
    else:
        ll = -theta0
        uu = PI2
        if alpha < 1:
            while np.isclose(g1((ll + PI2) / 2), 0):
                ll = (ll + PI2) / 2

            if np.isclose(g1((ll + PI2) / 2), 1):
                while np.isclose(g1((ll + uu) / 2), 1):
                    uu = (ll + uu) / 2

            if np.isclose(uu + ll, 0, 1e-13):
                return -np.inf if log else 0

        r1 = find_root(lambda t: g1(t) - 1, ll, uu)
        gr1 = xexp(r1 + 1)

        try:
            r2 = find_root(lambda t: np.log(g1(t)), ll, uu)
            gr2 = xexp(np.exp(r2))
        except (AssertionError, RuntimeError):
            r2, gr2 = np.nan, -np.inf

        theta2, mid = (r1, gr1) if gr1 >= gr2 else (r2, gr2)

    eps = 1e-4
    if mid > eps > g2(-theta0):
        theta1 = find_root(lambda x: g2(x) - eps, -theta0, theta2)
        r1 = integrate(g2, -theta0, theta1)
        r2 = integrate(g2, theta1, theta2)
    else:
        r1 = 0
        r2 = integrate(g2, -theta0, theta2)

    if mid > eps > g2(PI2):
        theta3 = find_root(lambda x: g2(x) - eps, theta2, PI2)
        r3 = integrate(g2, theta2, theta3)
        r4 = integrate(g2, theta3, PI2)
    else:
        r3 = integrate(g2, theta2, PI2)
        r4 = 0

    c = (alpha / (np.pi * abs(alpha - 1) * x_m_zeta))
    r = r1 + r2 + r3 + r4

    return (np.log(c) + np.log(r)) if log else (c * r)


def aux_f2(x, beta, log=False):
    r"""
    Auxiliary for pdf calculation when :math:`\alpha = 1`

    Parameters
    ----------
    x: scalar
        Input vector

    beta: float
        Skew parameter

    log: bool
        If True, returns the log of the auxiliary function

    Returns
    -------
    scalar
        Output of auxiliary function
    """
    i2b = 1 / (2 * beta)
    p2b = PI * i2b
    ea = -p2b * x

    if np.isinf(ea):
        return -np.inf if log else 0

    def g1(u):
        if np.isclose(abs(u + np.size(beta)), 0, atol=1e-10):
            return 0
        h = p2b + u * PI2
        return h / p2b * np.exp(ea + h * tanpi2(u)) / cospi2(u)

    def g2(u):
        return xexp(g1(u))

    root = find_root(lambda u: g1(u) - 1, -1, 1)

    r1 = integrate(g2, -1, root)
    r2 = integrate(g2, root, 1)

    if log:
        return np.log(PI2) + np.log(abs(i2b)) + np.log(r1 + r2)
    else:
        return PI2 * abs(i2b) * (r1 + r2)


def pareto(x, alpha, beta, log=False):
    """Tail approximation density for stable pdf"""
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.ravel()

    neg = x < 0
    if np.any(neg):
        x[neg] *= -1
        beta = np.repeat(beta, len(x))
        beta[neg] *= -1

    if log:
        return np.log(alpha) + np.log1p(beta) + stable_tail(alpha, log=True) - (1 + alpha) * np.log(x)
    else:
        return alpha * (1 + beta) * stable_tail(alpha) * x ** (-(1 + alpha))
