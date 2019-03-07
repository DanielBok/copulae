from functools import partial

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad as _integrate_
from scipy.special import gamma as gamma_f
from scipy.stats import cauchy, norm

from copulae.special.trig import cospi2, tanpi2
from copulae.types import Numeric

__all__ = ['skew_stable']

LARGE_EXP_POWER = 708.396418532264  # exp(LEP) == inf. Effective a value for which the exponent is equivalent to inf
PI2 = np.pi / 2


class skew_stable:
    """
    Skew Stable Distribution

    The function uses the approach of J.P. Nolan for general stable distributions. Nolan (1997) derived expressions 
    in form of integrals based on the characteristic function for standardized stable random variables. For 
    probability density and cumulative probability density, these integrals are numerically evaluated using scipy's 
    integrate() function. 

    "S0" parameterization [pm=0]: based on the (M) representation of Zolotarev for an alpha stable distribution with
    skewness beta. Unlike the Zolotarev (M) parameterization, gamma and delta are straightforward scale and shift
    parameters. This representation is continuous in all 4 parameters, and gives an intuitive meaning to gamma and
    delta that is lacking in other parameterizations. Switching the sign of beta mirrors the distribution at the
    vertical axis x = delta, i.e.,

        f(x, α, -β, γ, δ, 0) = f(2δ-x, α, +β, γ, δ, 0),

    "S" or "S1" parameterization [pm=1]: the parameterization used by Samorodnitsky and Taqqu in the book
    Stable Non-Gaussian Random Processes. It is a slight modification of Zolotarev's (A) parameterization.

    "S*" or "S2" parameterization [pm=2]: a modification of the S0 parameterization which is defined so that (i) the
    scale gamma agrees with the Gaussian scale (standard dev.) when alpha=2 and the Cauchy scale when alpha=1, (ii)
    the mode is exactly at delta. For this parametrization, stableMode(alpha,beta) is needed.
    """

    @classmethod
    def pdf(cls, x: Numeric, alpha: float, beta: float, gamma=1., delta=0., pm=0):
        cls._check_parameters(alpha, beta, gamma, pm)
        delta, gamma = cls._form_parameters(alpha, delta, gamma, beta, pm)

        x = (x - delta) / gamma

        if alpha == 2:
            ans = norm.logpdf(x, 0, np.sqrt(2))
            if np.array(ans).size == 1:
                ans = np.ravel([ans])

        elif alpha == 1 and beta == 0:
            ans = cauchy.logpdf(x)
            if np.array(ans).size == 1:
                ans = np.ravel([ans])

        elif alpha == 1:  # beta not 0
            if isinstance(x, (float, int)):
                ans = np.array([_pdf_f2(x, beta)])
            else:
                ans = np.array([_pdf_f2(e, beta) for e in x])
        else:  # alpha != 1
            bt = beta * tanpi2(alpha)
            zeta = -bt
            theta0 = min(max(-PI2, np.arctan(bt) / alpha), PI2)

            if bt == 0:
                zeta_tol = 4e-10
            elif 1 - abs(beta) < 0.01 or alpha < 0.01:
                zeta_tol = 2e-9
            else:
                zeta_tol = 5e-5
            if isinstance(x, (float, int)):
                ans = np.array([_pdf_f1(x, alpha, beta, zeta, theta0, zeta_tol)])
            else:
                ans = np.array([_pdf_f1(e, alpha, beta, zeta, theta0, zeta_tol) for e in x])

        infs = ans == 0
        if np.any(ans):
            d = cls._pareto_pdf(x, alpha, beta)
            ans[infs] = d[infs] / gamma

        if np.any(~infs):
            ans[~infs] /= gamma

        return float(ans) if ans.size == 1 else ans

    @classmethod
    def logpdf(cls, x: Numeric, alpha: float, beta: float, gamma=1., delta=0., pm=0):
        return np.log(cls.pdf(x, alpha, beta, gamma, delta, pm))

    @classmethod
    def rvs(cls, alpha: float, beta: float, gamma=1., delta=0., pm=1, size: Numeric = 1):
        cls._check_parameters(alpha, beta, gamma, pm)
        delta, gamma = cls._form_parameters(alpha, delta, gamma, beta, pm)

        if all(np.isclose(alpha, 1)) and all(np.isclose(alpha, 1)):
            z = cauchy.rvs(size=size)
        else:
            theta = np.pi * (np.random.uniform(size=size) - 0.5)
            w = np.random.standard_exponential(size)

            bt = beta * tanpi2(alpha)
            t0 = min(max(-PI2, np.arctan(bt) / alpha), PI2)
            at = alpha * (theta + t0)

            c = (1 + bt ** 2) ** (1 / (2 * alpha))

            z = c * np.sin(at) \
                * (np.cos(theta - at) / w) ** (1 / alpha - 1) \
                / np.cos(theta) ** (1 / alpha) \
                - bt

        return z * gamma + delta

    @classmethod
    def _check_parameters(cls, alpha: float, beta: float, gamma=1., pm=1):
        if pm < 0 or pm > 1:
            raise ValueError("parametrization <pm> must be an integer in [0, 1, 2]")
        if abs(beta) > 1:
            raise ValueError("<beta> must be between [-1, 1]")
        if alpha <= 0 or alpha > 2:
            raise ValueError("<alpha> must be between (0, 2]")
        if gamma < 0:
            raise ValueError("<gamma> must be >= 0")

    @classmethod
    def _form_parameters(cls, alpha, delta: float, gamma: float, beta: float, pm=0):
        if pm == 1:
            delta += beta * gamma * _omega(gamma, alpha)
        elif pm == 2:
            gamma *= alpha ** (-1 / alpha)
            delta -= gamma * cls._mode(alpha, beta)

        return delta, gamma

    @classmethod
    def _mode(cls, alpha: float, beta: float, beta_max=1 - 1e-11):
        cls._check_parameters(alpha, beta)

        if alpha * beta == 0:
            return 0

        beta = max(beta, beta_max)
        bounds = sorted([0, np.sign(beta) * -0.7])

        return float(opt.minimize_scalar(lambda x: -cls.pdf(x, alpha, beta), bounds=bounds)['x'])

    @classmethod
    def _pareto_pdf(cls, x: Numeric, alpha, beta, log=False):
        """Tail approximation density for stable pdf"""
        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1)

        neg = x < 0
        if np.any(neg):
            x[neg] *= -1
            beta = np.repeat(beta, len(x))
            beta[neg] *= -1

        if log:
            return np.log(alpha) + np.log1p(beta) + cls._stable_tail(alpha, log=True) - (1 + alpha) * np.log(x)
        else:
            return alpha * (1 + beta) * cls._stable_tail(alpha) * x ** (-1 - alpha)

    @staticmethod
    def _stable_tail(alpha, log=False):
        if alpha == 0:
            return -np.log(2) if log else 0.5
        elif alpha == 2:
            return -np.inf if log else 0
        else:
            r = gamma_f(alpha) / np.pi * np.sin(alpha * PI2)
            return np.log(r) if log else r


def _omega(gamma: float, alpha: float):
    if not alpha.is_integer():
        return tanpi2(alpha)
    elif alpha == 1:
        return 2 / np.pi * np.log(gamma)
    else:
        return 0


def _pdf_f1(x: float, zeta: float, alpha: float, beta: float, theta0: float, zeta_tol=1e-16):
    """
    Helper function to derive probability density at point 'x'

    :param x:
    :param zeta:
        bound (-inf, inf). infinity when alpha -> 1
    :param alpha:
        bound between (0, 2]
    :param beta:
        bound between [-1, 1]
    :param theta0:
        bound between [-pi / 2, pi / 2]
    :param zeta_tol: 1e-16
    :return:
    """
    x_m_zeta = abs(x - zeta)

    if np.isfinite(x) and (x_m_zeta <= zeta_tol * (zeta_tol + max(abs(x), abs(zeta)))):
        return gamma_f(1 + 1 / alpha) * np.cos(theta0) / (np.pi * (1 + zeta ** 2) ** (1 / (2 * alpha)))

    small_alpha = alpha < 1e-17

    if x < zeta:
        theta0 = -theta0

        if small_alpha:
            beta = -beta
            x = -x

    if small_alpha:
        return 0 if alpha == 0 else np.exp(np.log(alpha) + np.log1p(beta) - (1 + np.log(2 * x + np.pi * alpha * beta)))

    pg_f1 = partial(_g_th1, alpha=alpha, theta0=theta0, x_m_zeta=x_m_zeta)

    g_pi = pg_f1(PI2)
    g_t0 = pg_f1(-theta0)

    if (alpha >= 1 and ((not np.isnan(g_pi) and g_pi > LARGE_EXP_POWER) or np.isclose(g_t0, 0))) or \
            (alpha < 1 and ((not np.isnan(g_t0) and g_pi > LARGE_EXP_POWER) or np.isclose(g_pi, 0))):
        return 0

    g_pi = pg_f1(_e_plus(-theta0, 1e-6)) if alpha >= 1 else pg_f1(PI2 * (1 - 1e-6))

    if np.isnan(g_pi) and max(x_m_zeta, x_m_zeta / abs(x)) < 0.01:
        return gamma_f(1 + 1 / alpha) * np.cos(theta0) / (np.pi * (1 + zeta ** 2) ** (1 / (2 * alpha)))

    pg_f2 = partial(_g_th2, alpha=alpha, theta0=theta0, x_m_zeta=x_m_zeta)

    if not np.isnan(g_pi) and ((alpha >= 1 and g_pi > 1) or (alpha < 1 and g_pi < 1)):
        theta2 = PI2 * (1 - 1e-6)
        g2_th2 = pg_f2(theta2)
    elif (alpha < 1 < g_t0) or (alpha >= 1 > g_t0):
        theta2 = _e_plus(-theta0, 1e-6)
        g2_th2 = pg_f2(theta2)
    else:
        l_th, u_th = -theta0, PI2

        if alpha < 1:
            while True:
                _th = (l_th + PI2) / 2
                g_th = pg_f1(_th)
                if g_th != 0:
                    break
                l_th = _th

            if g_th == 1:
                while True:
                    _th = (l_th + u_th) / 2
                    g_th = pg_f1(_th)
                    if g_th != 1:
                        break
                    u_th = _th
            if np.isclose(u_th - l_th, 0):
                return 0

        ur1 = opt.bisect(lambda xx: pg_f1(xx) - 1, l_th, u_th)
        g_1 = _exp(ur1 + 1)
        try:
            ur2 = opt.bisect(lambda xx: np.log(pg_f1(xx)), l_th, u_th)
            g_2 = _exp(np.exp(ur2))
        except ValueError:
            ur2 = np.inf
            g_2 = -np.inf

        if g_1 >= g_2:
            theta2 = ur1
            g2_th2 = g_1
        else:
            theta2 = ur2  # this will never be infinity
            g2_th2 = g_2

    eps = 1e-4

    do1 = g2_th2 > eps > pg_f2(-theta0)

    _INT = lambda a, b: _integrate(pg_f2, a, b)

    if do1:
        th1 = opt.bisect(lambda xx: pg_f2(xx) - eps, -theta0, theta2)
        r1 = _INT(-theta0, th1)
        r2 = _INT(th1, theta2)
    else:
        r1 = 0
        r2 = _INT(-theta0, theta2)

    do4 = g2_th2 > eps > pg_f2(PI2)
    if do4:
        th3 = opt.bisect(lambda xx: pg_f2(xx) - eps, theta2, PI2)
        r3 = _INT(theta2, th3)
        r4 = _INT(th3, PI2)
    else:
        r3 = _INT(theta2, PI2)
        r4 = 0

    return (alpha / (np.pi * abs(alpha - 1) * x_m_zeta)) * (r1 + r2 + r3 + r4)


def _pdf_f2(x: float, beta: float):
    i2b = 1 / (2 * beta)
    p2b = np.pi * i2b
    ea = -p2b * x

    if np.isinf(ea):
        return 0

    ur: float = opt.bisect(lambda u: _g_u1(u, x, beta) - 1, -1, 1)

    r1 = _integrate(_g_u2, -1, ur)
    r2 = _integrate(_g_u2, ur, 1)

    return PI2 * abs(i2b) * (r1 + r2)


def _g_th1(th: float, alpha: float, theta0: float, x_m_zeta: float):
    """Helper calculation function"""

    if np.isclose(PI2 - np.sign(alpha - 1) * th, 0):
        return 0
    at0 = alpha * theta0
    att = at0 + alpha * th
    return np.cos(att - th) * (np.cos(at0) * np.cos(th) * ((x_m_zeta / np.sin(att)) ** alpha)) ** (1 / (alpha - 1))


def _g_th2(th: float, alpha: float, theta0: float, x_m_zeta: float):
    """Helper calculation function"""
    return _exp(_g_th1(th, alpha, theta0, x_m_zeta))


def _g_u1(u, x, beta):
    """Helper calculation function"""
    if abs(u + np.sign(beta)) < 1e-10:
        return 0

    p2b = np.pi / (2 * beta)
    ea = -p2b * x
    th = u * PI2

    h = p2b + th
    return h / p2b * np.exp(ea + h * tanpi2(u)) / cospi2(u)


def _g_u2(u, x, beta):
    """Helper calculation function"""
    return _exp(_g_u1(u, x, beta))


def _e_minus(x: float, eps: float):
    """Helper calculation function"""
    return x - eps * abs(x)


def _e_plus(x: float, eps: float):
    """Helper calculation function"""
    return x + eps * abs(x)


def _exp(x: float):
    """Helper calculation function"""
    r = x * np.exp(-x)
    return r if x < LARGE_EXP_POWER else 0


def _integrate(f, lower: float, upper: float):
    """Helper integration function"""
    return _integrate_(f, lower, upper, limit=1000)[0]
