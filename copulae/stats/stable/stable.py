from collections import abc

import numpy as np
import scipy.optimize as opt
from scipy.stats import cauchy, norm

from copulae.special.trig import tanpi2
from copulae.stats.stable import pdf
from copulae.types import Numeric

__all__ = ['skew_stable']

LARGE_EXP_POWER = 708.396418532264  # exp(LEP) == inf. Effective a value for which the exponent is equivalent to inf
PI2 = np.pi / 2


# noinspection PyPep8Naming
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
    def pdf(cls, x, alpha, beta, gamma=1., delta=0., pm=0, log=False):
        """
        Returns density for stable DF

        Parameters
        ----------
        x: {array_like, scalar}
            Numeric vector of quantiles.

        alpha: float
            Value of the index parameter alpha in the interval = (0, 2]

        beta: float
            Skewness parameter in the range [-1, 1]

        gamma: float
            Scale parameter

        delta: float
            Location (or ‘shift’) parameter delta.

        pm: {0, 1, 2}
            Type of parameterization, an integer in {0, 1, 2}. Defaults to 0, 'S0' parameterization.

        log: bool
            If True, returns the log of the density

        Returns
        -------
        {array_like, scalar}
            Numeric vectors of density
        """
        cls._check_parameters(alpha, beta, gamma, pm)
        delta, gamma = cls._parameterize(alpha, delta, gamma, beta, pm)

        if isinstance(x, abc.Iterable):
            x = np.asarray(x)

        x = (x - delta) / gamma

        if alpha == 2:
            ans = norm.logpdf(x, 0, np.sqrt(2)) if log else norm.pdf(x, 0, np.sqrt(2))

        elif alpha == 1 and beta == 0:
            ans = cauchy.logpdf(x) if log else cauchy.pdf(x)

        elif alpha == 1:  # beta not 0
            if isinstance(x, (complex, float, int)):
                ans = np.array([pdf.aux_f2(x, beta, log)])
            else:
                ans = np.array([pdf.aux_f2(e, beta, log) for e in x.ravel()]).reshape(x.shape)
        else:  # alpha != 1
            if isinstance(x, (complex, float, int)):
                ans = np.array([pdf.aux_f1(x, alpha, beta, log)])
            else:
                ans = np.array([pdf.aux_f1(e, alpha, beta, log) for e in x.ravel()]).reshape(x.shape)

        if ans.size == 1:
            ans = ans.ravel()

        infs = ans == 0
        if np.any(ans):
            d = pdf.pareto(x, alpha, beta, log)[infs]
            ans[infs] = (d - np.log(gamma)) if log else (d / gamma)

        if np.any(~infs):
            d = ans[~infs]
            ans[~infs] = (d - np.log(gamma)) if log else (d / gamma)

        return ans.item(0) if ans.size == 1 else ans

    @classmethod
    def logpdf(cls, x: Numeric, alpha: float, beta: float, gamma=1., delta=0., pm=0):
        """
        Returns log of density for stable DF

        Parameters
        ----------
        x: {array_like, scalar}
            Numeric vector of quantiles.

        alpha: float
            Value of the index parameter alpha in the interval = (0, 2]

        beta: float
            Skewness parameter in the range [-1, 1]

        gamma: float
            Scale parameter

        delta: float
            Location (or ‘shift’) parameter delta.

        pm: {0, 1, 2}
            Type of parameterization, an integer in {0, 1, 2}. Defaults to 0, 'S0' parameterization.

        Returns
        -------
        {array_like, scalar}
            Numeric vectors of density
        """
        return cls.pdf(x, alpha, beta, gamma, delta, pm, log=True)

    @classmethod
    def rvs(cls, alpha: float, beta: float, gamma=1., delta=0., pm=1, size: Numeric = 1):
        cls._check_parameters(alpha, beta, gamma, pm)
        delta, gamma = cls._parameterize(alpha, delta, gamma, beta, pm)

        if np.isclose(alpha, 1) and np.isclose(beta, 0):
            z = cauchy.rvs(size=size)
        else:
            theta = np.pi * (np.random.uniform(size=size) - 0.5)
            w = np.random.standard_exponential(size)

            bt = beta * tanpi2(alpha)
            t0 = min(max(-PI2, np.arctan(bt) / alpha), PI2)
            at = alpha * (theta + t0)

            c = (1 + bt ** 2) ** (1 / (2 * alpha))

            z = (c * np.sin(at)
                 * (np.cos(theta) ** (-1 / alpha))
                 * ((np.cos(theta - at) / w) ** ((1 - alpha) / alpha))
                 - bt)

        return z * gamma + delta

    @classmethod
    def _check_parameters(cls, alpha: float, beta: float, gamma=1., pm=1):
        assert pm in (0, 1, 2), "parametrization `pm` must be an integer in [0, 1, 2]"
        assert abs(beta) <= 1, "`beta` must be in the interval [-1, 1]"
        assert 0 < alpha <= 2, "`alpha` must be in the interval (0, 2]"
        assert gamma >= 0, "`gamma` must be >= 0"

    @classmethod
    def _parameterize(cls, alpha, delta: float, gamma: float, beta: float, pm=0):
        if pm == 1:
            delta = delta + beta * gamma * _omega(gamma, alpha)
        elif pm == 2:
            gamma = gamma * alpha ** (-1 / alpha)
            delta = delta - gamma * cls._mode(alpha, beta)

        return delta, gamma

    @classmethod
    def _mode(cls, alpha: float, beta: float, beta_max=1 - 1e-11):
        cls._check_parameters(alpha, beta)

        if alpha * beta == 0:
            return 0

        beta = max(beta, beta_max)
        bounds = sorted([0, np.sign(beta) * -0.7])

        return float(opt.minimize_scalar(lambda x: -cls.pdf(x, alpha, beta), bounds=bounds)['x'])


def _omega(gamma: float, alpha: float):
    if not alpha.is_integer():
        return tanpi2(alpha)
    elif alpha == 1:
        return 2 / np.pi * np.log(gamma)
    else:
        return 0
