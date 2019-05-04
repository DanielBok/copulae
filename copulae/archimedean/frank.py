import numpy as np

from copulae.copula import Summary, TailDep
from copulae.core import valid_rows_in_u
from copulae.special.debye import debye_1, debye_2
from copulae.special.optimize import find_root
from copulae.special.special_func import log1mexp, log1pexp, poly_log
from copulae.stats import random_uniform
from copulae.stats.log import random_log_series_ln1p
from copulae.types import Array
from copulae.utility import array_io, as_array
from .abstract import AbstractArchimedeanCopula


class FrankCopula(AbstractArchimedeanCopula):
    r"""
    A Frank copula is an Archimedean copula. In the bivariate case, its parameters can interpolate between
    a lower limit of :math:`-\infty` (countermonotonicity) and an upper limit of :math:`\infty` (comonotonicity).
    A Frank copula is defined as

    .. math::

        C_\theta (u_1, \dots, u_d) = \frac{1}{\theta}
            \log \left(1 + \frac{ \prod_i (e^{\theta u_i} - 1) }{e^{-\theta} - 1} \right)
    """

    def __init__(self, theta=np.nan, dim=2):
        """
        Creates a Frank copula instance

        Parameters
        ----------
        theta: float, optional
            Number specifying the copula parameter

        dim: int, optional
            Dimension of the copula
        """
        super().__init__(dim, theta, 'Frank')
        assert not (dim != 2 and theta < 0), 'Frank Copula parameter must be >= 0 when dimension == 2'

        self._bounds = (-np.inf if dim == 2 else 0), np.inf

    @array_io
    def dipsi(self, u: Array, degree=1, log=False):
        assert degree in (1, 2), 'degree can only be 1 or 2'

        s = 1 if log or degree % 2 == 0 else -1

        ut = u * self.params
        if degree == 1:
            v = self.params / np.expm1(ut)
        else:
            v = (self.params ** 2 * np.exp(ut)) / np.expm1(ut) ** 2

        return s * (np.log(v) if log else v)

    @array_io(optional=True)
    def drho(self, x=None):  # pragma: no cover
        if x is None:
            x = self.params
        return 12 * (x / np.expm1(x) - 3 * debye1(x) + 2 * debye1(x)) / x ** 2

    @array_io(optional=True)
    def dtau(self, x=None):  # pragma: no cover
        if x is None:
            x = self.params
        return (x / np.expm1(x) + 1 - debye1(x) / x) * (2 / x) ** 2

    @array_io
    def ipsi(self, u, log=False):
        r = np.asarray(u) * self.params

        res = np.copy(r)
        res[np.isnan(r)] = np.nan
        em = np.expm1(-self.params)

        #  for small inputs, u <= 0.01
        small_mask = np.abs(r) <= 0.01 * abs(self.params)
        res[small_mask] = -np.log(np.expm1(-r[small_mask]) / em)

        big_mask = np.abs(r) > 0.01 * abs(self.params)
        e = np.exp(-self.params)
        mid_mask = (e > 0) & (np.abs(self.params - r) < 0.5)  # theta * (1 - u) < 0.5

        m1 = big_mask & mid_mask
        m2 = big_mask & ~mid_mask
        r[m1] = -np.log1p(e * np.expm1((self.params - r[m1])) / em)
        r[m2] = -np.log1p((np.exp(-r[m2]) - e) / em)

        return np.log(r) if log else r

    def irho(self, rho: Array):  # pragma: no cover
        # TODO frank: add inverse rho
        return NotImplemented

    @array_io
    def itau(self, tau):
        res = np.array([find_root(lambda x: self._tau(x) - t,
                                  2.2e-16 if t > 0 else -1e20,
                                  1e20 if t > 0 else -2.2e-16) for t in tau.ravel()])
        res = res.reshape(tau.shape)
        res[tau == 0] = tau[tau == 0]
        return res

    @property
    def lambda_(self):  # pragma: no cover
        return TailDep(0, 0)

    @property
    def params(self):
        return self._theta

    @params.setter
    def params(self, theta):
        if self.dim > 2 and theta < 0:
            raise ValueError('theta must be positive when dim > 2')
        self._theta = float(theta)

    @array_io(dim=2)
    def pdf(self, u: Array, log=False):
        assert not np.isnan(self.params), "Copula must have parameters to calculate parameters"

        n, d = u.shape
        theta = self.params

        ok = valid_rows_in_u(u)
        res = np.repeat(np.nan, n)

        u_ = u[ok]
        u_sum = u_.sum(1)
        lp = log1mexp(theta)
        lpu = log1mexp(theta * u_)
        lu = lpu.sum(1)

        li_arg = np.exp(lp + (lpu - lp).sum(1))
        li = poly_log(li_arg, 1 - d, log=True)

        res[ok] = (d - 1) * np.log(theta) + li - theta * u_sum - lu

        return res if log else np.exp(res)

    def psi(self, s):
        assert not np.isnan(self.params), "Copula must have parameters to calculate psi"

        s = np.asarray(s)
        if self.params <= -36:
            return -log1pexp(-s - self.params) / self.params
        elif self.params < 0:
            return -np.log1p(np.exp(-s) * np.expm1(-self.params)) / self.params
        elif self.params == 0:
            return np.exp(-s)
        else:
            const = log1mexp(self.params)
            m = np.less(s, const, where=~np.isnan(s))

            s[m] = np.nan
            s[~m] = -log1mexp(s[~m] - log1mexp(self.params)) / self.params
            return s.item(0) if s.size == 1 else s

    def random(self, n: int, seed: int = None):
        u = random_uniform(n, self.dim, seed)
        if abs(self.params) < 1e-7:
            return u

        if self.dim == 2:
            v = u[:, 1]
            a = -abs(self.params)
            v = -1 / a * np.log1p(-v * np.expm1(-a) / (np.exp(-a * u[:, 0]) * (v - 1) - v))
            u[:, 1] = 1 - v if self.params > 0 else v
            return u

        # alpha too large
        if log1mexp(self.params) == 0:
            return np.ones((n, self.dim))

        fr = random_log_series_ln1p(-self.params, n)[:, None]
        return self.psi(-np.log(u) / fr)

    @property
    def rho(self):
        return self._rho(self.params)

    def summary(self):
        return Summary(self, {"theta": self.params})

    @property
    def tau(self):
        t = self.params
        if np.isclose(t, 0):
            return t / 9
        return self._tau(self.params)

    @staticmethod
    def _rho(theta):
        if np.isclose(theta, 0):
            return theta / 6
        return 1 + 12 / theta * (debye2(theta) - debye1(theta))

    @staticmethod
    def _tau(theta):
        theta = np.asarray(theta)
        if theta.size == 1:
            theta = float(theta)
        return 1 + 4 * (debye1(theta) - 1) / theta


def debye1(x):
    """
    Custom debye order 1 that takes care of negative numbers or non-finite numbers

    Parameters
    ----------
    x: array_like
        Numeric vector

    Returns
    -------
    ndarray or scalar
        Debye order 1 numbers

    See Also
    --------
    :code:`copulae.special.debye.debye_1`: The debye order 1 function
    """
    x = as_array(x)
    fin = np.isfinite(x)
    d = np.ravel(np.abs(x))

    with np.errstate(invalid='ignore'):
        if np.all(fin):
            d = debye_1(d)
        else:
            d[fin] = debye_1(d[fin])
            d = np.ravel(d)

            pinf = np.isinf(x) & (x > 0)
            if np.any(pinf):
                d[pinf] = 0  # set positive infinity to 0 (but not na, thus can't use ~fin)

        d = np.ravel(d)
        d[x < 0] -= x[x < 0] / 2
        return d.item(0) if d.size == 1 else d


def debye2(x):
    """
    Custom debye order 2 that takes care of negative numbers or non-finite numbers

    Parameters
    ----------
    x: array_like
        Numeric vector

    Returns
    -------
    ndarray or scalar
        Debye order 2 numbers

    See Also
    --------
    :code:`copulae.special.debye.debye_2`: The debye order 2 function
    """
    x = as_array(x)
    fin = np.isfinite(x)
    d = np.ravel(np.abs(x))

    with np.errstate(invalid='ignore'):
        if np.all(fin):
            d = debye_2(d)
        else:
            d[fin] = debye_2(d[fin])
            d = np.ravel(d)

            pinf = np.isposinf(x)
            if np.any(pinf):
                d[pinf] = 0  # set positive infinity to 0 (but not na, thus can't use ~fin)

        d = np.ravel(d)
        d[x < 0] -= 2 / 3 * x[x < 0]
        return d.item(0) if d.size == 1 else d
