import numpy as np

from copulae.copula import TailDep
from copulae.core.roots import find_root
from copulae.special import log1mexp, log1pexp
from copulae.special.debye import debye_1, debye_2
from copulae.types import Array
from copulae.utility import array_io
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
        super().__init__(dim, theta, 'gumbel')
        assert not (dim != 2 and theta < 0), 'Frank Copula parameter must be >= 0 when dimension == 2'

        self._ext = None
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
        return 12 * (x / np.expm1(x) - 3 * debye_2(x) + 2 * debye_1(x)) / x ** 2

    @array_io(optional=True)
    def dtau(self, x=None):  # pragma: no cover
        if x is None:
            x = self.params
        return (x / np.expm1(x) + 1 - debye_1(x) / x) * (2 / x) ** 2

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
                                  1e-8 if t > 0 else -1e20,
                                  1e20 if t > 0 else -1e-8) for t in tau.ravel()])
        res = res.reshape(tau.shape)
        res[tau == 0] = tau[tau == 0]
        return res

    @property
    def lambda_(self):  # pragma: no cover
        return TailDep(0, 0)

    @property
    def params(self):
        return self._theta

    def pdf(self, x: Array, log=False):
        pass

    def psi(self, s):
        assert not np.isnan(self.params), "Copula must have parameters to calculate psi"

        if self.params <= -36:
            return -log1pexp(-s - self.params) / self.params
        elif self.params < 0:
            return -np.log1p(np.exp(-s) * np.expm1(-self.params)) / self.params
        elif self.params == 0:
            return np.exp(-s)
        else:
            const = log1mexp(self.params)
            if s < const:
                return np.nan
            return -log1mexp(s - log1mexp(self.params)) / self.params

    def random(self, n: int, seed: int = None):
        pass

    @property
    def rho(self):
        t = self.params
        if np.isclose(t, 0):
            return t / 6
        return 1 + 12 / t * (debye_2(t) - debye_1(t))

    def summary(self):
        # TODO Summary
        return NotImplemented

    @property
    def tau(self):
        t = self.params
        if np.isclose(t, 0):
            return t / 9
        return self._tau(self.params)

    def _tau(self, theta):
        theta = np.asarray(theta)
        if theta.size == 1:
            theta = float(theta)
        return 1 + 4 * (debye_1(theta) - 1) / theta
