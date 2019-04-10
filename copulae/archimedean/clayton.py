import numpy as np
import numpy.random as rng

from copulae.copula import Summary, TailDep
from copulae.core import EPS, valid_rows_in_u
from copulae.stats import random_uniform
from copulae.types import Array
from copulae.utility import array_io
from .abstract import AbstractArchimedeanCopula


class ClaytonCopula(AbstractArchimedeanCopula):
    r"""
    The Clayton copula is a copula that allows any specific non-zero level of (lower) tail dependency between
    individual variables. It is an Archimedean copula and exchangeable. A Clayton copula is defined as

    .. math::

        C_\theta (u_1, \dots, u_d) = \left(\sum_i^d (u_{i}^{-\theta}) - d + 1 \right)^{-1/\theta}
    """

    def __init__(self, theta=np.nan, dim=2):
        """
        Creates a Clayton copula instance

        Parameters
        ----------
        theta: float, optional
            Number specifying the copula parameter

        dim: int, optional
            Dimension of the copula
        """
        super().__init__(dim, theta, 'Clayton')
        assert not (dim != 2 and theta < 0), 'Clayton Copula parameter must be >= 0 when dimension == 2'

        self._bounds = (-1 + EPS) if dim == 2 else 0, np.inf

    @array_io
    def dipsi(self, u, degree=1, log=False):
        assert degree in (1, 2), 'degree can only be 1 or 2'

        s = 1 if log or degree % 2 == 0 else -1
        t = self.params

        if degree == 1:
            a = np.log(t) - (1 + t) * np.log(u)
        else:
            a = np.log(t) + np.log1p(t) - (t + 2) * np.log(u)

        return s * (a if log else np.exp(a))

    @array_io(optional=True)
    def drho(self, x=None):  # pragma: no cover
        # TODO Clayton: add rho derivative function
        # if x is None:
        #     x = self._theta
        # if np.isnan(x):
        #     return np.nan
        # return self._ext.drho(x)
        return NotImplemented

    @array_io(optional=True)
    def dtau(self, theta=None):
        if theta is None:
            theta = self._theta
        return 2 / (theta + 2) ** 2

    @array_io
    def ipsi(self, u: Array, log=False):
        v = np.sign(self._theta) * (u ** -self._theta - 1)
        return np.log(v) if log else v

    @array_io
    def itau(self, tau: Array):
        return 2 * tau / (1 - tau)

    @property
    def lambda_(self):
        if np.isnan(self._theta):
            return TailDep(self._theta, self._theta)

        return TailDep(2 ** (-1 / self._theta) if self._theta > 0 else 0, 0)

    @property
    def params(self):
        return self._theta

    @params.setter
    def params(self, theta: float):
        theta = float(theta)

        if self.dim == 2 and theta < -1:
            raise ValueError('theta must be greater than -1 in 2 dimensional Clayton copula')
        elif self.dim > 2 and theta < 0:
            raise ValueError('theta must be positive when dim > 2')

        self._theta = theta

    @array_io(dim=2)
    def pdf(self, u: Array, log=False):
        assert not np.isnan(self.params), "Copula must have parameters to calculate parameters"

        n, d = u.shape

        theta = self.params
        ok = valid_rows_in_u(u)
        log_pdf = np.repeat(np.nan, n)
        if not ok.any():
            return log_pdf
        elif theta == 0:
            log_pdf[ok] = 0
            return log_pdf

        lu = np.log(u).sum(1)
        t = self.ipsi(u).sum(1)

        if theta < 0:  # dim == 2
            pos_t = t < 1
            log_pdf = np.log1p(theta) - (1 + theta) * lu - (d + 1 / theta) * np.log1p(-t)
            log_pdf[~ok] = np.nan
            log_pdf[ok & ~pos_t] = -np.inf
        else:
            p = np.log1p(theta * np.arange(1, d)).sum()
            log_pdf = p - (1 + theta) * lu - (d + 1 / theta) * np.log1p(t)

        return log_pdf if log else np.exp(log_pdf)

    @array_io
    def psi(self, s: Array):
        return np.maximum(1 + np.sign(self._theta) * s, np.zeros_like(s)) ** (-1 / self._theta)

    def random(self, n: int, seed: int = None) -> np.ndarray:
        theta = self._theta
        if np.isnan(theta):
            raise RuntimeError('Clayton copula parameter cannot be nan')

        if abs(theta) < 1e-7:
            return random_uniform(n, self.dim, seed)

        r = random_uniform(n, self.dim, seed)
        if self.dim == 2:
            r[:, 1] = (r[:, 0] ** (-theta) * (r[:, 1] ** (-theta / (theta + 1)) - 1) + 1) ** (-1 / theta)
            return r
        else:
            gam = rng.standard_gamma(1 / theta, n)[:, None]
            r = -np.log(r) / gam
            return self.psi(r)

    @property
    def rho(self):
        return self._rho(self.params)

    def summary(self):
        return Summary(self, {"theta": self.params})

    @property
    def tau(self):
        return self._tau(self.params)

    @staticmethod
    def _rho(theta):
        # TODO Clayton: add rho function
        return NotImplemented

    @staticmethod
    def _tau(theta):
        return theta / (theta + 2)
