import numpy as np
import numpy.random as rng

from copulae.copula.extensions import ClaytonExt
from copulae.core import EPS, valid_rows_in_u
from copulae.indep.utils import random_uniform
from copulae.types import Array, Numeric, OptNumeric
from copulae.utils import reshape_data
from .abstract import AbstractArchimedeanCopula


class ClaytonCopula(AbstractArchimedeanCopula):
    def __init__(self, theta=np.nan, dim=2):
        super().__init__(dim, theta, 'clayton')
        self._ext = ClaytonExt(self)

    @reshape_data
    def cdf(self, u: Array, log=False) -> np.ndarray:
        return self.psi(self.ipsi(u).sum(1))

    @reshape_data
    def dipsi(self, u, degree=1, log=False):
        s = 1 if log or degree % 2 == 0 else -1
        t = self._theta

        if t < 0:
            raise NotImplementedError('have not implemented dipsi for theta < 0')

        if degree == 1:
            if log:
                a = np.log(t) - (1 - t) * np.log(u)
            else:
                a = t * u ** (- (1 + t))
        elif degree == 2:
            if log:
                a = np.log(t) - (1 + t) * np.log(u)
            else:
                a = t * (1 + t) * u ** (-t - 2)
        else:
            raise NotImplementedError('have not implemented absdiPsi for degree > 2')

        return s * a

    def dtau(self, x: OptNumeric = None):
        if x is None:
            x = self._theta
        return 2 / (x + 2) ** 2

    def drho(self, x: OptNumeric = None):
        if x is None:
            x = self._theta
        if np.isnan(x):
            return np.nan
        return self._ext.drho(x)

    def irho(self, rho: Numeric):
        return self._ext.irho(rho)

    @reshape_data
    def ipsi(self, u: Array):
        u = np.asarray(u)
        return np.sign(self._theta) * (u ** -self._theta - 1)

    @reshape_data
    def itau(self, tau: Array):
        tau = np.asarray(tau)
        return 2 * tau / (1 - tau)

    @property
    def params(self):
        return self._theta

    @params.setter
    def params(self, theta: float):
        theta = float(theta)

        if self.dim == 2 and theta < -1:
            raise ValueError('theta must be greater than -1 in 2 dimensional clayton copulas')
        elif self.dim > 2 and theta < 0:
            raise ValueError('theta must be positive when dim > 2')

        self._theta = theta

    @reshape_data
    def pdf(self, x: Array, log=False):

        n, d = x.shape
        if d != self.dim:
            raise ValueError("input array does not match copula's dimension")
        elif d < 2:
            raise ValueError("input array should at least be bivariate")

        theta = self.params
        ok = valid_rows_in_u(x)
        log_pdf = np.repeat(np.nan, n)
        if not ok.any():
            return log_pdf
        elif theta == 0:
            log_pdf[ok] = 0
            return log_pdf

        lu = np.log(x).sum(1)
        t = self.ipsi(x).sum(1)

        if theta < 0:  # dim == 2
            pos_t = t < 1
            log_pdf = np.log1p(theta) - (1 + theta) * lu - (d + 1 / theta) * np.log1p(-t)
            log_pdf[~ok] = np.nan
            log_pdf[ok & ~pos_t] = -np.inf
        else:
            p = np.log1p(theta * np.arange(1, d)).sum()
            log_pdf = p - (1 + theta) * lu - (d + 1 / theta) * np.log1p(t)

        return log_pdf if log else np.exp(log_pdf)

    def psi(self, s: Array):
        s = np.asarray(s)
        return np.maximum(1 + np.sign(self._theta) * s, np.zeros_like(s)) ** (-1 / self._theta)

    def random(self, n: int, seed: int = None) -> np.ndarray:
        theta = self._theta
        if np.isnan(theta):
            raise RuntimeError('theta cannot be nan')

        if abs(theta) < EPS:
            return random_uniform(n, self.dim, seed)

        r = random_uniform(n, self.dim, seed)
        if self.dim == 2:
            r[:, 1] = (r[:, 0] ** (-theta) * (r[:, 1] ** (-theta / (theta + 1)) - 1) + 1) ** (-1 / theta)
            return r
        else:
            gam = rng.standard_gamma(1 / theta, n)[:, None]
            r = -np.log(r) / gam
            return self.psi(r)

    def summary(self):
        pass

    @property
    def rho(self):
        if np.isnan(self._theta):
            return np.nan
        return self._ext.rho(self._theta)

    @property
    def tau(self):
        return self._theta / (self._theta + 2)

    @property
    def __lambda__(self):
        if np.isnan(self._theta):
            return self._theta, self._theta

        return 2 ** (-1 / self._theta) if self._theta > 0 else 0, 0
