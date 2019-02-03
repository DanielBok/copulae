from typing import Optional

import numpy as np
import numpy.random as rng

from copulae.core import EPS
from copulae.core import valid_rows_in_u
from copulae.indep.utils import random_uniform
from copulae.types import Array
from .abstract import AbstractArchimedeanCopula


class ClaytonCopula(AbstractArchimedeanCopula):
    def __init__(self, theta=np.nan, dim=2):
        super().__init__(dim, theta, 'clayton')

    def psi(self, s: Array):
        s = np.asarray(s)
        return np.maximum(1 + np.sign(self._theta) * s, np.zeros_like(s)) ** (-1 / self._theta)

    def ipsi(self, u: Array):
        u = np.asarray(u)
        return np.sign(self._theta) * (u ** -self._theta - 1)

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

    @property
    def tau(self):
        return self._theta / (self._theta + 2)

    def itau(self, tau: Array):
        tau = np.asarray(tau)
        return 2 * tau / (1 - tau)

    def dtau(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self._theta
        return 2 / (x + 2) ** 2

    @property
    def rho(self):
        theta = self._theta
        if np.isnan(theta):
            return theta

    def irho(self, rho: Array):
        pass

    def drho(self, x: Optional[np.ndarray] = None):
        pass

    @property
    def __lambda__(self):
        if np.isnan(self._theta):
            return self._theta, self._theta

        return 2 ** (-1 / self._theta) if self._theta > 0 else 0, 0

    def cdf(self, u: Array, log=False) -> np.ndarray:
        u = np.asarray(u)
        theta = self._theta
        if u.ndim == 1:
            if len(u) == self.dim:
                u = u.reshape(-1, self.dim)
            else:
                raise ValueError("input array does not match copula's dimension")
        elif u.ndim != 2:
            raise ValueError("input array must be a vector or matrix")

        n, d = u.shape
        if d != self.dim:
            raise ValueError("input array does not match copula's dimension")
        elif d < 2:
            raise ValueError("input array should at least be bivariate")

        if np.isnan(theta) or np.isinf(theta) or theta < (-1 if self.dim == 2 else 0):
            return np.repeat(np.inf, n) if log else np.zeros(n)

        ok = valid_rows_in_u(u)
        res = np.repeat(np.nan, n)
        if not ok.any():
            return res
        elif theta == 0:
            res[ok] = 0
            return res

        lu = np.log(u).sum(1)
        t = self.ipsi(u).sum(1)

        if theta < 0:  # dim == 2
            pos_t = t < 1
            res = np.log1p(theta) - (1 + theta) * lu - (d + 1 / theta) * np.log1p(-t)
            res[~ok] = np.nan
            res[ok & ~pos_t] = -np.inf
        else:
            p = np.log1p(theta * np.arange(1, d)).sum()
            res = p - (1 + theta) * lu - (d + 1 / theta) * np.log1p(t)

        return res if log else np.exp(res)

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
