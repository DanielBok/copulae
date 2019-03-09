from collections import abc
from typing import Optional

import numpy as np
import numpy.random as rng
from scipy.interpolate import UnivariateSpline, interp1d

from copulae.copula import TailDep
from copulae.core import EPS, valid_rows_in_u
from copulae.stats import random_uniform
from copulae.types import Array, Numeric, OptNumeric
from copulae.utility import reshape_data, reshape_output
from ._data_ext import _Ext
from .abstract import AbstractArchimedeanCopula


class ClaytonCopula(AbstractArchimedeanCopula):
    def __init__(self, theta=np.nan, dim=2):
        """
        The Clayton copula is a copula that allows any specific non-zero level of (lower) tail dependency between
        individual variables. It is an Archimedean copula and exchangeable. A Clayton copula is defined as

        .. math::

            C_\\theta (u_1, \dots, u_d) = \left(\sum_i^d (u_{i}^{-\\theta}) - d + 1 \\right)^{-1/\\theta}

        Parameters
        ----------
        theta: float, optional
            Number specifying the copula parameter

        dim: int, optional
            Dimension of the copula
        """
        super().__init__(dim, theta, 'clayton')
        self._ext = ClaytonExt(self)
        # TODO ADD BOUNDS
        self._bounds = (-1 + EPS, np.inf) if dim == 2 else (0, np.inf)

    def A(self, w: Numeric):
        return NotImplemented

    def dAdu(self, w: Numeric):
        return NotImplemented

    @reshape_output
    def dipsi(self, u, degree=1, log=False):
        s = 1 if log or degree % 2 == 0 else -1
        t = self._theta

        if t < 0:
            raise NotImplementedError('have not implemented dipsi for theta < 0')

        if degree == 1:
            if log:
                a = np.log(t) - (1 + t) * np.log(u)
            else:
                a = t * u ** (- (1 + t))
        elif degree == 2:
            if log:
                a = np.log(t) + np.log1p(t) - (t + 2) * np.log(u)
            else:
                a = t * (1 + t) * u ** (-t - 2)
        else:
            raise NotImplementedError('have not implemented absdiPsi for degree > 2')

        return s * a

    def drho(self, x: OptNumeric = None):
        # TODO Clayton: add rho derivative function
        # if x is None:
        #     x = self._theta
        # if np.isnan(x):
        #     return np.nan
        # return self._ext.drho(x)
        return NotImplemented

    @reshape_output
    def dtau(self, x: OptNumeric = None):
        if x is None:
            x = self._theta
        return 2 / (x + 2) ** 2

    @reshape_output
    def ipsi(self, u: Array, log=False):
        u = np.asarray(u)
        v = np.sign(self._theta) * (u ** -self._theta - 1)
        return np.log(v) if log else v

    def irho(self, rho: Numeric):
        # TODO Clayton: add inverse rho function
        return NotImplemented
        # return self._ext.irho(rho)

    @reshape_output
    def itau(self, tau: Array):
        tau = np.asarray(tau)
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
            raise ValueError('theta must be greater than -1 in 2 dimensional clayton copulae')
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

    @reshape_output
    def psi(self, s: Array):
        s = np.asarray(s)
        return np.maximum(1 + np.sign(self._theta) * s, np.zeros_like(s)) ** (-1 / self._theta)

    def random(self, n: int, seed: int = None) -> np.ndarray:
        theta = self._theta
        if np.isnan(theta):
            raise RuntimeError('Clayton copula parameter cannot be nan')

        if abs(theta) < 6.06e-6:  # magic number
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
        # TODO Clayton: add rho function
        # if np.isnan(self._theta):
        #     return np.nan
        # return self._ext.rho(self._theta)
        return NotImplemented

    def summary(self):
        # TODO Clayton: add summary
        return NotImplemented

    @property
    def tau(self):
        return self._theta / (self._theta + 2)


# TODO Clayton: write extension
class ClaytonExt(_Ext):
    """
    Clayton Extension class is used to derive values that have no analytical solutions. The values are derived
    numerically and stored in an interpolation object that is called whenever we need to call those values
    """

    def __init__(self, copula, seed: Optional[int] = None):
        super().__init__(copula, 10, seed)
        pos_rho_func_name = 'clayton_pos_rho'
        neg_rho_func_name = 'clayton_neg_rho'
        pos_irho_func_name = 'clayton_pos_irho'
        neg_irho_func_name = 'clayton_neg_irho'

        if self.file_exists:
            self._pos_rho: UnivariateSpline = self.load_copula_data(pos_rho_func_name)
            self._neg_rho: UnivariateSpline = self.load_copula_data(neg_rho_func_name)
            self._neg_irho = self.load_copula_data(neg_irho_func_name)
            self._pos_irho = self.load_copula_data(pos_irho_func_name)
        else:
            self.form_interpolator('spearman')

            self.save_copula_data(pos_rho_func_name, self._pos_rho)
            self.save_copula_data(neg_rho_func_name, self._neg_rho)
            self.save_copula_data(pos_irho_func_name, self._pos_irho)
            self.save_copula_data(neg_irho_func_name, self._neg_irho)

    def drho(self, alpha: Numeric):
        alpha = np.asarray(alpha, np.float)
        theta = self._forward_transfer(alpha)
        rhos = np.where(alpha <= 0,
                        self._neg_rho(theta),
                        self._pos_rho(theta) * self._forward_derivative(alpha))

        return float(rhos) if rhos.ndim == 0 else rhos

    def form_interpolator(self, method: str, df: int = 5, s=1.1, **kwargs):
        neg_param, neg_val = [-1, 0], [-1, 0]
        pos_param, pos_val = [0, 1], [0, 1]

        theta_grid_neg = np.arange(-0.999, 0, 0.001)
        theta_grid_pos = np.arange(0.001, 1, 0.001)

        self._neg_rho = super().form_interpolator(theta_grid_neg, method, pos_param, pos_val, False, df, s)
        self._pos_rho = super().form_interpolator(theta_grid_pos, method, neg_param, pos_val, False, df, s)

        # forms the reverse interpolator
        x_neg = [neg_param[0], *theta_grid_neg, neg_param[1]]
        x_pos = [pos_param[0], *theta_grid_pos, pos_param[1]]
        y_neg_smth = [neg_val[0], *self._neg_rho(theta_grid_neg), neg_val[1]]
        y_pos_smth = [pos_val[0], *self._pos_rho(theta_grid_pos), pos_val[1]]
        self._neg_irho = interp1d(y_neg_smth, x_neg)
        self._pos_irho = interp1d(y_pos_smth, x_pos)

    def irho(self, rho: Numeric):
        if isinstance(rho, abc.Iterable):
            rho = np.asarray(rho)
            shape = rho.shape

            res = np.ones(np.prod(shape))
            for i, e in enumerate(rho.ravel()):
                if -1 <= e < 0:
                    res[i] = self._neg_irho(e)
                elif 0 <= e <= 1:
                    res[i] = self._pos_irho(e)
                else:
                    res[i] = np.nan

            return res.reshape(shape)

        else:
            if -1 <= rho < 0:
                return self._neg_irho(rho)
            elif 0 <= rho <= 1:
                return self._pos_irho(rho)
            return np.nan

    def rho(self, alpha: Numeric):
        alpha = np.asarray(alpha, np.float)
        theta = self._forward_transfer(alpha)
        rhos = np.where(alpha <= 0, self._neg_rho(theta), self._pos_rho(theta))

        return float(rhos) if rhos.ndim == 0 else rhos

    def _forward_transfer(self, x: np.ndarray):
        return np.where(x <= 0, x, np.tanh(x / self.ss))

    def _backward_transfer(self, x: np.ndarray):
        return np.where(x <= 0, x, self.ss * np.arctanh(x))

    def _forward_derivative(self, x: np.ndarray):
        return np.where(x <= 0, x, (1 - np.tanh(x / self.ss) ** 2) / self.ss)

    def _set_param(self, alpha: float):
        self.copula.params = alpha
