from typing import Optional

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

from copulae.types import Numeric
from ._common import _DataMixin

__all__ = ['ClaytonExt']


class ClaytonExt(_DataMixin):
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
        if hasattr(rho, '__iter__'):
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
