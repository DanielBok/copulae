import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.special import gammaln

from copulae.core import valid_rows_in_u
from copulae.special.special_func import polyn_eval, sign_ff, stirling_second_all, stirling_first_all
from copulae.stats import poisson
from copulae.types import Array, Numeric
from copulae.utility.utils import reshape_data, reshape_output
from ._data_ext import _Ext
from .abstract import AbstractArchimedeanCopula
from .auxiliary import dsum_sibuya


class GumbelCopula(AbstractArchimedeanCopula):
    def __init__(self, theta=np.nan, dim=2):
        super().__init__(dim, theta, 'clayton')
        self._ext = GumbelExt(self)

    @reshape_output
    def A(self, w: Numeric):
        bnd = (w == 0) | (w == 1)
        r = (w ** self.params + (1 - w) ** self.params) ** (1 / self.params)
        r[bnd] = 1
        return r

    @reshape_output
    def dAdu(self, w: Numeric):

        alpha = self.params

        expr1 = 1 - w
        expr2 = w ** alpha + expr1 ** alpha
        expr3 = 1 / alpha
        expr4 = expr3 - 1
        expr5 = expr2 ** expr4
        expr6 = alpha - 1
        expr7 = alpha * w ** expr6 - alpha * expr1 ^ expr6
        expr8 = expr3 * expr7
        expr9 = expr6 - 1
        # value = expr4 ** expr5

        grad = expr5 * expr8
        hess = expr4 * expr7 * expr8 * expr2 ** (expr4 - 1) + \
               alpha * expr3 * expr5 * expr6 * (w ** expr9 + expr1 ** expr9)

        if type(grad) is float and type(hess) is float:
            return grad, hess
        else:
            res = np.zeros((len(grad), 2))
            res[:, 0] = grad
            res[:, 1] = hess
            return res

    @reshape_output
    def dipsi(self, u, degree=1, log=False) -> np.ndarray:
        s = 1 if log or degree % 2 == 0 else -1
        lu = np.log(u)
        if degree == 1:
            v = self.params * ((-lu) ** (self.params - 1)) / u
            if log:
                v = np.log(v)
        elif degree == 2:
            v = self.params * (self.params - 1 - lu) * ((-lu) ** (self.params - 2)) / (u ** 2)
            if log:
                v = np.log(v)
        else:
            raise NotImplementedError('have not implemented absdiPsi for degree > 2')

        return s * v

    def drho(self, x: Optional[np.ndarray] = None):
        return NotImplemented

    @reshape_data
    def dtau(self, x: Optional[np.ndarray] = None):
        return self.params ** -2

    def irho(self, rho: Array):
        return NotImplemented

    @reshape_output
    def ipsi(self, u: Array, log=False):
        v = (-np.log(u)) ** self.params
        return np.log(v) if log else v

    @reshape_output
    def itau(self, tau: Array):
        warning_message = "For the Gumbel copula, tau must be >= 0. Replacing negative values by 0."
        if hasattr(tau, '__iter__'):
            tau = np.asarray(tau)
            neg = tau < 0
            if np.any(neg):
                warnings.warn(warning_message)
                tau[neg] = 0
        else:
            if tau < 0:
                warnings.warn(warning_message)
                tau = 0.0

        return 1 / (1 - tau)

    @reshape_data
    def pdf(self, u: Array, log=False):
        n, d = u.shape
        if d != self.dim:
            raise ValueError("input array does not match copula's dimension")
        elif d < 2:
            raise ValueError("input array should at least be bivariate")

        theta = self.params
        ok = valid_rows_in_u(u)
        pdf = np.repeat(np.nan, n)

        if theta == 1:
            pdf[ok] = 0 if log else 1
            return pdf

        mlu = -np.log(u)
        lmlu = np.log(u)
        ip = self.ipsi(u)
        ln = np.log(np.exp(ip).sum(1))

        alpha = 1 / self.params
        lx = alpha * ln

    @reshape_output
    def psi(self, s: Array) -> np.ndarray:
        return np.exp(-s ** (1 / self.params))

    def random(self, n: int, seed: int = None):
        pass

    @property
    def rho(self):
        return NotImplemented

    def summary(self):
        return NotImplemented

    @property
    def tau(self):
        return 1 - 1 / self.params

    @property
    def __lambda__(self):
        return 0, 2 - 2 ** (1 / self.params)


class GumbelExt(_Ext):
    def __init__(self, copula, seed: Optional[int] = None):
        super().__init__(copula, 10, seed)


def gumbel_coef(d: int, alpha: float, method='sort', log=False) -> np.ndarray:
    """
    Coefficients of Polynomial used for Gumbel Copula

    Compute the coefficients a[d,k](θ) involved in the generator (psi) derivatives and the copula density of Gumbel
    copulas.

    :param d: int
        the dimension of the Gumbel copula
    :param alpha: float
        the inverse of the theta parameter
    :param method: str, default 'sort'
        string specifying computation method. One of sort, horner, direct, log, ds.direct, diff,
    :param log: boolean, default False
        If True, the logarithm of the result is returned
    :return: ndarray
        the coefficients of the polynomial
    """
    if not (0 < alpha <= 1):
        raise ValueError("<alpha> used in calculating the gumbel polynomial must be (0, 1]")

    if type(d) is not int or d < 1:
        raise ValueError("dimension of copula must be an integer and >= 1")

    method = method.lower()
    if method == 'sort':
        ls = np.log(np.abs(stirling_first_all(d)))
        lS = [np.log(stirling_second_all(i + 1)) for i in range(d)]

        a = np.zeros(d)
        for i in range(d):
            ds = np.arange(i, d)
            b = (ds + 1) * np.log(alpha) + ls[ds] + [lS[x][i] for x in ds]
            exponents = np.exp(b - b.max())

            # sum odd components of exponents first
            sum_ = exponents[::2].sum() - exponents[1::2].sum()
            a[i] = np.log(sum_) + b.max() if log else np.exp(b.max()) * sum_
        return a

    elif method == 'horner':
        s = np.abs(stirling_first_all(d))
        ds = np.arange(d)
        S = [stirling_second_all(i + 1) for i in ds]

        pol = np.repeat(np.nan, d)
        for i in ds:
            js = np.arange(i, d)
            c_j = s[js] * [S[j][i] for j in js]
            pol[i] = polyn_eval(c_j, -alpha)

        return (ds + 1) * np.log(alpha) + np.log(pol) if log else pol * alpha ** (ds + 1)
    elif method == 'direct':
        s = np.asarray(stirling_first_all(d))
        ds = np.arange(d)
        S = [stirling_second_all(i + 1) for i in ds]

        a = np.zeros(d)
        for i in ds:
            js = np.arange(i, d)
            S_ = [S[j][i] for j in js]
            sum_ = np.sum(alpha ** (js + 1) * s[js] * S_)
            a[i] = np.log(abs(sum_)) if log else (-1) ** (d - i + 1) * sum_

        return a
    else:
        if method in ('log', 'ds.direct', 'diff'):
            method = 'direct' if method.startswith('ds.') else method
            ds = np.arange(d) + 1
            ck = np.array([1, *np.cumprod(np.arange(d, 1, -1))])[::-1]
            if log:
                ck = np.log(ck)
            p = dsum_sibuya(d, ds, alpha, method, log)

            return p + ck if log else p * ck
        else:
            raise ValueError(f"Unknown method: '{method}'. Use one of sort, horner, direct, log, ds.direct, diff")


def gumbel_poly(log_x: np.ndarray, alpha: float, d: int, method='default', log=False):
    """
    Compute the polynomial involved in the generator derivatives and the copula density of a Gumbel copula

    :param log_x: ndarray
        1d vector, log of x
    :param alpha: float
        the inverse of the theta parameter
    :param d: int
        the dimension of the Gumbel copula
    :param method: str, default 'default'
        A string which determines the method used to calculate the polynomial. Must be one default, pois, direct, log,
        sort. If set to 'default', an algorithm will automatically determine best method to use
    :param log: boolean, default False
        If True, the logarithm of the result is returned
    :return:
    """

    if not (0 < alpha <= 1):
        raise ValueError("<alpha> used in calculating the gumbel polynomial must be (0, 1]")

    if type(d) is not int or d < 1:
        raise ValueError("dimension of copula must be an integer and >= 1")

    log_x = np.ravel(log_x)
    method = method.lower()
    if method == 'default':
        _methods = defaultdict(list)
        for i, lx in enumerate(log_x):
            _methods[_get_poly_method(lx, alpha, d)].append(i)

        res = np.repeat(np.nan, len(log_x))
        for meth, indices in _methods.items():
            res[indices] = _calculate_gumbel_poly(log_x[indices], alpha, d, meth, log)

        return res

    return _calculate_gumbel_poly(log_x, alpha, d, method, log)


def _calculate_gumbel_poly(lx: np.ndarray, alpha: float, d: int, method: str, log: bool):
    """Inner function that does the actual Gumbel polynomial calculation"""
    k = np.arange(d) + 1

    if method == 'pois':
        n = len(lx)
        x = np.exp(lx)  # n x 1 vector

        lppois = np.array([poisson.logcdf(d - k, xx) for xx in x]).T  # d x n matrix
        llx = k.reshape(-1, 1) @ lx.reshape(1, -1)  # d x n matrix
        labs_poch = np.array([np.sum(np.log(np.abs(alpha * j - (k - 1)))) for j in k])
        lfac = gammaln(k + 1)  # d x 1 vector

        lxabs = llx + lppois + np.tile(labs_poch - lfac, (n, 1)).T + np.tile(x, (d, 1))

        signs = sign_ff(alpha, k, d)
        offset = np.max(lxabs, 0)
        sum_ = np.sum(signs[:, None] * np.exp(lxabs - offset[None, :]), 0)
        res = np.log(sum_) + offset

        return res if log else np.exp(res)
    elif method in ('direct', 'log', 'sort'):
        log_a_dk = gumbel_coef(d, alpha, method, True)

        log_x = log_a_dk[:, None] + k.reshape(-1, 1) @ lx.reshape(1, -1)
        x = np.exp(log_x).sum(0)
        return np.log(x) if log else x
    else:
        raise ValueError(f"Unknown <method>: {method}. Use one of pois, direct, log, sort")


def _get_poly_method(lx: float, alpha: float, d: int):
    """Determines the method to apply for for each log x argument to gumbel_poly"""

    if d <= 30:
        return 'direct'
    elif d <= 50:
        return 'direct' if alpha <= 0.8 else 'log'
    elif d <= 70:
        return 'direct' if alpha <= 0.7 else 'log'
    elif d <= 90:
        if d <= 0.5:
            return 'direct'
        elif lx <= 4.08:
            return 'pois'
        return 'log'
    elif d <= 120:
        if alpha < 0.003:
            return 'sort'
        elif alpha <= 0.4:
            return 'direct'
        elif lx <= 3.55:
            return 'pois'
        elif lx >= 5.92:
            return 'direct'
        return 'log'
    elif d <= 170:
        if alpha < 0.01:
            return 'sort'
        elif alpha <= 0.3:
            return 'direct'
        elif lx <= 3.55:
            return 'pois'
        return 'log'
    elif d <= 200:
        return 'pois' if lx <= 2.56 else 'log'
    else:
        return 'log'
