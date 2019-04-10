import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.special import gammaln

from copulae.copula import Summary, TailDep
from copulae.core import valid_rows_in_u
from copulae.special.special_func import polyn_eval, sign_ff, stirling_second_all, stirling_first_all
from copulae.special.trig import cospi2
from copulae.stats import poisson, random_uniform, skew_stable
from copulae.types import Array, Numeric
from copulae.utility import array_io
from .abstract import AbstractArchimedeanCopula
from .auxiliary import dsum_sibuya


class GumbelCopula(AbstractArchimedeanCopula):
    r"""
    The Gumbel copula is a copula that allows any specific level of (upper) tail dependency between individual
    variables. It is an Archimedean copula, and exchangeable. A Gumbel copula is defined as

    .. math::

        C_\theta (u_1, \dots, u_d) = \exp( - (\sum_u^i (-\log u_{i})^{-\theta} )^{1/\theta})
    """

    def __init__(self, theta=np.nan, dim=2):
        """
        Creates a Gumbel copula instance

        Parameters
        ----------
        theta: float, optional
            Number specifying the copula parameter

        dim: int, optional
            Dimension of the copula
        """
        assert theta >= 1 or np.isnan(theta), 'Gumbel Copula parameter must be >= 1'

        super().__init__(dim, theta, 'Gumbel')
        self._bounds = (1.0, np.inf)

    @array_io
    def A(self, w: Numeric):
        r"""
        The Pickands dependence function. This can be seen as the generator function of an
        extreme-value copula.

        A bivariate copula C is an extreme-value copula if and only if

        .. math::

            C(u, v) = (uv)^{A(log(v) / log(uv))}, (u,v) in (0,1]^2 w/o {(1,1)}

        where :math:`A: [0,1] \rightarrow [1/2, 1]` is convex and satisfies
        :math:`max(t,1-t) \leq A(t) \leq 1 \forall t \in [0, 1]`

        Parameters
        ----------
        w: array like
            A numeric scalar or vector

        Returns
        -------
        ndarray
            Array containing values of the dependence function
        """
        bnd = (w == 0) | (w == 1)
        r = (w ** self.params + (1 - w) ** self.params) ** (1 / self.params)
        r[bnd] = 1
        return r

    @array_io
    def dAdu(self, w):
        """
        First and second derivative of A

        Parameters
        ----------
        w: array_like
            A numeric scalar or vector

        Returns
        -------
        ndarray
            Array containing the first and second derivative of the dependence function

        See Also
        --------
        A: Dependence function
        """

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

        if isinstance(grad, float) and isinstance(hess, float):
            return grad, hess
        else:
            res = np.zeros((len(grad), 2))
            res[:, 0] = grad
            res[:, 1] = hess
            return res

    @array_io
    def dipsi(self, u, degree=1, log=False) -> np.ndarray:
        assert degree in (1, 2), 'degree can only be 1 or 2'

        s = 1 if log or degree % 2 == 0 else -1
        lu = np.log(u)
        if degree == 1:
            v = self.params * ((-lu) ** (self.params - 1)) / u
        else:
            v = self.params * (self.params - 1 - lu) * ((-lu) ** (self.params - 2)) / (u ** 2)

        return s * (np.log(v) if log else v)

    def drho(self, x: Optional[np.ndarray] = None):
        # TODO Gumbel: add rho derivative function
        return NotImplemented

    @array_io(optional=True)
    def dtau(self, x=None):
        if x is None:
            x = self.params
        return x ** -2

    @array_io
    def ipsi(self, u: Array, log=False):
        v = (-np.log(u)) ** self.params
        return np.log(v) if log else v

    @array_io
    def itau(self, tau):
        warning_message = "For the Gumbel copula, tau must be >= 0. Replacing negative values by 0."
        if np.size(tau) == 1:
            tau = float(tau)
            if tau < 0:
                warnings.warn(warning_message)
                return 1
        else:
            if np.any(tau < 0):
                warnings.warn(warning_message)
                tau[tau < 0] = 0
        return 1 / (1 - tau)

    def lambda_(self):
        return TailDep(0, 2 * 2 ** (1 / self.params))

    @property
    def params(self):
        return self._theta

    @params.setter
    def params(self, theta: float):
        theta = float(theta)

        if theta < 1:
            raise ValueError('<theta> must be >= 1 for Gumbel copula')

        self._theta = theta

    @array_io(dim=2)
    def pdf(self, u: np.ndarray, log=False):
        assert not np.isnan(self.params), "Copula must have parameters to calculate parameters"

        n, d = u.shape

        theta = self.params
        ok = valid_rows_in_u(u)

        if theta == 1:
            pdf = np.repeat(np.nan, n)
            pdf[ok] = 0 if log else 1
            return pdf

        nlu = -np.log(u)  # negative log u
        lnlu = np.log(nlu)  # log negative log u
        lip = self.ipsi(u, log=True)  # log ipsi u

        # get sum of logs
        if u.ndim == 1:
            offset = u.max()
            ln = offset + np.log(np.exp(lip - offset).sum(1))
        else:
            offset = u.max(1)
            ln = offset + np.log(np.exp(lip - offset[:, None]).sum(1))

        alpha = 1 / self.params
        lx = alpha * ln

        ls = gumbel_poly(lx, alpha, d, log=True) - d * lx / alpha
        lnc = -np.exp(lx)
        log_pdf = lnc + d * np.log(theta) + ((theta - 1) * lnlu + nlu).sum(1) + ls

        return log_pdf if log else np.exp(log_pdf)

    @array_io
    def psi(self, s: Array) -> np.ndarray:
        return np.exp(-s ** (1 / self.params))

    def random(self, n: int, seed: int = None):
        if np.isnan(self.params):
            raise RuntimeError('Clayton copula parameter cannot be nan')

        u = random_uniform(n, self.dim, seed)
        if self.params - 1 < 1e-7:
            return random_uniform(n, self.dim, seed)

        if np.isclose(self.params, 1):
            return u

        alpha = 1 / self.params
        fr = skew_stable.rvs(alpha, beta=1, gamma=cospi2(alpha) ** self.params, pm=1, size=n)

        return self.psi(-np.log(u) / fr[:, None])

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
        # TODO Gumbel: add rho function
        return NotImplemented

    @staticmethod
    def _tau(theta):
        return 1. - 1 / theta


def gumbel_coef(d: int, alpha: float, method='sort', log=False) -> np.ndarray:
    """
    Coefficients of Polynomial used for Gumbel Copula

    Compute the coefficients a[d,k](θ) involved in the generator (psi) derivatives and the copula density of Gumbel
    copulas.

    Parameters
    ----------
    d: int
        Dimension of the Gumbel copula

    alpha: float
        Inverse of the theta parameter (that describes the Gumbel copula)

    method: { 'sort', 'horner', 'direct', 'log', 'ds.direct', 'diff' }, optional
        String specifying computation method to compute coefficient

    log: bool, optional
        If True, the logarithm of the result is returned

    Returns
    -------
    ndarray
        The coefficients of the polynomial
    """
    assert (0 < alpha <= 1), "`alpha` used in calculating the gumbel polynomial must be (0, 1]"
    assert isinstance(d, int) and d >= 1, "dimension of copula must be an integer and >= 1"

    method = method.lower()
    assert method in ('sort', 'horner', 'direct', 'log', 'ds.direct', 'diff'), \
        "Method must be one of 'sort', 'horner', 'direct', 'log', 'ds.direct', 'diff'"

    if method == 'sort':
        ls = np.log(np.abs(stirling_first_all(d)))
        lS = [np.log(stirling_second_all(i + 1)) for i in range(d)]

        a = np.zeros(d)
        for i in range(d):
            ds = np.arange(i, d)
            b = (ds + 1) * np.log(alpha) + ls[ds] + np.asarray([lS[x][i] for x in ds])
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
        method = 'direct' if method.startswith('ds.') else method
        ds = np.arange(d) + 1
        ck = np.array([1, *np.cumprod(np.arange(d, 1, -1))])[::-1]
        if log:
            ck = np.log(ck)
        p = dsum_sibuya(d, ds, alpha, method, log)

        return p + ck if log else p * ck


def gumbel_poly(log_x: np.ndarray, alpha: float, d: int, method='default', log=False):
    """
    Compute the polynomial involved in the generator derivatives and the copula density of a Gumbel copula

    Parameters
    ----------
    log_x: ndarray
        1d vector, log of `x`

    alpha: float
        Inverse of the theta parameter (that describes the Gumbel copula)

    d: int
        Dimension of the Gumbel copula

    method: { 'default', 'pois', 'direct', 'log' }, optional
        String specifying computation method to compute polynomial. If set to 'default', an algorithm will
        automatically determine best method to use

    log: bool, optional
        If True, the logarithm of the result is returned

    Returns
    -------
    ndarray
        polynomials of the Gumbel copula
    """
    assert 0 < alpha <= 1, "`alpha` used in calculating the gumbel polynomial must be (0, 1]"
    assert isinstance(d, int) and d >= 1, "dimension of copula must be an integer and >= 1"

    log_x = np.asarray(log_x)
    shape = log_x.shape
    log_x = np.ravel(log_x)

    method = method.lower()
    if method == 'default':
        _methods = defaultdict(list)
        for i, lx in enumerate(log_x):
            _methods[_get_poly_method(lx, alpha, d)].append(i)

        res = np.repeat(np.nan, len(log_x))
        for meth, indices in _methods.items():
            res[indices] = _calculate_gumbel_poly(log_x[indices], alpha, d, meth, log)

    else:
        res = _calculate_gumbel_poly(log_x, alpha, d, method, log)

    return res.reshape(shape)


def _calculate_gumbel_poly(lx: np.ndarray, alpha: float, d: int, method: str, log: bool):
    """Inner function that does the actual Gumbel polynomial calculation"""
    k = np.arange(d) + 1

    method = method.lower()
    assert method in ('pois', 'direct', 'log', 'sort'), "Method must be one of 'pois', 'direct', 'log', 'sort'"

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
    else:
        log_a_dk = gumbel_coef(d, alpha, method, True)

        log_x = log_a_dk[:, None] + k.reshape(-1, 1) @ lx.reshape(1, -1)
        x = np.exp(log_x).sum(0)
        return np.log(x) if log else x


def _get_poly_method(lx: float, alpha: float, d: int):  # pragma: no cover
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
