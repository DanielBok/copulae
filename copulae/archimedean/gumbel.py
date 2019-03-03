import numpy as np

from copulae.special.special_func import polyn_eval, stirling_second_all, stirling_first_all
from .auxiliary import dsum_sibuya


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
