import numpy as np

from copulae.types import Numeric


def random_log_series_ln1p(alpha: float, size: Numeric = 1):
    """"""
    if alpha < -3:
        # random variate following the logarithmic (series) distribution Log(alpha)
        p = - alpha / np.log(1 - np.expm1(alpha))
        u = np.random.uniform(size=size)

        up = u > p
        x = 1
        while np.any(up):
            u[up] -= p
            x += 1
            p *= alpha * (x - 1) / x

            up = u > p
        return u
    else:
        # random variate following the logarithmic (series) distribution Log(1 - e^h)
        e = -np.expm1(alpha)
        u1 = np.random.uniform(size=size)
        u2 = np.random.uniform(size=size)

        h = u2 * alpha
        q = -np.expm1(h)
        log_q = np.where(h > -0.69314718055994530942, np.log(-np.expm1(h)), np.log1p(-np.exp(h)))

        mask1 = u1 > e
        mask2 = u1 < q * q
        mask3 = mask2 & (log_q == 0)
        mask4 = mask2 & (log_q != 0)
        mask5 = ~mask2 & (u1 > q)
        mask6 = ~mask2 & (u1 <= q)

        u1[mask6] = 2
        with np.errstate(invalid='ignore'):
            u1[mask4] = 1 + np.floor(np.log(u1 / log_q))[mask4]
        u1[mask3] = np.inf
        u1[mask1 | mask5] = 1

        return u1
