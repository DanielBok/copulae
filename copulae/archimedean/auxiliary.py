import numpy as np

from copulae.special import comb, sign_ff


def dsum_sibuya(x, n, alpha: float, method='log', log=False):
    """
    Inner probability mass function for a nested Joe copula, i.e. a Sibuya sum

    Also used in gumbel_coef and gumbel_poly for Gumbel's copula. Methods must be one of log, direct, diff.

    Parameters
    ----------
    x: iterable of int
        Vector of integer values ("quantiles") `x` at which to compute the probability mass or cumulative probability.

    n: ndarray
        The number of summands

    alpha: float
        Parameter in (0, 1]

    method: { 'log', 'direct', 'diff' }, optional
        String specifying computation method. One of 'log', 'direct', 'diff'

    log: bool, optional
        If True, the logarithm of the result is returned

    Returns
    -------
    ndarray
         Vector of probabilities, positive if and only if `x` ≥ `n` and of the same length as `x`
         (or `n` if that is longer).
    """

    try:
        x = np.asarray(x, int)
        n = np.asarray(n, int)

        if x.size == 1:
            x = x.reshape(1)
        if n.size == 1:
            n = n.reshape(1)

    except ValueError:  # pragma: no cover
        raise TypeError("`x` and `n` must be integer vectors or scalars")

    assert np.all(n >= 0), "integers in `n` must >= 0"
    assert 0 < alpha <= 1, "`alpha` must be between (0, 1]"

    len_x, len_n = len(x), len(n)

    len_long = max(len_x, len_n)
    if len_x != len_n:
        if len_x < len_long:
            x = np.repeat(x, len_long)
        else:
            n = np.repeat(n, len_long)

    if alpha == 1:
        return (x == n).astype(float)

    method = method.lower()
    assert method in ('log', 'direct', 'diff')

    s = np.zeros(len_long)
    if method == 'log':
        signs = sign_ff(alpha, np.arange(np.max(n)) + 1, x)

        for i, xx, nn in zip(range(len_long), x, n):
            if xx < nn:
                s[i] = -np.inf
                continue
            js = np.arange(nn)
            lxabs = comb(nn, js + 1, log=True) + comb(alpha * (js + 1), xx, log=True)

            offset = np.max(lxabs)
            sum_ = np.sum(signs[js] * np.exp(lxabs - offset))
            s[i] = np.log(sum_) + offset

        return s if log else np.exp(s)
    elif method == 'direct':
        for i, xx, nn in zip(range(len_long), x, n):
            if xx < nn:
                continue
            js = np.arange(nn) + 1
            s[i] = np.sum(comb(nn, js) * comb(alpha * js, xx) * (-1) ** (xx - js))

        return np.log(s) if log else s

    else:  # diff
        for i, xx, nn in zip(range(len_long), x, n):
            dd = comb(np.array([*range(nn, 0, -1), 0]) * alpha, xx)
            s[i] = np.diff(dd, nn) * (-1) ** xx

        return np.log(s) if log else s
