import numpy as np

from copulae.core import pseudo_obs, rank_data
from ._exchangeability import exch_replication, exch_test_cn, exch_test_stat
from .common import TestStatistic

__all__ = ['exch_test']


def exch_test(x, y, N=1000, m=0, ties='average'):
    r"""
    Test of Exchangeability for a Bivariate Copula.

    Test for assessing the exchangeability of the underlying bivariate copula based on the empirical copula.
    The test statistics are defined in the first two references. Approximate p-values for the test statistics are
    obtained by means of a multiplier technique if there are no ties in the component series of the bivariate
    data, or by means of an appropriate bootstrap otherwise.

    A random vector X is called exchangeable iff :math:`(X1, ..., Xd) = (X_{\pi(1)}, ..., X_{\pi(d)})`
    for any permutation :math:`(\pi(1), \pi(2), \dots, \pi(d))` of :math:`(1, \dots, d)`.

    A copula C is called exchangeable iff C is the distribution function of an exchangeable random vector
    (with uniform marginal distributions on [0, 1]). For such a copula
    :math:`C(u1, u2, ..., ud ) = C(u\pi(1), u\pi(2), ..., u\pi(d))` holds for any permutation
    :math:`(\pi(1), \pi(2), \dots, \pi(d))` of :math:`(1, \dots, d)`.

    Examples of exchangeable copulas:
        Gumbel, Clayton, and also the Gaussian copula :math:`C_P^{Ga}` and the t-Copula :math:`C_{ν,P}^t`, if
        P is an equicorrelation matrix, i.e. :math:`R = \rho J_d + (1 − \rho)I_d`. :math:`J_d \in R^{d×d}`
        is a matrix consisting only of ones, and :math:`I_d \in R^{d×d}` is the d-dimensional identity matrix.

    For bivariate exchangeable copulas we have:

    .. math::

        P(U_2 \leq u_2|U_1 = u_1) = P(U_1 \leq u_2|U_2 = u_1).

    Parameters
    ----------
    x: array_like
        first vector for the exchangeability test

    y: array_like
        second vector for the exchangeability test

    N: int
        Number of multiplier or bootstrap iterations to be used to simulate realizations of the test statistic under
        the null hypothesis.

    m: int
        If m = 0, integration in the Cramér–von Mises statistic is carried out with respect to the empirical copula.
        If m > 0, integration is carried out with respect to the Lebesgue measure and m specifies the size of the
        integration grid.

    ties: str, optional
        String specifying how ranks should be computed if there are ties in any of the coordinate samples. Options
        include 'average', 'min', 'max', 'dense', 'ordinal'.

    Returns
    -------
    TestStatistic
        Test statistics for the exchangeability test. The null hypothesis assumes that the vectors are exchangeable.
        Thus a small p-value will indicate evidence against exchangeability

    Examples
    --------
    >>> from copulae.datasets import load_danube
    >>> from copulae.gof import exch_test

    >>> danube = load_danube().values
    >>> test_stats = exch_test(danube[:, 0], danube[:, 1])
    >>> print(test_stats.p_value)

    A small p-value here indicates strong evidence against exchangeability

    References
    ----------
    Genest, C., G. Nešlehová, J. and Quessy, J.-F. (2012). Tests of symmetry for bivariate copulas. Annals of the
    Institute of Statistical Mathematics 64, 811–834.

    Kojadinovic, I. and Yan, J. (2012). A nonparametric test of exchangeability for extreme-value and left-tail
    decreasing bivariate copulas. The Scandinavian Journal of Statistics 39:3, 480–496.

    Kojadinovic, I. (2017). Some copula inference procedures adapted to the presence of ties. Computational Statistics
    and Data Analysis 112, 24–41, http://arxiv.org/abs/1609.05519.
    """
    x = pseudo_obs(x, ties)
    y = pseudo_obs(y, ties)
    u = np.vstack([x, y]).T

    assert isinstance(m, int) and m >= 0, "size of the integration grid must be an integer >= 0"
    assert x.ndim == 1 and y.ndim == 1, "x and y must be vectors. Exchangeability tests is bivariate"
    assert isinstance(N, int) and N >= 1, "number of replications for exchangeability test must be a positive integer"

    n = len(u)
    if m > 0:
        xis = np.linspace(1 / m, 1 - 1 / m, m)
        g = np.stack([np.tile(xis, m), np.repeat(xis, m)]).T
        ng = m * m
    else:
        g = u
        ng = n

    s = exch_test_stat(u, g, n, ng)

    has_ties = len(np.unique(x)) != n or len(np.unique(y)) != n

    if has_ties:
        ir = np.floor(rank_data(np.sort(u, 0), axis=1)).astype(int) - 1
        s0 = np.asarray([exch_replication(ir, u, g, n, m, ng) for _ in range(N)])

    else:
        s0 = exch_test_cn(u, g, n, ng, N)

    return TestStatistic(
        s,
        (np.sum(s0 >= s) + 0.5) / (N + 1),
        "Test of exchangeability for bivariate copulas"
    )
