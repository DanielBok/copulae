import numpy as np

from copulae.core import pseudo_obs, rank_data
from ._radial_symmetry import rad_sym_test_stat, rad_sym_replicate
from .common import TestStatistic

__all__ = ['rad_sym_test']


def rad_sym_test(x, N=1000, ties='average'):
    r"""
    Test of Radial Symmetry for a Multivariate Copula.

    Test for assessing the radial symmetry of the underlying multivariate copula based on the empirical copula. The
    test statistic is a multivariate extension of the definition adopted in the first reference. An approximate
    p-value for the test statistic is obtained by means of a appropriate bootstrap which can take the presence of
    ties in the component series of the data into account; see the second reference.

    A random vector :math:`X` is called radially symmetric (for d = 1 simply symmetric) about :math:`a \in R^d` if
    :math:`X − a = a − X`, that is, if :math:`X − a` and :math:`a − X` are equal in distribution. In a hand-wavy
    manner, perhaps the consequence of the radial symmetry test is to verify if an elliptical copula should be used
    to fit the data as elliptical copulas are radial symmetric.

    Parameters
    ----------
    x: {array_like, pandas.DataFrame}
        A matrix like data structure

    N: int
        Number of bootstrap iterations to be used to simulate realizations of the test statistic under the null
        hypothesis

    ties: str, optional
        String specifying how ranks should be computed if there are ties in any of the coordinate samples. Options
        include 'average', 'min', 'max', 'dense', 'ordinal'.

    Returns
    -------
    TestStatistic
        Test statistics for the radial symmetry test. The null hypothesis assumes that the vectors are radially
        symmetric. Thus a small p-value will indicate evidence against radial symmetry

    Examples
    --------
    >>> from copulae.datasets import load_danube
    >>> from copulae.gof import rad_sym_test

    >>> danube = load_danube()
    >>> test_stats = rad_sym_test(danube)
    >>> print(test_stats.p_value)

    A small p-value here indicates strong evidence against radial symmetry.

    References
    ----------
    Genest, C. and G. Nešlehová, J. (2014). On tests of radial symmetry for bivariate copulas. Statistical Papers 55,
    1107–1119.

    Kojadinovic, I. (2017). Some copula inference procedures adapted to the presence of ties. Computational Statistics
    and Data Analysis 112, 24–41, http://arxiv.org/abs/1609.05519.
    """
    x = np.asarray(x)

    assert isinstance(N, int) and N >= 1, "number of replications for exchangeability test must be a positive integer"
    assert x.ndim == 2, "input data must be a 2-dimensional matrix"

    n, p = x.shape
    u = pseudo_obs(x, ties)

    s = rad_sym_test_stat(u.ravel('F'), n, p)

    has_ties = False
    for i in range(p):
        if len(np.unique(x[:, i])) != n:
            has_ties = True
            break

    ir = np.floor(rank_data(np.sort(u, 0), axis=1)).astype(int) - 1
    s0 = np.array([rad_sym_replicate(u, ir, n, p, has_ties) for _ in range(N)])

    return TestStatistic(
        s,
        (np.sum(s0 >= s) + 0.5) / (N + 1),
        "Test of radial symmetry based on the empirical copula"
    )
