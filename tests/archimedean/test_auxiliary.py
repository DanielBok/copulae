import pytest
from numpy.testing import assert_almost_equal

from copulae.archimedean.auxiliary import dsum_sibuya


@pytest.mark.parametrize("x, n, alpha, expected", [
    (5, [1, 2, 3, 4, 5], 0.78, [0.01247110, 0.03619696, 0.08091112, 0.16286625, 0.28871744]),
    ([46, 8, 29, 13, 2, 25], 1, 0.4, [0.001270308, 0.015155712, 0.002432143, 0.007571182, 0.12, 0.002998585]),
    ([46, 8, 29, 13, 2, 25], 3, 0.4, [0.003224508, 0.029309952, 0.005915379, 0.016429824, 0.0, 0.007173205]),
])
def test_dsum_sibuya(x, n, alpha, expected):
    for method in ('log', 'diff', 'direct'):
        assert_almost_equal(dsum_sibuya(x, n, alpha, method), expected, 5)
