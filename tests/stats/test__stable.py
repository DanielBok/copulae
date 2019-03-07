from copulae.stats._stable import _pdf_f1
import pytest
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize('x, alpha, beta, zeta, theta0, exp', [
    (0.5, 0.5, 0.5, 0.5, 0.5, 0.446949128627526),
    (0.3, 0.3, 1.6, -0.3, 0.4, 0.25587568222612),
    (0.3, -0.4, 0, -0.3, 0.4, 0),

])
def test__pdf_f1(x, alpha, beta, zeta, theta0, exp):
    y = _pdf_f1(x, alpha, beta, zeta, theta0, zeta_tol=1e-16)
    assert_almost_equal(y, exp, 6)
