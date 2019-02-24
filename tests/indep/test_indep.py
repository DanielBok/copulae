import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from copulae import IndepCopula

DP = 3


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = IndepCopula(dim)
    cop.fit(residual_data)
    return cop


def test_copula_pdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    log_pdf = copula.pdf(U, log=True)
    expected_log_pdf = np.zeros(5)
    assert_array_almost_equal(log_pdf, expected_log_pdf, DP)

    pdf = copula.pdf(U)
    assert_array_almost_equal(pdf, np.exp(expected_log_pdf), DP)


def test_copula_cdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    log_cdf = copula.cdf(U, log=True)
    expected_log_cdf = -2.478547, -9.366887, -8.695993, -5.701305, -1.599828
    assert_array_almost_equal(log_cdf, expected_log_cdf, DP)

    cdf = copula.cdf(U)
    assert_array_almost_equal(cdf, np.exp(expected_log_cdf), DP)


def test_copula_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)
