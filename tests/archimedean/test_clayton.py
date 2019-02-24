import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from copulae import ClaytonCopula

DP = 3


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = ClaytonCopula(dim=dim)
    cop.params = 0.3075057
    return cop


def test_fitted_parameters_match_target(residual_data: np.ndarray):
    target_param = 0.3075057
    cop = ClaytonCopula(dim=residual_data.shape[1])
    cop.fit(residual_data)
    assert np.allclose(cop.params, target_param, atol=1e-3)


def test_fitted_log_likelihood_match_target(copula, residual_data):
    target_log_lik = 147.4181
    U = copula.pobs(residual_data)
    log_lik = copula.log_lik(U)

    assert np.isclose(log_lik, target_log_lik, atol=1e-2)


def test_copula_pdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]

    pdf = copula.pdf(U)
    expected_pdf = 3.5433140, 0.2732326, 0.2439287, 1.1865992, 6.8052550
    assert_array_almost_equal(pdf, expected_pdf, DP)

    log_pdf = copula.pdf(U, log=True)
    assert_array_almost_equal(log_pdf, np.log(expected_pdf), DP)


def test_copula_cdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]

    cdf = copula.cdf(U)
    expected_cdf = 0.141910076, 0.003616266, 0.004607600, 0.026352224, 0.251302560
    assert_array_almost_equal(cdf, expected_cdf, DP)

    log_cdf = copula.cdf(U, log=True)
    assert_array_almost_equal(log_cdf, np.log(expected_cdf), DP)


def test_copula_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)
