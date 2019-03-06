import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from copulae import GaussianCopula

DP = 3

target_params = np.array([
    0.19108579, -0.36593809, 0.12821787, 0.12885945, 0.11054193, 0.30997872, 0.51268273, -0.02703581, -0.08222982,
    -0.03201687, 0.20790589, 0.05828219, -0.00647038, 0.05512774, 0.01065161, 0.62408405, 0.93609999, 0.59009292,
    0.71106162, 0.41605434, 0.56242001
])


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = GaussianCopula(dim)
    cop.params = target_params
    return cop


def test_fitted_log_likelihood_match_target(copula, residual_data):
    target_log_lik = 810.931
    U = copula.pobs(residual_data)
    log_lik = copula.log_lik(U)

    assert_almost_equal(log_lik, target_log_lik, DP)


def test_fitted_parameters_match_target(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = GaussianCopula(dim)
    cop.fit(residual_data)
    assert_array_almost_equal(cop.params, target_params, 4)


def test_gaussian_cdf(copula, U5):
    cdf = copula.cdf(U5)

    expected_cdf = 0.153405034, 0.001607509, 0.002487501, 0.039130689, 0.245082925
    assert_array_almost_equal(cdf, expected_cdf, DP)

    log_cdf = copula.cdf(U5, log=True)
    assert_array_almost_equal(log_cdf, np.log(expected_cdf), DP)


def test_gaussian_pdf(copula, U5):
    pdf = copula.pdf(U5)

    expected_pdf = 6.145473, 16.026015, 8.344821, 5.240921, 41.002535
    assert_array_almost_equal(pdf, expected_pdf, DP)

    log_pdf = copula.pdf(U5, log=True)
    assert_array_almost_equal(log_pdf, np.log(expected_pdf), DP)


def test_gaussian_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)
