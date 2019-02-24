import numpy as np
import pytest

from copulae import GaussianCopula


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = GaussianCopula(dim)
    cop.fit(residual_data)
    return cop


def test_fitted_parameters_match_target(copula):
    target_params = np.array([
        0.19108579, -0.36593809, 0.12821787, 0.12885945, 0.11054193, 0.30997872, 0.51268273, -0.02703581, -0.08222982,
        -0.03201687, 0.20790589, 0.05828219, -0.00647038, 0.05512774, 0.01065161, 0.62408405, 0.93609999, 0.59009292,
        0.71106162, 0.41605434, 0.56242001
    ])
    assert np.allclose(copula.params, target_params, atol=5e-5)


def test_fitted_log_likelihood_match_target(copula, residual_data):
    target_log_lik = 810.931
    U = copula.pobs(residual_data)
    log_lik = copula.log_lik(U)

    assert np.isclose(log_lik, target_log_lik, atol=1e-2)


def test_copula_pdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    pdf = copula.pdf(U)
    expected_pdf = 6.145473, 16.026015, 8.344821, 5.240921, 41.002535
    assert np.allclose(pdf, expected_pdf, atol=2e-4)  # tolerance is a little to tight, maybe should increase

    log_pdf = copula.pdf(U, log=True)
    assert np.allclose(log_pdf, np.log(pdf), atol=2e-4)


def test_copula_cdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    cdf = copula.cdf(U)
    expected_cdf = 0.153405034, 0.001607509, 0.002487501, 0.039130689, 0.245082925
    assert np.allclose(cdf, expected_cdf, atol=2e-4)

    log_cdf = copula.cdf(U, log=True)
    assert np.allclose(log_cdf, np.log(cdf), atol=2e-4)


def test_copula_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)
