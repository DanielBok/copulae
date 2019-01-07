import numpy as np
import pytest

from copulae import StudentCopula


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = StudentCopula(dim)
    cop.fit(residual_data)
    return cop


def test_fitted_parameters_match_target(copula):
    target_rho = np.array([
        0.17744184, -0.37436649, 0.09228916, 0.11114309, 0.07197019, 0.2650227, 0.52527897, -0.05499694,
        -0.0773855, -0.06585167, 0.18124078, 0.05795289, 0.01324174, 0.06096022, 0.01836235, 0.63010478,
        0.93983371, 0.57962059, 0.71667641, 0.41154285, 0.5589378
    ])
    target_df = 10.54433123
    params = copula.params

    assert np.allclose(target_rho, params.rho, atol=5e-5)
    assert np.isclose(target_df, params.df, atol=1e-3)


def test_fitted_log_likelihood_match_target(copula, residual_data):
    target_log_lik = 838.7959
    U = copula.pobs(residual_data)
    log_lik = copula.log_lik(U)

    assert np.isclose(log_lik, target_log_lik, atol=1e-2)


def test_copula_pdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    pdf = copula.pdf(U, log=True)
    expected_pdf = 1.621063, 2.762239, 1.563267, 1.998821, 3.088964
    assert np.allclose(pdf, expected_pdf, atol=2e-4)  # tolerance is a little to tight, maybe should increase

    log_pdf = copula.pdf(U)
    assert np.allclose(log_pdf, np.exp(pdf), atol=2e-4)


def test_copula_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)
