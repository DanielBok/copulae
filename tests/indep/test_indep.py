import numpy as np
import pytest

from copulae import IndepCopula


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = IndepCopula(dim)
    cop.fit(residual_data)
    return cop


def test_copula_pdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    log_pdf = copula.pdf(U, log=True)
    expected_pdf = np.zeros(5)
    assert np.allclose(log_pdf, expected_pdf, atol=2e-4)  # tolerance is a little to tight, maybe should increase

    pdf = copula.pdf(U)
    assert np.allclose(pdf, np.exp(log_pdf), atol=2e-4)


def test_copula_cdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    log_cdf = copula.cdf(U, log=True)
    expected_cdf = -2.478547, -9.366887, -8.695993, -5.701305, -1.599828
    assert np.allclose(log_cdf, expected_cdf, atol=2e-4)  # tolerance is a little to tight, maybe should increase

    cdf = copula.cdf(U)
    assert np.allclose(cdf, np.exp(log_cdf), atol=2e-4)


def test_copula_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)
