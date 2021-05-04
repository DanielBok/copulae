import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from copulae import StudentCopula

DP = 3

target_df = 10.54433123

target_rho = np.array([
    0.17744184, -0.37436649, 0.09228916, 0.11114309, 0.07197019, 0.2650227, 0.52527897, -0.05499694,
    -0.0773855, -0.06585167, 0.18124078, 0.05795289, 0.01324174, 0.06096022, 0.01836235, 0.63010478,
    0.93983371, 0.57962059, 0.71667641, 0.41154285, 0.5589378
])


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = StudentCopula(dim)
    cop.params = (target_df, *target_rho)
    return cop


def test_fitted_log_likelihood_match_target(copula, residual_data):
    target_log_lik = 838.7959
    U = copula.pobs(residual_data)
    log_lik = copula.log_lik(U)

    assert np.isclose(log_lik, target_log_lik, atol=1e-2)


@pytest.mark.parametrize("as_df", [False, True])
def test_fitted_parameters_match_target(residual_data, as_df):
    if as_df:
        residual_data = pd.DataFrame(residual_data, columns=[f'V{i}' for i in range(residual_data.shape[1])])

    cop = StudentCopula(residual_data.shape[1])
    cop.fit(residual_data)

    assert_array_almost_equal(target_rho, cop.params.rho, DP)
    assert_almost_equal(target_df, cop.params.df, DP)


def test_student_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)


def test_student_pdf(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    pdf = copula.pdf(U)
    expected_pdf = 5.058462, 15.835262, 4.774393, 7.380350, 21.954320
    assert_array_almost_equal(pdf, expected_pdf, DP)  # tolerance is a little to tight, maybe should increase

    log_pdf = copula.pdf(U, log=True)
    assert_array_almost_equal(log_pdf, np.log(expected_pdf), DP)


def test_summary(residual_data):
    cop = StudentCopula(residual_data.shape[1])
    cop.fit(residual_data)
    summary = cop.summary()

    assert isinstance(str(summary), str)
    assert isinstance(summary.as_html(), str)
