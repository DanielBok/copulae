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


def test_itau_fit(residual_data: np.ndarray):
    target_param = 0.7848391
    cop = ClaytonCopula(dim=residual_data.shape[1])
    cop.fit(residual_data, method='itau')
    assert np.isclose(cop.params, target_param, 1e-3)


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


def test_dipsi(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    dp = copula.dipsi(U)

    target = np.array([
        [-0.429362, -0.464782, -0.580397, -0.339745, -0.800636, -0.353574, -0.59639],
        [-0.31799, -5.115089, -73.363718, -0.542165, -0.89823, -0.583534, -1.597917],
        [-73.363718, -0.518216, -0.390545, -1.227847, -0.743835, -1.263912, -1.315],
        [-0.531285, -0.418598, -0.884908, -1.369847, -0.668919, -1.239665, -2.00929],
        [-0.603007, -0.310586, -0.408301, -0.342187, -0.711275, -0.350986, -0.322362]
    ])

    assert_array_almost_equal(dp, target, 3)
    log_dp = copula.dipsi(U, log=True)

    log_target = np.array([
        [-0.845455, -0.766188, -0.544044, -1.079561, -0.222348, -1.039662, -0.51686],
        [-1.145734, 1.632195, 4.29543, -0.612186, -0.107329, -0.538652, 0.468701],
        [4.29543, -0.657363, -0.940213, 0.205262, -0.295936, 0.234212, 0.273836],
        [-0.632458, -0.870844, -0.122272, 0.314699, -0.402093, 0.214841, 0.697782],
        [-0.505826, -1.169293, -0.89575, -1.072397, -0.340696, -1.047008, -1.132079]
    ])
    assert_array_almost_equal(log_dp, log_target, 3)


def test_psi(copula, residual_data):
    U = copula.pobs(residual_data)[:5]
    psi = copula.psi(U)

    target = np.array([
        [0.154834, 0.168502, 0.210311, 0.118544, 0.278835, 0.124292, 0.215762],
        [0.109408, 0.698909, 0.952157, 0.196958, 0.30515, 0.211387, 0.444069],
        [0.952157, 0.188358, 0.139403, 0.379728, 0.262445, 0.386776, 0.396441],
        [0.193074, 0.150601, 0.301688, 0.406425, 0.239507, 0.382059, 0.499464],
        [0.217994, 0.106275, 0.146519, 0.119563, 0.252666, 0.12322, 0.111253]
    ])

    assert_array_almost_equal(psi, target, 3)
