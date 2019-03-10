import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from copulae import ClaytonCopula

DP = 3


@pytest.fixture
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = ClaytonCopula(dim=dim)
    cop.params = 0.3075057
    return cop


def test_clayton_cdf(copula, U5):
    cdf = copula.cdf(U5)
    expected_cdf = 0.141910076, 0.003616266, 0.004607600, 0.026352224, 0.251302560
    assert_array_almost_equal(cdf, expected_cdf, DP)

    log_cdf = copula.cdf(U5, log=True)
    assert_array_almost_equal(log_cdf, np.log(expected_cdf), DP)


def test_clayton_pdf(copula, U5):
    pdf = copula.pdf(U5)
    expected_pdf = 3.5433140, 0.2732326, 0.2439287, 1.1865992, 6.8052550
    assert_array_almost_equal(pdf, expected_pdf, DP)

    log_pdf = copula.pdf(U5, log=True)
    assert_array_almost_equal(log_pdf, np.log(expected_pdf), DP)


def test_clayton_random_generates_correctly(copula):
    n = 10
    assert copula.random(n).shape == (n, copula.dim)

    expected = (n, 2)
    cop = ClaytonCopula(0)
    assert cop.random(n).shape == expected

    cop.params = 1.5
    assert cop.random(n).shape == expected


@pytest.mark.parametrize('degree, log, expected', [
    (1, False, np.array([
        [-0.429362, -0.464782, -0.580397, -0.339745, -0.800636, -0.353574, -0.59639],
        [-0.31799, -5.115089, -73.363718, -0.542165, -0.89823, -0.583534, -1.597917],
        [-73.363718, -0.518216, -0.390545, -1.227847, -0.743835, -1.263912, -1.315],
        [-0.531285, -0.418598, -0.884908, -1.369847, -0.668919, -1.239665, -2.00929],
        [-0.603007, -0.310586, -0.408301, -0.342187, -0.711275, -0.350986, -0.322362]
    ])),
    (1, True, np.array([
        [-0.845455, -0.766188, -0.544044, -1.079561, -0.222348, -1.039662, -0.51686],
        [-1.145734, 1.632195, 4.29543, -0.612186, -0.107329, -0.538652, 0.468701],
        [4.29543, -0.657363, -0.940213, 0.205262, -0.295936, 0.234212, 0.273836],
        [-0.632458, -0.870844, -0.122272, 0.314699, -0.402093, 0.214841, 0.697782],
        [-0.505826, -1.169293, -0.89575, -1.072397, -0.340696, -1.047008, -1.132079]
    ])),
    (2, False, np.array([
        [0.724675, 0.833484, 1.233557, 0.479416, 2.176318, 0.51439, 1.294179],
        [0.426574, 57.429652, 6314.964511, 1.093785, 2.666117, 1.245351, 7.368466],
        [6314.964511, 1.009964, 0.613078, 4.628758, 1.911267, 4.87139, 5.224239],
        [1.055345, 0.692921, 2.596726, 5.614904, 1.584739, 4.707672, 11.039657],
        [1.319628, 0.409201, 0.663124, 0.485516, 1.766099, 0.507765, 0.436979]
    ])),
    (2, True, np.array([
        [-0.322033, -0.182141, 0.209902, -0.735187, 0.777635, -0.664773, 0.257877],
        [-0.85197, 4.050561, 8.750677, 0.089644, 0.980623, 0.219418, 1.99721],
        [8.750677, 0.009914, -0.489264, 1.532289, 0.647766, 1.583379, 1.653309],
        [0.053868, -0.36684, 0.954251, 1.725425, 0.46042, 1.549193, 2.401494],
        [0.27735, -0.893548, -0.410794, -0.722544, 0.568773, -0.677736, -0.827871]
    ]))
])
def test_dipsi(copula, U5, degree, log, expected):
    dp = copula.dipsi(U5, degree, log)
    assert_array_almost_equal(dp, expected, 3)


@pytest.mark.parametrize('theta, degree, err', [
    (-1, 1, 'have not implemented dipsi for theta < 0'),
    (2, 3, 'have not implemented absdiPsi for degree > 2')
])
def test_dipsi_raises_errors(theta, degree, err):
    cop = ClaytonCopula(theta, dim=2)
    with pytest.raises(NotImplementedError, match=err):
        cop.dipsi([0.6534646, 0.3093621], degree=degree)


def test_fitted_parameters_match_target(residual_data: np.ndarray):
    cop = ClaytonCopula(dim=residual_data.shape[1])
    cop.fit(residual_data)
    assert_almost_equal(cop.params, 0.3075057, 4)


def test_fitted_log_likelihood_match_target(copula, U):
    target_log_lik = 147.4181
    log_lik = copula.log_lik(U)

    assert np.isclose(log_lik, target_log_lik, atol=1e-2)


def test_lambda_(copula):
    assert_array_almost_equal(copula.lambda_, [0.1049685, 0])

    copula.params = np.nan
    assert_array_almost_equal(copula.lambda_, [np.nan, np.nan])


def test_itau_fit(residual_data: np.ndarray):
    cop = ClaytonCopula(dim=residual_data.shape[1])
    cop.fit(residual_data, method='itau')
    assert_almost_equal(cop.params, 0.7848391, 4)


@pytest.mark.parametrize("dim, theta, err", [
    [2, -5, 'theta must be greater than -1 in 2 dimensional clayton copulae'],
    [3, -1, 'theta must be positive when dim > 2']
])
def test_param_set_raises_error(dim, theta, err):
    cop = ClaytonCopula(dim=dim)
    with pytest.raises(ValueError, match=err):
        cop.params = theta


def test_pdf_bivariate():
    # the dim > 2 case is tested when we fitted the copula. Test the bivariate case here
    cop = ClaytonCopula(-0.3075057, 2)

    u = np.array([
        [0.199296, 0.645346],
        [0.135279, 0.30006]
    ])
    assert_array_almost_equal(cop.pdf(u), [1.152112, 1.018026])


@pytest.mark.parametrize("data, err", [
    (np.random.uniform(size=(3, 5)), 'number of columns in input data does not match copula dimension'),
    (np.array([1, 4, 5]), 'number of columns in input data does not match copula dimension')
])
def test_pdf_raises_errors(copula, data, err):
    with pytest.raises(ValueError, match=err):
        copula.pdf(data)


def test_psi(copula, U5):
    psi = copula.psi(U5)

    target = np.array([
        [0.154834, 0.168502, 0.210311, 0.118544, 0.278835, 0.124292, 0.215762],
        [0.109408, 0.698909, 0.952157, 0.196958, 0.30515, 0.211387, 0.444069],
        [0.952157, 0.188358, 0.139403, 0.379728, 0.262445, 0.386776, 0.396441],
        [0.193074, 0.150601, 0.301688, 0.406425, 0.239507, 0.382059, 0.499464],
        [0.217994, 0.106275, 0.146519, 0.119563, 0.252666, 0.12322, 0.111253]
    ])

    assert_array_almost_equal(psi, target, 3)


def test_tau(copula):
    assert_almost_equal(copula.dtau(), 0.3756163)
