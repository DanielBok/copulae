import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_almost_equal

from copulae.archimedean.gumbel import GumbelCopula, gumbel_coef, gumbel_poly

DP = 5


@pytest.fixture(scope='module')
def copula(residual_data: np.ndarray):
    dim = residual_data.shape[1]
    cop = GumbelCopula(dim=dim)
    cop.params = 1.171789
    return cop


def test_gumbel_cdf(copula, U5):
    cdf = copula.cdf(U5)
    expected_cdf = 0.1456360421, 0.0005107288, 0.0009219520, 0.0125531052, 0.2712652274
    assert_array_almost_equal(cdf, expected_cdf, DP)

    log_cdf = copula.cdf(U5, log=True)
    assert_array_almost_equal(log_cdf, np.log(expected_cdf), DP)


@pytest.mark.parametrize('d, alpha, expected', [
    (4, 0.5, [0.937499999999999, 0.9375, 0.375, 0.0625]),
    (9, 0.3, [4929.14899392301, 3386.53392115501, 1019.23574791499, 176.49924705, 19.324096533, 1.375329942,
              0.062368866, 0.001653372, 1.9683e-05]),
    (4, 0.5, [0.937499999999999, 0.9375, 0.375, 0.0625]),
    (9, 0.3, [4929.14899392301, 3386.53392115501, 1019.23574791499, 176.49924705, 19.324096533, 1.375329942,
              0.062368866, 0.001653372, 1.9683e-05]),
    (5, 0.38, [3.6199299168, 2.799182448, 0.86752632, 0.129278432, 0.0079235168]),
    (7, 0.95, [6.37797790546874, 8.88826645546841, 6.57189190546899, 3.52783019531255, 1.62493996875,
               0.77184648515625, 0.69833729609375]),
    (4, 0.5, [0.937499999999999, 0.9375, 0.375, 0.0625]),
    (9, 0.3, [4929.14899392301, 3386.53392115501, 1019.23574791499, 176.49924705, 19.324096533, 1.375329942,
              0.062368866, 0.001653372, 1.9683e-05]),
])
@pytest.mark.parametrize('method', ['sort', 'horner', 'direct', 'ds.direct', 'log', 'diff'])
def test_gumbel_coef(d, alpha, method, expected):
    assert_array_almost_equal(gumbel_coef(d, alpha, method), expected, 5)


@pytest.mark.parametrize('d, alpha', [
    (0.5, 0.25),
    (5, -0.1),
    (5, 0),
    (5, 1.25)
])
def test_gumbel_coef_raises_error(d, alpha):
    with pytest.raises(AssertionError):
        gumbel_coef(d, alpha)


@pytest.mark.parametrize('degree, log, expected', [
    (1, False, [[-1.1963634, -1.3185243, -1.6825998, -0.8127349, -2.3088903, -0.8877997, -1.7303602],
                [-0.6407114, -11.4764199, -98.6582937, -1.5663222, -2.5708457, -1.6920077, -4.3002524],
                [-98.6582937, -1.4916994, -1.0506974, -3.4118984, -2.1527366, -3.500707, -3.6256003],
                [-1.5326107, -1.1575186, -2.5355094, -3.7585628, -1.9416819, -3.4410592, -5.2395599],
                [-1.7499836, -0.5109072, -1.1193696, -0.8270009, -2.0617878, -0.8746662, -0.6865766]]),
    (1, True, [[0.1792865, 0.2765131, 0.5203401, -0.2073503, 0.836767, -0.1190091, 0.5483296],
               [-0.4451761, 2.4402945, 4.5916623, 0.4487303, 0.9442349, 0.5259158, 1.4586737],
               [4.5916623, 0.399916, 0.0494542, 1.2272689, 0.7667399, 1.252965, 1.2880199],
               [0.4269726, 0.1462786, 0.9303946, 1.3240366, 0.6635545, 1.2357793, 1.6562375],
               [0.5596064, -0.6715673, 0.1127657, -0.1899495, 0.7235735, -0.1339129, -0.3760375]]),
    (2, False, [[2.5834848, 2.7917326, 3.7022266, 2.8532213, 5.9267741, 2.5772514, 3.8456254],
                [5.0612285, 106.4207541, 6761.4807276, 3.3740593, 7.059027, 3.7300914, 17.2331832],
                [6761.4807276, 3.1804178, 2.4467844, 11.4331451, 5.3062488, 11.95908, 12.7191019],
                [3.28484, 2.5327075, 6.8997444, 13.5543241, 4.5350002, 11.604507, 24.6519845],
                [3.905924, 12.1151323, 2.4919784, 2.7837092, 4.9641826, 2.6113856, 4.1003485]]),
    (2, True, [[0.9491392, 1.0266624, 1.3089344, 1.0484486, 1.7794801, 0.9467235, 1.3469362],
               [1.6216092, 4.6674006, 8.8189972, 1.2161166, 1.9543072, 1.3164327, 2.8468368],
               [8.8189972, 1.1570126, 0.8947747, 2.4365166, 1.6688851, 2.4814908, 2.5431049],
               [1.189318, 0.9292889, 1.9314844, 2.6067056, 1.5118251, 2.4513936, 3.2048574],
               [1.3624944, 2.4944553, 0.9130769, 1.0237843, 1.6022487, 0.959881, 1.411072]]
     ),
    (3, False, None)
])
def test_gumbel_dipsi(copula, U5, degree, log, expected):
    if degree not in (1, 2):
        with pytest.raises(AssertionError, match='degree can only be 1 or 2'):
            copula.dipsi(U5, degree, log)
    else:
        assert_array_almost_equal(copula.dipsi(U5, degree, log), expected, 4)


@pytest.mark.parametrize("as_df", [False, True])
def test_fitted_parameters_match_target(residual_data, as_df):
    if as_df:
        residual_data = pd.DataFrame(residual_data, columns=[f'V{i}' for i in range(residual_data.shape[1])])

    cop = GumbelCopula(dim=residual_data.shape[1])
    cop.fit(residual_data)
    assert_almost_equal(cop.params, 1.171789, 5)


@pytest.mark.parametrize('log_x, alpha, d, log, expected', [
    ([3.204613, 2.253136, 3.422148, 2.106722, 2.11856, 3.716989], 0.4, 4, False,
     [13397.4244, 489.5018, 29942.3549, 306.3865, 318.0584, 90751.3460]),
    ([3.204613, 2.253136, 3.422148, 2.106722, 2.11856, 3.716989], 0.4, 4, True,
     [9.502818, 6.193388, 10.307029, 5.724847, 5.762235, 11.415879]),
    ([1.0051537, 1.0049213, 0.7952184, 1.1184151], 0.7, 3, False,
     [11.034945, 11.028364, 6.495272, 14.792584]),
    ([1.0051537, 1.0049213, 0.7952184, 1.1184151], 0.7, 3, True,
     [2.401067, 2.400470, 1.871074, 2.694126])
])
@pytest.mark.parametrize('method', ['default', 'pois', 'direct', 'log', 'sort'])
def test_gumbel_poly(log_x, alpha, d, method, log, expected):
    log_x = np.asarray(log_x)
    assert_allclose(gumbel_poly(log_x, alpha, d, method, log), expected, rtol=1e-4)


def test_gumbel_pdf(copula, U5):
    pdf = copula.pdf(U5)
    expected_cdf = 3.1295301, 0.5511432, 0.7869483, 1.2362505, 5.2847504
    assert_array_almost_equal(pdf, expected_cdf, DP)

    log_cdf = copula.pdf(U5, log=True)
    assert_array_almost_equal(log_cdf, np.log(expected_cdf), DP)


def test_gaussian_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)


def test_summary(residual_data):
    cop = GumbelCopula(dim=residual_data.shape[1])
    cop.fit(residual_data)
    smry = cop.summary()

    assert isinstance(str(smry), str)
    assert isinstance(smry.as_html(), str)
