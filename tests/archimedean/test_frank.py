import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from copulae import FrankCopula
from copulae.archimedean.frank import debye1, debye2
from copulae.special import log1mexp


@pytest.fixture
def copula(residual_data):
    cop = FrankCopula(1.295, dim=residual_data.shape[1])
    return cop


@pytest.fixture(scope="module")
def fitted_frank(residual_data):
    cop = FrankCopula(dim=residual_data.shape[1])
    cop.fit(residual_data, scale=1e6)
    return cop


@pytest.mark.parametrize('order, x, exp', [
    (1, [-0.5, 0.5], [1.13192715679061, 0.881927156790606]),
    (2, [-0.5, 0.5], [1.17705452668059, 0.843721193347254]),
    (1, [-np.inf, np.inf, np.nan, 1.5, -1.5], [np.inf, 0, np.nan, 0.686145310789402, 1.4361453107894]),
    (2, [-np.inf, np.inf, np.nan, 1.5, -1.5], [np.inf, 0, np.nan, 0.591496372256713, 1.59149637225671]),
    (1, -0.8, 1.21766522318744),
    (2, 0.8, 0.759812510194521)
])
def test_debye(order, x, exp):
    if order == 1:
        d = debye1(x)
    else:
        d = debye2(x)

    assert_almost_equal(d, exp)


@pytest.mark.parametrize('degree, log, expected', [
    (1, False, [[-0.7498397, -0.8244404, -1.0630926, -0.5582222, -1.4982391, -0.5880289, -1.0955267],
                [-0.5112031, -7.9557253, -65.1879561, -0.9849964, -1.6838453, -1.0694667, -2.9188229],
                [-65.1879561, -0.9356651, -0.6673127, -2.2840205, -1.3882791, -2.347519, -2.4368172],
                [-0.9626244, -0.7270339, -1.6587446, -2.5318735, -1.2409069, -2.3048703, -3.5878326],
                [-1.1089049, -0.4951701, -0.7051608, -0.5634924, -1.3245653, -0.5824573, -0.5206642]]),
    (1, True, [[-0.2878958, -0.1930504, 0.0611822, -0.5829982, 0.4042905, -0.5309792, 0.0912352],
               [-0.6709883, 2.0738918, 4.1772747, -0.0151173, 0.52108, 0.0671601, 1.0711804],
               [4.1772747, -0.0664977, -0.4044965, 0.8259372, 0.3280649, 0.853359, 0.8906928],
               [-0.0380919, -0.3187822, 0.5060611, 0.9289595, 0.2158424, 0.8350244, 1.2775483],
               [0.1033729, -0.7028539, -0.3493294, -0.5736015, 0.2810843, -0.5404993, -0.6526499]]),
    (2, False, [[1.5333021, 1.7473522, 2.5068706, 1.0345098, 4.1849402, 1.1072755, 2.6188857],
                [0.9233367, 73.596229, 4333.8880284, 2.2457883, 5.0159146, 2.5287183, 12.2994025],
                [4333.8880284, 2.0871554, 1.3094762, 8.174556, 3.7251401, 8.5508823, 9.0937566],
                [2.1732444, 1.4700871, 4.8995081, 9.6891597, 3.1468242, 8.2972339, 17.5187859],
                [2.6657018, 0.8864387, 1.4104351, 1.0472463, 3.4697854, 1.0935388, 0.9453515]]),
    (2, True, [[0.4274237, 0.5581016, 0.9190352, 0.0339276, 1.4314924, 0.1019025, 0.9627489],
               [-0.0797614, 4.2985938, 8.3742203, 0.8090566, 1.6126158, 0.9277126, 2.5095507],
               [8.3742203, 0.7358021, 0.2696272, 2.1010264, 1.3151045, 2.1460345, 2.2075881],
               [0.7762212, 0.3853217, 1.5891348, 2.2710077, 1.1463938, 2.1159222, 2.8632738],
               [0.9804674, -0.1205433, 0.3438982, 0.0461641, 1.2440927, 0.089419, -0.0561985]]),
    (3, False, None)
])
def test_dipsi(copula, U5, degree, log, expected):
    if degree not in (1, 2):
        with pytest.raises(AssertionError, match='degree can only be 1 or 2'):
            copula.dipsi(U5, degree, log)
    else:
        assert_array_almost_equal(copula.dipsi(U5, degree, log), expected, 4)


def test_fit(fitted_frank):
    assert_almost_equal(fitted_frank.params, 1.29505, 5)


@pytest.mark.parametrize('log, expected', [
    (False, [[0.1367441, 0.1725768, 0.2792778, 0.0383506, 0.4486266, 0.0543063, 0.2929384],
             [0.0126516, 1.6461267, 3.6183703, 0.2455986, 0.5129605, 0.2819772, 0.859795],
             [3.6183703, 0.2237245, 0.0955484, 0.6965139, 0.4084643, 0.7141002, 0.73832],
             [0.2357378, 0.1255286, 0.5044985, 0.7634729, 0.351976, 0.7023225, 1.0071502],
             [0.2985192, 0.0037354, 0.1146523, 0.0411903, 0.3844331, 0.0513431, 0.0178761]]),
    (True, [[-1.9896437, -1.7569129, -1.2755484, -3.2609861, -0.8015643, -2.9131147, -1.2277929],
            [-4.369969, 0.4984251, 1.2860237, -1.4040568, -0.6675565, -1.2659291, -0.1510613],
            [1.2860237, -1.4973399, -2.3481219, -0.3616676, -0.8953508, -0.336732, -0.3033779],
            [-1.4450349, -2.0752217, -0.6841905, -0.2698777, -1.0441922, -0.3533626, 0.0071248],
            [-1.2089212, -5.5899093, -2.1658511, -3.189552, -0.9559855, -2.9692249, -4.0242913]]
     ),
])
def test_ipsi(copula, U5, log, expected):
    assert_array_almost_equal(copula.ipsi(U5, log), expected)


def test_itau(copula, U5):
    assert_array_almost_equal(copula.itau(U5),
                              [[15.9183071, 12.8806454, 8.3496639, 52.7849147, 5.4014431, 37.7801859, 7.9976228],
                               [156.3375744, 1.0597816, 0.1367344, 9.3731482, 4.7476127, 8.277582, 2.7334516],
                               [0.1367344, 10.1929662, 22.1625828, 3.469787, 5.8996837, 3.3770169, 3.2553548],
                               [9.7248162, 17.217456, 4.8253228, 3.1358966, 6.7691291, 3.4387144, 2.2462485],
                               [7.8624339, 525.0165626, 18.7160445, 49.2659883, 6.2415492, 39.8632145, 111.1875078]])

    assert_almost_equal(copula.itau(0.5), 5.73628270702243)


def test_pdf(copula, U5):
    expected = [4.02910800346931, 0.608665049979427, 0.669917791901045, 1.02347002860614, 11.3229231831187]
    density = copula.pdf(U5)
    assert_array_almost_equal(density, expected)


@pytest.mark.parametrize('theta, expected', [
    (1.295, 0.448292484741403),
    (0, 0.606530659712633),
    (-15, 0.966666679896355),
    (-37, 0.986486486486487)
])
def test_psi(theta, expected):
    copula = FrankCopula(theta, 2)
    assert_almost_equal(copula.psi(0.5), expected)

    if theta > 0:
        assert np.isnan(copula.psi(log1mexp(theta) - 1e-6))


def test_gaussian_random_generates_correctly(copula):
    assert copula.random(10).shape == (10, copula.dim)


@pytest.mark.parametrize('theta, expected', [
    (1.295, 0.211156879535562),
    (1e-9, 1.66666666666667e-10)
])
def test_rho(theta, expected):
    assert_almost_equal(FrankCopula(theta, 2).rho, expected)


def test_summary(fitted_frank):
    smry = fitted_frank.summary()
    assert isinstance(str(smry), str)
    assert isinstance(smry.as_html(), str)


@pytest.mark.parametrize('theta, expected', [
    (1.295, 0.141542489375599),
    (1e-9, 1.11111111111111e-10)
])
def test_tau(theta, expected):
    assert_almost_equal(FrankCopula(theta).tau, expected)
