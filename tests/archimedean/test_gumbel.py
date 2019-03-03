import pytest
from numpy.testing import assert_array_almost_equal

from copulae.archimedean.gumbel import gumbel_coef


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
    with pytest.raises(ValueError):
        gumbel_coef(d, alpha)
