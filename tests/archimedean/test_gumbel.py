import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_allclose

from copulae.archimedean.gumbel import gumbel_coef, gumbel_poly


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
