import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from copulae.special.debye import debye_1, debye_2


@pytest.mark.parametrize('order, expected', [
    (1, [np.nan,
         1,
         0.538789569077856,
         0.430624598125269,
         0.388148021297938,
         0.244097102433199,
         0.104109604916145,
         0.0456926129680061,
         0.015740995855007,
         0.00298373674378419,
         0.00234120988734447,
         0.00160387487017183,
         0.000245059005251211,
         0.00016046258651165
         ]),
    (2, [np.nan,
         1,
         0.410794135797497,
         0.286509626630673,
         0.240553687521279,
         0.103803191576259,
         0.0192603338788639,
         0.00371005217024517,
         0.000440303803725957,
         1.58200930368215e-05,
         9.74021912478141e-06,
         4.57118706191541e-06,
         1.06715956352981e-07,
         4.57546886987235e-08],
     ),
])
@pytest.mark.filterwarnings()
def test_debye(order, expected):
    values = [-1, 0, 2.5, 3.5, 4, 6.7, 15.8, 36, 104.5, 551.3, 702.6, 1025.6, 6712.4, 10251.2]
    if order == 1:
        res = debye_1(values)
    else:
        res = debye_2(values)
    assert_array_almost_equal(res, expected, 5)
