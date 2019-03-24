import pytest
import numpy as np
from copulae.special.clausen import clausen
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize('x, exp, dp', [
    (np.arange(-2, 4.1, 0.4), [-0.727146050863279,
                               -0.905633219234944,
                               -1.00538981376486,
                               -0.98564887439532,
                               -0.767405894042677,
                               0,
                               0.767405894042678,
                               0.98564887439532,
                               1.00538981376486,
                               0.905633219234944,
                               0.727146050863279,
                               0.496799302036956,
                               0.23510832439665,
                               -0.0404765846184049,
                               -0.313708773770116,
                               -0.56814394442987
                               ], 6),
    (0.25, 0.596790672033802, 6),
    (1e-16, 3.78413614879047e-15, 20)
])
def test_clausen(x, exp, dp):
    assert_almost_equal(clausen(x), exp, dp)
