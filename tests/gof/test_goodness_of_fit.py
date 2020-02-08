import pytest
from numpy.testing import assert_almost_equal

from copulae import GumbelCopula
from copulae.gof import gof_copula
from .data import gof_data


@pytest.mark.parametrize("multiprocess", [False, True])
def test_gof_sn(multiprocess):
    data = gof_data()
    stat = gof_copula(GumbelCopula, data, reps=1000, multiprocess=multiprocess)

    assert_almost_equal(stat.parameter, 2.091908, 4)
    assert_almost_equal(stat.statistic, 0.17978, 4)
    assert_almost_equal(stat.pvalue, 0.0004995, 4)
