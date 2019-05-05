from copulae.datasets import load_smi, load_danube
from copulae.gof import exch_test

from numpy.testing import assert_almost_equal


def test_exch_stats():
    danube = load_danube().values
    smi = load_smi(as_returns=True).values

    # no ties
    stat = exch_test(danube[:, 0], danube[:, 1])
    assert_almost_equal(stat.statistic, 0.05196866)

    # ties
    stat = exch_test(smi[:, 0], smi[:, 1])
    assert_almost_equal(stat.statistic, 0.01316327)

    stat = exch_test(smi[:, 1], smi[:, 2])
    assert_almost_equal(stat.statistic, 0.02795918)
