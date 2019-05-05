from copulae.datasets import load_smi, load_danube
from copulae.gof import rad_sym_test

from numpy.testing import assert_almost_equal


def test_exch_stats():
    danube = load_danube()
    smi = load_smi(as_returns=True)

    # no ties
    stat = rad_sym_test(danube)
    assert_almost_equal(stat.statistic, 0.229749862416269)

    # ties
    stat = rad_sym_test(smi)
    assert_almost_equal(stat.statistic, 0.0582653061224491)
