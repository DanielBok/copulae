import pytest
from numpy.testing import assert_almost_equal

from copulae.special.combinatorics import *


@pytest.mark.parametrize("n, r, exp", [
    (7, 4, 35),
    (5.5, 3, 14.4375),
    (5, 7.6, 0.003003713),
    ([5, 9], 7.6, [0.003003713, 16.895886772])
])
def test_comb(n, r, exp):
    assert_almost_equal(comb(n, r), exp, 4)


@pytest.mark.parametrize("n, r, exp", [
    (7, 4, 840),
    (5, 3.5, 90.27033),
    (5, 7.6, 51.93495),
    (6.5, 4, 563.0625),
    (6.5, 8.4, -177.0250183803),
    ([5, 9], 7.6, [51.93495, 292134.08108])
])
def test_perm(n, r, exp):
    assert_almost_equal(perm(n, r), exp, 4)
