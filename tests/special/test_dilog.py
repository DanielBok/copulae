import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from copulae.special.dilog import dilog, dilog_complex


@pytest.mark.parametrize('x, exp', [
    (np.arange(0, 3.1, 0.1),
     [0, 0.102617791099391, 0.211003775439705, 0.326129510075476, 0.449282974471282, 0.582240526465013,
      0.727586307716333, 0.889377624286039, 1.07479460000825, 1.29971472300496, 1.64493406684823, 1.96199910130557,
      2.12916943038396, 2.24088783985365, 2.31907303630966, 2.37439527027248, 2.41313113797463, 2.43935427088584,
      2.4558764585043, 2.46472330248959, 2.46740110027234, 2.46505797538081, 2.45858660199974, 2.44869251574337,
      2.43594109904594, 2.42079080656593, 2.40361721226863, 2.38473076153774, 2.36439010232858, 2.34281224729354,
      2.3201804233131]),
    (1.0075, 1.68897951570505),
    ([0.05, 1],
     [0.050639292464496, 1.64493406684823])
])
def test_dilog(x, exp):
    assert_array_almost_equal(dilog(x), exp)


@pytest.mark.parametrize('x, exp', [
    # Case 1
    (1.5, 2.37439527027248 - 1.2738062049196j),
    ([-2, 0.5, 4], [-1.43674636688368 + 0j, 0.582240526465013 + 0j, 2.06130946677732 - 4.3551721806072j]),
    # Case 2
    (0.75 + 0.661437827766148j, 0.640251982195514 + 0.962672985187488j),
    ([-0.4 + 0.916515138991168j, 0.875 - 0.484122918275927j],
     [-0.48648480481521 + 0.736302270619478j, 0.914962944925344 - 0.852055426132055j]),
    # Case 3
    (0.59 + 0.8j, 0.39378816738142 + 1.00269347958312j),  # series 3
    (0.4 + 0.4j, 0.378202523127816 + 0.491497909931901j),  # series 2
    (0.2 + 0.1j, 0.20765936375487 + 0.111389714232424j),  # series 1
    (0.8 + 0.3j, 0.950054236249761 + 0.541339991172987j),
    # Case 4
    (1.4 + 0.6j, 1.32883448711453 + 1.53499183728498j),
])
@pytest.mark.filterwarnings('ignore:invalid value encountered in log:RuntimeWarning')
def test_diog_complex(x, exp):
    numbers = dilog_complex(x)

    if isinstance(x, (complex, float, int)):
        re = numbers.real
        im = numbers.imag

        t_re = exp.real
        t_im = exp.imag
    else:
        re = [n.real for n in numbers]
        im = [n.imag for n in numbers]

        t_re = [n.real for n in exp]
        t_im = [n.imag for n in exp]

    assert_array_almost_equal(re, t_re)
    assert_array_almost_equal(im, t_im)
