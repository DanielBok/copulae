import pytest
from numpy.testing import assert_almost_equal

from copulae.stats import skew_stable


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('x, alpha, beta, exp', [
    ([0.4, 0.6], 2, 0.4, [0.271033696776216, 0.257815227404741]),
    (0.7, 2, 0.3, 0.249570928036152),
    ([0.4, 0.6], 1, 0, [0.274405074296371, 0.234051386899846]),
    (0.7, 1, 0, 0.213630796096504),
    ([0.4, 0.6], 1, 0.6, [0.235618702903308, 0.208686907242902]),
    (0.7, 1, 0.6, 0.195767388717312),
    (0, 1.5, 0, 0.287352751452164),
    (0.3, 1.5, 0.4, 0.27237094986795),
    ([0.6, 1.1], 0.45, -0.6, [0.150825004226331, 0.0404642061548807])
])
def test_pdf(x, alpha, beta, exp):
    res = skew_stable.pdf(x, alpha, beta)
    assert_almost_equal(res, exp)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('x, alpha, beta, exp', [
    ([0.4, 0.6], 2, 0.4, [-1.30551212348465, -1.35551212348465]),
    (0.7, 2, 0.3, -1.38801212348465),
    ([0.4, 0.6], 1, 0, [-1.29314989096767, -1.45221458559736]),
    (0.7, 1, 0, -1.5435060058067),
    ([0.4, 0.6], 1, 0.6, [-1.44554044621578, -1.56692020168768]),
    (0.7, 1, 0.6, -1.63082811680585),
    (0, 1.5, 0, -1.24704471881004),
    (0.3, 1.5, 0.4, -1.30059035551878),
    ([0.6, 1.1], 0.45, -0.6, [-1.89163502663389, -3.20733749429485])
])
def test_log_pdf(x, alpha, beta, exp):
    res = skew_stable.logpdf(x, alpha, beta)
    assert_almost_equal(res, exp)
