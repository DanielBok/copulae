import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal

from copulae import EmpiricalCopula, pseudo_obs
from copulae.copula import Summary
from copulae.datasets import load_marginal_data, load_smi


@pytest.fixture(scope="module")
def u():
    return np.array([
        [0.1666667, 0.3333333, 0.6666667, 0.1666667],
        [0.8333333, 0.5000000, 0.5000000, 0.3333333],
        [0.3333333, 0.1666667, 0.1666667, 0.8333333],
        [0.5000000, 0.6666667, 0.8333333, 0.6666667],
        [0.6666667, 0.8333333, 0.3333333, 0.5000000],
    ])


@pytest.fixture(scope="module")
def smi():
    return pseudo_obs(load_smi().iloc[:, :4])


@pytest.mark.parametrize("smoothing, expected", [
    (None, [0.05673759, 0.26241135, 0.09929078, 0.41843972, 0.29078014]),
    ("beta", [0.06482708, 0.26010368, 0.10219117, 0.42249506, 0.29251128]),
    ("checkerboard", [0.05673762, 0.26595741, 0.10283691, 0.41843972, 0.29078014]),
])
def test_empirical_cdf(smi, u, smoothing, expected):
    cop = EmpiricalCopula(smi, smoothing=smoothing)
    assert_almost_equal(cop.cdf(u), expected)


@pytest.mark.parametrize("log", [True, False])
def test_empirical_pdf(smi, u, log):
    cop = EmpiricalCopula(smi, smoothing="beta")
    pdf = cop.pdf(u, log)
    if not log:
        pdf = np.log(pdf)

    expected = [-38.96873715, -7.30477533, -14.89215812, -5.42307982, -24.47846229]
    assert_almost_equal(pdf, expected, decimal=6)


def test_empirical_param_returns_none(smi):
    cop = EmpiricalCopula(smi)
    assert cop.params is None


@pytest.mark.parametrize("seed", [None, 888])
def test_empirical_random(smi, seed):
    cop = EmpiricalCopula(smi)
    rvs = cop.random(5, seed)

    assert rvs.shape == (5, smi.shape[1])


def test_empirical_summary(smi):
    cop = EmpiricalCopula(smi)
    assert isinstance(cop.summary(), Summary)


def test_empirical_to_margins():
    original = load_marginal_data()
    test_input = pseudo_obs(original)[:6]
    output = EmpiricalCopula.to_marginals(test_input, original)

    assert_frame_equal(output, original.iloc[:6])
