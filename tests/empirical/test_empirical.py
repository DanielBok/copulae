import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from copulae import EmpiricalCopula
from copulae.datasets import load_smi
from copulae.errors import NotApplicableError


@pytest.fixture
def smi():
    return load_smi().iloc[:, :4]


@pytest.fixture(scope="module")
def u():
    return np.array([
        [0.1666667, 0.3333333, 0.6666667, 0.1666667],
        [0.8333333, 0.5000000, 0.5000000, 0.3333333],
        [0.3333333, 0.1666667, 0.1666667, 0.8333333],
        [0.5000000, 0.6666667, 0.8333333, 0.6666667],
        [0.6666667, 0.8333333, 0.3333333, 0.5000000],
    ])


@pytest.mark.parametrize("log", [True, False])
def test_empirical_pdf(smi, u, log):
    cop = EmpiricalCopula(4, smi, smoothing="beta")
    pdf = cop.pdf(u, log) if log else np.log(cop.pdf(u))

    expected = [-38.968738, -7.304777, -14.892165, -5.423080, -24.478464]

    assert_almost_equal(pdf, expected, decimal=6)


@pytest.mark.parametrize("func", [
    "cop.drho()",
    "cop.dtau()",
    "cop.irho(np.arange(4))",
    "cop.itau(np.arange(4))",
    "cop.lambda_",
    "cop.params",
    "cop.rho",
    "cop.tau",
])
def test_empirical_non_applicable_methods_raises_error(smi, func):
    cop = EmpiricalCopula(data=smi)
    with pytest.raises(NotApplicableError):
        eval(func)


@pytest.mark.parametrize("seed", [None, 888])
def test_empirical_random(smi, seed):
    cop = EmpiricalCopula(data=smi)
    rvs = cop.random(5, seed)

    assert rvs.shape == (5, smi.shape[1])
