import numpy as np
import pytest

from copulae import MarginalCopula, NormalCopula
from copulae.copula.exceptions import InputDataError


@pytest.fixture(scope="module")
def copula():
    np.random.seed(8888)

    series = np.random.normal(0.3, 0.5, 100)
    data = np.vstack([
        series,
        3 + series * 1.5 + np.random.normal(0.6, 1.2, 100)
    ]).T

    cop = MarginalCopula(NormalCopula(2), ['norm', 'norm'])
    cop.fit(data)

    return cop


def test_bounds(copula):
    assert copula.bounds is NotImplemented


def test_cdf(copula):
    # smoke tests
    data = np.random.normal(size=(50, 2))
    cdf = copula.cdf(data)
    assert cdf.shape == (50,)


@pytest.mark.parametrize("margins", [
    "norm",
    ['norm', 'norm'],
    {"type": "norm"},
    [{"type": "norm"}, {"type": "norm"}]
])
def test_copula_init_setup(margins):
    # test no error
    MarginalCopula(NormalCopula(2), margins)


@pytest.mark.parametrize("margins, error", [
    ([], AssertionError),  # empty collection
    ([1, 2], TypeError),  # did not pass in a marginal "type"
    ('bad type margins definition type', TypeError)
])
def test_copula_init_setup_raises_error(margins, error):
    with pytest.raises(error):
        MarginalCopula(NormalCopula(2), margins)


@pytest.mark.parametrize("data", [
    np.random.normal(size=(50, 2, 2)),
    np.random.normal(size=(50, 3)),
])
def test_copula_fit_with_bad_data_raises_error(data):
    with pytest.raises(InputDataError):
        cop = MarginalCopula(NormalCopula(2), ["norm", "norm"])
        cop.fit(data)


def test_params(copula):
    params = copula.params
    assert isinstance(params, dict)
    assert params.keys() == {"copula", "marginals"}


def test_pdf(copula):
    # smoke tests
    data = np.random.normal(size=(50, 2))
    cdf = copula.pdf(data)
    assert cdf.shape == (50,)


def test_random(copula):
    rvs = copula.random(50)
    assert rvs.shape == (50, 2)


def test_summary(copula):
    smry = copula.summary()
    assert isinstance(smry.as_html(), str)
    assert isinstance(repr(smry), str)
