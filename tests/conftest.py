import pytest

from copulae.core import pseudo_obs
from copulae.datasets import load_residuals


@pytest.fixture(scope='session')
def residual_data():
    return load_residuals().values


@pytest.fixture
def U(residual_data):
    return pseudo_obs(residual_data)


@pytest.fixture
def U5(U):
    return U[:5]
