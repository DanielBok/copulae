import pytest

from copulae.datasets import load_residuals


@pytest.fixture(scope='session')
def residual_data():
    return load_residuals().values
