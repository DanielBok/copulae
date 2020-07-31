from functools import lru_cache

import numpy as np
import pytest

from copulae import GaussianCopula
from copulae.core import cov2corr


@lru_cache(3)
def gen_corr(d=3) -> np.ndarray:
    np.random.seed(10)
    a = np.random.uniform(size=d * d).reshape(d, d)
    return cov2corr(a @ a.T)


def test_set_parameter():
    corr = gen_corr()
    cop = GaussianCopula(3)
    cop[:] = corr

    cop[:, 0] = corr[:, 0]
    cop[1, :] = corr[1, :]
    cop[:, :] = corr
    cop[1, 2] = 0.5
    cop[0] = 0.3


@pytest.mark.parametrize("category, value", [
    ["full", gen_corr() + 2],  # values above 1
    ["full", gen_corr() - 2],  # values below -1
    ["full", gen_corr(4)],
    ["slice", np.repeat(0.1, 4)],  # one more than required
])
def test_set_parameter_value_error(category, value):
    cop = GaussianCopula(3)
    with pytest.raises(ValueError):
        if category == 'full':
            cop[:] = value
        elif category == 'slice':
            cop[:, 0] = value


@pytest.mark.parametrize("index", [
    (0, 1, 2),
    (-1, 0),
    (0, 4),
    (1, 1)
])
def test_set_parameter_index_error(index):
    cop = GaussianCopula(3)
    with pytest.raises(IndexError):
        cop[index] = 0.3


def test_get_parameter():
    cop = GaussianCopula(3)
    cop[:] = gen_corr()

    assert isinstance(cop[:], np.ndarray)
    assert isinstance(cop[:, 0], np.ndarray)
    assert isinstance(cop[0, 0], (float, int))


def test_get_parameter_index_error():
    cop = GaussianCopula(3)
    cop[:] = gen_corr()

    with pytest.raises(IndexError):
        cop[0, 0, 0]

    with pytest.raises(IndexError):
        cop[{}]
