from copulae.utils import merge_dicts, reshape_data
import numpy as np
import pytest

from copulae import GaussianCopula


def test_merge_dicts():
    d1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
    d2 = {'e': 5, 'b': {'c': 3, 'f': 4}}

    final = merge_dicts(d1, d2)
    expected = {'a': 1, 'b': {'c': 3, 'd': 3, 'f': 4}, 'e': 5}

    assert final == expected
    assert merge_dicts(d1) == d1


def test_reshape_data_decorator():
    copula = GaussianCopula(2)

    func = reshape_data(lambda cop, x: x)

    res = func(copula, np.zeros(2))
    assert res.ndim == 2

    with pytest.raises(ValueError):
        func(copula, np.zeros((3, 3, 3)))

    with pytest.raises(ValueError):
        func(copula, np.zeros((10, 4)))

    func = reshape_data(lambda cop, x: np.array([0.0]))
    assert func(copula, np.zeros(2)) == 0
