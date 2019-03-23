import numpy as np
import pytest

from copulae import GaussianCopula
from copulae.utility import merge_dicts, array_io


def test_merge_dicts():
    d1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
    d2 = {'e': 5, 'b': {'c': 3, 'f': 4}}

    final = merge_dicts(d1, d2)
    expected = {'a': 1, 'b': {'c': 3, 'd': 3, 'f': 4}, 'e': 5}

    assert final == expected
    assert merge_dicts(d1) == d1


def test_reshape_data_decorator():
    copula = GaussianCopula(2)

    # test that scalar goes through fine
    assert isinstance(array_io(lambda cop, x: x)(copula, 0.5), float)
    assert isinstance(array_io(lambda cop, x: x, dim=1)(copula, 0.5), float)

    # test that vector goes through okay
    assert array_io(lambda cop, x: x, dim=1)(copula, [0.5, 0.2]).ndim == 1

    # test that 1D array gets converted to 2D array
    res = array_io(lambda cop, x: x, dim=2)(copula, np.zeros(2))
    assert res.ndim == 2

    # Non-optional input
    with pytest.raises(AssertionError):
        array_io(lambda cop, x: x, optional=False)(copula, None)
