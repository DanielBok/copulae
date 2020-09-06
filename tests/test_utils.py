import pytest

from copulae import GaussianCopula
from copulae.utility.array import array_io
from copulae.utility.dict import merge_dicts


def test_merge_dicts():
    d1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
    d2 = {'e': 5, 'b': {'c': 3, 'f': 4}}

    final = merge_dicts(d1, d2)
    expected = {'a': 1, 'b': {'c': 3, 'd': 3, 'f': 4}, 'e': 5}

    assert final == expected
    assert merge_dicts(d1) == d1


def test_reshape_data_decorator():
    copula = GaussianCopula(2)

    # test that array output with 1 element becomes scalar and is float
    x1 = array_io(lambda cop, x: [1])(copula, [0.5, 0.5])
    assert x1 == 1
    assert isinstance(x1, float)

    # test that vector output gets squeezed if one of the dimension is 1
    for output in (
            [[1, 2]],
            [[1], [2]]
    ):
        x2 = array_io(lambda cop, x: output)(copula, [0.5, 0.5])
        assert x2.ndim == 1, "vector output must have squeezed"

    # test that 2D output stays 2D
    x3 = array_io(lambda cop, x: [[1, 2], [3, 4]])(copula, [0.5, 0.5])
    assert x3.ndim == 2, "2D output should remain 2D"

    # Non-optional input
    with pytest.raises(ValueError):
        array_io(lambda cop, x: x, optional=False)(copula, None)
