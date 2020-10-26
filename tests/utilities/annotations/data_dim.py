import pytest

from copulae.utility.annotations import validate_data_dim


class TestCls:
    @validate_data_dim({"data1": 2, "data2": [1, 2], "data3": 1})
    def __init__(self, data1, _, data2, *, data3):
        pass

    @validate_data_dim({"data4": 1})
    def method(self, smoothing, data4):
        pass


@pytest.mark.parametrize("args, kwargs", [
    [([[1, 1]], "_arg", [1]), {"data3": [1, 2, 3]}],
    [([[2, 2]], "_arg", [[1]]), {"data3": [1, 2, 3]}],
    [([[3, 3]], "_arg"), {"data2": [1], "data3": [1, 2, 3]}],
])
def test_validation_data_dim(args, kwargs):
    cls = TestCls(*args, **kwargs)
    cls.method("none", data4=[1, 2, 3])
    cls.method("none", [1, 2, 3])
    cls.method(data4=[1, 2, 3], smoothing="none")


def test_empty_validation_rule_raises_error():
    with pytest.raises(AssertionError):
        class BadClass:
            @validate_data_dim({})
            def __init__(self):
                pass


@pytest.mark.parametrize("dims", [
    {"data1": -1},
    {"data1": [-1, 1]},
])
def test_non_integer_shape_raises_error(dims):
    print(dims)
    with pytest.raises(AssertionError):
        class BadClass:
            @validate_data_dim(dims)
            def __init__(self, data1):
                pass
