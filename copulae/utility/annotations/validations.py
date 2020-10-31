import inspect
from typing import Any, Callable, Collection, Dict, Union

import numpy as np
import pandas as pd
from wrapt import decorator

__all__ = ['validate_data_dim']


# noinspection PyPep8Naming
class validate_data_dim:
    def __init__(self, dims: Dict[str, Union[int, Collection[int]]]):
        """
        Annotation to assert that the method has a specific variable with a specific dimension

        As a useful side effect, it will also cast the parameters (if they are not numpy arrays or
        pandas DataFrames) into numpy arrays. That is, list of floats will be cast as numpy arrays.

        Parameters
        ----------
        dims
            A dictionary where the keys are the variable (argument) names to check and the value is
            either a list of int or an int. The integers represent the ideal dimension(s) of the argument

        Examples
        --------
        >>> from copulae.utility.annotations import validate_data_dim
        >>> class Test:
        ...     @validate_data_dim({"data1": [1, 2], "data2": 2, "data3": [1]})
        ...     def __init__(self, data1, data2, *, data3):
        ...         pass
        >>> d1 = [1]
        >>> d2 = [[2, 2]]
        >>> d3 = [3, 3]
        >>> works1 = Test(d1, d2, data3=d3)
        >>> works2 = Test(d1, data2=d2, data3=d3)
        >>> errors = Test([[1, 2]], [1, 2,], [[3, 3]])
        """
        assert len(dims) > 0, "dims cannot be empty"

        def _valid_dim(s: int):
            return isinstance(s, int) and s >= 1

        self.dims = {}
        for arg_name, arg_dim in dims.items():
            if np.isscalar(arg_dim):
                assert _valid_dim(arg_dim), "dimension must be an integer"
                arg_dim = arg_dim,
            else:
                arg_dim = tuple(arg_dim)
                assert all(_valid_dim(i) for i in arg_dim), "dimension must be an integer"
            self.dims[arg_name] = arg_dim

    @decorator
    def __call__(self, method: Callable[..., Any], _, args, kwargs):
        specs = inspect.getfullargspec(method)
        # skip first because first is "self"
        assert {*specs.args[1:], *specs.kwonlyargs} >= self.dims.keys(), \
            f"missing arguments names in {method.__name__}"

        # organize the inputs arguments
        args = list(args)
        errors = {k: "" for k in self.dims}

        # checks args for errors
        for arg_name, (i, item) in zip(specs.args[1:], enumerate(args)):
            if arg_name in self.dims:
                args[i] = self.cast(item)
                self.update_errors(errors, arg_name, args[i])

        # check kwargs for errors
        for arg_name, item in kwargs.items():
            if arg_name in self.dims:
                kwargs[arg_name] = self.cast(item)
                self.update_errors(errors, arg_name, kwargs[arg_name])

        errors = {k: v for k, v in errors.items() if v != ""}
        if any(errors):
            raise ValueError("errors in input dimensions: \n\t" + '\n\t'.join(errors.values()))

        return method(*args, **kwargs)

    @staticmethod
    def cast(item: Any):
        return item if isinstance(item, (np.ndarray, pd.Series, pd.DataFrame)) else np.array(item)

    def update_errors(self, errors: Dict[str, str], arg_name: str, arg: Union[np.ndarray, pd.DataFrame]):
        ideal_dims = self.dims[arg_name]
        dim = arg.ndim
        if dim not in ideal_dims:
            message = f"expected {arg_name} to have "
            message += (f"dimensions {ideal_dims}" if len(ideal_dims) > 1 else f"dimension {ideal_dims[0]}")
            message += f" but got {dim} instead"

            errors[arg_name] = message
