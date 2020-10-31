from inspect import getfullargspec
from typing import List, Union

import numpy as np
import pandas as pd
import wrapt

from copulae.copula import BaseCopula

__all__ = ['cast_input', 'cast_output', 'shape_first_input_to_cop_dim', "squeeze_output"]


# noinspection PyPep8Naming
class cast_input:
    """Cast inputs as numpy arrays if not already a pandas Series, DataFrame or numpy array"""

    def __init__(self, arguments: List[str], optional=False):
        self.arguments = set(arguments)
        self.optional = optional

    @wrapt.decorator
    def __call__(self, method, _, args, kwargs):
        args = list(args)
        specs = getfullargspec(method)

        for arg_name, (i, a) in zip(specs.args[1:], enumerate(args)):
            # cast when:
            # argument must be specified
            # argument must not already be a DataFrame, Series or ndarray
            # argument must not be null while optional is True
            if arg_name in self.arguments:
                args[i] = self.cast(a, arg_name)

        for arg_name, a in kwargs.items():
            if arg_name is self.arguments:
                kwargs[arg_name] = self.cast(a, arg_name)

        return method(*args, **kwargs)

    def cast(self, item, arg_name: str):
        if not self.optional and item is None:
            raise ValueError(f"{arg_name} cannot be None when it is mandatory")

        if self.optional and item is None:
            return item

        return item if isinstance(item, (np.ndarray, pd.Series, pd.DataFrame)) else np.array(item)


@wrapt.decorator
def cast_output(method, instance: BaseCopula, args, kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    Attempts to cast output to a DataFrame if applicable. Class instance must have '_columns' attribute
    """
    columns = getattr(instance, "_columns", None)
    output = method(*args, **kwargs)

    return pd.DataFrame(output, columns=columns) if columns is not None else output


@wrapt.decorator
def shape_first_input_to_cop_dim(method, instance: BaseCopula, args, kwargs):
    """
    Shapes the first input argument to a 2D matrix-like item which has the the same number of columns as
    the copula's dimension.

    As a side effect, also reorders the DataFrame or Series input if the fitted data was a DataFrame.
    With a fitted DataFrame, the _column will be present to allow for ordering
    """
    x = args[0]
    args = list(args)

    columns = getattr(instance, "_columns", None)
    if isinstance(x, pd.Series):
        if columns is not None and set(x.index) == set(columns):
            x = x[columns]  # order by columns
        x = x.to_numpy()[None, :]  # cast to matrix
    elif isinstance(x, pd.DataFrame):
        if columns is not None:
            x = x.loc[:, columns]  # order by columns
    else:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, float)
        if x.ndim == 1:
            x = x[None, :]

    if x.ndim == 1:
        x = x[None, :]  # convert series or 1D vector to 2D vector

    if x.ndim != 2:
        raise ValueError("Input array must have 1 or 2 dimensions")
    elif x.shape[1] != instance.dim:
        raise ValueError("Input array must have same dimension as copula")

    args[0] = x

    return method(*args, **kwargs)


@wrapt.decorator
def squeeze_output(method, _, args, kwargs) -> Union[np.ndarray, pd.Series, float]:
    """Squeezes the output to a float if the size is one"""
    output: Union[float, np.ndarray, pd.Series] = method(*args, **kwargs)
    return float(output) if np.isscalar(output) or output.size == 1 else output
