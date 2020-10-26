from typing import Union

import numpy as np
import pandas as pd
import wrapt

from copulae.copula import BaseCopula

__all__ = ['cast_output', 'shape_first_input_to_cop_dim', "squeeze_output"]


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
def squeeze_output(method, _, args, kwargs):
    """Squeezes the output to a float if the size is one"""
    output: pd.Series = method(*args, **kwargs)
    return float(output) if output.size == 1 else output
