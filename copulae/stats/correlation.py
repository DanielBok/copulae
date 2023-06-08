from functools import wraps
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ['corr', 'kendall_tau', 'pearson_rho', 'spearman_rho', 'CorrMethod', 'CorrDataUse']

CorrMethod = Literal['pearson', 'kendall', 'spearman', 'tau', 'rho']
CorrDataUse = Literal['everything', 'complete', 'pairwise.complete']


def format_docstring(notes):
    def decorator(func):
        func.__doc__ = func.__doc__ % notes
        return func

    return decorator


def format_output(f):
    """Formats the output as a DataFrame based on the inputs"""

    @wraps(f)
    def decorator(x: Union[np.ndarray, pd.Series, pd.DataFrame],
                  y: Union[np.ndarray, pd.Series] = None,
                  method: CorrMethod = 'pearson',
                  use: CorrDataUse = 'everything') -> Union[np.ndarray, pd.DataFrame]:
        _corr = f(np.asarray(x), y if y is None else np.asarray(y), method, use)
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(_corr, index=x.columns, columns=x.columns)

        elif isinstance(x, pd.Series) and isinstance(y, pd.Series):
            x_name = x.name or 'X'
            y_name = y.name or 'Y'

            if x_name == y_name:
                x_name, y_name = f'{x_name}1', f'{y_name}2'
            names = [x_name, y_name]
            return pd.DataFrame(_corr, index=names, columns=names)

        return _corr

    return decorator


__notes__ = """
    If x and y are vectors, calculates the correlation between the 2 vectors. If x is a matrix, calculates the
    correlation between every 2 columns. 3 types of correlation are supported: Pearson, Kendall and Spearman.

    'use' determines how missing values are handled. 'everything' will propagate NA values resulting in NA for the
    correlation. 'complete' will remove rows where NA exists. 'pairwise.complete' will remove rows for the pairwise 
    columns being compared where NA exists.
    
    For example:
    b = [[NA, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    if comparison is 'everything', when comparing
    corr 1: 
        column: 1, 2 
        data used: [NA, 3, 6], [1, 4, 7]
    corr 2:
        column: 1, 3
        data used: [NA, 3, 6], [2, 5, 8]
    corr 3:
        data used: [1, 4, 7], [2, 5, 8]
    
    if comparison is 'complete', when comparing
    corr 1: 
        column: 1, 2 
        data used: [3, 6], [4, 7]
    corr 2:
        column: 1, 3
        data used: [3, 6], [5, 8]
    corr 3:
        data used: [4, 7], [5, 8]
    
    if comparison is 'pairwise.complete', when comparing
    corr 1: 
        column: 1, 2 
        data used: [3, 6], [4, 7]
    corr 2:
        column: 1, 3
        data used: [3, 6], [5, 8]
    corr 3:
        data used: [1, 4, 7], [2, 5, 8]
""".strip()


@format_output
@format_docstring(__notes__)
def corr(x: Union[np.ndarray, pd.Series, pd.DataFrame],
         y: Union[np.ndarray, pd.Series] = None,
         method: CorrMethod = 'pearson',
         use: CorrDataUse = 'everything') -> Union[pd.DataFrame, np.ndarray]:
    """
    Calculates the correlation

    Parameters
    ----------
    x: ndarray, pd.Series, pd.DataFrame
        Numeric vector to compute correlation. If matrix, `y` can be omitted and correlation will be calculated
        amongst the different columns

    y: ndarray, pd.Series, optional
        Numeric vector to compute correlation against `x`. If `y` is provided, both `x` and `y` must be a
        1-dimensional vector

    method: {'pearson', 'spearman', 'kendall'}, optional
        The method for calculating correlation

    use: {'everything', 'complete', 'pairwise.complete'}
        The method to handle missing data

    Returns
    -------
    DataFrame or ndarray
        Correlation matrix

    Notes
    -----
    %s
    """
    x = np.asarray(x)

    use = _validate_use(use)
    compute_corr = _get_corr_func(method)

    if y is not None:
        y = np.asarray(y)
        c = np.identity(2)
        c[0, 1] = c[1, 0] = compute_corr(*_form_xy_vector(x, y, use))

    else:
        if len(x.shape) != 2:
            raise ValueError('x must be a matrix with dimension 2')
        c = np.identity(x.shape[1])
        for (i, j), (c1, c2) in _yield_vectors(x, use):
            c[i, j] = c[j, i] = compute_corr(c1, c2)

    return c


@format_docstring(__notes__)
def pearson_rho(x: np.ndarray, y: np.ndarray = None, use: CorrDataUse = 'everything') -> Union[
    pd.DataFrame, np.ndarray]:
    """
    Calculates the Pearson correlation

    Parameters
    ----------
    x: ndarray
        Numeric vector to compute correlation. If matrix, `y` can be omitted and correlation will be calculated
        amongst the different columns

    y: ndarray, optional
        Numeric vector to compute correlation against `x`

    use: {'everything', 'complete', 'pairwise.complete'}
        The method to handle missing data

    Returns
    -------
    DataFrame or ndarray
        Correlation matrix

    Notes
    -----
    %s
    """
    return corr(x, y, 'pearson', use)


@format_docstring(__notes__)
def kendall_tau(x: np.ndarray, y: np.ndarray = None, use: CorrDataUse = 'everything') -> Union[
    pd.DataFrame, np.ndarray]:
    """
    Calculates the Kendall's Tau correlation

    Parameters
    ----------
    x: ndarray
        Numeric vector to compute correlation. If matrix, `y` can be omitted and correlation will be calculated
        amongst the different columns

    y: ndarray, optional
        Numeric vector to compute correlation against `x`

    use: {'everything', 'complete', 'pairwise.complete'}
        The method to handle missing data

    Returns
    -------
    DataFrame or ndarray
        Correlation matrix

    Notes
    -----
    %s
    """
    return corr(x, y, 'kendall', use)


@format_docstring(__notes__)
def spearman_rho(x: np.ndarray, y: np.ndarray = None, use: CorrDataUse = 'everything') -> Union[
    pd.DataFrame, np.ndarray]:
    """
    Calculates the Spearman's Rho correlation

    Parameters
    ----------
    x: ndarray
        Numeric vector to compute correlation. If matrix, `y` can be omitted and correlation will be calculated
        amongst the different columns

    y: ndarray, optional
        Numeric vector to compute correlation against `x`

    use: {'everything', 'complete', 'pairwise.complete'}
        The method to handle missing data

    Returns
    -------
    DataFrame or ndarray
        Correlation matrix

    Notes
    -----
    %s
    """
    return corr(x, y, 'spearman', use)


def _get_corr_func(method: str):
    """
    Determines the correlation function

    Parameters
    ----------
    method: str
        Correlation function name
    """

    method = method.lower()
    valid_methods = {'pearson', 'kendall', 'spearman', 'tau', 'rho'}

    if method not in valid_methods:
        raise ValueError(f"method must be one of {', '.join(valid_methods)}")

    if method in {'kendall', 'tau'}:
        corr_func = stats.kendalltau
    elif method in {'spearman', 'rho'}:
        corr_func = stats.spearmanr
    else:
        corr_func = stats.pearsonr

    def compute_corr(u: np.ndarray, v: np.ndarray) -> float:
        return np.nan if any(~np.isfinite(u) | ~np.isfinite(v)) else corr_func(u, v)[0]

    return compute_corr


def _validate_use(use: str):
    """Validates the 'use' argument. If invalid, raises an error"""
    use = use.lower()
    valid_use_keywords = {'everything', 'pairwise.complete', 'complete'}

    if use not in valid_use_keywords:
        raise ValueError(f"use must be one of {', '.join(valid_use_keywords)}")

    return use


def _yield_vectors(x: np.ndarray, use: str):
    """
    Yields valid column vectors depending on 'use' argument from the data provided

    :param x: ndarray
        data array
    :param use: str
        determines how missing values are handled. 'everything' will propagate NA values resultinng in NA for the
        correlation. 'complete' will remove rows where NA exists. 'pairwise.complete' will remove rows for the pairwise
        columns being compared where NA exists.
    """
    mask = ~np.isnan(x)
    if use == 'complete':
        x = x[mask.all(1)]

    cols = x.shape[1]
    for i in range(cols):
        for j in range(i + 1, cols):
            if use == 'pairwise.complete':
                v = mask[:, (i, j)].all(1)
                yield (i, j), (x[v, i], x[v, j])
            else:
                yield (i, j), (x[:, i], x[:, j])


def _form_xy_vector(x: np.ndarray, y: np.ndarray, use: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yields valid data vectors based on the `use` argument

    Parameters
    ----------
    x: ndarray
        Numeric vector to compute correlation. If matrix, `y` can be omitted and correlation will be calculated
        amongst the different columns

    y: ndarray, optional
        Numeric vector to compute correlation against `x`

    use: {'everything', 'complete', 'pairwise.complete'}
        The method to handle missing data. 'everything' will propagate NA values resultinng in NA for the correlation.
        'complete' will remove rows where NA exists. 'pairwise.complete' will remove rows for the pairwise
        columns being compared where NA exists.

    Returns
    -------
    Tuple[ndarray, ndarray]
        2 column vectors
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('x and y must be 1 dimension vector for correlation function')
    if len(x) != len(y):
        raise ValueError('x and y must be similar in length')

    if use == 'everything':
        return x, y

    v = ~(np.isnan(x) | np.isnan(y))
    return x[v], y[v]
