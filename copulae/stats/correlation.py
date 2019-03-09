from typing import Tuple

import numpy as np
from scipy import stats

__all__ = ['corr', 'kendall_tau', 'pearson_rho', 'spearman_rho']


def format_docstring(notes):
    def decorator(func):
        func.__doc__ = func.__doc__ % notes
        return func

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


@format_docstring(__notes__)
def corr(x: np.ndarray, y: np.ndarray = None, method='pearson', use='everything'):
    """
    Calculates the correlation

    Parameters
    ----------
    x: ndarray
        Numeric vector to compute correlation. If matrix, `y` can be omitted and correlation will be calculated
        amongst the different columns

    y: ndarray, optional
        Numeric vector to compute correlation against `x`

    method: {'pearson', 'spearman', 'kendall'}, optional
        The method for calculating correlation

    use: {'everything', 'complete', 'pairwise.complete'}
        The method to handle missing data

    Returns
    -------
    ndarray
        Correlation matrix

    Notes
    -----
    %s
    """
    use = _validate_use(use)
    corr_func = _get_corr_func(method)

    if y is not None:
        x, y = _form_xy_vector(x, y, use)
        c = np.identity(2)
        c[0, 1] = c[1, 0] = corr_func(x, y)[0]

    else:
        if len(x.shape) != 2:
            raise ValueError('x must be a matrix with dimension 2')
        c = np.identity(x.shape[1])
        for (i, j), (c1, c2) in _yield_vectors(x, use):
            c[i, j] = c[j, i] = corr_func(c1, c2)[0]

    return c


@format_docstring(__notes__)
def pearson_rho(x: np.ndarray, y: np.ndarray = None, use='everything'):
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
    ndarray
        Correlation matrix

    Notes
    -----
    %s
    """
    return corr(x, y, 'pearson', use)


@format_docstring(__notes__)
def kendall_tau(x: np.ndarray, y: np.ndarray = None, use='everything'):
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
    ndarray
        Correlation matrix

    Notes
    -----
    %s
    """
    return corr(x, y, 'kendall', use)


@format_docstring(__notes__)
def spearman_rho(x: np.ndarray, y: np.ndarray = None, use='everything'):
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
    ndarray
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

    Returns
    -------
    Callable
        The correlation function
    """

    method = method.lower()
    valid_methods = {'pearson', 'kendall', 'spearman', 'tau', 'rho'}

    if method not in valid_methods:
        raise ValueError(f"method must be one of {', '.join(valid_methods)}")

    if method in {'kendall', 'tau'}:
        return stats.kendalltau
    elif method in {'spearman', 'rho'}:
        return stats.spearmanr
    else:
        return stats.pearsonr


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
