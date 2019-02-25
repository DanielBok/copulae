from typing import Tuple

import numpy as np
from scipy import stats

from copulae.utility import format_docstring

__all__ = ['corr', 'kendall_tau', 'pearson_rho', 'spearman_rho']

__corr_doc__ = """
    If x and y are vectors, calculates the correlation between the 2 vectors. If x is a matrix, calculates the
    correlation between every 2 columns. 3 types of correlation are supported: Pearson, Kendall and Spearman.

    'use' determines how missing values are handled. 'everything' will propagate NA values resultinng in NA for the
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


@format_docstring(corr_doc=__corr_doc__)
def corr(x: np.ndarray, y: np.ndarray = None, method='pearson', use='everything'):
    """
    Calculates the correlation

    {corr_doc}

    :param x: numpy array
        A numeric vector, matrix
    :param y: optional, numpy array
        A numeric vector, matrix. Optional
    :param method: str
        One of 'pearson' (default), 'spearman' or 'kendall'
    :param use: str
        One of 'everything' (default), 'complete', 'pairwise.complete'
    :return: numpy array
        Correlation matrix
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


@format_docstring(corr_doc=__corr_doc__)
def pearson_rho(x: np.ndarray, y: np.ndarray = None, use='everything'):
    """
    Calculates the Pearson correlation

    {corr_doc}

    :param x: numpy array
        A numeric vector, matrix
    :param y: optional, numpy array
        A numeric vector, matrix. Optional
    :param use: str
        One of 'everything' (default), 'complete', 'pairwise.complete'
    :return: numpy array
        Pearson Correlation matrix
    """
    return corr(x, y, 'pearson', use)


@format_docstring(corr_doc=__corr_doc__)
def kendall_tau(x: np.ndarray, y: np.ndarray = None, use='everything'):
    """
    Calculates the Kendall Tau correlation

    {corr_doc}

    :param x: numpy array
        A numeric vector, matrix
    :param y: optional, numpy array
        A numeric vector, matrix. Optional
    :param use: str
        One of 'everything' (default), 'complete', 'pairwise.complete'
    :return: numpy array
        Kendall Tau Correlation matrix
    """
    return corr(x, y, 'kendall', use)


@format_docstring(corr_doc=__corr_doc__)
def spearman_rho(x: np.ndarray, y: np.ndarray = None, use='everything'):
    """
    Calculates the Spearman Rho correlation

    {corr_doc}

    :param x: numpy array
        A numeric vector, matrix
    :param y: optional, numpy array
        A numeric vector, matrix. Optional
    :param use: str
        One of 'everything' (default), 'complete', 'pairwise.complete'
    :return: numpy array
        Spearman Rho Correlation matrix
    """
    return corr(x, y, 'spearman', use)


def _get_corr_func(method: str):
    """
    Returns the correlation

    :param method: str
        correlation function name
    :return: callable
        A correlation function
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
    """
    Validates the 'use' argument. If invalid, raises an error
    """
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
    Yields valid data vectors based on the 'use' argument

    :param x: ndarray
        data vector
    :param y: ndarray
        data vector
    :param use: str
        determines how missing values are handled. 'everything' will propagate NA values resultinng in NA for the
        correlation. 'complete' will remove rows where NA exists. 'pairwise.complete' will remove rows for the pairwise
        columns being compared where NA exists.
    :return: Tuple[ndarray, ndarray]
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
