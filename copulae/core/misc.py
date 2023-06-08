from typing import Literal, Tuple, TypeVar

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ['create_cov_matrix', 'EPS', 'pseudo_obs', 'rank_data', 'tri_indices']

EPS = np.finfo('float').eps
"""Machine Epsilon"""

ArrayVar = TypeVar('ArrayVar', np.ndarray, pd.DataFrame)


def create_cov_matrix(params: np.ndarray):
    """
    Creates a matrix from a given vector of parameters.

    Useful for elliptical copulae where we translate the rhos to the covariance matrix

    Parameters
    ----------
    params: array like
        (1, N) vector of parameters

    Returns
    -------
    numpy array
        (N x N) matrix where the upper and lower triangles are the parameters and the diagonal is a vector of 1
    """
    c = len(params)
    d = int(1 + (1 + 4 * 2 * c) ** 0.5) // 2  # dimension of matrix, determine this from the length of params

    sigma = np.identity(d)
    sigma[tri_indices(d, 1)] = np.tile(params, 2)
    return sigma


def pseudo_obs(data: ArrayVar, ties='average') -> ArrayVar:
    """
    Compute the pseudo-observations for the given data matrix

    Parameters
    ----------
    data: (N, D) ndarray
        Random variates to be converted to pseudo-observations

    ties: str, optional
        The method used to assign ranks to tied elements. The options are 'average', 'min', 'max', 'dense'
        and 'ordinal'.

        **average**
            The average of the ranks that would have been assigned to all the tied values is
            assigned to each value.
        **min**
            The minimum of the ranks that would have been assigned to all the tied values is
            assigned to each value. (This is also referred to as "competition" ranking.)
        **max**
            The maximum of the ranks that would have been assigned to all the tied values is
            assigned to each value.
        **dense**
            Like *min*, but the rank of the next highest element is assigned the rank immediately
            after those assigned to the tied elements. 'ordinal': All values are given a distinct rank,
            corresponding to the order that the values occur in `a`.

    Returns
    -------
    numpy.array or pandas.DataFrame
        matrix or vector of the same dimension as `data` containing the pseudo observations

    Examples
    --------
    >>> from copulae import pseudo_obs
    >>> from copulae.datasets import load_marginal_data
    >>> import numpy as np
    >>> data = load_marginal_data()
    >>> data.head(3)
        STUDENT      NORM       EXP
    0 -0.485878  2.646041  0.393322
    1 -1.088878  2.906977  0.253731
    2 -0.462133  3.166951  0.480696
    >>> pseudo_obs(data).head(3)  # pseudo-obs is a DataFrame because input is a DataFrame
        STUDENT      NORM       EXP
    0  0.325225  0.188604  0.557814
    1  0.151616  0.399533  0.409530
    2  0.336221  0.656115  0.626458
    >>> np.random.seed(1)
    >>> rand = np.random.normal(size=(100, 3))
    >>> rand[:3].round(3)
    array([[ 1.624, -0.612, -0.528],
           [-1.073,  0.865, -2.302],
           [ 1.745, -0.761,  0.319]])
    >>> pseudo_obs(rand)[:3].round(3)  # otherwise returns numpy arrays
    array([[0.921, 0.208, 0.248],
           [0.168, 0.792, 0.01 ],
           [0.941, 0.178, 0.584]])
    >>> pseudo_obs(rand.tolist())[:3].round(3)
    array([[0.921, 0.208, 0.248],
           [0.168, 0.792, 0.01 ],
           [0.941, 0.178, 0.584]])
    """
    u = rank_data(data, 1, ties) / (len(data) + 1)
    if isinstance(data, pd.DataFrame):
        u = pd.DataFrame(u, index=data.index, columns=data.columns)

    return u


def rank_data(obs: np.ndarray, axis=0, ties='average'):
    """
    Assign ranks to data, dealing with ties appropriately. This function works on core as well as vectors

    Parameters
    ----------
    obs
        Data to be ranked. Can only be 1 or 2 dimensional.
    axis: {0, 1}, optional
        The axis to perform the ranking. 0 means row, 1 means column.
    ties: { 'average', 'min', 'max', 'dense', 'ordinal' }, default 'average'
        The method used to assign ranks to tied elements. The options are 'average', 'min', 'max', 'dense'
        and 'ordinal'.

    Returns
    -------
    numpy array
        matrix or vector of the same dimension as X containing the pseudo observations

    See Also
    --------
    :py:func:`~copulae.core.misc.pseudo_obs`
        The pseudo-observations function
    """
    obs = np.asarray(obs)
    ties = ties.lower()
    assert obs.ndim in (1, 2), "Data can only be 1 or 2 dimensional"

    if obs.ndim == 1:
        return stats.rankdata(obs, ties)
    elif obs.ndim == 2:
        if axis == 0:
            return np.array([stats.rankdata(obs[i, :], ties) for i in range(obs.shape[0])])
        elif axis == 1:
            return np.array([stats.rankdata(obs[:, i], ties) for i in range(obs.shape[1])]).T
        else:
            raise ValueError(f"No axis named 3 for object type {type(obs)}")


def tri_indices(n: int, m=0, side: Literal['lower', 'upper', 'both'] = 'both') -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the indices for the triangle of an (n, n) array

    Parameters
    ----------
    n : int
        dimension of square matrix
    m : int
        offset
    side : { 'lower', 'upper', 'both' }
        Side of triangle to return. Supported values are 'lower', 'upper', 'both'

    Returns
    -------
    (numpy array, numpy array)
        Tuple of row indices and column indices

    Examples
    --------
    >>> import numpy as np
    >>> from copulae.core import tri_indices
    >>> x = np.arange(9).reshape(3, 3)
    # To get lower indices of matrix
    >>> x[tri_indices(3, 1, 'lower')]
    array([3, 6, 7])
    # To form covariance matrix
    >>> c = np.eye(3)
    >>> c[tri_indices(3, 1)] = np.tile([0.1, 0.2, 0.3], 2)
    >>> c
    array([[1. , 0.1, 0.2],
           [0.1, 1. , 0.3],
           [0.2, 0.3, 1. ]])
    """
    side = side.lower()

    if side not in {'lower', 'upper', 'both'}:
        raise ValueError("side option must be one of 'lower', 'upper' or 'both'")

    l_i = [], []  # lower indices
    if side in {'lower', 'both'}:
        for i in range(n - m):
            for j in range(i + m, n):
                l_i[1].append(i)
                l_i[0].append(j)

        if side == 'lower':
            return np.array(l_i[0]), np.array(l_i[1])

    u_i = [[], []]  # upper indices
    if side in {'upper', 'both'}:
        for i in range(n - m):
            for j in range(i + m, n):
                u_i[0].append(i)
                u_i[1].append(j)
        if side == 'upper':
            return np.array(u_i[0]), np.array(u_i[1])

    return np.array([*u_i[0], *l_i[0]]), np.array([*u_i[1], *l_i[1]])
