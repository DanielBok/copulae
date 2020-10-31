from abc import ABC
from typing import Collection, Tuple, Union

import numpy as np

from copulae.copula import BaseCopula, Param
from copulae.core import EPS, create_cov_matrix, is_psd, near_psd, tri_indices
from copulae.utility.annotations import *


class AbstractEllipticalCopula(BaseCopula[Param], ABC):
    """
    The abstract base class for Elliptical Copulas
    """

    def __init__(self, dim: int, name: str):
        super().__init__(dim, name)
        self._rhos = np.zeros(sum(range(dim)))

    @cast_input(['x'], optional=True)
    @squeeze_output
    def drho(self, x=None):
        if x is None:
            x = self._rhos
        return 6 / (np.pi * np.sqrt(4 - x ** 2))

    @cast_input(['x'], optional=True)
    @squeeze_output
    def dtau(self, x=None):
        if x is None:
            x = self._rhos
        return 2 / (np.pi * np.sqrt(1 - x ** 2))

    @cast_input(['tau'])
    @squeeze_output
    def itau(self, tau):
        return np.sin(np.asarray(tau) * np.pi / 2)

    def log_lik(self, data: np.ndarray, *, to_pobs=True, ties="average"):
        if not is_psd(self.sigma):
            return -np.inf

        if hasattr(self, '_df') and getattr(self, "_df", 0) <= 0:  # t copula
            return -np.inf

        return super().log_lik(data, to_pobs=to_pobs, ties=ties)

    @property
    def rho(self):
        return np.arcsin(self._rhos / 2) * 6 / np.pi

    @property
    def sigma(self):
        """
        The covariance matrix for the elliptical copula

        :return: numpy array
            Covariance matrix for elliptical copula
        """
        return create_cov_matrix(self._rhos)

    @property
    def tau(self):
        return 2 * np.arcsin(self._rhos) / np.pi

    def __getitem__(self, index: Union[int, Tuple[Union[slice, int], Union[slice, int]], slice]) \
            -> Union[np.ndarray, float]:
        if isinstance(index, slice):
            return self.sigma
        elif isinstance(index, int):
            return self._rhos[index]
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError('only 2-dimensional indices supported')
            return self.sigma[index]
        raise IndexError("invalid index type")

    def __setitem__(self,
                    index: Union[int, Tuple[Union[slice, int], Union[slice, int]], slice],
                    value: Union[float, Collection[float], np.ndarray]):
        d = self.dim

        if np.isscalar(value):
            if value < -1 or value > 1:
                raise ValueError("correlation value must be between -1 and 1")
        else:
            value = np.asarray(value)
            if not np.all((value >= -1 - EPS) & (value <= 1 + EPS)):
                raise ValueError("correlation value must be between -1 and 1")

        if isinstance(index, slice):
            value = near_psd(value)
            if value.shape != (d, d):
                raise ValueError(f"The value being set should be a matrix of dimension ({d}, {d})")
            self._rhos = value[tri_indices(d, 1, 'lower')]
            return

        if isinstance(index, int):
            self._rhos[index] = value

        else:
            index = tuple(index)
            if len(index) != 2:
                raise IndexError('index can only be 1 or 2-dimensional')

            x, y = index
            # having 2 slices for indices is equivalent to self[:]
            if isinstance(x, slice) and isinstance(y, slice):
                self[:] = value
                return
            elif isinstance(x, slice) or isinstance(y, slice):
                value = np.repeat(value, d) if np.isscalar(value) else np.asarray(value)
                if len(value) != d:
                    raise ValueError(f"value must be a scalar or be a vector with length {d}")

                # one of the item is
                for i, v in enumerate(value):
                    idx = (i, y) if isinstance(x, slice) else (x, i)
                    if idx[0] == idx[1]:  # skip diagonals
                        continue

                    idx = _get_rho_index(d, idx)
                    self._rhos[idx] = v
            else:
                # both are integers
                idx = _get_rho_index(d, index)
                self._rhos[idx] = float(value)

        self._force_psd()

    def _force_psd(self):
        """
        Forces covariance matrix to be positive semi-definite. This is useful when user is overwriting covariance
        parameters
        """
        cov = near_psd(self.sigma)
        self._rhos = cov[tri_indices(self.dim, 1, 'lower')]


def _get_rho_index(dim: int, index: Tuple[int, int]) -> int:
    x, y = index
    if x < 0 or y < 0:
        raise IndexError('Only positive indices are supported')
    elif x >= dim or y >= dim:
        raise IndexError('Index cannot be greater than dimension of copula')
    elif x == y:
        raise IndexError('Cannot set values along the diagonal')

    for j, v in enumerate(zip(*tri_indices(dim, 1, 'upper' if x < y else 'lower'))):
        if (x, y) == v:
            return j

    raise IndexError(f"Unable to find index {(x, y)}")
