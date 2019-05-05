from abc import ABC
from typing import Tuple

import numpy as np

from copulae.copula.base import BaseCopula
from copulae.core import create_cov_matrix, is_psd, near_psd, tri_indices
from copulae.utility import array_io


class AbstractEllipticalCopula(BaseCopula, ABC):
    """
    The abstract base class for Elliptical Copulas
    """

    def __init__(self, dim: int, name: str):
        super().__init__(dim, name)
        self._rhos = np.zeros(sum(range(dim)))

    @array_io(optional=True)
    def drho(self, x=None):
        if x is None:
            x = self._rhos
        return 6 / (np.pi * np.sqrt(4 - x ** 2))

    @array_io(optional=True)
    def dtau(self, x=None):
        if x is None:
            x = self._rhos
        return 2 / (np.pi * np.sqrt(1 - x ** 2))

    @array_io
    def itau(self, tau):
        return np.sin(np.asarray(tau) * np.pi / 2)

    def log_lik(self, data: np.ndarray):
        if not is_psd(self.sigma):
            return -np.inf

        if hasattr(self, '_df') and self._df <= 0:  # t copula
            return -np.inf

        return super().log_lik(data)

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

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._rhos[i]
        elif isinstance(i, (slice, tuple, list, np.ndarray)):
            if len(i) == 2:
                return self.sigma[i]
            else:
                raise IndexError('only 2-dimensional indices supported')
        raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean "
                         "arrays are valid indices")

    def __setitem__(self, i, value):
        d = self.dim

        if isinstance(i, slice):
            value = near_psd(value)
            if value.shape != (d, d):
                return IndexError(f"The value being set should be a matrix of dimension ({d}, {d})")
            self._rhos = value[tri_indices(d, 1, 'lower')]
            return

        assert -1.0 <= value <= 1.0, "correlation value must be between -1 and 1"
        if isinstance(i, int):
            self._rhos[i] = value

        else:
            i = _get_rho_index(d, i)
            self._rhos[i] = value

        self._force_psd()

    def __delitem__(self, i):
        d = self.dim

        if isinstance(i, slice):
            self._rhos = np.zeros(len(self._rhos))

        elif isinstance(i, int):
            self._rhos[i] = 0

        else:
            i = _get_rho_index(d, i)
            self._rhos[i] = 0

        self._force_psd()

    def _force_psd(self):
        """
        Forces covariance matrix to be positive semi-definite. This is useful when user is overwriting covariance
        parameters
        """
        cov = near_psd(self.sigma)
        self._rhos = cov[tri_indices(self.dim, 1, 'lower')]


def _get_rho_index(d: int, i: Tuple[int, int]) -> int:
    i = tuple(i)
    if not isinstance(i, tuple):
        raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean "
                         "arrays are valid indices")

    if len(i) != 2:
        raise IndexError('only 2-dimensional indices supported')

    x, y = i
    if x < 0 or y < 0:
        raise IndexError('Only positive indices are supported')
    elif x >= d or y >= d:
        raise IndexError('Index cannot be greater than dimension of copula')
    elif x == y:
        raise IndexError('Cannot set values along the diagonal')

    for j, v in enumerate(zip(*tri_indices(d, 1, 'upper' if x < y else 'lower'))):
        if i == v:
            return j

    raise IndexError(f"Unable to find index {i}")
