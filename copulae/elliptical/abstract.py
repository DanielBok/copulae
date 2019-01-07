from abc import ABC
from typing import Optional

import numpy as np

from copulae.copula.base import BaseCopula
from copulae.math_tools import is_PSD, tri_indices
from copulae.types import Array
from .utils import create_cov_matrix


class AbstractEllipticalCopula(BaseCopula, ABC):
    """
    The abstract base class for Elliptical Copulas
    """

    def __init__(self, dim: int, name: str):
        super().__init__(dim, name)
        self._rhos = np.zeros(sum(range(dim)))
        self.is_elliptical = True

    def log_lik(self, data: np.ndarray):
        if not is_PSD(self.sigma):
            return -np.inf
        return super().log_lik(data)

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

    @property
    def rho(self):
        return np.arcsin(self._rhos / 2) * 6 / np.pi

    def drho(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self._rhos
        return 6 / (np.pi * np.sqrt(4 - x ** 2))

    def dtau(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self._rhos
        return 2 / (np.pi * np.sqrt(1 - x ** 2))

    def itau(self, tau: Array):
        return np.sin(tau * np.pi / 2)

    def __getitem__(self, i):
        if type(i) is int:
            return self._rhos[i]
        elif hasattr(i, '__len__'):
            if len(i) == 2:
                return self.sigma[i]
            else:
                raise IndexError('only 2-dimensional indices supported')
        raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean "
                         "arrays are valid indices")

    def __setitem__(self, i, value):
        d = self.dim
        if type(i) is int:
            self._rhos[i] = value
            return

        if type(i) is slice:
            value = np.asarray(value)
            if value.shape != (d, d):
                return IndexError(f"The value being set shoud be a matrix of dimension ({d}, {d})")
            self._rhos = value[tri_indices(d, 1, 'lower')]
            return

        if hasattr(i, '__len__'):
            if len(i) == 2:
                x, y = i
                if x < 0 or y < 0:
                    raise IndexError('Only positive indices are supported')
                elif x >= d or y >= d:
                    raise IndexError('Index cannot be greater than dimension of copula')
                elif x == y:
                    raise IndexError('Cannot set values along the diagonal')

                for j, v in enumerate(zip(*tri_indices(d, 1, 'upper' if x < y else 'lower'))):
                    if i == v:
                        self._rhos[j] = value
                        return
                else:
                    raise IndexError(f"Unable to find index {i}")
            else:
                raise IndexError('only 2-dimensional indices supported')
        raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean "
                         "arrays are valid indices")

    def __delitem__(self, i):
        d = self.dim
        if type(i) is int:
            self._rhos[i] = 0
            return

        if type(i) is slice:
            self._rhos = np.zeros(len(self._rhos))
            return

        if hasattr(i, '__len__'):
            if len(i) == 2:
                x, y = i
                if x < 0 or y < 0:
                    raise IndexError('Only positive indices are supported')
                elif x >= d or y >= d:
                    raise IndexError('Index cannot be greater than dimension of copula')
                elif x == y:
                    raise IndexError('Cannot set values along the diagonal')

                for j, v in enumerate(zip(*tri_indices(d, 1, 'upper' if x < y else 'lower'))):
                    if i == v:
                        self._rhos[j] = 0
                        return
                else:
                    raise IndexError(f"Unable to find index {i}")
            else:
                raise IndexError('only 2-dimensional indices supported')
        raise IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean "
                         "arrays are valid indices")
