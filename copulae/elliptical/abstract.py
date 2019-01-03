from abc import ABC
from typing import Optional

import numpy as np
from copulae.copula.base import BaseCopula
from copulae.math_tools import tri_indices
from copulae.types import Array


class AbstractEllipticalCopula(BaseCopula, ABC):
    """
    The abstract base class for Elliptical Copulas
    """

    def __init__(self, dim: int, name: str):
        super().__init__(dim, name)
        self._rhos = np.zeros(sum(range(dim)))
        self.is_elliptical = True

    @property
    def sigma(self):
        d = self.dim
        sigma = np.identity(d)
        sigma[tri_indices(d, 1)] = np.tile(self._rhos, 2)
        return sigma

    @sigma.setter
    def sigma(self, sigma: Array):
        sigma = np.array(sigma)

        d = self.dim
        if sigma.shape != (d, d):
            raise ValueError(f'correlation matrix needs to be of dimension ({d}, {d}) for copula')

        self._rhos = sigma[tri_indices(d, 1, 'lower')]

    def drho(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self._rhos
        return 6 / (np.pi * np.sqrt(4 - x ** 2))

    def dtau(self, x: Optional[np.ndarray] = None):
        if x is None:
            x = self._rhos
        return 2 / (np.pi * np.sqrt(1 - x ** 2))
