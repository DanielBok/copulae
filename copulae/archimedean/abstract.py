from abc import ABC, abstractmethod

import numpy as np

from copulae.copula.base import BaseCopula
from copulae.types import Array
from copulae.utility import array_io


class AbstractArchimedeanCopula(BaseCopula, ABC):
    def __init__(self, dim: int, theta: float, family: str):
        family = family.lower()

        try:
            self._theta = float(theta)
        except ValueError:
            raise ValueError('theta must be a float')

        families = ('clayton', 'frank', 'amh', 'gumbel', 'joe')
        assert family in families, f"Unknown family of Archimedean copula: {family}. Use one of {', '.join(families)}"

        super().__init__(dim, family)

    @array_io(dim=2)
    def cdf(self, u: Array, log=False) -> np.ndarray:
        cdf = self.psi(self.ipsi(u).sum(1))
        return np.log(cdf) if log else cdf

    @abstractmethod
    def dipsi(self, u, degree=1, log=False):
        """
        Derivative of the inverse of the generator function for Archimedean copulae

        Parameters
        ----------
        u: {array_like, scalar}
            Numerical vector at which the derivative of the inverse generator function is to be evaluated against

        degree: int
            The degree of the derivative

        log: bool, optional
            If True, the log of the derivative will be returned

        Returns
        -------
        ndarray
            Derivative of the inverse generator value for the Archimedean copula
        """
        pass

    @abstractmethod
    def ipsi(self, u, log=False):
        """
        The inverse generator function for Archimedean copulae

        Currently only computes the first two derivatives of iPsi()

        Parameters
        ----------
        u: {array_like, scalar}
            Numerical vector at which the inverse generator function is to be evaluated against

        log: bool, optional
            If True, log of ipsi will be returned

        Returns
        -------
        ndarray
            Inverse generator value for the Archimedean copula
        """
        pass

    @abstractmethod
    def psi(self, s):
        """
        Generator function for Archimedean copulae.

        Parameters
        ----------
        s: {array_like, scalar}
            Numerical vector at which the generator function is to be evaluated against

        Returns
        -------
        ndarray
            Generator value for the Archimedean copula
        """
        pass
