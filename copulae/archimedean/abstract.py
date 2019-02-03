from abc import ABC, abstractmethod

import numpy as np

from copulae.copula.base import BaseCopula
from copulae.types import Array


class AbstractArchimedeanCopula(BaseCopula, ABC):
    def __init__(self, dim: int, theta: float, family: str):
        family = family.lower()

        try:
            self._theta = float(theta)
        except ValueError:
            raise ValueError('theta must be a float')

        if dim < 2:
            raise ValueError('dim must be >= 2')
        if dim > 2 and self._theta < 0:
            raise ValueError('theta can only be negative when dim = 2')

        families = ('clayton', 'frank', 'amh', 'gumbel', 'joe')
        if family not in families:
            raise ValueError(f"Unknown family of Archimedean copula: {family}. Use one of {', '.join(families)}")

        super().__init__(dim, family)

    @abstractmethod
    def psi(self, s: Array) -> np.ndarray:
        """
        Generator function for Archimedean copulas.

        :param s: ndarray
            numerical vector at which these functions are to be evaluated.
        :return: ndarray
            generator value for Archimedean copula
        """
        raise NotImplementedError

    @abstractmethod
    def ipsi(self, u: Array) -> np.ndarray:
        """
        The inverse generator function for Archimedean copulas

        Currently only computes the first two derivatives of iPsi()

        :param u: ndarray
            numerical vector at which these functions are to be evaluated.
        :return: ndarray
            inverse generator value for Archimedean copula
        """
        raise NotImplementedError

    @abstractmethod
    def dipsi(self, u, degree=1, log=False) -> np.ndarray:
        """
        Derivative of the inverse of the generator function for Archimedean copulas

        :param u: ndarray
            numerical vector at which these functions are to be evaluated.
        :param degree: int, default 1
            the degree of the derivative (defaults to 1)
        :param log: bool, default False
            If True, log of derivative will be returned
        :return: ndarray
            derivative of psi
        """
        raise NotImplementedError

    def pdf(self, x: Array, log=False):
        pdf = self.psi(self.ipsi(x).sum(1))
        return np.log(pdf) if log else pdf
