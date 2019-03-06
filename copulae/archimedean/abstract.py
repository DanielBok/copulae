from abc import ABC, abstractmethod

import numpy as np

from copulae.copula.base import BaseCopula
from copulae.types import Array, Numeric
from copulae.utility import reshape_data


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

    def A(self, w: Numeric):
        """
        The Pickands dependence function. This can be seen as the generator function of an extreme-value copula.

        A bivariate copula C is an extreme-value copula if and only if

        C(u, v) = (uv)^A(log(v) / log(uv)), (u,v) in (0,1]^2 w/o {(1,1)},

        where A: [0,1] -> [1/2, 1] is convex and satisfies max(t,1-t) <= A(t) <= 1 for all t in [0,1].

        In the d-variate case, the Pickands dependence function A is defined on the d-dimensional unit simplex.

        :param w: int, float, Iterable[int, float]
            A numeric

        :return: ndarray
            Value of the dependence function
        """
        raise NotImplementedError

    @reshape_data
    def cdf(self, u: Array, log=False) -> np.ndarray:
        cdf = self.psi(self.ipsi(u).sum(1))
        return np.log(cdf) if log else cdf

    def dAdu(self, w: Numeric):
        """
        First and second derivative of A

        :param w: int, float, Iterable[int, float]
            A numeric

        :return: ndarray
            First and second derivative of A
        """
        raise NotImplementedError

    @abstractmethod
    def psi(self, s: Array):
        """
        Generator function for Archimedean copulae.

        :param s: ndarray
            numerical vector at which these functions are to be evaluated.
        :return: ndarray
            generator value for Archimedean copula
        """
        raise NotImplementedError

    @abstractmethod
    def ipsi(self, u: Array, log=False):
        """
        The inverse generator function for Archimedean copulae

        Currently only computes the first two derivatives of iPsi()

        :param u: ndarray
            numerical vector at which these functions are to be evaluated.
        :param log: boolean, default False
            If True, log of psi inverse will be returned
        :return: ndarray
            inverse generator value for Archimedean copula
        """
        raise NotImplementedError

    @abstractmethod
    def dipsi(self, u, degree=1, log=False):
        """
        Derivative of the inverse of the generator function for Archimedean copulae

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
