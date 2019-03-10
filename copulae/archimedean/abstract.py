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

        .. math::

            C(u, v) = (uv)^{A(log(v) / log(uv))}, (u,v) in (0,1]^2 w/o {(1,1)}

        where :math:`A: [0,1] \rightarrow [1/2, 1]` is convex and satisfies :math:`max(t,1-t) \leq A(t) \leq 1` for all
        :math:`t \in [0, 1]`

        Parameters
        ----------
        w: scalar or array like
            A numeric scalar or vector

        Returns
        -------
        ndarray
            Array containing values of the dependence function
        """
        raise NotImplementedError

    @reshape_data
    def cdf(self, u: Array, log=False) -> np.ndarray:
        cdf = self.psi(self.ipsi(u).sum(1))
        return np.log(cdf) if log else cdf

    def dAdu(self, w: Numeric):
        """
        First and second derivative of A

        Parameters
        ----------
        w: scalar or array like
            A numeric scalar or vector

        Returns
        -------
        ndarray
            Array containing the first and second derivative of the dependence function

        See Also
        --------
        A: Dependence function
        """
        raise NotImplementedError

    @abstractmethod
    def psi(self, s: Array):
        """
        Generator function for Archimedean copulae.

        Parameters
        ----------
        s: array like
            Numerical vector at which the generator function is to be evaluated against

        Returns
        -------
        ndarray
            Generator value for the Archimedean copula
        """
        raise NotImplementedError

    @abstractmethod
    def ipsi(self, u: Array, log=False):
        """
        The inverse generator function for Archimedean copulae

        Currently only computes the first two derivatives of iPsi()

        Parameters
        ----------
        u: array like
            Numerical vector at which the inverse generator function is to be evaluated against

        log: bool, optional
            If True, log of ipsi will be returned

        Returns
        -------
        ndarray
            Inverse generator value for the Archimedean copula
        """
        raise NotImplementedError

    @abstractmethod
    def dipsi(self, u: Array, degree=1, log=False):
        """
        Derivative of the inverse of the generator function for Archimedean copulae

        Parameters
        ----------
        u: array like
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
        raise NotImplementedError
