from typing import Optional

import numpy as np
from scipy.stats import beta

from copulae.copula import BaseCopula, TailDep
from copulae.core import rank_data
from copulae.errors import NotApplicableError
from copulae.special import log_sum
from copulae.types import Array
from .distribution import emp_dist_func
from copulae.copula import Summary


class EmpiricalCopula(BaseCopula):
    """
    Given pseudo-observations from a distribution with continuous margins and copula, the empirical copula is
    the (default) empirical distribution function of these pseudo-observations. It is thus a natural nonparametric
    estimator of the copula.
    """

    def __init__(self, dim: Optional[int] = None, data: Optional[np.ndarray] = None, smoothing: Optional[str] = None,
                 ties="average", offset: float = 0):
        """
        Creates an empirical copula

        Parameters
        ----------
        dim
            Dimension of the copula. If this is not provided, it will be derived from the dimension of the data set

        data
            The data set for the empirical copula. The data set dimension must match the copula's dimension. If
            dim is not set, the dimension of the copula will be derived from the data's dimension. Data must be
            a matrix

        smoothing
            If not specified (default), the empirical distribution function or copula is computed. If "beta", the
            empirical beta copula is computed. If "checkerboard", the empirical checkerboard copula is computed.

        ties
            The method used to assign ranks to tied elements. The options are 'average', 'min', 'max', 'dense'
            and 'ordinal'.
            'average': The average of the ranks that would have been assigned to all the tied values is assigned to each
                value.
            'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to each
                value. (This is also referred to as "competition" ranking.)
            'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to each value.
            'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after those
                assigned to the tied elements. 'ordinal': All values are given a distinct rank, corresponding to
                the order that the values occur in `a`.

        offset
            Used in scaling the result for the density and distribution functions. Defaults to 0.
        """
        dim, data = self._validate_dim_and_data(dim, data)

        super().__init__(dim, "EmpiricalCopula")
        self._smoothing = self._validate_smoothing(smoothing)
        self._ties = ties
        self._offset = offset
        self._data = self._validate_data(data)

    def cdf(self, x: Array, log=False) -> np.ndarray:
        cdf = emp_dist_func(self.pobs(x), self.data, self._smoothing, self._ties, self._offset)
        return np.log(cdf) if log else cdf

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        self._data = self._validate_data(data)

    def drho(self, x=None):
        raise NotApplicableError

    def dtau(self, x=None):
        raise NotApplicableError

    def irho(self, rho: Array):
        raise NotApplicableError

    def itau(self, tau):
        raise NotApplicableError

    @property
    def lambda_(self) -> 'TailDep':
        raise NotApplicableError

    @property
    def params(self):
        raise NotApplicableError

    def pdf(self, u: Array, log=False):
        assert self.smoothing == "beta", "Empirical Copula only has density (PDF) for smoothing = 'beta'"
        assert isinstance(self.data, np.ndarray), "data is still undefined for EmpiricalCopula"
        u = self.pobs(u, self._ties)

        data_rank = rank_data(self.data, 1, self._ties)
        n = len(self.data)

        if log:
            return np.array([
                log_sum(
                    np.array([
                        sum(beta.logpdf(row, a=row_rank, b=n + 1 - row_rank))
                        for row_rank in data_rank
                    ])
                ) for row in u]) - np.log(n + self._offset)
        else:
            return np.array([
                sum([
                    np.prod(beta.pdf(row, a=row_rank, b=n + 1 - row_rank))
                    for row_rank in data_rank
                ]) for row in u]) / (n + self._offset)

    def random(self, n: int, seed: int = None):
        assert isinstance(self.data, np.ndarray), "data is still undefined for EmpiricalCopula"
        assert n <= len(self.data), "random samples desired must not exceed number of rows in data"

        if seed is not None:
            np.random.seed(seed)

        return self.data[np.random.choice(np.arange(len(self.data)), size=n, replace=False)]

    @property
    def rho(self):
        raise NotApplicableError

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value: Optional[str]):
        self._smoothing = self._validate_smoothing(value)

    def summary(self):
        return Summary(self, {
            "Dimensions": self.dim,
            "Ties method": self._ties,
            "Offset": self._offset,
            "Smoothing": self._smoothing,
        })

    @property
    def tau(self):
        raise NotApplicableError

    def _validate_data(self, data: np.ndarray):
        data = np.asarray(data)
        assert data.ndim == 2, "data must be 2 dimensional"
        assert self.dim == data.shape[1], "data and copula dimension do not match"
        return self.pobs(data, self._ties)

    @staticmethod
    def _validate_dim_and_data(dim: Optional[int] = None, data: Optional[np.ndarray] = None):
        assert dim is not None or data is not None, "Either dimension or data must be specified"

        dim = np.shape(data)[1] if dim is None else int(dim)
        assert dim > 1, "Dimension must be an integer greater than 1"

        return dim, data

    @staticmethod
    def _validate_smoothing(smoothing: Optional[str] = None):
        if smoothing is None:
            smoothing = "none"

        smoothing = smoothing.lower()
        assert smoothing in ("none", "beta", "checkerboard"), "Smoothing must be 'none', 'beta' or 'checkerboard'"

        return smoothing
