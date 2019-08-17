from typing import Optional

import numpy as np

from copulae.copula import BaseCopula, TailDep
from copulae.core import rank_data
from copulae.errors import NotApplicableError
from copulae.special import log_sum
from copulae.types import Array
from scipy.stats import beta


class EmpiricalCopula(BaseCopula):

    def summary(self):
        pass

    def cdf(self, x: Array, log=False) -> np.ndarray:
        pass

    def __init__(self, dim: Optional[int] = None, data: Optional[np.ndarray] = None, smoothing: Optional[str] = None,
                 ties="average", offset: float = 0):
        dim, data = self._validate_dim_and_data(dim, data)

        super().__init__(dim, "EmpiricalCopula")
        self._data = data
        self._smoothing = self._validate_smoothing(smoothing)
        self._ties = ties
        self._offset = offset

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

    @property
    def tau(self):
        raise NotApplicableError

    @staticmethod
    def _validate_data(data: np.ndarray):
        data = np.asarray(data)
        assert data.ndim == 2, "data must be 2 dimensional"
        return data

    def _validate_dim_and_data(self, dim: Optional[int] = None, data: Optional[np.ndarray] = None):
        assert dim is not None or data is not None, "Either dimension or data must be specified"
        if data is not None:
            data = self._validate_data(data)

        dim = data.shape[1] if dim is None else int(dim)
        assert dim > 1, "Dimension must be an integer greater than 1"

        return dim, data

    @staticmethod
    def _validate_smoothing(smoothing: Optional[str] = None):
        if smoothing is None:
            smoothing = "none"

        smoothing = smoothing.lower()
        assert smoothing in ("none", "beta", "checkerboard"), "Smoothing must be 'none', 'beta' or 'checkerboard'"

        return smoothing
