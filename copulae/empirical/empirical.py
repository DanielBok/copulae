from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import beta

from copulae.copula import BaseCopula
from copulae.copula import Summary
from copulae.core import rank_data
from copulae.special import log_sum
from copulae.types import Array
from .distribution import emp_dist_func

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

Smoothing = Literal['none', 'beta', 'checkerboard']


class EmpiricalCopula(BaseCopula[None]):
    """
    Given pseudo-observations from a distribution with continuous margins and copula, the empirical copula is
    the (default) empirical distribution function of these pseudo-observations. It is thus a natural nonparametric
    estimator of the copula.
    """

    def __init__(self, dim: Optional[int] = None, data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 smoothing: Optional[Smoothing] = None, ties="average", offset: float = 0):
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
        assert dim is not None or data is not None, "Either dimension or data must be specified"

        self._dim = data.shape[1] if dim is None else int(dim)
        assert self.dim > 1, "Dimension must be >= 2"

        self.data = data
        self._name = "Empirical"
        self.smoothing = smoothing
        self._ties = ties
        self._offset = offset
        self.init_validate()

    def cdf(self, x: Array, log=False) -> np.ndarray:
        cdf = emp_dist_func(self.pobs(x), self.data, self._smoothing, self._ties, self._offset)
        return np.log(cdf) if log else cdf

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        data = np.asarray(data)
        assert data.ndim == 2, "data must be 2 dimensional"
        assert self.dim == data.shape[1], "data and copula dimensions do not match"
        self._data = self.pobs(data, self._ties)

    @property
    def params(self):
        return None

    def pdf(self, u: Array, log=False):
        assert self.smoothing == "beta", "Empirical Copula only has density (PDF) for 'beta' smoothing"
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

        return self.data[np.random.randint(0, len(self.data), n)]

    @property
    def smoothing(self):
        return self._smoothing

    @smoothing.setter
    def smoothing(self, smoothing: Optional[Smoothing]):
        if smoothing is None:
            smoothing: Smoothing = "none"

        assert smoothing in ("none", "beta", "checkerboard"), "Smoothing must be 'none', 'beta' or 'checkerboard'"
        self._smoothing = smoothing

    def summary(self):
        return Summary(self, {
            "Dimensions": self.dim,
            "Ties method": self._ties,
            "Offset": self._offset,
            "Smoothing": self._smoothing,
        })
