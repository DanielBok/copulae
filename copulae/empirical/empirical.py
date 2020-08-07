from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import beta

from copulae.copula import BaseCopula
from copulae.copula import Summary
from copulae.core import rank_data
from copulae.special import log_sum
from copulae.types import Array, EPSILON, Ties
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

    Examples
    --------
    >>> from copulae import EmpiricalCopula
    >>> from copulae.datasets import load_marginal_data
    >>> df = load_marginal_data()
    >>> df.head(3)
        STUDENT      NORM       EXP
    0 -0.485878  2.646041  0.393322
    1 -1.088878  2.906977  0.253731
    2 -0.462133  3.166951  0.480696
    >>> emp_cop = EmpiricalCopula(3, df, smoothing="beta")
    >>> data = emp_cop.data  # getting the pseudo-observation data (this is the converted df)
    >>> data[:3]
    array([[0.32522493, 0.1886038 , 0.55781406],
           [0.15161613, 0.39953349, 0.40953016],
           [0.33622126, 0.65611463, 0.62645785]])
    # must feed pseudo-observations into cdf
    >>> emp_cop.cdf(data[:2])
    array([0.06865595, 0.06320104])
    >>> emp_cop.pdf([[0.5, 0.5, 0.5]])
    0.009268568506099015
    >>> emp_cop.random(3, seed=10)
    array([[0.59046984, 0.98467178, 0.16494502],
           [0.31989337, 0.28090636, 0.09063645],
           [0.60379873, 0.61779407, 0.54215262]])
    """

    def __init__(self, dim: Optional[int] = None, data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                 smoothing: Optional[Smoothing] = None, ties: Ties = "average", offset: float = 0):
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
            'average': The average of the ranks that would have been assigned to all the tied values is assigned
                to each value.
            'min': The minimum of the ranks that would have been assigned to all the tied values is assigned to
                each value. (This is also referred to as "competition" ranking.)
            'max': The maximum of the ranks that would have been assigned to all the tied values is assigned to
                each value.
            'dense': Like 'min', but the rank of the next highest element is assigned the rank immediately after
                those assigned to the tied elements. 'ordinal': All values are given a distinct rank, corresponding
                to the order that the values occur in `a`.

        offset
            Used in scaling the result for the density and distribution functions. Defaults to 0.
        """
        self.ties = ties
        self._offset = offset
        self._name = "Empirical"
        self.smoothing = smoothing

        assert dim is not None or data is not None, "Either dimension or data must be specified"
        self._dim = data.shape[1] if dim is None else int(dim)
        assert self.dim > 1, "Dimension must be >= 2"

        self.data = data
        self.init_validate()

    def cdf(self, u: Array, log=False) -> np.ndarray:
        if np.any(u > (1 + EPSILON)) or np.any(u < -EPSILON):
            raise ValueError("input array must be pseudo observations")
        cdf = emp_dist_func(u, self.data, self._smoothing, self._offset)
        return np.log(cdf) if log else cdf

    @property
    def data(self):
        """
        The empirical data source from which to compare against. Note that when setting the data, it will
        be automatically transformed to pseudo-observations by default based on the
        :meth:`~EmpiricalCopula.ties` property
        """
        if self._data is None:
            self._data = self.pobs(self._source, self.ties)
        return self._data

    @data.setter
    def data(self, data: Union[pd.DataFrame, np.ndarray]):
        data = np.asarray(data)
        assert data.ndim == 2, "data must be 2 dimensional"
        assert self.dim == data.shape[1], "data and copula dimensions do not match"
        self._source = data
        self._data = None

    @property
    def params(self):
        """
        By default, the Empirical copula has no "parameters" as everything is defined by the input data
        """
        return None

    def pdf(self, u: Array, log=False):
        assert self.smoothing == "beta", "Empirical Copula only has density (PDF) for 'beta' smoothing"
        assert isinstance(self.data, np.ndarray), "data is still undefined for EmpiricalCopula"
        u = self.pobs(u, self.ties)

        data_rank = rank_data(self.data, 1, self.ties)
        n = len(self.data)

        if log:
            value = np.array([
                log_sum(
                    np.array([
                        sum(beta.logpdf(row, a=row_rank, b=n + 1 - row_rank))
                        for row_rank in data_rank
                    ])
                ) for row in u]) - np.log(n + self._offset)
        else:
            value = np.array([
                sum([
                    np.prod(beta.pdf(row, a=row_rank, b=n + 1 - row_rank))
                    for row_rank in data_rank
                ]) for row in u]) / (n + self._offset)

        return value.item() if value.size == 1 else value

    def random(self, n: int, seed: int = None):
        assert isinstance(self.data, np.ndarray), "data is still undefined for EmpiricalCopula"
        assert n <= len(self.data), "random samples desired must not exceed number of rows in data"

        if seed is not None:
            np.random.seed(seed)

        return self.data[np.random.randint(0, len(self.data), n)]

    @property
    def smoothing(self):
        """
        The smoothing parameter. "none" provides no smoothing. "beta" and "checkerboard" provide a smoothed
        version of the empirical copula. See equations (2.1) - (4.1) in Segers, Sibuya and Tsukahara

        References
        ----------
        `The Empirical Beta Copula <https://arxiv.org/pdf/1607.04430.pdf>`
        """
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
            "Ties method": self.ties,
            "Offset": self._offset,
            "Smoothing": self._smoothing,
        })

    @property
    def ties(self):
        """
        The method used to assign ranks to tied elements. The options are 'average', 'min', 'max', 'dense'
        and 'ordinal'.

        'average':
            The average of the ranks that would have been assigned to all the tied values is assigned
            to each value.
        'min':
            The minimum of the ranks that would have been assigned to all the tied values is assigned to
            each value. (This is also referred to as "competition" ranking.)
        'max':
            The maximum of the ranks that would have been assigned to all the tied values is assigned to
            each value.
        'dense':
            Like 'min', but the rank of the next highest element is assigned the rank immediately after
            those assigned to the tied elements. 'ordinal': All values are given a distinct rank, corresponding
            to the order that the values occur in `a`.
        """
        return self._ties

    @ties.setter
    def ties(self, value: Ties):
        if getattr(self, "_ties", "") != value:
            self._ties = value
            self._data = None
