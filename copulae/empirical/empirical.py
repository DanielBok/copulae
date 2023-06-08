from typing import Literal, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import beta

from copulae.copula import BaseCopula
from copulae.copula import Summary
from copulae.core import rank_data
from copulae.special import log_sum
from copulae.types import Array, EPSILON, Matrix, Ties
from copulae.utility.annotations import *
from .distribution import emp_dist_func
from .summary import FitSummary

Smoothing = Literal['none', 'beta', 'checkerboard']


class EmpiricalCopula(BaseCopula[None]):
    """
    Given pseudo-observations from a distribution with continuous margins and copula, the empirical copula is
    the (default) empirical distribution function of these pseudo-observations. It is thus a natural nonparametric
    estimator of the copula.

    Examples
    --------
    >>> from copulae import EmpiricalCopula, pseudo_obs
    >>> from copulae.datasets import load_marginal_data
    >>> df = load_marginal_data()
    >>> df.head(3)
        STUDENT      NORM       EXP
    0 -0.485878  2.646041  0.393322
    1 -1.088878  2.906977  0.253731
    2 -0.462133  3.166951  0.480696
    >>> u = pseudo_obs(df)  # data must be converted to pseudo observations
    >>> emp_cop = EmpiricalCopula(u, smoothing="beta")
    >>> data = emp_cop.data  # getting the pseudo-observation data (this is the converted df)
    >>> data[:3]
        STUDENT      NORM       EXP
    0  0.325225  0.188604  0.557814
    1  0.151616  0.399533  0.409530
    2  0.336221  0.656115  0.626458
    # must feed pseudo-observations into cdf
    >>> emp_cop.cdf(data[:2])
    array([0.06865595, 0.06320104])
    >>> emp_cop.pdf([[0.5, 0.5, 0.5],
    ...              [0.6, 0.4, 0.6]])
    array([0.00926857, 0.00868833])
    >>> emp_cop.random(3, seed=10)
        STUDENT      NORM       EXP
    0  0.590470  0.984672  0.164945
    1  0.319893  0.280906  0.090636
    2  0.603799  0.617794  0.542153
    """

    @validate_data_dim({"data": 2})
    def __init__(self,
                 data: Matrix,
                 smoothing: Optional[Smoothing] = None,
                 ties: Ties = "average",
                 offset: float = 0):
        """
        Creates an empirical copula

        Parameters
        ----------
        data
            The margins data set for the empirical copula. This data need not be the pseudo-observations.
            The data set dimension must match the copula's dimension. If dim is not set, the dimension of
            the copula will be derived from the data's dimension. Data must be a matrix

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
        self.smoothing = smoothing

        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            data = np.array(data)

        columns = list(data.columns) if isinstance(data, pd.DataFrame) else None
        super().__init__(data.shape[1], "Empirical", FitSummary(*data.shape), columns)
        self._data = data

    @property
    def data(self):
        return self._data

    @validate_data_dim({"u": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def cdf(self, u: Array, log=False):
        """
        Returns the cumulative distribution function (CDF) of the copulae.

        The CDF is also the probability of a RV being less or equal to the value specified. Equivalent to the 'p'
        generic function in R.

        Parameters
        ----------
        u: ndarray
            Vector or matrix of the pseudo-observations of the observed data. This vector must be (n x d)
            where `d` is the dimension of the copula and must have values between 0 and 1. The caller must
            have specified the density.

        log: bool
            If True, the log of the probability is returned

        Returns
        -------
        ndarray or float
            The CDF of the random variates
        """
        if np.any(u > (1 + EPSILON)) or np.any(u < -EPSILON):
            raise ValueError("input array must be pseudo observations")

        uu = self.pobs(self._data, self.ties)  # pseudo-observations of source marginal to compare against
        cdf = emp_dist_func(u, uu, self._smoothing, self._offset)
        return np.log(cdf) if log else cdf

    def fit(self, data, x0=None, method='ml', optim_options=None, ties='average', verbose=1, to_pobs=True, scale=1.0):
        if verbose > 1:
            warn("EmpiricalCopula does not need 'fitting'")
        return self

    @property
    def params(self):
        """
        By default, the Empirical copula has no "parameters" as everything is defined by the input data
        """
        return None

    @validate_data_dim({"u": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def pdf(self, u: Array, log=False):
        assert self.smoothing == "beta", "Empirical Copula only has density (PDF) for 'beta' smoothing"

        data_rank = rank_data(self.data, 1, self.ties)
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

    @cast_output
    def random(self, n: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        data = np.asarray(self.data)
        return self._format_output(data[np.random.randint(0, len(data), n)])

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

    @select_summary
    def summary(self, category: Literal['copula', 'fit'] = 'copula'):
        return Summary(self, {
            "Dimensions": self.dim,
            "Ties method": self.ties,
            "Offset": self._offset,
            "Smoothing": self._smoothing,
        })

    @validate_data_dim({"u": 2, "original": 2})
    @classmethod
    def to_marginals(cls, u: Matrix, original: Matrix) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transforms a sample marginal data (pseudo-observations) to empirical margins based on the
        input dataset

        Parameters
        ----------
        u
            Sample marginals (pseudo observations). Values must be between [0, 1]
        original
            The original dataset (do not cast these into pseudo-observations)

        Returns
        -------
        np.ndarray or pd.DataFrame
            Transformed marginals

        Examples
        --------
        >>> from copulae import EmpiricalCopula, pseudo_obs
        >>> from copulae.datasets import load_marginal_data
        >>> src = load_marginal_data()
        >>> src.head(3)
            STUDENT      NORM       EXP
        0 -0.485878  2.646041  0.393322
        1 -1.088878  2.906977  0.253731
        2 -0.462133  3.166951  0.480696
        # Pretend to get some psuedo marginals (~new data)
        >>> sample_marginals = pseudo_obs(src)[:3] + 0.01
        >>> sample_marginals
            STUDENT      NORM       EXP
        0  0.335225  0.198604  0.567814
        1  0.161616  0.409533  0.419530
        2  0.346221  0.666115  0.636458
        # Get the empirical marginals
        >>> EmpiricalCopula.to_marginals(sample_marginals, src)
            STUDENT      NORM       EXP
        0 -0.465186  2.663224  0.408747
        1 -1.054686  2.917943  0.261550
        2 -0.428498  3.176723  0.492262
        """
        # because it is pseudo-observations, and already ranked within the columns, we can just
        # multiply by the number of rows in the original data set to get the position rank
        index = np.array(np.floor(u * len(original)), int)
        data = np.array(original)
        source = np.take_along_axis(data, data.argsort(axis=0), axis=0)

        # marginals derived row by row based on 'lowest' position, we could offer an interpolation in
        # the future, but not sure how popular this method is
        marginals = np.transpose([source[r, c] for c, r in enumerate(index.T)])

        return pd.DataFrame(marginals, columns=original.columns) if isinstance(original, pd.DataFrame) else marginals

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
        self._ties = value

    def _format_output(self, output: np.ndarray):
        """Converts output array to DataFrame if the input data is a DataFrame"""
        if isinstance(output, pd.DataFrame):
            return pd.DataFrame(output, columns=self.data.columns)
        return output
