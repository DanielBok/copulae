from typing import Collection, Dict, List, Literal, TypedDict, Union

import numpy as np
import pandas as pd

from copulae.copula import BaseCopula, Param
from copulae.copula.exceptions import InputDataError
from copulae.types import Array
from copulae.utility.annotations import *
from .summary import Summary
from .univariate import DistDetail, create_univariate, get_marginal_detail


class MarginalCopulaParam(TypedDict):
    copula: Param
    marginals: List[Dict[str, float]]


MarginsInput = Union[str, Collection[str], DistDetail, Collection[DistDetail]]


class MarginalCopula(BaseCopula[MarginalCopulaParam]):
    """
    MarginalCopula enables users to specify different marginal distributions for a given dependency
    structure. For example, we could have a 2D Gaussian Copula but have the Student and exponential
    marginals.

    Examples
    --------
    >>> from copulae.datasets import load_marginal_data
    >>> from copulae import GaussianCopula, MarginalCopula
    >>> data = load_marginal_data()
    >>> cop = MarginalCopula(GaussianCopula(3), [
    ...     {"type": "t"},
    ...     {"type": "norm"},
    ...     {"type": "exp"}
    ... ])
    >>> cop.fit(data)
    >>> cop.summary()
    Marginal Copula Summary
    ================================================================================
    Marginal Copula with 3 dimensions
    Marginal Parameters
    -------------------
    Dist Type: t
        df: 15.457419183265923
        loc: -0.02197911363613804
        scale: 0.9875829850778358
    Dist Type: norm
        loc: 3.0026994935016047
        scale: 0.40913830848310995
    Dist Type: expon
        loc: 5.9475127982985763e-05
        scale: 0.4841057217117129
    ================================================================================
    Inner Joint Copula Parameter
    ----------------------------
    Gaussian Copula Summary
    ================================================================================
    Gaussian Copula with 3 dimensions
    Parameters
    --------------------------------------------------------------------------------
     Correlation Matrix
     1.000000  0.259487  0.404228
     0.259487  1.000000  0.178879
     0.404228  0.178879  1.000000
    """

    def __init__(self, copula: BaseCopula[Param], margins: MarginsInput):
        """
        Creates a MarginalCopula instance

        Parameters
        ----------
        copula
            The dependency structure. Must be another copula that is not the MarginalCopula
        margins
            Note that string names of the distribution must exist in :module:`scipy.stats`. If a single
            string (i.e. norm) or a single dictionary, all marginal distributions will be defined similarly.
            If a collection of string or dictionary, each instance will describe the marginal at that position.
            Dictionary form allows users to be more detailed in describing the distribution
        """
        self._copula = copula
        self._marginals = _process_margins_input(margins, copula.dim)
        super().__init__(copula.dim, "Marginal")

    @property
    def bounds(self):
        """Bounds is not implemented for the :class:`MarginalCopula`"""
        return NotImplemented

    @validate_data_dim({"x": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def cdf(self, x: Array, log=False) -> Union[np.ndarray, float]:
        x = self._check_x_input(x)
        u = np.empty_like(x)
        for i, m in enumerate(self._marginals):
            u[:, i] = m.cdf(x[:, i])

        return self._copula.cdf(u, log)

    def fit(self, data: Union[pd.DataFrame, np.ndarray], x0: Union[Collection[float], np.ndarray] = None, method='ml',
            verbose=1, optim_options: dict = None, ties='average', to_pobs=True, scale=1.0):
        if isinstance(data, pd.DataFrame):
            self._columns = list(data.columns)

        data = np.asarray(data)
        if data.ndim != 2:
            raise InputDataError('Data must be a matrix of dimension (n x d)')
        elif self.dim != data.shape[1]:
            raise InputDataError('Dimension of data does not match copula')

        for i, m in enumerate(self._marginals):
            self._marginals[i] = m.dist(*m.dist.fit(data[:, i]))

        # writing it as such to get past the type-hints flags
        self._copula.fit(data, x0, method, optim_options, ties, verbose, to_pobs, scale)
        self._fit_smry = self._copula._fit_smry
        return self

    @property
    def params(self):
        return {
            "copula": self._copula.params,
            "marginals": [get_marginal_detail(m) for m in self._marginals]
        }

    @validate_data_dim({"x": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def pdf(self, x: Array, log=False) -> Union[np.ndarray, float]:
        x = self._check_x_input(x)
        u = np.empty_like(x)
        density_margin = 0 if log else 1

        for i, m in enumerate(self._marginals):
            xx = x[:, i]
            u[:, i] = m.cdf(xx)
            if log:
                density_margin += m.logpdf(xx)
            else:
                density_margin *= m.pdf(xx)

        d = self._copula.pdf(u, log)
        return (d + density_margin) if log else (d * density_margin)

    @cast_output
    def random(self, n: int, seed: int = None) -> Union[np.ndarray, float]:
        u = self._copula.random(n, seed)
        x = np.empty_like(u)
        for i, m in enumerate(self._marginals):
            x[:, i] = m.ppf(u[:, i])

        return x

    @select_summary
    def summary(self, category: Literal['copula', 'fit'] = 'copula'):
        return Summary(self.name, self.dim, self._copula, self._marginals)

    def _check_x_input(self, x: Array):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("data must be 1 or 2 dimensional")
        if self.dim != x.shape[1]:
            raise ValueError(f"data should have {self.dim} columns")
        return x


def _process_margins_input(margins: MarginsInput, dim: int):
    if isinstance(margins, str):
        # single string describing margin type
        return [create_univariate({"type": margins}) for _ in range(dim)]
    elif isinstance(margins, dict):
        return [create_univariate(margins) for _ in range(dim)]  # DistDetail dict
    elif isinstance(margins, Collection):
        margins = list(margins)
        assert len(margins) == dim, f"margins collection must have {dim} elements"
        _output = []
        for m in margins:
            if isinstance(m, str):
                _output.append(create_univariate({"type": m}))
            elif isinstance(m, dict):
                _output.append(create_univariate(m))
            else:
                raise TypeError(f"{m} is not a supported argument for a marginal")

        return _output

    raise TypeError(f"{margins} is not a valid input for margins")
