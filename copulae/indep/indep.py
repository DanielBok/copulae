from typing import Collection, Literal, Union
from warnings import warn

import numpy as np
import pandas as pd

from copulae.copula import BaseCopula, EstimationMethod, Summary
from copulae.stats import random_uniform
from copulae.types import Array, Ties
from copulae.utility.annotations import *
from .summary import FitSummary


class IndepCopula(BaseCopula[int]):
    def __init__(self, dim=2, fields: Collection[str] = None):
        r"""
        The Independence copula is the copula that results from a dependency structure in which each individual
        variable is independent of each other. It has no parameters and is defined as

        .. math::

            C(u_1, \dots, u_d) = \prod_i u_i

        Parameters
        ----------
        dim: int, optional
            The dimension of the copula

        fields: list of str
            The names of the data's columns
        """
        columns = list(fields) if fields is not None else None
        if columns is not None:
            assert len(columns) == dim, "number of fields must match copula dimension"

        super().__init__(dim, "Independent", FitSummary(dim), columns)

    @validate_data_dim({"x": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def cdf(self, x: Array, log=False):
        return np.log(x).sum(1) if log else x.prod(1)

    def fit(self, data: Union[pd.DataFrame, np.ndarray], x0: Union[Collection[float], np.ndarray] = None,
            method: EstimationMethod = 'ml', optim_options: dict = None, ties: Ties = 'average', verbose=1,
            to_pobs=True, scale=1.0):
        if verbose > 1:
            warn("IndepCopula does not need 'fitting'")
        return self

    @property
    def params(self):
        return self.dim

    @validate_data_dim({"x": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def pdf(self, x: Array, log=False):
        return np.repeat(0 if log else 1, len(x))

    @cast_output
    def random(self, n: int, seed: int = None):
        return random_uniform(n, self.dim, seed)

    @select_summary
    def summary(self, category: Literal['copula', 'fit'] = 'copula'):
        return Summary(self, {})
