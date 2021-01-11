import os
from concurrent.futures import ProcessPoolExecutor
from typing import Type, TypeVar, Union

import numpy as np
import pandas as pd

from copulae.copula import BaseCopula
from copulae.empirical.distribution import emp_dist_func
from .utils import GofData, GofStat

__all__ = ["gof_copula"]

from ...types import Ties

Copula = TypeVar("Copula", bound=BaseCopula, covariant=True)


def gof_copula(copula: Type[Copula], data: Union[pd.DataFrame, np.ndarray], reps: int, ties: Ties = "average",
               fit_ties: Ties = "average", multiprocess=False, **fit_options):
    r"""
    Computes the goodness of fit statistic for the class of copula.

    Performs the "Sn" gof test, described in Genest et al. (2009) for the parametric case.

    Compares the empirical copula against a parametric estimate of the copula derived under the null hypothesis.

    Notes
    -----
    Given the pseudo-observations :math:`U_{ij} \forall i = 1, \dots, n, j = 1, \dots, d` and the empirical copula
    given by :math:`C_n(\textbf{u}) = \frac{1}{n} \sum^n_{i=1} \textbf{I}(U_{i1} \leq u_1, \dots, U_{id} \leq u_d)`
    where :math:`\textbf{u} \in [0, 1]^d`, the null hypothesis, :math:`H_0` thus tests if

    .. math::

        C \in C_0

    where :math:`C_0` is the true class of the copulae under :math:`H_0`. The test statistic T is defined as

    .. math::

        T = n \int_{[0, 1]^d} [C_n(\textbf{u}) - C_\theta_n(\textbf{u})]^2 dC_n(\textbf{u})

    where :math:`C_\theta_n(\textbf{u})` is the estimation of :math:`C` under :math:`H_0`.

    The approximate p-value is then given by:

    .. math::

        \sum^M_{k=1} \textbf{I}(|T_k| \geq |T|) / M

    Parameters
    ----------
    copula
        The class type of the copula type

    data
        The data used to test how well the class of copula fits

    reps
        number of bootstrap or multiplier replications to be used to obtain approximate realizations of the test
        statistic under the null hypothesis

    ties
        Method used to determine how ranks of the data used for calculating the test statistics should be
        computed. See :code:`pseudo_obs` for more information

    fit_ties
        Method used to determine how ranks of the data used for fitting the copula should be computed.
        See :code:`pseudo_obs` for more information

    multiprocess
        If True, uses multiprocessing to speed up the tests. Note that if the number of reps  is small and/or
        if the time taken to fit each empirical copula os short, running multiprocess could be slower than
        running the tests in a single-process fashion.

    fit_options
        Arguments to pass into the :code:`.fit()` method of the copula


    Returns
    -------
    GofStat
        Test statistics
    """
    data = GofData(data, ties, fit_ties)
    return GofParametricBootstrap(copula, data, reps, multiprocess, verbose=0, **fit_options).fit()


class GofParametricBootstrap:
    def __init__(self, copula: Type[Copula], data: GofData, reps, multiprocess, **fit_options):
        self._cop_factory = copula
        self.data = data

        self.copula = self.new_copula()
        self.fit_options = fit_options
        self.reps = reps
        self.multiprocess = multiprocess

    def fit(self):
        t = self.t_stat(self.copula, self.data)

        if self.multiprocess:
            with ProcessPoolExecutor(os.cpu_count()) as P:
                ts = np.array([f.result() for f in (P.submit(self._process_result) for _ in range(self.reps))])
        else:
            ts = np.array([self._process_result() for _ in range(self.reps)])

        return GofStat(method=f"Parametric bootstrap-based goodness-of-fit of {self.copula.name} with Sn",
                       parameter=self.copula.params,
                       statistic=t,
                       pvalue=(sum(np.abs(ts) >= abs(t)) + 0.5) / (self.reps + 1))

    def _process_result(self):
        u = self.copula.random(self.data.n_row)

        if self.data.has_ties:
            u = self._sort_data_by_column_inplace(u)

        data = GofData(u, self.data.ties, self.data.fit_ties)
        cop = self.new_copula()
        return self.t_stat(cop, data)

    def t_stat(self, copula: Copula, data: GofData):
        """Calculates the T statistic"""
        copula.fit(data.fitted_pobs, **self.fit_options)
        return gof_t_stat(copula, data.pobs)

    def new_copula(self) -> Copula:
        return self._cop_factory(dim=self.data.n_dim)

    def _sort_data_by_column_inplace(self, data: np.ndarray):
        """
        Use pre-calculated duplicate ranks array to sort random data with. This ensures that the
        resulting data (column-by-column) is ordered from lowest to highest
        """
        assert data.ndim == 2 and self.data.duplicated_ranks_array.shape == data.shape
        data.sort(0)
        for i in range(data.shape[1]):
            data[:, i] = data[self.data.duplicated_ranks_array[:, i], i]

        return data


def gof_t_stat(copula: Copula, data: np.ndarray) -> float:
    """Computes the T Statistic of the copula"""
    cop_cdf = copula.cdf(data)
    emp_cdf = emp_dist_func(data, data, smoothing='none')
    return sum((emp_cdf - cop_cdf) ** 2)
