from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from copulae.copula import AbstractCopula
from copulae.core import pseudo_obs, rank_data
from copulae.empirical.distribution import emp_dist_func


class GofCopula:
    def __init__(self, copula: AbstractCopula, data: Union[pd.DataFrame, np.ndarray], reps: int, ties="average",
                 fit_ties="average"):
        self._copula = copula
        self._data = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
        self._has_ties = self._data_has_ties(self._data)
        self._reps = int(reps)
        self._fit_ties = fit_ties
        self._ties = ties
        self._u = pseudo_obs(self._data, ties=ties)

    def gof_sn(self):
        r"""
        Performs the "Sn" gof test, described in Genest et al. (2009)

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


        Returns
        -------
        dict
            A dictionary of fitted results
        """
        t = gof_t_stat(self._copula, self._u, to_pobs=False)

        # pre-calculate dupe ranks to sort random data with letter
        dupe_ranks = np.sort(rank_data(self._data, 1), 0).astype(int) - 1 if self._has_ties else None

        t0 = np.repeat(np.nan, self._reps)
        for i in range(self._reps):
            u_r = self._copula.random(len(self._data))

            if self._has_ties:
                u_r = self._sort_data_by_column_inplace(u_r, dupe_ranks)

            t0[i] = gof_t_stat(self._copula, u_r, self._ties)

        return {
            "method": f"Parametric bootstrap-based goodness-of-fit of {self._copula.name} with Sn",
            "parameter": self._copula.params,
            "statistic": t,
            "pvalue": (sum(np.abs(t0) >= abs(t)) + 0.5) / (self._reps + 1)
        }

    @staticmethod
    def _data_has_ties(data):
        nrow, ncol = data.shape
        for i in range(ncol):
            if len(np.unique(data[:, i])) != nrow:
                return True
        return False

    def _generate_new_copula(self):
        u_fit = (pseudo_obs(self._data, ties=self._fit_ties)
                 if self._has_ties and self._fit_ties != self._ties else
                 deepcopy(self._u))

        cop = type(self._copula)(dim=self._data.shape[2])
        cop.fit(u_fit)
        return cop

    @staticmethod
    def _sort_data_by_column_inplace(data: np.ndarray, dupe_rank: np.ndarray):
        assert data.ndim == 2 and dupe_rank.shape == data.shape
        data.sort(0)
        for i in range(data.shape[1]):
            data[:, i] = data[dupe_rank[:, i], i]

        return data


def gof_t_stat(copula: AbstractCopula, data: np.ndarray, ties="average", *, to_pobs=True):
    """Computes the T Statistic of the copula"""
    if to_pobs:
        data = pseudo_obs(data, ties)

    cop_cdf = copula.cdf(data)
    emp_cdf = emp_dist_func(data, data, smoothing='none', ties=ties)
    return sum((emp_cdf - cop_cdf) ** 2)
