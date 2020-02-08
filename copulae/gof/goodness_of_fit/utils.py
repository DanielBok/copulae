from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from copulae.core import pseudo_obs, rank_data

__all__ = ["GofData", "GofStat"]


class GofData:
    def __init__(self, data: Union[pd.DataFrame, np.ndarray], ties: str, fit_ties):
        self.data = data.to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)
        self.ties = ties
        self.fit_ties = fit_ties

        self.has_ties = False
        nrow, ncol = self.data.shape
        for i in range(ncol):
            if len(np.unique(self.data[:, i])) != nrow:
                self.has_ties = True
                break

        # data used for fitting the main copula
        self.pobs = pseudo_obs(self.data, ties=ties)

        # data used fo
        self.fitted_pobs = pseudo_obs(self.data, ties=fit_ties) if self.has_ties and ties != fit_ties else self.pobs
        self._duplicated_rank_array = np.sort(rank_data(self.data, 1), 0).astype(int) - 1

    @property
    def duplicated_ranks_array(self) -> np.ndarray:
        return self._duplicated_rank_array

    @property
    def n_dim(self):
        return self.data.shape[1]

    @property
    def n_row(self):
        return len(self.data)


@dataclass
class GofStat:
    method: str
    parameter: Union[float, np.ndarray]
    statistic: float
    pvalue: float

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"""
Goodness-of-Fit statistics
==========================
Method     : {self.method}
Parameter  : {self.parameter}
Statistic  : {round(self.statistic, 9)}
P-Value    : {round(self.pvalue, 9)}
        """.strip()
