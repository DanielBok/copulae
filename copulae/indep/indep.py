from typing import Optional

import numpy as np

from copulae.copula import BaseCopula
from copulae.types import Array


class IndepCopula(BaseCopula):
    def __init__(self, dim=2):
        super().__init__(dim, 'Independence')
        self.fit_stats = 'Unavailable'

    @property
    def tau(self):
        return 0

    @property
    def rho(self):
        return 0

    @property
    def __lambda__(self):
        return 0, 0

    def itau(self, tau: Array):
        return 0

    def irho(self, rho: Array):
        return 0

    def dtau(self, x: Optional[np.ndarray] = None):
        return 0

    def drho(self, x: Optional[np.ndarray] = None):
        return 0

    def cdf(self, x: Array, log=False) -> np.ndarray:
        self._check_data_dimension(x)
        x = np.asarray(x)
        return np.log(x).sum(1) if log else x.prod(1)

    def pdf(self, x: Array, log=False) -> np.ndarray:
        self._check_data_dimension(x)
        return np.repeat(0 if log else 1, len(x))

    @property
    def params(self):
        return self.dim

    def __random__(self, n: int, seed: int = None) -> np.ndarray:
        return np.random.uniform(size=(n, self.dim))

    def summary(self):
        print(self)

    def __str__(self):
        return f"Independence Copula with {self.dim} dimensions"

    def log_lik(self, data: np.ndarray):
        print('Log Likelihood not available for indepedence copula')
        return

    def fit(self, data: np.ndarray, x0: np.ndarray = None, method='mpl', est_var=False, verbose=1,
            optim_options: dict = None):
        print('Fitting not required for independence copula')
        return
