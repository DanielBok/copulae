from typing import Optional

import numpy as np

from copulae.copula import BaseCopula, TailDep
from copulae.stats import random_uniform
from copulae.types import Array
from copulae.utility import reshape_data


class IndepCopula(BaseCopula):
    def __init__(self, dim=2):
        super().__init__(dim, 'Independence')
        self.fit_stats = 'Unavailable'

    @reshape_data
    def cdf(self, x: Array, log=False) -> np.ndarray:
        return np.log(x).sum(1) if log else x.prod(1)

    def drho(self, x: Optional[np.ndarray] = None):
        return 0

    def dtau(self, x: Optional[np.ndarray] = None):
        return 0

    def fit(self, data: np.ndarray, x0: np.ndarray = None, method='mpl', est_var=False, verbose=1,
            optim_options: dict = None):
        print('Fitting not required for independence copula')
        return

    def irho(self, rho: Array):
        return 0

    def itau(self, tau: Array):
        return 0

    @property
    def lambda_(self):
        return TailDep(0, 0)

    def log_lik(self, data: np.ndarray):
        print('Log Likelihood not available for indepedence copula')
        return

    @property
    def params(self):
        return self.dim

    @reshape_data
    def pdf(self, x: Array, log=False) -> np.ndarray:
        return np.repeat(0 if log else 1, len(x))

    @property
    def rho(self):
        return 0

    @property
    def tau(self):
        return 0

    def random(self, n: int, seed: int = None) -> np.ndarray:
        return random_uniform(n, self.dim, seed)

    def summary(self):
        return str(self)

    def __str__(self):
        return f"Independence Copula with {self.dim} dimensions"
