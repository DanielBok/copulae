from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from copulae.types import Array, Numeric


class AbstractCopula(ABC):

    @abstractmethod
    def __init__(self, dim: int, name: str):
        self.dim = dim
        self.name = name
        self.fit_stats: FitStats = None
        self._bounds: Tuple[Numeric, Numeric] = (0.0, 0.0)

        if dim < 1 or int(dim) != dim:
            raise ValueError("Copula dimension must be an integer greater than 1.")

    @abstractmethod
    def fit(self, data: np.ndarray, method='mpl', x0: np.ndarray = None, verbose=1, optim_options: dict = None):
        pass

    @abstractmethod
    def cdf(self, x: Array, log=False) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def drho(self, x: Optional[np.ndarray]):
        pass

    @abstractmethod
    def dtau(self, x: Optional[np.ndarray]):
        pass

    @abstractmethod
    def irho(self, rho: Array):
        pass

    @abstractmethod
    def itau(self, tau: Array):
        pass

    @abstractmethod
    def lambda_(self):
        pass

    @abstractmethod
    def log_lik(self, data: Array) -> float:
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @params.setter
    @abstractmethod
    def params(self, data: Array):
        pass

    @property
    @abstractmethod
    def params_bounds(self):
        pass

    @params_bounds.setter
    def params_bounds(self, bounds: Tuple[Array, Array]):
        pass

    @abstractmethod
    def pdf(self, x: Array, log=False) -> Union[float, np.ndarray]:
        pass

    @staticmethod
    @abstractmethod
    def pobs(data: Array, ties='average') -> np.ndarray:
        pass

    @abstractmethod
    def random(self, n: int, seed: int = None):
        pass

    @property
    @abstractmethod
    def rho(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @property
    @abstractmethod
    def tau(self):
        pass


class FitStats:
    """
    Statistics on the fit of the copula

    Attributes
    ----------
    params: named tuple, numpy array
        parameters of the copula after fitting

    var_est: numpy array
        estimate of the variance

    method: str
        method used to fit the copula

    log_lik: float
        log likelihood value of the copula after fitting

    nsample: int
        number of data points used to fit copula

    setup: dict
        optimizer set up options

    results: dict
        optimization results
    """

    def __init__(self, params: np.ndarray, var_est: np.ndarray, method: str, log_lik: float, nsample: int,
                 setup: Optional[dict] = None, results: Optional[dict] = None):
        self.params = params
        self.var_est = var_est
        self.method = method
        self.log_lik = log_lik
        self.nsample = nsample
        self.setup = setup
        self.results = results

    def __str__(self):
        msg = f"""
Log. Lik        : {self.log_lik}
Var. Est.       : Not Implemented Yet
Method          : {self.method}
Data Pts.       : {self.nsample}
""".strip()

        skip_keys = {'final_simplex'}
        for title, dic in [('Optim Options', self.setup), ('Results', self.results)]:
            if dic is not None:
                string = "\n".join(f'\t{k:15s}: {v}' for k, v in dic.items() if k not in skip_keys)
                msg += f"\n\n{title}\n{string}"

        return msg

    def summary(self):
        print(self)
