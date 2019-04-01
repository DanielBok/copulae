from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from copulae.copula.summary import FitSummary
from copulae.types import Array, Numeric


class AbstractCopula(ABC):

    @abstractmethod
    def __init__(self, dim: int, name: str):
        assert dim >= 2, 'Copula must have more than 2 dimensions'
        assert isinstance(dim, int), 'Copula dimension must be an integer'

        self.__dim = dim  # prevent others from messing around
        self.name = name
        self.fit_smry: Optional[FitSummary] = None
        self._bounds: Tuple[Numeric, Numeric] = (0.0, 0.0)

    @property
    def dim(self):
        return self.__dim

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
    def pdf(self, u: Array, log=False) -> Union[float, np.ndarray]:
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
