import numpy as np

from copulae.copula import BaseCopula, Summary
from copulae.stats import random_uniform
from copulae.types import Array
from copulae.utility import array_io


class IndepCopula(BaseCopula[int]):
    def __init__(self, dim=2):
        r"""
        The Independence copula is the copula that results from a dependency structure in which each individual
        variable is independent of each other. It has no parameters and is defined as

        .. math::

            C(u_1, \dots, u_d) = \prod_i u_i

        Parameters
        ----------
        dim: int, optional
            The dimension of the copula
        """
        self._dim = dim
        self._name = 'Independent'
        self.init_validate()

    @array_io
    def cdf(self, x: Array, log=False) -> np.ndarray:
        return np.log(x).sum(1) if log else x.prod(1)

    def fit(self, data: np.ndarray, x0: np.ndarray = None, method='mpl', verbose=1, optim_options: dict = None,
            ties='average'):
        print('Fitting not required for Independent Copula')
        return self

    @property
    def params(self):
        return self.dim

    @array_io
    def pdf(self, x: Array, log=False) -> np.ndarray:
        return np.repeat(0 if log else 1, len(x))

    def random(self, n: int, seed: int = None) -> np.ndarray:
        return random_uniform(n, self.dim, seed)

    def summary(self):
        return Summary(self, {})
