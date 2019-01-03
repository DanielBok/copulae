from abc import ABC, abstractmethod
from collections import namedtuple
from typing import NamedTuple

import numpy as np

from copulae.copula.abstract import AbstractCopula
from copulae.estimators import CopulaEstimator, __estimator_params_docs__
from copulae.math_tools.misc import pseudo_obs
from copulae.types import Array
from copulae.utils import format_docstring

TailDep = NamedTuple('Lambda', lower=np.ndarray, upper=np.ndarray)


class BaseCopula(AbstractCopula, ABC):
    """
    The base copula object. All implemented copulae should inherit this class as it creates a common API for the fit
    method.
    """

    def __init__(self, dim: int, name: str):
        """
        Creates a new abstract Copula.

        :param dim: integer (greater than 1)
            The dimension of the copulae

        :param name: string
            Default copulae. one of Gaussian, Student (T)
        """
        super().__init__(dim, name)

    def _check_dimension(self, x: Array):
        x = np.array(x)
        if len(x.shape) != 2:
            raise ValueError("Data passed in must be a matrix where rows represent the number of data and columns "
                             "represent the dimension of the copula")

        dim = x.shape[1]
        if dim != self.dim:
            raise ValueError(f"Expected vector of dimension {self.dim}, get matrix of dimension {dim}")

    @format_docstring(params_doc=__estimator_params_docs__)
    def fit(self, data: np.ndarray, x0: np.ndarray = None, method='mpl', est_var=False, verbose=1,
            optim_options: dict = None):
        """
        Fit the copula with specified data

        {params_doc}
        :return: (float, float)
            Tuple containing parameters of the Gaussian Copula
        """
        self._check_dimension(data)
        data = self.pobs(data)

        CopulaEstimator(self, data, x0=x0, method=method, est_var=est_var, verbose=verbose, optim_options=optim_options)

    @property
    def tau(self):
        """
        Computes the Kendall's Tau for bivariate copulas

        :return: numpy array
            Kendall's Tau
        """
        raise NotImplementedError

    @property
    def rho(self):
        """
        Computes the Spearman's Rho for bivariate copulas
        :return: numpy array
            Spearman's Rho
        """
        raise NotImplementedError

    @property
    def lambda_(self) -> TailDep:
        """
        Computes the tail dependence index for bivariate copulas
        :return: named tuple
            Tail dependence index (lambda) with keys
                lower: numpy array
                upper: numpy array
        """
        raise NotImplementedError

    @staticmethod
    def _lambda_(lower: np.array, upper: np.array) -> TailDep:
        Lambda = namedtuple('lambda', ['lower', 'upper'])
        return Lambda(lower, upper)

    def itau(self, tau: Array):
        """
        Computes the inverse Kendall's Tau

        The inverse tau is sometimes called the calibration function. Together with the inverse rho, it helps determine
        ("calibrate") the copula parameter (which must be 1-dimensional) given the values of Kendall's Tau and
        Spearman's Rho

        :param tau: numpy array
            numerical values of Kendall's tau in [-1, 1]
        :return:
        """
        raise NotImplementedError

    def irho(self, rho: Array):
        """
        Computes the inverse Spearman's Rho

        The inverse tau is sometimes called the calibration function. Together with the inverse rho, it helps determine
        ("calibrate") the copula parameter (which must be 1-dimensional) given the values of Kendall's Tau and
        Spearman's Rho

        :param rho: numpy array
            numerical values of Spearman's rho in [-1, 1].
        :return:
        """
        raise NotImplementedError

    def cdf(self, x: Array, log=False) -> np.ndarray:
        """
        Returns the cumulative distribution function (CDF) of the copulae. The CDF is also the probability of a RV being
        less or equal to the value specified

        :param x: numpy array of size (n x d)
            Vector or matrix of observed data
        :param log: bool
            If True, the density of d are given as log(p)
        :return: numpy array
            The probability (CDF) of the RV
        """
        raise NotImplementedError

    def pdf(self, x: Array, log=False) -> np.ndarray:
        """
        Returns the probability distribution function (PDF) of the copulae. The PDF is also the density of the RV at for
        the particular distribution

        :param x: numpy array of size (n x d)
            Vector or matrix of observed data
        :param log: bool
            If True, the density of d are given as log(d)
        :return: numpy array
            The density (PDF) of the RV
        """
        raise NotImplementedError

    def ppf(self, x: Array) -> np.ndarray:
        """
        Returns the percent point function (inverse of cdf) of the given RV. The ppf is also the quantile that the
        RV belongs to.

        :param x: numpy array (of size d)
            Values to compute ppf

        :return: numpy array
            The ppf of the RV
        """
        raise NotImplementedError

    @property
    def params(self):
        """
        The parameter set which describes the copula

        :return: numpy array:
            parameters of the copulae
        """
        raise NotImplementedError

    @params.setter
    def params(self, params: Array):
        """
        Sets the parameter which describes the copula
        :param params: numpy array:
            parameters of the copulae
        """
        raise NotImplementedError

    def log_lik(self, data: np.ndarray = None) -> float:
        """
        Returns the log likelihood (NLL) of the copula.

        The greater the NLL (closer to 0 from -inf) the better. If copula is fitted and data is not supplied, return
        the fitted NLL. If data is given, calculate NLL given data. Otherwise error.

        :param data: numpy array
            Calculate NLL given new data set
        :return: float
            NLL
        """
        if data is not None:
            return self.pdf(data, log=True).sum()

        if self.fit_stats is None:
            raise AttributeError("Unable to give log likelihood of fitted copula as copula is not fitted yet")

        return self.fit_stats.log_lik

    def concentration_down(self, x):
        """
        Returns the theoretical lower concentration function.

        Parameters
        ----------
        x : float (between 0 and 0.5)
        """
        if x > 0.5 or x < 0:
            raise ValueError("The argument must be included between 0 and 0.5.")
        return self.cdf([x, x]) / x

    def concentration_up(self, x):
        """
        Returns the theoretical upper concentration function.

        Parameters
        ----------
        x : float (between 0.5 and 1)
        """
        if x < 0.5 or x > 1:
            raise ValueError("The argument must be included between 0.5 and 1.")
        return (1. - 2 * x + self.cdf([x, x])) / (1. - x)

    def concentration_function(self, x):
        """
        Returns the theoretical concentration function.

        Parameters
        ----------
        x : float (between 0 and 1)
        """
        if x < 0 or x > 1:
            raise ValueError("The argument must be included between 0 and 1.")
        if x < 0.5:
            return self.concentration_down(x)
        return self.concentration_up(x)

    @staticmethod
    def pobs(data: np.ndarray, ties='average'):
        """
        Compute the pseudo-observations for the given data matrix

        :param data: numpy array
            n x d-matrix (or d-vector) of random variates to be converted to pseudo-observations

        :param ties: str
            string specifying how ranks should be computed if there are ties in any of the coordinate samples                    The options are 'average', 'min', 'max', 'dense' and 'ordinal'. Passed to scipy.stats.rankdata

        :return: numpy array
            matrix or vector of the same dimension as X containing the pseudo observations
        """
        return pseudo_obs(data, ties)

    def random(self, n: int, seed: int=None):
        """
        Generate random observations for the copula

        :param n: int
            number of observations to be generated
        :param seed: int, optional
            seed for the random generator
        :return: numpy array (n x d)
            array of generated observations
        """

        if self.fit_stats is None:
            raise RuntimeError("Copula must be fitted before it can generate random numbers")

    @abstractmethod
    def __random(self, n: int):
        raise NotImplementedError
