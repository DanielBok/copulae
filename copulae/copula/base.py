from abc import ABC, abstractmethod
from typing import NamedTuple, Union

import numpy as np

from copulae.copula.abstract import AbstractCopula
from copulae.copula.estimator import CopulaEstimator
from copulae.core import pseudo_obs
from copulae.types import Array, Numeric


class BaseCopula(AbstractCopula, ABC):
    """
    The base copula object. All implemented copulae should inherit this class as it creates a common API for the fit
    method.
    """

    def __init__(self, dim: int, name: str):
        super().__init__(dim, name)

    @abstractmethod
    def cdf(self, x: Array, log=False) -> np.ndarray:
        """
        Returns the cumulative distribution function (CDF) of the copulae.

        The CDF is also the probability of a RV being less or equal to the value specified. Equivalent to the 'p'
        generic function in R.

        Parameters
        ----------
        x: ndarray
            Vector or matrix of the observed data. This vector must be (n x d) where `d` is the dimension of
            the copula

        log: bool
            If True, the log of the probability is returned

        Returns
        -------
        ndarray
            The CDF of the random variates
        """
        raise NotImplementedError

    def fit(self, data: np.ndarray, x0: np.ndarray = None, method='mpl', est_var=False, verbose=1,
            optim_options: dict = None):
        """
        Fit the copula with specified data

        Parameters
        ----------
        data: ndarray
            Array of data used to fit copula. Usually, data should be the pseudo observations

        x0: ndarray
            Initial starting point. If value is None, best starting point will be estimated

        method: { 'ml', 'mpl', 'irho', 'itau' }, optional
            Method of fitting. Supported methods are: 'ml' - Maximum Likelihood, 'mpl' - Maximum Pseudo-likelihood,
            'irho' - Inverse Spearman Rho, 'itau' - Inverse Kendall Tau

        est_var: bool, optional
            If True, estimates variance of the fitted copula.

        verbose: int, optional
            Log level for the estimator. The higher the number, the more verbose it is. 0 prints nothing.

        optim_options: dict, optional
            Keyword arguments to pass into scipy.optimize.minimize

        See Also
        --------
        :code:`scipy.optimize.minimize`: the `scipy minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_ function use for optimization
        """
        data = self.pobs(data)
        if data.ndim != 2:
            raise ValueError('Data must be a matrix of dimension (n x d)')
        elif self.dim != data.shape[1]:
            raise ValueError('Dimension of data does not match copula')

        CopulaEstimator(self, data, x0=x0, method=method, est_var=est_var, verbose=verbose, optim_options=optim_options)

    @abstractmethod
    def drho(self, x=None):
        """
        Computes derivative of Spearman's Rho

        Parameters
        ----------
        x: array_like, optional
            1d vector to compute derivative of Spearman's Rho. If not supplied, will default to copulae parameters

        Returns
        -------
        ndarray
            Derivative of Spearman's Rho
        """
        raise NotImplementedError

    @abstractmethod
    def dtau(self, x=None):
        """
        Computes derivative of Kendall's Tau

        Parameters
        ----------
        x: array_like, optional
            1d vector to compute derivative of Kendall's Tau. If not supplied, will default to copulae parameters

        Returns
        -------
        ndarray:
            Derivative of Kendall's Tau
        """
        raise NotImplementedError

    @abstractmethod
    def irho(self, rho: Array):
        """
        Computes the inverse Spearman's Rho

        The inverse Rho can be used as the calibration function to determine the copula's parameters.

        Parameters
        ----------
        rho: array_like
            numerical values of Spearman's rho between [-1, 1].

        Returns
        -------
        ndarray
            Inverse Spearman Rho values

        See Also
        --------
        :code:`itau`: inverse Kendall's Tau
        """
        raise NotImplementedError

    @abstractmethod
    def itau(self, tau):
        """
        Computes the inverse Kendall's Tau

        The inverse Tau can be used as the calibration function to determine the copula's parameters.

        Parameters
        ----------
        tau: array_like
            numerical values of Spearman's rho between [-1, 1].

        Returns
        -------
        array_like
            Inverse Kendall's Tau values

        See Also
        --------
        :code:`irho`: inverse Spearman's Rho
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def lambda_(self) -> 'TailDep':
        """
        Computes the tail dependence index for bivariate copulae

        Returns
        -------
        TailDep
            Tail dependence index (lambda). This is a NamedTuple with keys `lower` and `upper`.
            Both of which contains either an int, float or ndarray
        """
        raise NotImplementedError

    def log_lik(self, data: np.ndarray) -> float:
        r"""
         Returns the log likelihood (LL) of the copula given the data.

        The greater the LL (closer to :math:`\infty`) the better.

        Parameters
        ----------
        data: ndarray
            Data set used to calculate the log likelihood

        Returns
        -------
        float
            Log Likelihood

        """
        return self.pdf(self.pobs(data), log=True).sum()

    @property
    @abstractmethod
    def params(self):
        """
        The parameter set which describes the copula

        Returns
        -------
        ndarray or tuple or float
            parameters of the copulae. If copula is Archimedean, this will be a float. Otherwise, it could either be
            a tuple or ndarray
        """
        raise NotImplementedError

    @params.setter
    @abstractmethod
    def params(self, params: Numeric):
        raise NotImplementedError

    @property
    def params_bounds(self):
        """
        Gets the bounds for the parameters

        Returns
        -------
        (scalar or array_like, scalar or array_like)
            Lower and upper bound of the copula's parameters
        """
        return self._bounds

    @abstractmethod
    def pdf(self, u: Array, log=False):
        """
        Returns the probability distribution function (PDF) of the copulae.

        The PDF is also the density of the RV at for the particular distribution. Equivalent to the 'd' generic function
        in R.

        Parameters
        ----------
        u: ndarray
            Vector or matrix of observed data

        log: bool, optional
            If True, the density 'd' is given as log(d)

        Returns
        -------
        ndarray or float
            The density (PDF) of the RV
        """
        raise NotImplementedError

    @staticmethod
    def pobs(data, ties='average'):
        """
        Compute the pseudo-observations for the given data matrix

        Parameters
        ----------
        data: {array_like, DataFrame}
            Random variates to be converted to pseudo-observations

        ties: { 'average', 'min', 'max', 'dense', 'ordinal' }, optional
            String specifying how ranks should be computed if there are ties in any of the coordinate samples

        Returns
        -------
        ndarray
            matrix or vector of the same dimension as `data` containing the pseudo observations
        """
        return pseudo_obs(data, ties)

    @abstractmethod
    def random(self, n: int, seed: int = None):
        """
        Generate random observations for the copula

        Parameters
        ----------
        n: int
            Number of observations to be generated

        seed: int, optional
            Seed for the random generator

        Returns
        -------
        ndarray
            array of generated observations
        """
        raise NotImplementedError

    @property
    def rho(self):
        """
        Computes the Spearman's Rho for bivariate copulae

        Returns
        -------
        ndarray
            Spearman's Rho
        """
        raise NotImplementedError

    @abstractmethod
    def summary(self):
        """Constructs the summary information about the copula"""
        raise NotImplementedError

    @property
    def tau(self):
        """
        Computes the Kendall's Tau for bivariate copulae

        Returns
        -------
        ndarray
            Kendall's Tau
        """
        raise NotImplementedError


class TailDep(NamedTuple):
    lower: Union[np.ndarray, float, int]
    upper: Union[np.ndarray, float, int]
