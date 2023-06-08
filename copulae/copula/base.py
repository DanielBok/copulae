from abc import ABC, abstractmethod
from numbers import Number
from typing import Collection, Generic, List, Literal, NamedTuple, Protocol, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from copulae.copula.estimator import EstimationMethod, fit_copula
from copulae.copula.exceptions import InputDataError
from copulae.copula.summary import SummaryType
from copulae.core import pseudo_obs
from copulae.types import Array, Numeric, Ties

__all__ = ["BaseCopula", "CopulaCorrProtocol", "Param", "EstimationMethod", "TailDep"]

Param = TypeVar("Param")


class BaseCopula(ABC, Generic[Param]):
    """
    The base copula object. All implemented copulae should inherit this class as it creates a common API
    such as the :py:meth:`BaseCopula.fit` method.
    """

    def __init__(self, dim: int, name: str, fit_smry: SummaryType = None, columns: List[str] = None):
        self._dim = dim
        self._name = name
        self._columns = columns
        self._fit_smry = fit_smry
        self._bounds: Tuple[Number, Number] = (0, 0)

        assert isinstance(self.dim, int) and self.dim >= 2, 'Copula must have more than 2 dimensions'

    @property
    def bounds(self):
        """
        Gets the bounds for the parameters

        Returns
        -------
        (scalar or array_like, scalar or array_like)
            Lower and upper bound of the copula's parameters
        """
        return self._bounds

    @abstractmethod
    def cdf(self, x: Array, log=False) -> Union[np.ndarray, float]:
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
        ndarray or float
            The CDF of the random variates
        """
        raise NotImplementedError

    @property
    def dim(self):
        """Number of dimensions in copula"""
        return self._dim

    def fit(self, data: Union[pd.DataFrame, np.ndarray], x0: Union[Collection[float], np.ndarray] = None,
            method: EstimationMethod = 'ml', optim_options: dict = None, ties: Ties = 'average', verbose=1,
            to_pobs=True, scale=1.0):
        """
        Fit the copula with specified data

        Parameters
        ----------
        data: ndarray
            Array of data used to fit copula. Usually, data should be the pseudo observations

        x0: ndarray
            Initial starting point. If value is None, best starting point will be estimated

        method: { 'ml', 'irho', 'itau' }, optional
            Method of fitting. Supported methods are: 'ml' - Maximum Likelihood, 'irho' - Inverse Spearman Rho,
            'itau' - Inverse Kendall Tau

        optim_options: dict, optional
            Keyword arguments to pass into :func:`scipy.optimize.minimize`

        ties: { 'average', 'min', 'max', 'dense', 'ordinal' }, optional
            Specifies how ranks should be computed if there are ties in any of the coordinate samples. This is
            effective only if the data has not been converted to its pseudo observations form

        verbose: int, optional
            Log level for the estimator. The higher the number, the more verbose it is. 0 prints nothing.

        to_pobs: bool
            If True, casts the input data along the column axis to uniform marginals (i.e. convert variables to
            values between [0, 1]). Set this to False if the input data are already uniform marginals.

        scale: float
            Amount to scale the objective function value of the numerical optimizer. This is helpful in
            achieving higher accuracy as it increases the sensitivity of the optimizer. The downside is
            that the optimizer could likely run longer as a result. Defaults to 1.

        See Also
        --------
        :py:meth:`~copulae.BaseCopula.pobs`
            The psuedo-observation method converts non-uniform data into uniform marginals.

        :code:`scipy.optimize.minimize`
            the `scipy minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_ function use for optimization
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            data = np.asarray(data)

        data = self.pobs(data, ties) if to_pobs else data

        if data.ndim != 2:
            raise InputDataError('Data must be a matrix of dimension (n x d)')
        elif self.dim != data.shape[1]:
            raise InputDataError('Dimension of data does not match copula')
        elif np.any((data < 0) | (data > 1)):
            raise InputDataError('Input data must be between [0, 1] (marginals). Set to_pobs=True if you want '
                                 'copulae to convert this automatically for you')

        x0 = np.asarray(x0) if x0 is not None and not isinstance(x0, np.ndarray) and isinstance(x0, Collection) else x0
        self._fit_smry = fit_copula(self, data, x0, method, verbose, optim_options, scale)

        if isinstance(data, pd.DataFrame):
            self._columns = list(data.columns)
            print(data.describe())

        return self

    def log_lik(self, data: Union[np.ndarray, pd.DataFrame], *, to_pobs=True, ties: Ties = 'average') -> float:
        r"""
         Returns the log likelihood (LL) of the copula given the data.

        The greater the LL (closer to :math:`\infty`) the better.

        Parameters
        ----------
        data
            Data set used to calculate the log likelihood

        to_pobs
            If True, converts the data input to pseudo observations.

        ties
            Specifies how ranks should be computed if there are ties in any of the coordinate samples. This is
            effective only if :code:`to_pobs` is True.

        Returns
        -------
        float
            Log Likelihood
        """
        data = self.pobs(data, ties) if to_pobs else data
        return self.pdf(data, log=True).sum()

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def params(self) -> Param:
        """The parameter set which describes the copula"""
        raise NotImplementedError

    @params.setter
    @abstractmethod
    def params(self, params: Numeric):
        raise NotImplementedError

    @abstractmethod
    def pdf(self, u: Array, log=False) -> Union[np.ndarray, float]:
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
    def pobs(data, ties: Ties = 'average'):
        """
        Compute the pseudo-observations for the given data matrix

        Parameters
        ----------
        data: { array_like, DataFrame }
            Random variates to be converted to pseudo-observations

        ties: { 'average', 'min', 'max', 'dense', 'ordinal' }, optional
            Specifies how ranks should be computed if there are ties in any of the coordinate samples

        Returns
        -------
        ndarray
            matrix or vector of the same dimension as `data` containing the pseudo observations

        See Also
        --------
        :py:func:`~copulae.core.misc.pseudo_obs`
            The pseudo-observations function
        """
        return pseudo_obs(data, ties)

    def random(self, n: int, seed: int = None) -> Union[pd.DataFrame, np.ndarray]:
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
        pd.DataFrame or ndarray
            array of generated observations
        """
        raise NotImplementedError

    def summary(self, category: Literal['copula', 'fit'] = 'copula') -> SummaryType:
        """Constructs the summary information about the copula"""
        raise NotImplementedError


class CopulaCorrProtocol(Protocol):
    """Additional protocol methods for the copula related to the correlation structure"""

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

    @property
    @abstractmethod
    def rho(self):
        """
        Computes the Spearman's Rho for bivariate copulae

        Returns
        -------
        ndarray
            Spearman's Rho
        """
        raise NotImplementedError

    @property
    @abstractmethod
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
