from typing import Collection, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn

from copulae.copula import BaseCopula
from copulae.core import pseudo_obs
from copulae.types import Array, Ties
from copulae.utility.annotations import *
from .estimators import expectation_maximization, gradient_descent, k_means
from .estimators.em import Criteria
from .exception import GMCFitMethodError, GMCNotFittedError, GMCParamMismatchError
from .parameter import GMCParam, GMCParamDict
from .random import random_gmcm
from .summary import Summary

__all__ = ['GaussianMixtureCopula', 'EstimateMethod']

EstimateMethod = Literal["pem", "sgd", "kmeans"]


class GaussianMixtureCopula(BaseCopula[GMCParam]):
    r"""
    The Gaussian Mixture Copula (GMC).

    A Gaussian Copula has many normal marginal densities bound together by a single multivariate and uni-model
    Gaussian density. However, if a dataset has multiple modes (peaks) with different dependence structure, the
    applicability of the Gaussian Copula gets severely limited. A Gaussian Mixture Copula on the other hand
    allows modeling of data with many modes (peaks).

    The GMC's dependence structure is obtained from a Gaussian Mixture Model (GMM). For a GMC with :math:`M`
    components and :math:`d` dimensions, the density (PDF) is given by

    .. math::

        \phi &= \sum_i^M w_i \phi_i (x_1, x_2, \dots, x_d; \theta_i)

        &\text{s.t.}

        &\sum^M_i w_i = 1

        &0 \leq w_i \leq 1 \quad \forall i \in [1, \dots, M]

        &\text{where}

        w_i &= \text{weight of the marginal density}

        \phi_i &= \text{marginal density}

        \theta_i &= \text{parameters of the Gaussian marginal}

    The Gaussian Mixture Copula is thus given by

    .. math::

        C(u_1, u_2, \dots, u_d; \Theta) &=
            \frac{\phi(\Phi_1^{-1}(u_1), \Phi_2^{-1}(u_2), \dots, \Phi_d^{-1}(u_d); \Theta}
            {\phi_1(\Phi_1^{-1}(u_1)) \cdot \phi_2(\Phi_2^{-1}(u_2)) \cdots \phi_d(\Phi_1^{-1}(u_d))}

        & \text{where}

        \Phi_i &= \text{Inverse function of GMM marginal CDF}

        \Theta &= (w_i, \theta_i) \forall i \in [1, \dots, M]
    """

    def __init__(self, n_clusters: int, ndim: int,
                 params: Union[GMCParam, np.ndarray, Collection[float], GMCParamDict] = None):
        """
        Creates a Gaussian Mixture Copula

        Parameters
        ----------
        n_clusters : int
            The number of clusters

        ndim : int
            The number of dimension for each Gaussian component

        params : GMCParam, optional
            The initial parameter for the model
        """
        super().__init__(ndim, 'GaussianMixtureCopula')
        self._clusters = n_clusters
        self._dim = ndim
        self.params = params
        self._fit_details = {"method": None}

    @property
    def bounds(self):
        """Bounds is not meaningful for :class:`GaussianMixtureCopula`"""
        return NotImplemented

    @validate_data_dim({"x": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def cdf(self, x: Array, log=False):
        """
        Returns the cumulative distribution function (CDF) of the copulae.

        The CDF is also the probability of a RV being less or equal to the value specified. Equivalent to the 'p'
        generic function in R.

        Parameters
        ----------
        x
            Vector or matrix of the observed data. This vector must be (n x d) where `d` is the dimension of
            the copula

        log
            If True, the log of the probability is returned

        Returns
        -------
        np.ndarray or float
            The CDF of the random variates
        """
        if self.params is None:
            raise GMCNotFittedError

        out = np.zeros(len(x))
        for prob, mean, cov in self.params:
            out += prob * (mvn.logcdf(x, mean, cov) if log else mvn.cdf(x, mean, cov))

        return out

    @property
    def clusters(self):
        """Number of clusters in the :class:`GaussianMixtureCopula`"""
        return self._clusters

    @property
    def dim(self):
        """Number of dimensions for each copula in the :class:`GaussianMixtureCopula`"""
        return super().dim

    def fit(self, data: Union[pd.DataFrame, np.ndarray], x0: Union[Collection[float], np.ndarray, GMCParam] = None,
            method: EstimateMethod = 'pem', optim_options: dict = None, ties: Ties = 'average', verbose=1,
            max_iter=3000, criteria: Criteria = 'GMCM', eps=1e-4):
        """
        Fit the copula with specified data

        Parameters
        ----------
        data
            Array of data used to fit copula. Usually, data should not be pseudo observations as this will
            skew the model parameters

        x0
            Initial starting point. If value is None, best starting point will be estimated

        method
            Method of fitting. Supported methods are: 'pem' - Expectation Maximization with pseudo log-likelihood,
            'kmeans' - K-means, 'sgd' - stochastic gradient descent

        optim_options : dict, optional
            Keyword arguments to pass into scipy.optimize.minimize. Only applicable for gradient-descent
            optimizations

        ties : { 'average', 'min', 'max', 'dense', 'ordinal' }, optional
            Specifies how ranks should be computed if there are ties in any of the coordinate samples. This is
            effective only if the data has not been converted to its pseudo observations form

        verbose:
            Log level for the estimator. The higher the number, the more verbose it is. 0 prints nothing.

        max_iter : int
            Maximum number of iterations

        criteria : { 'GMCM', 'GMM', 'Li' }
            The stopping criteria. Only applicable for Expectation Maximization (EM).  'GMCM' uses the absolute
            difference between the current and last based off the GMCM log likelihood, 'GMM' uses the absolute
            difference between the current and last based off the GMM log likelihood and 'Li' uses the stopping
            criteria defined by Li et. al. (2011)

        eps : float
            The epsilon value for which any absolute delta will mean that the model has converged

        Notes
        -----
        Maximizing the exact likelihood of GMCM is technically intractable using expectation maximization. The
        pseudo-likelihood

        See Also
        --------
        :code:`scipy.optimize.minimize`: the `scipy minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_ function use for optimization
        """
        method = method.lower()

        if x0 is None:
            self._fit_smry = k_means(data, self.clusters, self.dim)
            self.params = self._fit_smry.best_params
        elif isinstance(x0, GMCParam):
            self.params = x0
        else:
            self.params = GMCParam.from_vector(x0, self.clusters, self.dim)

        if method == 'pem':
            u = pseudo_obs(data, ties)
            self._fit_smry = expectation_maximization(u, self.params, max_iter, criteria, verbose, eps)

        elif method == 'sgd':
            u = pseudo_obs(data, ties)
            self._fit_smry = gradient_descent(u, self.params, max_iter=max_iter, **(optim_options or {}))
        elif method == 'kmeans':
            if x0 is not None:  # otherwise already fitted by default
                self._fit_smry = k_means(data, self.clusters, self.dim)
        else:
            raise GMCFitMethodError(f"Invalid method: {method}. Use one of (kmeans, pem, sgd)")

        self.params = self._fit_smry.best_params
        self._fit_details["method"] = method
        return self

    @property
    def params(self):
        """
        The parameter set which describes the copula

        Returns
        -------
        GMCParam
            The model parameter
        """
        return self._param

    @params.setter
    def params(self, params: Optional[Union[GMCParam, np.ndarray, Collection, GMCParamDict]]):
        if params is None:
            self._param = None
        elif isinstance(params, dict):
            self._param = GMCParam.from_dict(params)
        elif isinstance(params, GMCParam):
            if params.n_dim != self.dim or params.n_clusters != self.clusters:
                raise GMCParamMismatchError(f"Expected {self.clusters} clusters and {self.dim} dimensions "
                                            f"but got {params.n_clusters} and {params.n_dim} respectively instead")
            self._param = params
        elif np.all(np.isreal(params)):
            self._param = GMCParam.from_vector(params, self.clusters, self.dim)
        else:
            raise GMCParamMismatchError("Unsupported params type for GaussianMixtureCopula")

    @validate_data_dim({"x": [1, 2]})
    @shape_first_input_to_cop_dim
    @squeeze_output
    def pdf(self, x: Array, log=False):
        """
        Returns the probability distribution function (PDF) of the copulae.

        The PDF is also the density of the RV at for the particular distribution. Equivalent to the 'd' generic function
        in R.

        Parameters
        ----------
        x
            Vector or matrix of observed data

        log
            If True, the density 'd' is given as log(d)

        Returns
        -------
        np.ndarray or float
            The density (PDF) of the RV
        """
        if self.params is None:
            raise GMCNotFittedError

        out = np.zeros(len(x))
        for prob, mean, cov in self.params:
            out += prob * (mvn.logpdf(x, mean, cov) if log else mvn.pdf(x, mean, cov))

        return out

    @cast_output
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
        np.ndarray
            array of generated observations
        """
        if seed is not None:
            np.random.seed(seed)

        return random_gmcm(n, self.params)

    @select_summary
    def summary(self, category: Literal['copula', 'fit'] = 'copula'):
        """Constructs the summary information about the copula"""
        return Summary(self.params, self._fit_details)
