from typing import List, Literal
from warnings import warn

import numpy as np
from scipy.stats import multivariate_normal as mvn

from copulae.mixtures.gmc.loglik import gmcm_log_likelihood, gmm_log_likelihood
from copulae.mixtures.gmc.marginals import gmm_marginal_ppf
from copulae.mixtures.gmc.parameter import GMCParam
from .exceptions import FitException, InvalidStoppingCriteria
from .summary import FitSummary

Criteria = Literal["GMCM", "GMM", "Li"]


def expectation_maximization(u: np.ndarray, param: GMCParam, max_iter=3000, criteria: Criteria = 'GMCM', verbose=1,
                             eps=1e-4):
    """
    Executes a pseudo expectation maximization algorithm to obtain the best model parameters.

    The algorithm is very sensitive to the starting parameters. Thus, choose the initial parameters very carefully.

    Parameters
    ----------
    u : np.ndarray
        Pseudo observations

    param : GMCParam
        Initial model parameters

    max_iter : int
        Maximum number of iterations

    criteria : { 'GMCM', 'GMM', 'Li' }
        The stopping criteria. 'GMCM' uses the absolute difference between the current and last based off
        the GMCM log likelihood, 'GMM' uses the absolute difference between the current and last based off
        the GMM log likelihood and 'Li' uses the stopping criteria defined by Li et. al. (2011)

    eps : float
        The epsilon value for which any absolute delta will mean that the model has converged
    """
    q = gmm_marginal_ppf(u, param)
    log_lik = LogLik(criteria, eps)
    log_lik.set(
        gmm=gmm_log_likelihood(q, param),
        gmcm=gmcm_log_likelihood(u, param),
        param=param
    )  # set initial state

    for _ in range(max_iter):
        kappa = e_step(q, param)
        if any(kappa.sum(0) == 0):
            fouls = tuple(i + 1 for i, is_zero in enumerate(kappa.sum(0) == 0) if is_zero)
            raise FitException(f"Could not obtain expectation estimates from {fouls}. All posterior "
                               f"probabilities are zero. Try another start estimate or fewer components.")

        param = m_step(q, kappa)

        log_lik.set(gmm=gmm_log_likelihood(q, param),
                    gmcm=gmcm_log_likelihood(u, param),
                    param=param)
        q = gmm_marginal_ppf(u, param)

        if log_lik.has_converged:
            break
    else:
        if verbose:
            warn('Max iterations reached')

    return FitSummary(log_lik.best_param, log_lik.has_converged, 'pem', len(u),
                      {'Log. Likelihood': log_lik.best_log_lik, 'Criteria': criteria})


def e_step(q: np.ndarray, param: GMCParam):
    kappa = np.empty((len(q), param.n_clusters))

    for i, (p, m, s) in enumerate(zip(param.prob, param.means, param.covs)):
        kappa[:, i] = p * mvn.pdf(q, m, s, allow_singular=True)

    kappa /= kappa.sum(1)[:, None]
    kappa[np.isnan(kappa)] = 0

    return kappa


def m_step(q: np.ndarray, kappa: np.ndarray):
    _, d = q.shape
    _, m = kappa.shape
    prob = kappa.sum(0) / kappa.sum()
    means = np.array([(q * k[:, None]).sum(0) / k.sum() for k in kappa.T])

    # weighted covariance
    wt = kappa / kappa.sum(0)  # weights
    center = np.array([(q * w[:, None]).sum(0) for w in wt.T])
    wx = np.array([w[:, None] * (q - c) for c, w in zip(center, np.sqrt(wt).T)])
    covs = np.array([x.T @ x for x in wx])

    return GMCParam(m, d, prob, means, covs)


class LogLik:
    def __init__(self, criteria: Criteria = 'GMCM', eps=1e-4):
        """
        LogLik class houses the logic to determine the best model parameters and convergence criteria

        Parameters
        ----------
        criteria : { 'GMCM', 'GMM', 'Li' }
            The stopping criteria. 'GMCM' uses the absolute difference between the current and last based off
            the GMCM log likelihood, 'GMM' uses the absolute difference between the current and last based off
            the GMM log likelihood and 'Li' uses the stopping criteria defined by Li et. al. (2011)

        eps : float
            The epsilon value for which any absolute delta will mean that the model has converged
        """
        self.params: List[GMCParam] = []
        self._gmm: List[float] = []
        self._gmcm: List[float] = []

        self.eps = eps
        self.criteria = criteria.upper()
        self.count = 0
        if self.criteria not in ('GMCM', 'GMM', 'LI'):
            raise InvalidStoppingCriteria(f"Unknown stopping criteria: '{criteria}'. "
                                          f"Use one of {('GMCM', 'GMM', 'LI')}")

    @property
    def best_param(self):
        if self.criteria in ('GMCM', 'LI'):
            return self.params[int(np.argmax(self._gmcm))]
        else:
            return self.params[-1]

    def set(self, *, gmm: float = None, gmcm: float = None, param: GMCParam = None):
        self.count += 1
        if gmm is not None:
            self._gmm.append(gmm)

        if gmcm is not None:
            self._gmcm.append(gmcm)

        if param is not None:
            self.params.append(param)

    @property
    def has_converged(self):
        if self.criteria == 'GMCM':
            return abs(self._gmcm[-1] - self._gmcm[-2]) < self.eps
        elif self.criteria == 'GMM':
            return abs(self._gmm[-1] - self._gmm[-2]) < self.eps
        else:  # LI
            return abs(self._gmm[-1] - self._gmcm[-2]) < self.eps

    @property
    def best_log_lik(self):
        return self._gmcm[-1]
