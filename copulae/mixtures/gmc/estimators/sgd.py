import numpy as np
from scipy.optimize import OptimizeResult, minimize

from copulae.mixtures.gmc.loglik import gmcm_log_likelihood
from copulae.mixtures.gmc.parameter import GMCParam
from .summary import FitSummary

__all__ = ['gradient_descent']


def gradient_descent(u: np.ndarray, param: GMCParam, **kwargs):
    """
    Uses gradient descent to determine the optimal GMCParam

    Parameters
    ----------
    u : np.ndarray
        Pseudo observations
    param : GMCParam
        GMC model parameters
    kwargs :
        Other arguments to be passed into the minimize function
    """
    f = create_objective_function(u, param.n_clusters, param.n_dim)
    method = kwargs.pop("method", "nelder-mead")
    max_iter = kwargs.pop("max_iter", 3000)
    disp = kwargs.pop("disp", False)
    kwargs["options"] = kwargs.pop("options", get_default_option(method, disp, max_iter))
    kwargs.pop('args', ())  # drop args cause it's not needed

    optimal: OptimizeResult = minimize(f, param.to_vector(), args=(), method=method, **kwargs)

    return FitSummary(GMCParam.from_vector(optimal.x, param.n_clusters, param.n_dim),
                      optimal.success, 'sgd', len(u), {"algorithm": method, "Fn. Value": optimal.fun})


def create_objective_function(u: np.ndarray, m: int, d: int):
    def f(param: np.ndarray):
        p = GMCParam.from_vector(param, m, d)
        p.prob /= p.prob.sum()
        return -gmcm_log_likelihood(u, p)

    return f


def get_default_option(method: str, disp: bool, max_iter: int):
    method = method.lower()

    if method == 'nelder-mead':
        return {
            'maxiter': max_iter,
            'disp': disp,
            'xatol': 1e-4,
            'fatol': 1e-4,
        }
    elif method == 'bfgs':
        return {
            'maxiter': max_iter,
            'disp': disp,
            'gtol': 1e-4,
        }
    elif method == 'slsqp':
        return {
            'maxiter': max_iter,
            'ftol': 1e-06,
            'iprint': 1,
            'disp': disp,
            'eps': 1.5e-8
        }
    elif method == 'cobyla':
        return {
            'maxiter': max_iter,
            'rhobeg': 1.0,
            'disp': disp,
            'catol': 0.0002
        }
    elif method == 'trust-constr':
        return {
            'maxiter': max_iter,
            'disp': disp
        }
    else:
        return {}
