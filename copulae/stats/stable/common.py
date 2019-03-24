import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

__all__ = ['LARGE_EXP_POW', 'PI', 'PI2', 'PI2E', 'eplus', 'eminus', 'integrate', 'stable_tail', 'xexp']

LARGE_EXP_POW = 708.3964
PI = np.pi
PI2 = np.pi / 2
PI2E = PI2 * (1 - 1e6)


def eplus(x):
    return x + 1e-6 * abs(x)


def eminus(x):
    return x - 1e-6 * abs(x)


def integrate(f, lower: float, upper: float):
    """Helper integration function"""
    return quad(f, lower, upper, limit=1000)[0]


def stable_tail(alpha, log=False):
    if alpha == 0:
        return -np.log(2) if log else 0.5
    elif alpha == 2:
        return -np.inf if log else 0
    else:
        r = gamma(alpha) / np.pi * np.sin(alpha * PI2)
        return np.log(r) if log else r


def xexp(x):
    return 0 if x > LARGE_EXP_POW else (x * np.exp(-x))
