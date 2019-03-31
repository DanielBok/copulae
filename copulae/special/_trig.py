import numpy as np

from copulae.special import _machine as M
from copulae.utility import as_array


def angle_restrict_pos_err_e(theta):
    theta = as_array(theta)

    two_pi = 2 * np.pi

    y = 2 * np.floor(theta / two_pi)
    r = theta - y * (2 * two_pi)

    r[r > two_pi] -= two_pi
    r[r < 0] += two_pi

    m = abs(theta) > 0.0625 / M.DBL_EPSILON
    if np.any(m):
        r[m] = np.nan

    return r.item(0) if r.size == 1 else r
