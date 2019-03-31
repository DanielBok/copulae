import numpy as np

from copulae.special import _machine as M
from copulae.special._cheb import cheb_eval
from copulae.special._trig import angle_restrict_pos_err_e
from copulae.types import Numeric
from copulae.utility import as_array

__all__ = ['clausen']


def clausen(x) -> Numeric:
    r"""
    The Clausen function is defined by the following integral,

    .. math::

        Cl_2(x) = - \int_0^x \log(2 \sin(t/2)) dt

    See the `Wikipedia <https://en.wikipedia.org/wiki/Clausen_function>`_ article
    for more information.

    Parameters
    ----------
    x: {array_like, scalar}
        Numeric vector input

    Returns
    -------
    {array_like, scalar}
        Clausen output
    """
    x_cut = M.M_PI * M.DBL_EPSILON

    x = as_array(x, copy=True)
    sgn = as_array(np.ones_like(x))

    m = x < 0
    if np.any(m):
        sgn[m] = -1
        x[m] = -x[m]

    x = as_array(angle_restrict_pos_err_e(x))

    m = x > M.M_PI
    if np.any(m):
        x[m] = (6.28125 - x[m]) + 1.9353071795864769253e-03
        sgn[m] = -sgn[m]

    res = as_array(np.zeros_like(x))

    m = (x < x_cut) & (x != 0)
    if np.any(m):
        res[m] = x[m] * (1 - np.log(x[m]))

    m = (x >= x_cut) & (x != 0)
    if np.any(m):
        constants = [2.142694363766688447e+00,
                     0.723324281221257925e-01,
                     0.101642475021151164e-02,
                     0.3245250328531645e-04,
                     0.133315187571472e-05,
                     0.6213240591653e-07,
                     0.313004135337e-08,
                     0.16635723056e-09,
                     0.919659293e-11,
                     0.52400462e-12,
                     0.3058040e-13,
                     0.18197e-14,
                     0.1100e-15,
                     0.68e-17,
                     0.4e-18]
        res[m] = x[m] * (np.array([cheb_eval(constants, 2.0 * ((e / M.M_PI) ** 2 - 0.5)) for e in x[m]]) - np.log(x[m]))

    res *= sgn

    return res.item(0) if res.size == 1 else res
