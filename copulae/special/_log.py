import numpy as np

from copulae.utility import as_array


def complex_log_e(zr, zi):
    zr, zi = as_array(zr), as_array(zi)
    assert np.alltrue((zr != 0) | (zi != 0))

    ax, ay = np.abs(zr), np.abs(zi)
    mn, mx = np.min([ax, ay], 0), np.max([ax, ay], 0)
    re = np.log(mx) + 0.5 * np.log(1.0 + (mn / mx) * (mn / mx))
    im = np.arctan2(zi, zr)

    if np.size(re) == 1:
        re = float(re)
        im = float(im)
    return re, im
