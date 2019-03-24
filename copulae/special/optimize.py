import numpy as np
from scipy.optimize import brentq

__all__ = ['find_root']


def find_root(f, a, b, delta=1e-06, max_iter=1000):
    """
    Root finding algorithm. Iteratively expands the lower and upper bound if there are no
    roots in between them.

    Internally, it uses the brentq algorithm to find the root

    Parameters
    ----------
    f: callable
        Callable function

    a: float
        Lower bound

    b: float
        Upper bound

    delta: float
        Minimum expansion

    max_iter: int
        Maximum number of iterations to expand bounds

    Returns
    -------
    float
        value at root

    See Also
    --------
    :code:`scipy.minimize.brentq`: Brent's method for root finding

    Examples
    --------

    >>> import numpy as np
    >>> from copulae.special.optimize import find_root

    >>> find_root(lambda x: (84.5 - x ** 3) / (x ** 2 + 1), 4, 6)  # some random guesses (4, 6)

    >>> find_root(np.exp, -10, 10)  # no roots!
    """
    assert callable(f), "`f` must be a callable function"

    if a > b:
        a, b = b, a

    for _ in range(max_iter):
        assert np.isfinite(a) and np.isfinite(b), 'Could not find a root between intervals specified.'

        if f(a) * f(b) >= 0:
            step = 0.01 * np.abs([a, b])
            step = np.max([step, np.full_like(step, delta)], 0)

            a -= step[0]
            b += step[1]
        else:
            break

    assert np.isfinite(a) and np.isfinite(b) and f(a) * f(b) < 0, 'Could not find a root between intervals specified.'

    return brentq(f, a, b, full_output=False)
