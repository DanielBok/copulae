import numba as nb


@nb.njit()
def cheb_eval(constants, x: float):  # pragma: no cover
    """Evaluates the Chebyshev series"""
    d, dd = 0.0, 0.0

    for j in range(len(constants) - 1, 0, -1):
        dd, d = d, 2 * x * d - dd + constants[j]

    return x * d - dd + 0.5 * constants[0]
