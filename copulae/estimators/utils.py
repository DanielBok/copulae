from typing import Union
import numpy as np

InitialParam = Union[float, np.ndarray]


def warn_no_convergence():
    print("Warning: Possible convergence problem with copula fitting")
