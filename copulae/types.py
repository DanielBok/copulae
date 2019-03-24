from typing import Iterable, Union, Optional

import numpy as np

EPSILON = np.finfo('float').eps

Array = Union[Iterable[float], Iterable[int], np.ndarray]

Numeric = Union[np.ndarray, Iterable, int, float, complex]

OptNumeric = Optional[Numeric]
