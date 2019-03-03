from typing import Iterable, Union, Optional

import numpy as np

Array = Union[Iterable[float], Iterable[int], np.ndarray]
Numeric = Union[np.ndarray, Iterable[Union[float, int]], int, float]
OptNumeric = Optional[Numeric]
