from typing import Iterable, Union, Optional

import numpy as np

Array = Union[Iterable[float], np.ndarray]
Numeric = Union[np.ndarray, Iterable, int, float]
OptNumeric = Optional[Numeric]
