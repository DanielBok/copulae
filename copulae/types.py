from typing import Iterable, List, Union, Optional
import numpy as np

Array = Union[List[float], np.ndarray]
Numeric = Union[np.ndarray, Iterable, int, float]
OptNumeric = Optional[Numeric]
