from numbers import Number
from typing import Collection, Optional, Union

import numpy as np
import pandas as pd

EPSILON = np.finfo('float').eps

Array = Union[pd.Series, np.ndarray, Collection[Number]]

Numeric = Union[Array, Number]

OptNumeric = Optional[Numeric]
