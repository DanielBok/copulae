from numbers import Number
from typing import Collection, Optional, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

EPSILON = np.finfo('float').eps

Vector = Union[np.ndarray, Collection[Number], pd.Series]
Matrix = Union[np.ndarray, Collection[Collection[Number]], pd.DataFrame]
Array = Union[Vector, Matrix]

Numeric = Union[Array, Number]

OptNumeric = Optional[Numeric]

Ties = Literal['average', 'min', 'max', 'dense', 'ordinal']
