from numbers import Number
from typing import Collection, Optional, Union

import numpy as np
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

EPSILON = np.finfo('float').eps

Array = Union[pd.Series, np.ndarray, Collection[Number], Collection[Collection[Number]]]

Numeric = Union[Array, Number]

OptNumeric = Optional[Numeric]

Ties = Literal['average', 'min', 'max', 'dense', 'ordinal']
