from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


class SummaryType(ABC):
    def as_html(self):
        return self._repr_html_()

    @abstractmethod
    def _repr_html_(self) -> str:
        ...

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def __str__(self) -> str:
        ...


class Summary(SummaryType):
    """A general summary to describe the copula instance"""

    def __init__(self, copula, params: Dict[str, Any]):
        self.copula = copula
        self.params = params

    def _repr_html_(self):
        params = []
        for k, v in self.params.items():
            if isinstance(v, (int, float, complex, str)):
                params.append(f"<strong>{k}</strong><span>{v}</span>")
            if isinstance(v, np.ndarray) and v.ndim == 2:  # correlation matrix
                params.append(f"<strong>{k}</strong>" + pd.DataFrame(v).to_html(header=False, index=False))

        param_content = '' if len(params) == 0 else f"""
<div>
    <h3>Parameters</h3>
    {'<br/>'.join(params)}
</div>
"""

        return f"""
<div>
    <h2>{self.copula.name} Copula Summary</h2>
    <div>{self.copula.name} Copula with {self.copula.dim} dimensions</div>
    <hr/>
    {param_content}
</div>
"""

    def __str__(self):
        msg = [
            f"{self.copula.name} Copula Summary",
            "=" * 80,
            f"{self.copula.name} Copula with {self.copula.dim} dimensions",
            "\n",
        ]

        if len(self.params) > 0:
            msg.extend(["Parameters", "-" * 80])

            for k, v in self.params.items():
                if isinstance(v, (int, float, complex, str)):
                    msg.extend([f"{k:^20}: {v}", ''])
                if isinstance(v, np.ndarray) and v.ndim == 2:  # correlation matrix
                    msg.extend([f"{k:^20}", pd.DataFrame(v).to_string(header=False, index=False), ''])

        return '\n'.join(msg)
