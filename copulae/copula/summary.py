from itertools import zip_longest
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class FitSummary:
    """
    Statistics on the fit of the copula

    Attributes
    ----------
    params: named tuple, numpy array
        parameters of the copula after fitting

    var_est: numpy array
        estimate of the variance

    method: str
        method used to fit the copula

    log_lik: float
        log likelihood value of the copula after fitting

    nsample: int
        number of data points used to fit copula

    setup: dict
        optimizer set up options

    results: dict
        optimization results
    """

    def __init__(self, params: np.ndarray, var_est: np.ndarray, method: str, log_lik: float, nsample: int,
                 setup: Optional[dict] = None, results: Optional[dict] = None):
        self.params = params
        self.var_est = var_est
        self.method = method
        self.log_lik = log_lik
        self.nsample = nsample
        self.setup = setup
        self.results = results

    def as_html(self):
        return self._repr_html_()

    def _repr_html_(self):
        portions = []
        for s, r in zip_longest(self.setup.items(), self.results.items()):
            (sk, sv) = (None, None) if s is None else s
            (rk, rv) = (None, None) if r is None else r
            portions.append('<tr>' + ''.join(f"<td>{i}</td>" for i in (sk, sv, rk, rv)) + '</tr>')

        html = f"""
<div>
    <h3>Fit Summary</h2>
    <hr/>
    <table>
        <tr><th colspan="2">Fit Summary</th></tr>
        <tr><td>Log Likelihood</td><td>{self.log_lik}</td></tr>
        <tr><td>Variance Estimate</td><td>Not Implemented Yet</td></tr>
        <tr><td>Method</td><td>{self.method}</td></tr>
        <tr><td>Data Points</td><td>{self.nsample}</td></tr>
    </table>
    <br/>
    <table>
        <tr><th colspan="2">Optimization Setup</th><th colspan="2">Results</th></tr>
        {''.join(portions)}
    </table>
</div>
        """
        return html

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = [
            f"{'Fit Summary':^80s}",
            '=' * 80,
            f"""
Log. Likelihood      : {self.log_lik}
Variance Estimate    : Not Implemented Yet
Method               : {self.method}
Data Points          : {self.nsample}
""".strip(), '']

        skip_keys = {'final_simplex'}
        for title, dic in [('Optimization Setup', self.setup), ('Results', self.results)]:
            if dic is not None:
                string = "\n".join(f'\t{k:15s}: {v}' for k, v in dic.items() if k not in skip_keys)
                msg.extend([title, '-' * 80, string, ''])

        return '\n'.join(msg)


class Summary:
    def __init__(self, copula, params: Dict[str, Any]):
        self.copula = copula
        self.params = params

    def as_html(self):
        return self._repr_html_()

    def _repr_html_(self):
        params = []
        for k, v in self.params.items():
            if isinstance(v, (int, float, complex, str)):
                params.append(f"<strong>{k}</strong><span>{v}</span>")
            if isinstance(v, np.ndarray) and v.ndim == 2:  # correlation matrix
                params.append(f"<strong>{k}</strong>" + pd.DataFrame(v).to_html(header=False, index=False))

        fit_smry = self.copula.fit_smry
        fit_smry = fit_smry.as_html() if fit_smry else ""

        html = f"""
<div>
    <h2>{self.copula.name} Copula Summary</h2>
    <div>{self.copula.name} Copula with {self.copula.dim} dimensions</div>
    <hr/>
    <div>
        <h3>Parameters</h3>
        {'<br/>'.join(params)}
    </div>
    {fit_smry}
</div>
"""
        return html

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        msg = [
            f"{self.copula.name} Copula Summary",
            "=" * 80,
            f"{self.copula.name} Copula with {self.copula.dim} dimensions",
            "\n",
            "Parameters",
            "-" * 80,
        ]

        for k, v in self.params.items():
            if isinstance(v, (int, float, complex, str)):
                msg.extend([f"{k:^20}: {v}", ''])
            if isinstance(v, np.ndarray) and v.ndim == 2:  # correlation matrix
                msg.extend([f"{k:^20}", pd.DataFrame(v).to_string(header=False, index=False), ''])

        fit_smry = self.copula.fit_smry
        fit_smry = str(fit_smry) if fit_smry else ""
        msg.extend(['\n', fit_smry])

        return '\n'.join(msg)
